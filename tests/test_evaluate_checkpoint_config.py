"""Regression test for a real bug found 2026-08-14 while building an oracle-agreement
metric on a branch: evaluate.py's run_evaluation_suite never passed estimator_type,
intervention_type, or noise_scale to the FederatedCausalEnv it builds for frozen
evaluation -- every trace silently used FederatedCausalEnv's defaults
(estimator_type="analytic", intervention_type="soft_shift") and SCMConfig's default
noise_scale=1.0, regardless of what the checkpoint was actually trained with. Fixed by
saving these in the checkpoint dict (train.py) and reconstructing them in
evaluate_checkpoint. This test exercises that reconstruction directly against a
hand-built checkpoint with non-default values, so a regression would be caught even
without a full training run.
"""
import os
import pickle
import tempfile

import jax
import jax.numpy as jnp
import haiku as hk

from src.evaluate import evaluate_checkpoint
from src.marl.ppo_agent import IPPOActor


def _make_fake_checkpoint(d=4, **overrides):
    def actor_fwd(obs):
        return IPPOActor(d=d)(obs)
    actor_trans = hk.without_apply_rng(hk.transform(actor_fwd))
    obs_dim = 3 * d * d + 1 + d * d  # matches obs_feedback=True layout
    dummy_obs = jnp.zeros((1, obs_dim))
    params = actor_trans.init(jax.random.PRNGKey(0), dummy_obs)

    ckpt = {
        "actor_list": [params, params],
        "critic_list": [params, params],  # unused by evaluate_checkpoint's actor-only path
        "use_rnn": False,
        "use_inductive_graph_head": False,
        "d": d,
        "estimator_type": "analytic",
        "intervention_type": "hard",
        "noise_scale": 0.1,
        "mechanism_type": "LINEAR",
        "K": 2,
        "sample_count": 20,  # small, for a fast test
        "obs_feedback": True,
        "avici_max_context": 400,
    }
    ckpt.update(overrides)
    return ckpt


def test_evaluate_checkpoint_reconstructs_non_default_intervention_and_estimator_type():
    ckpt = _make_fake_checkpoint(intervention_type="hard", estimator_type="analytic", noise_scale=0.1)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pkl")
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

        trace = evaluate_checkpoint(ckpt_path=path, temperature=0.0, seed=0)

    # The whole point of the fix: metadata must reflect what was actually trained, not
    # FederatedCausalEnv's silent defaults (which would show "soft_shift").
    assert trace["metadata"]["intervention_type"] == "hard"
    assert trace["metadata"]["estimator_type"] == "analytic"


def test_evaluate_checkpoint_defaults_are_a_documented_compatibility_shim_not_silent():
    """Checkpoints saved before this fix lack these keys entirely -- evaluate_checkpoint
    must still run (backward compatibility) using the same fallback values the old code
    implicitly used, and those fallbacks must be visible in the returned trace's metadata
    rather than silently disappearing, so a caller can tell an old checkpoint was
    evaluated under assumed (not recovered) settings."""
    ckpt = _make_fake_checkpoint()
    del ckpt["estimator_type"]
    del ckpt["intervention_type"]
    del ckpt["noise_scale"]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pkl")
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
        trace = evaluate_checkpoint(ckpt_path=path, temperature=0.0, seed=0)

    assert trace["metadata"]["estimator_type"] == "analytic"
    assert trace["metadata"]["intervention_type"] == "soft_shift"
