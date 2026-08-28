"""`normalise_returns` -- off is byte-identical, on rescales without changing the argmax.

The discipline `turn_aware_credit` was held to on 2026-08-27: a flag that touches the
gradient path has to prove that OFF changes nothing before its ON results mean anything.
Two of this project's silent design decisions were changes that "obviously" defaulted to the
old behaviour and did not.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.env import MAConfig, ROUND_ROBIN, TwoAgentEnv, VARY
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import federated_topology


def _env(seed: int = 0) -> TwoAgentEnv:
    topology = federated_topology(2, 1, 3)
    config = MAConfig(topology=topology, n_obs=60, n_int=20, budget=6,
                      disclose_regime=True, turn_order=ROUND_ROBIN,
                      action_modes=(VARY,), belief_backend="factored",
                      policy_arch="gnn_portable", episode_mix="confounded",
                      reward_criterion="claims", claim_bar=1.0)
    return TwoAgentEnv(config, seed=seed)


def _buffer(rewards, dones, values=None):
    rewards = np.asarray(rewards, dtype=np.float32)
    return {"reward": rewards,
            "done": np.asarray(dones, dtype=np.float32),
            "value": np.asarray(values if values is not None
                                else np.zeros_like(rewards), dtype=np.float32)}


def test_off_is_byte_identical():
    """The default path must produce exactly the advantages it produced before the flag."""
    env = _env()
    buf = _buffer([0.0, 0.5, 0.0, 2.0], [0, 0, 0, 1])

    plain = IndependentPPO(env, PPOConfig(total_episodes=16))
    flagged = IndependentPPO(env, PPOConfig(total_episodes=16, normalise_returns=False))

    a_plain, r_plain = plain._advantages(dict(buf))
    a_flag, r_flag = flagged._advantages(dict(buf))
    assert np.array_equal(a_plain, a_flag)
    assert np.array_equal(r_plain, r_flag)


def test_on_rescales_but_preserves_ordering():
    """Scaling is a positive divisor, so the ORDER of the advantages cannot move.

    That is the property the change rests on: the critic's target shrinks, and the policy
    gradient still prefers exactly the transitions it preferred before.
    """
    env = _env()
    buf = _buffer([0.0, 0.5, 0.0, 2.0], [0, 0, 0, 1])

    plain = IndependentPPO(env, PPOConfig(total_episodes=16))
    scaled = IndependentPPO(env, PPOConfig(total_episodes=16, normalise_returns=True))

    a_plain, _ = plain._advantages(dict(buf))
    a_scaled, _ = scaled._advantages(dict(buf))

    assert not np.allclose(a_plain, a_scaled), "the flag did nothing"
    assert np.array_equal(np.argsort(a_plain), np.argsort(a_scaled))
    # A single positive factor, not a per-element reweighting.
    ratios = a_plain[a_scaled != 0] / a_scaled[a_scaled != 0]
    assert np.allclose(ratios, ratios[0])
    assert ratios[0] > 0


def test_all_zero_batch_is_a_no_op():
    """Before any episode scores, the honest scale is unknown and the floor must not blow up.

    An untrained policy on a hard rung sees nothing but zeros for many updates. Dividing by
    a near-zero estimate there would manufacture enormous advantages out of pure noise.
    """
    env = _env()
    buf = _buffer([0.0, 0.0, 0.0, 0.0], [0, 0, 0, 1])
    scaled = IndependentPPO(env, PPOConfig(total_episodes=16, normalise_returns=True))
    advantages, returns = scaled._advantages(dict(buf))
    assert np.all(np.isfinite(advantages))
    assert np.allclose(advantages, 0.0)
    assert np.allclose(returns, 0.0)


def test_scale_is_running_not_per_batch():
    """A second batch must be divided by the POOLED scale, not by its own.

    A per-batch divisor changes the effective learning rate every update; the running
    estimate is what makes the divisor settle. Feeding the same buffer twice must therefore
    give a DIFFERENT scale the second time -- the pool has grown -- while a per-batch
    implementation would give exactly the same numbers.
    """
    env = _env()
    scaled = IndependentPPO(env, PPOConfig(total_episodes=16, normalise_returns=True))
    first, _ = scaled._advantages(_buffer([0.0, 0.0, 0.0, 4.0], [0, 0, 0, 1]))
    # A quiet batch: per-batch scaling would divide this by its own tiny spread.
    second, _ = scaled._advantages(_buffer([0.0, 0.0, 0.0, 0.1], [0, 0, 0, 1]))
    assert scaled._return_count == 8
    assert np.all(np.isfinite(second))
    assert np.abs(second).max() < np.abs(first).max() * 2, \
        "a quiet batch was amplified -- the divisor is per-batch, not running"


def test_the_scale_tracks_the_reward_magnitude():
    """Ten times the reward must give ten times the scale, so the advantages barely move.

    This is the mechanism the agent-ladder result rests on: the same task paid at a larger
    magnitude should reach the optimiser at the same size.
    """
    small = IndependentPPO(_env(), PPOConfig(total_episodes=16, normalise_returns=True))
    large = IndependentPPO(_env(), PPOConfig(total_episodes=16, normalise_returns=True))
    rewards = np.array([0.0, 0.3, 0.0, 1.7], dtype=np.float32)
    a_small, _ = small._advantages(_buffer(rewards, [0, 0, 0, 1]))
    a_large, _ = large._advantages(_buffer(rewards * 10.0, [0, 0, 0, 1]))
    assert np.allclose(a_small, a_large, atol=1e-4)


def test_flag_reaches_the_config_from_the_trainer():
    """The knob has to be WIRED, not merely defined.

    A flag that parses and is then dropped on the floor is how a measured "no effect" gets
    recorded for a change that was never applied -- and `scripts/ma_train.py` builds its
    parser inside `main`, so there is no parser object to interrogate. Asserting on the
    source is blunt but it fails loudly if the argument is ever added without being passed
    through, which is the failure this guards.
    """
    import inspect

    import scripts.ma_train as ma_train

    source = inspect.getsource(ma_train.main)
    assert '"--normalise_returns"' in source, "the flag is not offered"
    assert "normalise_returns=args.normalise_returns" in source, \
        "the flag is parsed but never reaches PPOConfig"
    assert "normalise_returns" in inspect.getsource(ma_train._config_record), \
        "the setting would not be recorded, so a result file could not say it was used"

    assert PPOConfig(total_episodes=16, normalise_returns=True).normalise_returns is True
    with pytest.raises(TypeError):
        PPOConfig(total_episodes=16, normalize_returns=True)      # the US spelling is not it
