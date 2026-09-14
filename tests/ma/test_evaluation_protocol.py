"""Loading a checkpoint must not hijack the caller's random stream.

`IndependentPPO.__init__` seeded the global torch stream unconditionally, and `load` went
through it, so every evaluation of a loaded policy replayed ONE fixed sample path: repeating
an evaluation returned the identical number and every reported confidence interval excluded
policy stochasticity. Measured after the fix at w08: path SD 0.033 on a success rate of 0.47,
i.e. a +/-0.029 interval that was previously reported as zero.
"""
import numpy as np
import pytest
import torch

from ma.baselines import make_baselines
from ma.env import MAConfig, TwoAgentEnv
from ma.evaluate import run_arm_paths
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import federated_topology


def _env(claim_bar=1.0):
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(3, 2, 3), n_obs=40, n_int=10, budget=9,
        turn_order="round_robin", belief_backend="factored", action_modes=("vary",),
        claim_bar=claim_bar, reward_criterion="claims", policy_arch="gnn_portable",
        graph_model="sf", sf_m=2, episode_mix="confounded", vs_evidence="oracle"))


def test_constructing_a_learner_for_training_still_seeds_reproducibly():
    """Training must stay reproducible; only the LOAD path may leave the stream alone."""
    env = _env()
    a = IndependentPPO(env, PPOConfig(hidden=32, seed=7))
    first_a = next(iter(a.nets.values())).node_encoder[0].weight.clone()
    b = IndependentPPO(env, PPOConfig(hidden=32, seed=7))
    first_b = next(iter(b.nets.values())).node_encoder[0].weight.clone()
    assert torch.allclose(first_a, first_b)


def test_seed_torch_false_does_not_PIN_the_stream_to_a_fixed_point():
    """The precise bug. Building a network CONSUMES the global stream either way -- that is
    unavoidable and harmless. What was fatal is that `manual_seed` RESET it to a fixed
    point, so two constructions left the caller at the identical position and every
    subsequent evaluation drew the identical sample path."""
    env = _env()
    cfg = PPOConfig(hidden=32, seed=99)

    torch.manual_seed(1234)
    IndependentPPO(env, cfg, seed_torch=False)
    first = torch.randn(3)
    IndependentPPO(env, cfg, seed_torch=False)      # same config, again
    second = torch.randn(3)
    assert not torch.allclose(first, second), "stream was pinned despite seed_torch=False"


def test_seed_torch_true_DOES_pin_the_stream_which_is_why_load_must_not_use_it():
    """The contrast that shows the flag matters: with seeding on, two constructions leave
    the caller at exactly the same place -- which is the behaviour every past evaluation
    inherited through `load`."""
    env = _env()
    cfg = PPOConfig(hidden=32, seed=99)

    IndependentPPO(env, cfg, seed_torch=True)
    first = torch.randn(3)
    IndependentPPO(env, cfg, seed_torch=True)
    second = torch.randn(3)
    assert torch.allclose(first, second)


def test_multipath_reports_a_nonzero_interval_for_a_stochastic_policy():
    """The whole point: a sampled policy has evaluation variance, and it must be reported."""
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=32, seed=0))
    out = run_arm_paths(env, ppo.policies(deterministic=False), episodes=12, seed=0, paths=4)
    assert out["paths"] == 4
    assert "success__path_ci" in out
    assert out["success__path_sd"] >= 0.0
    assert len(out["per_path"]) == 4


def test_multipath_collapses_to_zero_spread_for_a_deterministic_policy():
    """Argmax has no sample path, so the interval must vanish -- otherwise the spread is
    coming from somewhere it should not (graph draws leaking across paths)."""
    env = _env()
    ppo = IndependentPPO(env, PPOConfig(hidden=32, seed=0))
    out = run_arm_paths(env, ppo.policies(deterministic=True), episodes=12, seed=0, paths=4)
    assert out["success__path_sd"] == pytest.approx(0.0, abs=1e-12)


def test_greedy_is_built_at_the_bar_the_task_grades_on():
    """A greedy at the class default of 0.7 stops scoring claims a task graded at 1.0 still
    counts open -- measured at +0.233 to greedy at four agents, enough to invert a headline.
    The claims backends require claim_bar=1.0, so 1.0 is the bar that matters, and 0.7 was
    exactly the wrong default to inherit."""
    env = _env(claim_bar=1.0)
    baselines = make_baselines(env, 0, 0)
    assert baselines["greedy_uncertainty"].bar == pytest.approx(1.0)
    assert baselines["greedy_partitioned"].bar == pytest.approx(1.0)


def test_greedy_falls_back_to_the_class_default_when_a_config_has_no_bar():
    """Old configs and non-claims backends must keep working rather than raising."""
    class _Bare:
        pass
    class _FakeEnv:
        config = _Bare()
        topology = federated_topology(3, 2, 3)
    assert make_baselines(_FakeEnv(), 0, 0)["greedy_uncertainty"].bar == pytest.approx(0.7)
