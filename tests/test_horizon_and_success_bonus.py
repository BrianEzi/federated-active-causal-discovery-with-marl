"""Tests for the budget-derived episode horizon and the terminal success bonus.

Both changes come from consolidating a parallel session's reward refactor. That refactor
made three changes; only two are adopted here, and the tests pin down why:

1. Deriving `max_steps` from the budget instead of carrying an independent horizon
   parameter -- adopted, but with the arithmetic corrected. The parallel version used
   `initial_budget * K`, which double-counts: budgets are per-agent and spent in parallel,
   so the horizon is `initial_budget / action_cost` regardless of K.
2. A terminal success bonus -- adopted, but moved so it applies under BOTH reward
   densities. The parallel version placed it inside the dense branch only, making it dead
   code under this project's `sparse` default.
3. A per-step `-0.1 * SHD` holding penalty -- NOT adopted. It is not potential-based
   (Ng, Harada & Russell 1999), so it silently changes the optimal policy, and under
   `sparse` the terminal `-SHD` term already penalises an unresolved graph. See
   docs/THEORY_NOTES.md #8.
"""
import numpy as np
import jax.numpy as jnp

from src.rewards import compute_ippo_rewards
from src.evaluator_env import FederatedCausalEnv
from src.types import SCMConfig, MechanismType, NoiseType


def _config(K=2):
    return SCMConfig(
        d=4, K=K,
        mechanism_type=int(MechanismType.LINEAR),
        noise_type=int(NoiseType.GAUSSIAN),
        noise_scale=0.1,
    )


def _env(initial_budget=20.0, action_cost=1.0, K=2, max_steps=None, **kw):
    return FederatedCausalEnv(
        _config(K), jnp.full(K, action_cost),
        initial_budget=initial_budget, max_steps=max_steps, **kw
    )


def test_horizon_defaults_to_budget_divided_by_action_cost():
    assert _env(initial_budget=20.0, action_cost=1.0).max_steps == 20


def test_horizon_is_independent_of_agent_count():
    """Budgets are per-agent (`jnp.full(K, initial_budget)`) and spent in parallel, so
    adding agents does not extend how long the episode can run. Guards against the
    `initial_budget * K` formulation, which would give 40 here at K=2."""
    assert _env(K=1).max_steps == _env(K=2).max_steps == 20


def test_horizon_tracks_a_changed_budget_and_cost():
    assert _env(initial_budget=10.0, action_cost=1.0).max_steps == 10
    assert _env(initial_budget=20.0, action_cost=2.0).max_steps == 10
    assert _env(initial_budget=15.0, action_cost=2.0).max_steps == 8  # ceil(7.5)


def test_explicit_horizon_still_overrides_the_derivation():
    assert _env(initial_budget=20.0, max_steps=5).max_steps == 5


def _rewards(density, is_terminal, bonus, shd_matrix, true_matrix):
    return compute_ippo_rewards(
        shd_matrix, true_matrix, has_cycle=False,
        reward_density=density, is_terminal=is_terminal, success_bonus=bonus,
    )


def test_success_bonus_applies_under_both_reward_densities():
    """The bug being guarded against: the bonus living inside the dense branch only, and
    therefore never firing under the project's `sparse` default."""
    solved = np.zeros((4, 4))
    for density in ("sparse", "dense"):
        without = _rewards(density, True, 0.0, solved, solved)
        with_bonus = _rewards(density, True, 5.0, solved, solved)
        for agent in ("agent_0", "agent_1"):
            assert with_bonus[agent] == pytest_approx(without[agent] + 5.0), (
                f"success_bonus did not apply under reward_density={density!r} for {agent}"
            )


def test_success_bonus_only_fires_on_a_solved_graph():
    true_dag = np.zeros((4, 4))
    wrong = np.zeros((4, 4))
    wrong[0, 1] = 1.0  # an error on agent_0's local edges
    r = _rewards("sparse", True, 5.0, wrong, true_dag)
    assert r["agent_0"] < 0.0, "agent_0 has an error and must not receive the bonus"


def test_success_bonus_does_not_fire_mid_episode():
    solved = np.zeros((4, 4))
    non_terminal = _rewards("sparse", False, 5.0, solved, solved)
    assert non_terminal["agent_0"] == pytest_approx(0.0)
    assert non_terminal["agent_1"] == pytest_approx(0.0)


def test_success_bonus_defaults_to_off():
    """Existing results must stay reproducible without passing new flags."""
    solved = np.zeros((4, 4))
    assert compute_ippo_rewards(solved, solved, has_cycle=False,
                                reward_density="sparse", is_terminal=True) == \
        _rewards("sparse", True, 0.0, solved, solved)


def pytest_approx(value, abs_tol=1e-6):
    import pytest
    return pytest.approx(value, abs=abs_tol)
