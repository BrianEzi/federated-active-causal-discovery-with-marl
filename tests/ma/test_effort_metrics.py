"""The effort metrics must separate PROTOCOL arithmetic from policy behaviour.

`free_rider_index` is min/max interventions, so higher is more even -- the name says the
opposite. Worse, under random turn order the actor is drawn uniformly, so the counts are
multinomial and the ratio is small however well the agents coordinate. Comparing turn orders
on the raw number therefore compares dice, not policies (docs/FINDINGS_TURN_ORDER_2026_08_29.md).
"""
import numpy as np
import pytest

from ma.env import MAConfig, RANDOM_TURN, ROUND_ROBIN, SIMULTANEOUS, TwoAgentEnv
from ma.evaluate import _evenness_null, _mean_where
from ma.topology import federated_topology


def _env(turn_order, n_agents=8, budget=24):
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(n_agents, 2, 4), n_obs=40, n_int=10, budget=budget,
        turn_order=turn_order, belief_backend="factored", action_modes=("vary",),
        claim_bar=1.0, reward_criterion="claims", policy_arch="gnn_portable"))


def test_round_robin_null_is_one_because_every_agent_is_handed_the_same_share():
    assert _evenness_null(_env(ROUND_ROBIN)) == 1.0


def test_simultaneous_null_is_one():
    assert _evenness_null(_env(SIMULTANEOUS)) == 1.0


def test_random_turn_null_is_small_by_arithmetic_alone():
    """No policy involved -- this is what uniform actor draws produce on their own."""
    null = _evenness_null(_env(RANDOM_TURN))
    assert 0.10 < null < 0.25, null
    # The measured free_rider_index of the rndturn arm was 0.140; the null must be close
    # enough that the raw number cannot be read as a behavioural finding.
    assert abs(null - 0.140) < 0.06, null


def test_random_turn_null_rises_as_budget_grows_relative_to_agents():
    """More rounds per agent means less relative sampling spread, so the null must rise."""
    lean = _evenness_null(_env(RANDOM_TURN, budget=24))
    rich = _evenness_null(_env(RANDOM_TURN, budget=240))
    assert rich > lean + 0.2, (lean, rich)


def test_mean_where_selects_and_returns_nan_when_empty():
    rows = [{"success": True, "k": 1}, {"success": False, "k": 0}, {"success": True, "k": 1}]
    assert _mean_where(rows, lambda r: r["k"] == 1) == 1.0
    assert _mean_where(rows, lambda r: r["k"] == 0) == 0.0
    assert np.isnan(_mean_where(rows, lambda r: r["k"] == 99))


def test_duplicate_coverage_floor_is_zero_until_the_shared_surface_saturates():
    env = _env(ROUND_ROBIN)
    env.reset(seed=0)
    assert env.duplicate_coverage_floor() == 0.0          # nothing spent yet
    shared = list(env.topology.exposed)
    for node in shared:                                    # one each: no forced duplication
        env.shared_touches[node] = 1
    assert env.duplicate_coverage_floor() == 0.0


def test_duplicate_coverage_floor_rises_once_spend_exceeds_the_surface():
    env = _env(ROUND_ROBIN)
    env.reset(seed=0)
    shared = list(env.topology.exposed)
    for node in shared:
        env.shared_touches[node] = 3                       # 3 * |shared| spent
    spent, surface = 3 * len(shared), len(shared)
    assert env.duplicate_coverage_floor() == pytest.approx((spent - surface) / spent)
    # And the floor is a genuine lower bound on what was actually measured.
    assert env.duplicate_coverage() >= env.duplicate_coverage_floor() - 1e-12
