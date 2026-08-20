"""Turn-taking, and the guards that keep it from silently degrading.

The reachability test is the important one here. `clean` rounds are what the whole
confounding design rests on, and there is a family of configurations in which they can
never occur at all -- in which case every regime rule quietly reduces to `pooled` and the
agent is measured on a criterion it cannot earn. That exact failure has already cost this
project once (see docs/SA_EXPERIMENT_LOG.md, the unearnable two-agent metric), so the
condition is asserted rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.env2 import (AGENTS, CLAMP, RANDOM_TURN, ROUND_ROBIN, SIMULTANEOUS,
                     MA2Config, TwoAgentEnv2)
from ma.topology import Topology

T_1_1_3 = Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
T_2_2_2 = Topology(name="T1", a_private=(0, 1), b_private=(2, 3), exposed=(4, 5))


def _env(**kw) -> TwoAgentEnv2:
    config = MA2Config(topology=T_1_1_3, n_obs=200, n_int=50, budget=3,
                       disclose_regime=True, **kw)
    return TwoAgentEnv2(config, seed=0)


def _clamp_own_private(env: TwoAgentEnv2, name: str) -> int:
    """The action index that clamps this agent's own private node."""
    window = env.windows[name]
    private = set(env.topology.a_private if name == "A" else env.topology.b_private)
    for index, (node, mode) in enumerate(window.actions):
        if node in private and mode == CLAMP:
            return index
    raise AssertionError(f"{name} has no clamp action on its own private node")


# -- protocol ---------------------------------------------------------------------------

def test_round_robin_alternates_and_only_one_agent_acts():
    env = _env(turn_order=ROUND_ROBIN)
    actors = []
    for _ in range(4):
        before = dict(env.n_interventions)
        env.step(_clamp_own_private(env, "A"), _clamp_own_private(env, "B"))
        moved = [n for n in AGENTS if env.n_interventions[n] > before[n]]
        assert len(moved) == 1, f"expected exactly one mover, got {moved}"
        actors.append(moved[0])
    assert actors == ["A", "B", "A", "B"]


def test_random_turn_order_uses_both_agents_and_still_moves_one():
    env = _env(turn_order=RANDOM_TURN)
    actors = []
    for _ in range(6):
        before = dict(env.n_interventions)
        result = env.step(_clamp_own_private(env, "A"), _clamp_own_private(env, "B"))
        moved = [n for n in AGENTS if env.n_interventions[n] > before[n]]
        assert len(moved) == 1
        actors.append(moved[0])
        if result.done:
            break
    assert set(actors) == set(AGENTS), "random selection never chose one of the agents"


def test_simultaneous_is_unchanged():
    """The legacy protocol must still let both agents move in one round -- every result
    before 2026-08-20 was measured under it and has to stay reproducible."""
    env = _env(turn_order=SIMULTANEOUS)
    env.step(_clamp_own_private(env, "A"), _clamp_own_private(env, "B"))
    assert env.n_interventions == {"A": 1, "B": 1}


def test_budget_is_per_agent_so_an_exhausted_agent_does_not_end_the_episode():
    """Round-robin must skip an agent with no budget left. Without the eligibility check
    the exhausted agent's forced pass reads as a voluntary pass and terminates the episode
    while its partner still has moves."""
    env = _env(turn_order=ROUND_ROBIN)
    env.n_interventions["A"] = env.config.budget          # A is spent, B is not
    result = env.step(_clamp_own_private(env, "A"), _clamp_own_private(env, "B"))
    assert env.active == "B"
    assert not result.done
    assert env.n_interventions["B"] == 1


def test_pass_by_the_active_agent_ends_the_episode():
    env = _env(turn_order=ROUND_ROBIN)
    result = env.step(env.windows["A"].pass_index, _clamp_own_private(env, "B"))
    assert result.info["passed"] and result.done
    assert env.n_interventions == {"A": 0, "B": 0}, "a passed round must cost nothing"


# -- the clean regime -------------------------------------------------------------------

@pytest.mark.parametrize("order", [SIMULTANEOUS, ROUND_ROBIN, RANDOM_TURN])
def test_clean_rounds_are_reachable(order):
    """A clean round must be EARNABLE under every protocol. If this fails the regime bit
    is constant, every rule collapses to `pooled`, and any confounding result measured on
    top of it is void."""
    env = _env(turn_order=order)
    env.step(_clamp_own_private(env, "A"), _clamp_own_private(env, "B"))
    # B clamping its private node is exactly what makes A's rows clean, and vice versa.
    assert env.clean["A"].any() or env.clean["B"].any()


def test_clean_marks_only_the_rows_of_the_clamped_round():
    env = _env(turn_order=ROUND_ROBIN)
    n_obs = env.config.n_obs
    env.step(_clamp_own_private(env, "A"), _clamp_own_private(env, "B"))   # A clamps
    clean_b = env.clean["B"]
    assert not clean_b[:n_obs].any(), "the observational block is never clean"
    assert clean_b[n_obs:].all(), "the clamped round should be clean end to end"


def test_a_vary_on_the_hidden_node_does_not_clean_anything():
    """Varying a hidden node leaves it a live variance source, so it de-confounds nothing.
    This is the asymmetry that motivates clamp-only."""
    env = _env(turn_order=ROUND_ROBIN)
    window = env.windows["A"]
    vary = next(i for i, (node, mode) in enumerate(window.actions)
                if node in env.topology.a_private and mode != CLAMP)
    env.step(vary, env.windows["B"].pass_index)
    assert not env.clean["B"].any()


def test_multi_private_topology_is_refused_rather_than_scored_wrong():
    """With two hidden nodes a single clamp leaves a block PARTIALLY clean, which the
    regime rules would score as fully confounding-free. Refuse until per-block confounding
    subsets exist."""
    config = MA2Config(topology=T_2_2_2, n_obs=100, n_int=20, budget=2)
    with pytest.raises(NotImplementedError, match="hides 2 nodes"):
        TwoAgentEnv2(config, seed=0)


def test_turn_order_is_validated():
    with pytest.raises(ValueError, match="turn_order"):
        TwoAgentEnv2(MA2Config(topology=T_1_1_3, turn_order="alternating"), seed=0)


# -- determinism ------------------------------------------------------------------------

def test_random_turn_order_is_reproducible_from_the_seed():
    def run() -> list:
        env = _env(turn_order=RANDOM_TURN)
        seen = []
        for _ in range(5):
            result = env.step(_clamp_own_private(env, "A"), _clamp_own_private(env, "B"))
            seen.append(env.active)
            if result.done:
                break
        return seen
    assert run() == run()
