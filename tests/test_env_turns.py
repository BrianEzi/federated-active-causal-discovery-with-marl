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

from ma.env import (CLAMP, MODES, RANDOM_TURN, ROUND_ROBIN, SIMULTANEOUS,
                    MAConfig, TwoAgentEnv)
from ma.topology import Topology, two_agent

T_1_1_3 = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
T_2_2_2 = two_agent(name="T1", a_private=(0, 1), b_private=(2, 3), exposed=(4, 5))


def _env(**kw) -> TwoAgentEnv:
    # BOTH modes -- see tests/ma/test_env.py:make. Turn-order semantics must hold
    # whichever action modes are enabled, and one test here is vary-specific.
    kw.setdefault("action_modes", MODES)
    config = MAConfig(topology=T_1_1_3, n_obs=200, n_int=50, budget=3,
                       disclose_regime=True, **kw)
    return TwoAgentEnv(config, seed=0)


def _clamp_own_private(env: TwoAgentEnv, agent: int) -> int:
    """The action index that clamps this agent's own private node."""
    window = env.windows[agent]
    private = set(env.topology.private[agent])
    for index, (node, mode) in enumerate(window.actions):
        if node in private and mode == CLAMP:
            return index
    raise AssertionError(f"Agent {agent} has no clamp action on its own private node")


# -- protocol ---------------------------------------------------------------------------

def test_round_robin_alternates_and_only_one_agent_acts():
    env = _env(turn_order=ROUND_ROBIN)
    actors = []
    for _ in range(4):
        before = dict(env.n_interventions)
        env.step({0: _clamp_own_private(env, 0), 1: _clamp_own_private(env, 1)})
        moved = [a for a in env.topology.agents if env.n_interventions[a] > before[a]]
        assert len(moved) == 1, f"expected exactly one mover, got {moved}"
        actors.append(moved[0])
    assert actors == [0, 1, 0, 1]


def test_random_turn_order_uses_both_agents_and_still_moves_one():
    env = _env(turn_order=RANDOM_TURN)
    actors = []
    for _ in range(6):
        before = dict(env.n_interventions)
        result = env.step({0: _clamp_own_private(env, 0), 1: _clamp_own_private(env, 1)})
        moved = [a for a in env.topology.agents if env.n_interventions[a] > before[a]]
        assert len(moved) == 1
        actors.append(moved[0])
        if result.done:
            break
    assert set(actors) == set(env.topology.agents), "random selection never chose one of the agents"


def test_simultaneous_is_unchanged():
    """The legacy protocol must still let both agents move in one round -- every result
    before 2026-08-20 was measured under it and has to stay reproducible."""
    env = _env(turn_order=SIMULTANEOUS)
    env.step({0: _clamp_own_private(env, 0), 1: _clamp_own_private(env, 1)})
    assert env.n_interventions == {0: 1, 1: 1}







# NOTE. Five tests were REMOVED here on 2026-08-21, not fixed. They pinned the pre-spec
# rules -- a per-agent intervention budget, the consecutive-pass tally, and "a forfeited turn
# generates no data" -- all of which `docs/TURN_BUDGET_SPEC.md` deliberately replaces. Their
# successors live in `tests/test_env_turn_budget.py`, numbered to the spec's section 12.
# A test that encodes a superseded decision is worse than no test: it argues for the old
# design every time someone runs it.


# -- the clean regime -------------------------------------------------------------------

@pytest.mark.parametrize("order", [SIMULTANEOUS, ROUND_ROBIN, RANDOM_TURN])
def test_clean_rounds_are_reachable(order):
    """A clean round must be EARNABLE under every protocol. If this fails the regime bit
    is constant, every rule collapses to `pooled`, and any confounding result measured on
    top of it is void."""
    env = _env(turn_order=order)
    env.step({0: _clamp_own_private(env, 0), 1: _clamp_own_private(env, 1)})
    # B clamping its private node is exactly what makes A's rows clean, and vice versa.
    assert env.clean[0].any() or env.clean[1].any()


def test_clean_marks_only_the_rows_of_the_clamped_round():
    env = _env(turn_order=ROUND_ROBIN)
    n_obs = env.config.n_obs
    env.step({0: _clamp_own_private(env, 0), 1: _clamp_own_private(env, 1)})   # A clamps
    clean_b = env.clean[1]
    assert not clean_b[:n_obs].any(), "the observational block is never clean"
    assert clean_b[n_obs:].all(), "the clamped round should be clean end to end"


def test_a_vary_on_the_hidden_node_does_not_clean_anything():
    """Varying a hidden node leaves it a live variance source, so it de-confounds nothing.
    This is the asymmetry that motivates clamp-only."""
    env = _env(turn_order=ROUND_ROBIN)
    window = env.windows[0]
    vary = next(i for i, (node, mode) in enumerate(window.actions)
                if node in env.topology.private[0] and mode != CLAMP)
    env.step({0: vary, 1: env.windows[1].pass_index})
    assert not env.clean[1].any()


def test_multi_hidden_topology_is_accepted_since_the_guard_was_removed():
    """CONVERTED 2026-08-23. This asserted `NotImplementedError` until the guard was
    removed by instruction; see the note in `ma/env.py.__init__`.

    The env accepting these topologies does NOT mean the exact belief path is sound on
    them -- it is not, and `test_clean_fraction_cannot_say_WHICH_node_was_clamped` below
    demonstrates exactly why. The soundness constraint moves to the backend when the
    backend boundary lands; it is no longer the environment's business.
    """
    config = MAConfig(topology=T_2_2_2, n_obs=100, n_int=20, budget=2)
    env = TwoAgentEnv(config, seed=0)
    assert max(len(config.topology.hidden_from(a))
               for a in config.topology.agents) > 1
    assert env.topology is config.topology


def test_clean_fraction_cannot_say_WHICH_node_was_clamped():
    """The defect the removed guard was protecting, demonstrated rather than asserted.

    THIS IS THE TEST THAT MATTERS. The guard's message is gone; the reason must not be.

    `clean[agent]` is a SCALAR fraction, `n_clamped / len(hidden)`. With two nodes hidden
    from an agent, clamping either ONE of them yields 0.5 -- an identical value for two
    materially different experiments. `belief_dp._assignment_weights` then mixes the clean
    and dirty score tables with that single number, so a confounding hypothesis about `h1`
    is scored identically whether `h1` or `h2` was the node actually clamped.

    That is unsoundness, not approximation: the exact path scores the wrong hypothesis.

    If someone makes `clean` per-node -- an `[n_rows, n_hidden]` mask, which is the real
    fix -- this test SHOULD start failing, and the fix is to assert the two clamps now
    produce DIFFERENT records. A green run here means the defect is still present.
    """
    topo = Topology(name="T_3agent_1each", private=((0,), (1,), (2,)), exposed=(3, 4, 5))
    config = MAConfig(topology=topo, n_obs=100, n_int=20, budget=4,
                      turn_order=SIMULTANEOUS, action_modes=(CLAMP,),
                      disclose_regime=True)

    def clean_after_clamping(node_by_agent) -> float:
        env = TwoAgentEnv(config, seed=0)
        actions = {}
        for agent in topo.agents:
            window = env.windows[agent]
            target = node_by_agent.get(agent)
            actions[agent] = (window.pass_index if target is None
                              else window.actions.index((target, CLAMP)))
        env.step(actions)
        # Agent 0 hides {1, 2}: one of the two was clamped, so the fraction is 0.5.
        return float(env.clean[0][-1])

    only_h1 = clean_after_clamping({1: 1})
    only_h2 = clean_after_clamping({2: 2})

    assert only_h1 == 0.5 and only_h2 == 0.5
    assert only_h1 == only_h2, (
        "two different experiments must currently be indistinguishable -- if this fails, "
        "`clean` has become per-node and the exact path may now be sound")


def test_turn_order_is_validated():
    with pytest.raises(ValueError, match="turn_order"):
        TwoAgentEnv(MAConfig(topology=T_1_1_3, turn_order="alternating"), seed=0)


# -- determinism ------------------------------------------------------------------------

def test_random_turn_order_is_reproducible_from_the_seed():
    def run() -> list:
        env = _env(turn_order=RANDOM_TURN)
        seen = []
        for _ in range(5):
            result = env.step({0: _clamp_own_private(env, 0), 1: _clamp_own_private(env, 1)})
            seen.append(env.active)
            if result.done:
                break
        return seen
    assert run() == run()
