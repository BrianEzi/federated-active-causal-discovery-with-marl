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


def test_multi_hidden_is_refused_ONLY_WHEN_THE_REGIME_BIT_IS_DISCLOSED():
    """NARROWED 2026-08-25. The hazard is unchanged; the condition was too wide.

    The unsound path is `_assignment_weights`'s `0 < f < 1` branch, which mixes the clean
    and dirty tables with one weight for every confounding edge -- so with more than one
    hidden node it knows how MANY were clamped, never WHICH. That branch is reachable only
    when the regime bit is disclosed. With `disclose_regime=False`, `_refresh`, `true_mass`
    and `dag_set_mass` all pass `clean` as zeros, `f` is exactly 0.0, and the `f == 0.0`
    branch reads the dirty table at the full parent set -- exact, at any number of hidden
    nodes. The old guard therefore refused the entire scale ladder over a branch it never
    entered. The refusal below is the combination that IS unsound.
    """
    config = MAConfig(topology=T_2_2_2, n_obs=100, n_int=20, budget=2,
                      disclose_regime=True)
    with pytest.raises(NotImplementedError, match="hide up to"):
        TwoAgentEnv(config, seed=0)


def test_multi_hidden_is_allowed_and_EXACT_without_the_regime_bit():
    """The other half, and the one that unblocks three and five agents.

    Asserts the mechanism, not just that construction succeeds: the vector actually handed
    to the belief must be all zeros, because that is what makes the score exact. A test that
    only checked `TwoAgentEnv(...)` did not raise would still pass if a future change routed
    a real fraction through, which is the exact failure the guard existed to prevent.
    """
    topology = Topology(name="3a_1p_2x", private=((0,), (1,), (2,)), exposed=(3, 4))
    assert max(len(topology.hidden_from(a)) for a in topology.agents) == 2
    config = MAConfig(topology=topology, n_obs=100, n_int=20, budget=2,
                      disclose_regime=False)
    env = TwoAgentEnv(config, seed=0)

    for agent in topology.agents:
        clean = (env.clean[agent] if config.disclose_regime
                 else np.zeros(len(env.samples), dtype=bool))
        assert not np.asarray(clean).any(), (
            f"agent {agent} would be scored through the inexact mixture")

    env.step({a: _clamp_own_private(env, a) for a in topology.agents})
    for agent in topology.agents:
        clean = (env.clean[agent] if config.disclose_regime
                 else np.zeros(len(env.samples), dtype=bool))
        assert not np.asarray(clean).any(), "still zero after a private clamp"


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
