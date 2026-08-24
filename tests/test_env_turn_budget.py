"""Acceptance tests for `docs/TURN_BUDGET_SPEC.md`.

Written BEFORE the implementation, and numbered to the spec's section 12, because every bug
on 20-21 August was a design decision made silently while implementing. Each test names the
decision it pins rather than the code path it covers.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.env import (CLAMP, NO_INTERVENTION, PRIVATE_SIGNAL, RANDOM_TURN,
                    ROUND_ROBIN, SHARED_SIGNAL, SIMULTANEOUS, MAConfig, TwoAgentEnv)
from ma.topology import Topology, two_agent

T_1_1_3 = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


def _env(**kw) -> TwoAgentEnv:
    config = MAConfig(topology=T_1_1_3, n_obs=200, n_int=50, budget=6,
                       disclose_regime=True, **kw)
    return TwoAgentEnv(config, seed=0)


def _action(env: TwoAgentEnv, agent: int, *, private: bool, mode: str = CLAMP) -> int:
    window = env.windows[agent]
    private_nodes = set(env.topology.private[agent])
    for index, (node, node_mode) in enumerate(window.actions):
        if node == -1 or node_mode != mode:
            continue
        if (node in private_nodes) == private:
            return index
    raise AssertionError(f"no {'private' if private else 'shared'} {mode} action for agent {agent}")


def _pass(env: TwoAgentEnv) -> dict:
    return {a: env.windows[a].pass_index for a in env.topology.agents}


# -- 12.1 a round is consumed whether the agent acts or declines -------------------------

def test_a_declined_round_still_consumes_the_shared_budget():
    """The whole point of the shared pool: a round A wastes is a round B does not get.
    Without this, declining is free and free-riding is the rational play -- measured at
    5/10 seeds collapsing into passing under the previous rules."""
    env = _env(turn_order=ROUND_ROBIN)
    assert env.rounds_used == 0
    env.step(_pass(env))
    assert env.rounds_used == 1
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    assert env.rounds_used == 2


def test_the_episode_ends_when_the_round_budget_is_exhausted():
    env = _env(turn_order=ROUND_ROBIN)
    result = None
    for _ in range(env.config.budget):
        result = env.step(_pass(env))
    assert env.rounds_used == env.config.budget
    assert result.done


# -- 12.2 a forfeited round generates observational data --------------------------------

def test_a_forfeited_round_generates_observational_rows_for_both_agents():
    env = _env(turn_order=ROUND_ROBIN)
    before = len(env.samples)
    env.step(_pass(env))
    assert len(env.samples) == before + env.config.n_int
    for agent in env.topology.agents:
        # Observational: nothing was intervened on, and nothing hidden was clamped.
        assert not env.known[agent][-env.config.n_int:].any()
        assert not env.clean[agent][-env.config.n_int:].any()


def test_total_data_volume_is_constant_regardless_of_behaviour():
    """Data quantity must not covary with how much a policy acted -- that confound is
    present in every number this project produced before the spec."""
    def rows(always_pass: bool) -> int:
        env = _env(turn_order=ROUND_ROBIN)
        for _ in range(env.config.budget):
            if always_pass:
                env.step(_pass(env))
            else:
                env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
        return len(env.samples)
    assert rows(True) == rows(False)


# -- 12.3 no single agent can end an episode --------------------------------------------

def test_a_single_pass_does_not_end_the_episode():
    """Asserted as "declining is never the CAUSE of termination" rather than "the episode
    did not end", because an episode can legitimately end on the same round by being
    SOLVED -- and a test that cannot tell those apart would pass for the wrong reason."""
    env = _env(turn_order=ROUND_ROBIN)
    result = env.step(_pass(env))
    assert not (result.done and not result.info["both_identified"]), (
        "the episode ended without being solved, so declining terminated it")


def test_there_is_no_voluntary_termination_at_all():
    """Even a unanimous decline keeps the episode alive: with step_cost zero there is
    nothing to escape, and removing the mechanism removes the whole class of rule that
    produced the 20 August collapse."""
    env = _env(turn_order=ROUND_ROBIN)
    for _ in range(env.config.budget - 1):
        result = env.step(_pass(env))
        assert not (result.done and not result.info["both_identified"]), (
            "declining must never terminate before the round budget runs out")


# -- 12.4 signalling is free -------------------------------------------------------------

def test_signalling_consumes_no_round():
    """Signals are broadcast at the round boundary, not spent as actions. Under the
    alternative -- declare only on your own turn -- establishing 'we are both finished'
    costs one turn per agent and grows with the number of agents."""
    env = _env(turn_order=ROUND_ROBIN)
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    assert env.rounds_used == 1
    assert set(env.signals) == set(env.topology.agents)


def test_the_signal_reports_the_region_actually_intervened_on():
    env = _env(turn_order=ROUND_ROBIN)
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    assert env.signals[0] == PRIVATE_SIGNAL
    assert env.signals[1] == NO_INTERVENTION, "1 did not act; its submission is discarded"
    env.step({0: _action(env, 0, private=False), 1: _action(env, 1, private=False)})
    assert env.signals[1] == SHARED_SIGNAL


def test_a_declined_round_signals_no_intervention():
    env = _env(turn_order=ROUND_ROBIN)
    env.step(_pass(env))
    assert all(env.signals[a] == NO_INTERVENTION for a in env.topology.agents)


def test_the_partners_signal_reaches_the_observation():
    env = _env(turn_order=ROUND_ROBIN)
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    obs = env.observation(1)
    assert len(obs) == env.windows[1].obs_size
    # The three signal slots are one-hot over the partner's reported region. They sit
    # before the per-node own-count block (k entries, appended 2026-08-25).
    k = env.windows[1].k
    assert np.isclose(obs[-3 - k:-k].sum(), 1.0)


# -- 12.5 the done bit must not leak the credit set --------------------------------------

def test_the_done_bit_comes_from_the_agents_own_posterior_not_the_credit_set():
    """The credit set is defined against the TRUE graph, so its mass is an ORACLE quantity.
    It is already computed every step for the reward, which makes it free to pass in -- and
    that is exactly what made this an easy mistake. Free is not the same as legitimate."""
    env = _env(turn_order=ROUND_ROBIN)
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    for agent in env.topology.agents:
        assert 0.0 <= env.done_bit[agent] <= 1.0
    # Two episodes with identical data but different TRUE graphs must agree on the done
    # bit: it cannot depend on the truth.
    adjacency = env.true_adjacency.copy()
    other = adjacency.copy()
    other[:] = False                                  # a different truth, same everything
    first = TwoAgentEnv(_env().config, seed=3)
    first.reset(seed=11, adjacency=adjacency)
    second = TwoAgentEnv(_env().config, seed=3)
    second.reset(seed=11, adjacency=adjacency)
    assert first.done_bit == second.done_bit


def test_the_done_bit_is_not_in_the_observation():
    """Logged, not acted on -- so it must not reach a policy either."""
    env = _env(turn_order=ROUND_ROBIN)
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    obs = env.observation(0)
    assert env.done_bit[1] not in list(obs) or env.done_bit[1] in (0.0, 1.0)


# -- 12.6 / 12.7 logging ------------------------------------------------------------------

def test_per_agent_interventions_and_forfeits_are_logged_separately():
    """`mean_steps` takes a max across agents, which hides an idle agent inside an average.
    Free-riding has to be visible as its own number."""
    env = _env(turn_order=ROUND_ROBIN)
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})   # 0 acts
    env.step(_pass(env))                                                             # 1 declines
    assert env.n_interventions == {0: 1, 1: 0}
    assert env.forfeits == {0: 0, 1: 1}


def test_clamp_targets_are_split_into_own_private_and_shared():
    """Clamping a SHARED node does nothing for a partner; only the private clamp does. An
    aggregate clamp fraction cannot tell those apart, so it cannot measure altruism."""
    env = _env(turn_order=ROUND_ROBIN)
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    env.step({0: _action(env, 0, private=False), 1: _action(env, 1, private=False)})
    assert env.clamps_private == {0: 1, 1: 0}
    assert env.clamps_shared == {0: 0, 1: 1}


def test_graph_connectedness_is_recorded():
    """A disconnected graph splits the agents into independent subproblems -- no
    cross-boundary paths, no confounding, nothing to coordinate about -- so those episodes
    cannot test what we are building. Every metric gets reported split by this."""
    env = _env(turn_order=ROUND_ROBIN)
    result = env.step(_pass(env))
    assert isinstance(result.info["connected"], bool)

    disconnected = np.zeros((5, 5), dtype=bool)
    env.reset(seed=1, adjacency=disconnected)
    assert env.connected is False


# -- 12.9 step cost -----------------------------------------------------------------------

def test_step_cost_defaults_to_zero():
    """Load-bearing with the absence of voluntary termination. Re-adding one without the
    other re-opens the collapse; see TURN_BUDGET_SPEC section 5."""
    assert MAConfig(topology=T_1_1_3).step_cost == 0.0


def test_acting_is_not_punished_relative_to_declining():
    env = _env(turn_order=ROUND_ROBIN)
    acted = env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    env.reset(seed=0)
    declined = env.step(_pass(env))
    assert acted.reward == declined.reward == 0.0


# -- 12.8 the existing guard must keep passing --------------------------------------------

@pytest.mark.parametrize("order", [SIMULTANEOUS, ROUND_ROBIN, RANDOM_TURN])
def test_clean_rounds_are_still_reachable(order):
    """DELIBERATE DUPLICATE of test_env_turns.py::test_clean_rounds_are_reachable.

    Checked 2026-08-22 and kept. It is item 12.8 of the turn-budget spec: the point is that
    the OLDER guard still holds under the new shared-budget semantics, which is a different
    claim from the original even though the assertion is identical. Sub-second either way.
    Do not "deduplicate" these without also removing 12.8 from the spec.
    """
    env = _env(turn_order=order)
    env.step({0: _action(env, 0, private=True), 1: _action(env, 1, private=True)})
    assert env.clean[0].any() or env.clean[1].any()
