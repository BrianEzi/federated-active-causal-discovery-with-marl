"""Cumulative partner counts, role-fixed modes, and the two behavioural metrics.

All four landed 2026-08-26 and all four are silent when wrong: a mis-sliced observation
still trains, a mode rule that never fires still runs, and a metric computed off the wrong
tally still prints a plausible number. See `docs/AGENDA_2026_08_26.md` items 2, 5, 7, 9.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.env import CLAMP, PASS_ACTION, ROUND_ROBIN, VARY, MAConfig, TwoAgentEnv
from ma.topology import federated_topology

TOPO = federated_topology(3, private_size=1, n_shared=3)


def _env(topology_override=None, **kw):
    kw.setdefault("belief_backend", "version_space")
    kw.setdefault("reward_criterion", "claims")
    kw.setdefault("claim_bar", 1.0)
    kw.setdefault("episode_mix", "confounded")
    kw.setdefault("disclose_regime", True)
    kw.setdefault("budget", 6)
    kw.setdefault("action_modes", (VARY,))
    return TwoAgentEnv(MAConfig(topology=topology_override or TOPO, n_obs=60, n_int=20,
                                turn_order=ROUND_ROBIN, **kw), seed=0)


def _act(env, agent, node):
    """Everyone passes except `agent`, which targets `node`."""
    actions = {a: env.windows[a].pass_index for a in TOPO.agents}
    actions[agent] = env.windows[agent].action_index(node, prefer=VARY)
    return env.step(actions)


# -- partner counts ---------------------------------------------------------------------


def test_partner_counts_widen_the_observation_by_exactly_their_shape():
    plain, counted = _env(), _env(observe_partner_counts=True)
    n_others, n_shared = TOPO.n_agents - 1, len(TOPO.exposed)
    assert counted.obs_size(0) - plain.obs_size(0) == n_others * (n_shared + 1)
    counted.reset(seed=4)
    for agent in TOPO.agents:
        assert counted.observation(agent).shape[0] == counted.obs_size(agent)


def test_a_partner_shared_intervention_is_counted_against_that_partner_and_that_node():
    env = _env(observe_partner_counts=True)
    env.reset(seed=4)
    shared = TOPO.exposed[1]
    # Agent 0 holds round 1 under round-robin.
    _act(env, 0, shared)

    # Agent 1 sees agent 0 in partner slot 0 (canonical order, self removed).
    window = env.windows[1]
    table = env.partner_counts[1]
    assert table[0, window.shared.index(shared)] == 1.0
    assert table.sum() == 1.0
    # And agent 2 sees the same event in ITS slot for agent 0, also slot 0.
    assert env.partner_counts[2][0, window.shared.index(shared)] == 1.0
    # Agent 0 does not count itself anywhere.
    assert env.partner_counts[0].sum() == 0.0


def test_a_partner_private_intervention_is_counted_without_naming_the_node():
    """The privacy claim, as an assertion rather than a comment: partners learn THAT you
    acted privately -- already broadcast by PRIVATE_SIGNAL each round -- and never where."""
    env = _env(observe_partner_counts=True)
    env.reset(seed=4)
    private = TOPO.private[0][0]
    _act(env, 0, private)

    table = env.partner_counts[1]
    assert table[0, -1] == 1.0                       # the unnamed private column
    assert table[0, :-1].sum() == 0.0                # nothing landed on a shared column
    assert table.shape[1] == len(env.windows[1].shared) + 1


def test_partner_counts_are_cumulative_across_rounds():
    env = _env(observe_partner_counts=True)
    env.reset(seed=4)
    shared = TOPO.exposed[0]
    _act(env, 0, shared)                             # round 1: agent 0
    _act(env, 1, shared)                             # round 2: agent 1
    _act(env, 2, shared)                             # round 3: agent 2
    # Agent 2's partners are 0 and 1, both of which hit the same shared node once.
    column = env.windows[2].shared.index(shared)
    assert env.partner_counts[2][:, column].tolist() == [1.0, 1.0]


def test_partner_counts_are_budget_normalised_in_the_observation():
    env = _env(observe_partner_counts=True, budget=6)
    env.reset(seed=4)
    shared = TOPO.exposed[0]
    _act(env, 0, shared)
    tail = env.observation(1)[-(TOPO.n_agents - 1) * (len(TOPO.exposed) + 1):]
    assert tail.max() == pytest.approx(1.0 / 6)
    assert 0.0 <= tail.min() and tail.max() <= 1.0


def test_nothing_is_counted_when_the_disclosures_are_switched_off():
    """The counts are the CUMULATIVE form of two existing per-round disclosures and must not
    become a third one: with both switched off they stay empty."""
    env = _env(observe_partner_counts=True, disclose_shared_targets=False,
               disclose_signals=False)
    env.reset(seed=4)
    _act(env, 0, TOPO.exposed[0])
    _act(env, 1, TOPO.private[1][0])
    assert env.partner_counts[2].sum() == 0.0


# -- mode by role -----------------------------------------------------------------------


def test_mode_by_role_gives_one_action_per_node_clamping_only_private_ones():
    env = _env(mode_by_role=True)
    window = env.windows[0]
    targets = [node for node, _ in window.actions if node != PASS_ACTION]
    assert sorted(targets) == sorted(window.authority)      # one action per node, no doubling
    for node, mode in window.actions:
        if node == PASS_ACTION:
            continue
        assert mode == (CLAMP if node in window.private else VARY)


def test_mode_by_role_actually_clamps_the_private_node_in_the_data():
    env = _env(mode_by_role=True)
    env.reset(seed=4)
    private = TOPO.private[0][0]
    _act(env, 0, private)
    # A clamp fixes the variable outright, so its interventional rows carry no variance.
    rows = env.samples[-env.config.n_int:, private]
    assert np.allclose(rows, 0.0)
    assert env.clamps_private[0] == 1 and env.clamps_shared[0] == 0


def test_mode_by_role_varies_a_shared_node():
    env = _env(mode_by_role=True)
    env.reset(seed=4)
    shared = TOPO.exposed[0]
    _act(env, 0, shared)
    rows = env.samples[-env.config.n_int:, shared]
    assert rows.std() > 0.0
    assert env.clamps_shared[0] == 0


def test_action_index_survives_a_mode_that_is_not_offered():
    """`(node, VARY)` is not a key under `mode_by_role`; asking the window is."""
    for by_role in (False, True):
        env = _env(mode_by_role=by_role, action_modes=(CLAMP,))
        window = env.windows[0]
        for node in window.authority:
            index = window.action_index(node, prefer=VARY)
            assert window.actions[index][0] == node


# -- behavioural metrics ----------------------------------------------------------------


def test_duplicate_coverage_is_zero_when_agents_divide_the_shared_nodes():
    env = _env()
    env.reset(seed=4)
    for agent, node in zip(TOPO.agents, TOPO.exposed):
        _act(env, agent, node)
    assert env.duplicate_coverage() == pytest.approx(0.0)


def test_duplicate_coverage_counts_every_intervention_past_the_first_on_a_node():
    env = _env()
    env.reset(seed=4)
    node = TOPO.exposed[0]
    for agent in TOPO.agents:
        _act(env, agent, node)
    # Three interventions, one distinct node covered: two rounds bought nothing new.
    assert env.duplicate_coverage() == pytest.approx(2 / 3)


def test_duplicate_coverage_ignores_private_interventions():
    """It measures contention on the SHARED surface. A private node is nobody else's to
    duplicate, so spending rounds there is a different failure and is not counted here."""
    env = _env()
    env.reset(seed=4)
    for agent in TOPO.agents:
        _act(env, agent, TOPO.private[agent][0])
    assert env.duplicate_coverage() == pytest.approx(0.0)
    assert sum(env.shared_touches.values()) == 0


def test_rounds_to_identification_censors_at_one_past_the_budget():
    env = _env(budget=4)
    result = env.reset(seed=4)
    while not result.done:
        result = env.step({a: env.windows[a].pass_index for a in TOPO.agents})
    rounds = env.rounds_to_identification()
    for agent in TOPO.agents:
        if env.identified_round[agent] is None:
            assert rounds[agent] == 5           # budget + 1, worse than using it all
        else:
            assert rounds[agent] == env.identified_round[agent]


def test_rounds_to_identification_latches_the_first_round():
    """The metric is "how many experiments did it take", so a window that comes undone
    later keeps the round it was first settled. The identification RATE answers the other
    question."""
    env = _env(budget=6)
    result = env.reset(seed=4)
    seen = {}
    while not result.done:
        result = env.step({a: env.windows[a].pass_index for a in TOPO.agents})
        for agent in TOPO.agents:
            if result.identified[agent] and agent not in seen:
                seen[agent] = env.rounds_used
    for agent, first in seen.items():
        assert env.identified_round[agent] == first


def test_baselines_are_built_lazily_and_actually_build():
    """`GreedyAgent` enumerates and refuses past window size 5, so building every arm
    eagerly crashed callers that never wanted it -- a trap that took three separate jobs.
    The lazy mapping fixes that, and this pins BOTH halves: a window too large for the
    enumerating arm still yields the others, and the mapping really does construct them.

    The second half is not hypothetical. The first version overrode `__contains__` to report
    what could be built, which made its own `key not in self` guard always false, so it
    never constructed anything and every lookup raised KeyError.
    """
    from ma.baselines import UncertaintyGreedyAgent, make_baselines

    env = _env(belief_backend="factored", topology_override=federated_topology(3, 3, 3))
    baselines = make_baselines(env, 0, seed=0)
    assert isinstance(baselines["greedy_uncertainty"], UncertaintyGreedyAgent)
    assert baselines["greedy_uncertainty"] is baselines["greedy_uncertainty"]   # cached
    assert "greedy" in baselines                                               # offered...
    with pytest.raises(ValueError):
        _ = baselines["greedy"]                                                # ...but refuses
