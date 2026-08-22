"""METRIC REACHABILITY -- can the thing we report ever be earned, in every regime?

This file exists because of a specific failure. The reported success criterion scored
EXACTLY 0.000 on confounded episodes, at every budget, for every policy -- not rarely, but
structurally, because the posterior was indexed by the AUGMENTED graph while the criterion
compared against the CAUSAL one. Every two-agent number gathered overnight was therefore an
unconfounded-only number wearing a general label, and nothing flagged it: the metric
returned plausible values, the tests passed, and the runs completed.

The general lesson, and the reason this is a separate file rather than another case in
test_evaluate.py: a metric can be WELL-FORMED and still be UNEARNABLE. Checking that the
truth belongs to its own credit set is not enough, because the bug was in the mapping from
posterior index to credit set, not in the set. What catches it is asking whether the metric
can be earned at all, separately WITHIN EACH REGIME the experiment claims to cover.

Any new metric, or any new regime, gets a case here before it is reported on.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.baselines import RandomAgent
from ma.env import MAConfig, TwoAgentEnv
from ma.evaluate import agent_report, credit_set, evaluate_episode
from ma.projection import bidirected_pairs
from ma.topology import Topology, two_agent


@pytest.fixture(scope="module")
def topology():
    return two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


def split_by_confounding(topology, episodes: int, budget: int = 8, seed: int = 2):
    """Run a clamping random pair and bucket episodes by whether confounding is present."""
    env = TwoAgentEnv(MAConfig(topology=topology, n_obs=1000, n_int=100, budget=budget,
                                 disclose_regime=True))
    policies = {a: RandomAgent(a, seed=seed, allow_clamp=True) for a in env.topology.agents}
    clean_eps, dirty_eps = [], []
    for episode in range(episodes):
        result = env.reset(seed=episode)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
        confounded = any(bool(bidirected_pairs(env.true_adjacency,
                                               env.topology.observed_by(a)))
                         for a in env.topology.agents)
        row = evaluate_episode(env)
        (dirty_eps if confounded else clean_eps).append(row)
    return clean_eps, dirty_eps


@pytest.fixture(scope="module")
def split_70(topology):
    """The 70-episode split, computed ONCE for the three tests that need it.

    All three called `split_by_confounding(topology, episodes=70)` with identical arguments,
    so the same 70 episodes were simulated three times over -- 179 s of the suite's 484 s,
    two thirds of it recomputing a deterministic function of its arguments.

    Module-scoped rather than session-scoped: nothing outside this file wants it, and a
    session fixture would build it even for a run that deselects these tests.

    Consumers must treat the returned rows as READ-ONLY; they are shared across the three
    tests now, and mutating them would couple tests through the fixture.
    """
    return split_by_confounding(topology, episodes=70)


@pytest.mark.slow
def test_confounded_episodes_can_be_scored(split_70, topology):
    """THE REGRESSION. Reported success was structurally 0.000 here.

    A rate of zero on a whole regime is not a hard task -- it is an unearnable metric, and
    the two look identical in a results table.
    """
    _, dirty = split_70
    assert len(dirty) >= 5, "need confounded episodes for this test to mean anything"
    credited = [r["per_agent"][agent]["mass_credit"]
                for r in dirty for agent in topology.agents]
    assert max(credited) > 0.0, (
        "no confounded episode gave ANY agent non-zero credit mass -- the metric cannot "
        "be earned in the regime the two-agent design exists to study")


@pytest.mark.slow
def test_unconfounded_episodes_can_be_scored(split_70, topology):
    """The control. If this fails too, the metric is broken outright rather than blind to
    one regime, which is a different diagnosis."""
    clean, _ = split_70
    assert len(clean) >= 5
    assert max(r["per_agent"][agent]["mass_credit"]
               for r in clean for agent in topology.agents) > 0.0


@pytest.mark.slow
def test_success_is_attainable_at_all(split_70):
    """Somewhere in a reasonable run, the full three-part criterion must actually fire.
    A criterion that never returns True cannot distinguish any two policies."""
    clean, dirty = split_70
    assert any(r["success"] for r in clean + dirty), (
        "the reported success criterion never fired in 70 episodes")


def test_credit_mass_never_exceeds_equivalence_mass(topology):
    """Sanity on the nesting: the credit set is a SUBSET of the equivalence class, so its
    mass cannot be larger. Catches an indexing mismatch between the two paths."""
    env = TwoAgentEnv(MAConfig(topology=topology, n_obs=600, n_int=100, budget=3,
                                 disclose_regime=True))
    for episode in range(8):
        env.reset(seed=episode)
        env.step({0: 0, 1: 2})
        for agent in env.topology.agents:
            row = agent_report(env, agent)
            assert row["mass_credit"] <= row["mass_equivalent"] + 1e-9


def test_the_truth_is_in_its_own_credit_set(topology):
    """Kept from the earlier suite. Necessary but NOT sufficient -- it passed throughout the
    period when the metric was scoring 0.000 on confounded episodes, because the bug was in
    the posterior indexing rather than in the set."""
    env = TwoAgentEnv(MAConfig(topology=topology, n_obs=400, n_int=100, budget=2))
    from ma.baselines import _Window
    for episode in range(10):
        env.reset(seed=episode)
        for agent in env.topology.agents:
            window = env.windows[agent]
            truth = window.induced(env.true_adjacency)
            mask = credit_set(window, truth)
            index = next(i for i, dag in enumerate(_Window.get(window.k).dags)
                         if np.array_equal(dag, truth))
            assert mask[index]
