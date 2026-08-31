"""The factored attribution backend must agree with the enumerated one where both can run.

WHY A CROSSCHECK AND NOT ONLY UNIT TESTS. `cb/factored_attribution.py` exists because the
structure enumeration (3^edges) and the attribution enumeration are independent and only the
first is expensive -- at k=12, 5.0e10 against 482. Swapping the structure half for the
factored belief is only sound if the attribution half is genuinely unchanged, and the way to
establish that is to run both on identical episodes at a size the enumerated one survives.

WHAT MUST MATCH, and what legitimately need not. The candidate SETS should agree: the same
owner hypotheses survive the same partner messages, because that pruning is the same code on
the same evidence. What may differ is WHEN a pair enters the attribution scope at all -- the
factored structure belief is deliberately conservative and settles a mark later than the
enumeration, which can only make it slower to attribute, never wrong. So the assertion is
one-sided: anything the factored backend settles, the enumerated one must also settle, and
they must never disagree on an answer.
"""
from __future__ import annotations

import numpy as np
import pytest

from cb.attribution import AttributedVersionSpaceBackend, score_groups
from cb.factored_attribution import FactoredAttributedBackend
from ma.env import MAConfig, TwoAgentEnv
from ma.topology import federated_topology


def _env(private=2, shared=2, agents=3):
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(agents, private, shared), n_obs=200, n_int=40, budget=16,
        turn_order="round_robin", belief_backend="factored", action_modes=("vary",),
        claim_bar=1.0, reward_criterion="claims", policy_arch="gnn_portable",
        graph_model="sf", sf_m=2, episode_mix="confounded", vs_evidence="oracle"))


def _drive(env, backends, episode):
    """Sweep every window position, feeding both backends the identical evidence."""
    from cb.attribution import response_signature
    result = env.reset(seed=episode)
    for agent, backend in backends.items():
        backend.reset(env._true_mag(agent), adjacency=env.true_adjacency,
                      topology=env.topology)
    turns = {a: 0 for a in env.topology.agents}
    while not result.done:
        active = env.active_agent()
        actions = {a: env.windows[a].action_index(
                       env.windows[a].nodes[turns[a] % env.windows[a].k], "vary")
                   for a in env.topology.agents}
        result = env.step(actions)
        for agent, backend in backends.items():
            backend.edge_marginals(env.samples[:, env.windows[agent].nodes], env.known[agent])
        if active is not None:
            node, _ = env.last_chosen[active]
            if node is not None and node not in env.topology.exposed:
                for agent, backend in backends.items():
                    if agent == active or not backend.true_groups:
                        continue
                    hit = response_signature(env.true_adjacency, env.topology, agent,
                                             backend.true_groups, node)
                    moved = frozenset(p for g, h in zip(backend.true_groups, hit) if h
                                      for p in g.pairs())
                    if moved:
                        backend.observe_partner(active, moved)
            turns[active] += 1
    return backends


@pytest.mark.parametrize("episode", range(6))
def test_factored_attribution_never_contradicts_the_enumerated_backend(episode):
    """One-sided: what the factored backend settles, the enumerated one must settle the same.

    Not equality of the unsure counts -- the factored structure belief reaches a mark later,
    so it is entitled to be less decided. What it is never entitled to be is DIFFERENTLY
    decided.
    """
    env = _env()
    agents = list(env.topology.agents)
    fast = {a: FactoredAttributedBackend(env.windows[a].k, n_agents=len(agents), agent=a,
                                         evidence="oracle") for a in agents}
    _drive(env, fast, episode)

    env2 = _env()
    slow = {a: AttributedVersionSpaceBackend(env2.windows[a].k, n_agents=len(agents),
                                             agent=a) for a in agents}
    _drive(env2, slow, episode)

    for agent in agents:
        f = score_groups(fast[agent].last, fast[agent].true_groups, bar=1.0)
        s = score_groups(slow[agent].last, slow[agent].true_groups, bar=1.0)
        assert f["total"] == s["total"], "the two backends disagree on the TRUE groups"
        # Neither may report a confident misattribution.
        assert f["wrong"] == 0, (
            f"episode {episode}, agent {agent}: factored backend settled {f['wrong']} "
            f"attribution(s) WRONG -- the soundness guarantee has broken")
        # Anything the factored one settles right, the enumerated one must too.
        settled_fast = {g for g, outcome, _ in f["detail"] if outcome == "right"}
        settled_slow = {g for g, outcome, _ in s["detail"] if outcome == "right"}
        assert settled_fast <= settled_slow, (
            f"episode {episode}, agent {agent}: the factored backend settled "
            f"{settled_fast - settled_slow} that the enumerated one did not -- it cannot be "
            f"MORE decided than the belief it approximates")


def test_the_factored_backend_says_nothing_without_partner_evidence():
    """The soundness floor: with no partner messages, attribution is UNSURE, never wrong.

    This is the property that matters most, because a confident misattribution is worse than
    no attribution at all -- and it is exactly what a scope bug produced before `scope`
    existed: 16 of 76 true groups reported WRONG with no evidence in play.
    """
    env = _env()
    agents = list(env.topology.agents)
    backends = {a: FactoredAttributedBackend(env.windows[a].k, n_agents=len(agents), agent=a,
                                             evidence="oracle") for a in agents}
    total = wrong = 0
    for episode in range(8):
        result = env.reset(seed=100 + episode)
        for agent, backend in backends.items():
            backend.reset(env._true_mag(agent), adjacency=env.true_adjacency,
                          topology=env.topology)
        turns = {a: 0 for a in agents}
        while not result.done:
            active = env.active_agent()
            result = env.step({a: env.windows[a].action_index(
                                   env.windows[a].nodes[turns[a] % env.windows[a].k], "vary")
                               for a in agents})
            for agent, backend in backends.items():
                backend.edge_marginals(env.samples[:, env.windows[agent].nodes],
                                       env.known[agent])
            if active is not None:
                turns[active] += 1
        for agent, backend in backends.items():
            score = score_groups(backend.last, backend.true_groups, bar=1.0)
            total += score["total"]
            wrong += score["wrong"]
    assert total > 0, "no true groups in any episode -- the test proves nothing"
    assert wrong == 0, (f"{wrong} of {total} true groups settled WRONG with no partner "
                        f"evidence at all")
