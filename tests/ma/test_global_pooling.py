"""The pooled global graph -- the object a federated causal discovery paper reports on.

Replaces `union_graph`, which reads `_Window.get(k).dags[map_index]` and therefore ENUMERATES:
it raises above k=5, so the entire factored ladder was out of its reach and the project had
no working global-graph metric at the sizes it actually reports.
"""
from itertools import combinations

import numpy as np
import pytest

from ma.baselines import RandomAgent, UncertaintyGreedyAgent
from ma.env import MAConfig, TwoAgentEnv
from ma.evaluate import global_graph_report, pooled_global_belief, union_graph
from ma.topology import federated_topology


def _env(priv, shared, agents, budget=20, seed=0):
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(agents, priv, shared), n_obs=60, n_int=20, budget=budget,
        turn_order="round_robin", belief_backend="factored", action_modes=("vary",),
        claim_bar=1.0, reward_criterion="claims", policy_arch="gnn_portable",
        graph_model="sf", sf_m=2, episode_mix="confounded", vs_evidence="oracle"), seed=seed)


def _play(env, make, episodes=4):
    pol = {a: make(a) for a in env.topology.agents}
    for ep in range(episodes):
        r = env.reset(seed=ep)
        while not r.done:
            r = env.step({a: pol[a](env, r) for a in env.topology.agents})
        yield env


def test_runs_at_k20_where_the_enumerating_union_cannot():
    """The whole point of the replacement."""
    env = _env(10, 10, 4)
    next(_play(env, lambda a: UncertaintyGreedyAgent(a, 0, bar=1.0), episodes=1))
    report = global_graph_report(env)
    assert report["global_pairs"] > 0
    with pytest.raises(Exception):          # enumeration wall, whatever it raises
        union_graph(env, {a: 0 for a in env.topology.agents})


def test_covers_exactly_the_pairs_some_window_contains_and_no_others():
    """Cross-private pairs cannot exist -- `allowed_edges` forbids them -- so no site holds a
    belief about them and they must be absent rather than scored as easy true non-edges."""
    env = _env(10, 10, 4)
    next(_play(env, lambda a: RandomAgent(a, 0, allow_clamp=False), episodes=1))
    pooled = pooled_global_belief(env)
    expected = set()
    for agent in env.topology.agents:
        nodes = env.windows[agent].nodes
        expected |= {tuple(sorted(p)) for p in combinations(nodes, 2)}
    assert set(pooled) == expected
    total = env.topology.d * (env.topology.d - 1) // 2
    assert len(pooled) < total          # cross-private pairs genuinely excluded


def test_pooling_never_excludes_the_truth_so_contradictions_stay_zero():
    """Every site's mark set contains the truth, so the intersection does too. A non-zero
    contradiction rate would mean a site's belief was unsound, not that the graph is hard."""
    env = _env(10, 10, 4)
    for done in _play(env, lambda a: UncertaintyGreedyAgent(a, 0, bar=1.0), episodes=6):
        assert global_graph_report(done)["global_contradiction"] == 0.0


def test_pooled_belief_is_at_least_as_tight_as_any_single_site():
    env = _env(6, 6, 3)
    next(_play(env, lambda a: UncertaintyGreedyAgent(a, 0, bar=1.0), episodes=1))
    pooled = pooled_global_belief(env)
    for agent, window in env.windows.items():
        belief = window.belief.last
        for u, v in combinations(range(window.k), 2):
            key = tuple(sorted((window.nodes[u], window.nodes[v])))
            entry = pooled[key]
            if entry["mark_disagreement"] or entry["contradiction"]:
                continue
            masses = np.array([1.0 - belief.adjacency[u, v], belief.directed[u, v],
                               belief.directed[v, u], belief.bidirected[u, v]])
            site_open = int((masses > 1e-9).sum())
            pooled_open = round(1.0 / (1.0 - entry["soft"])) if entry["soft"] < 1 else 0
            assert pooled_open <= site_open


def test_it_separates_a_competent_arm_from_a_random_one():
    """A metric that cannot tell greedy from random tells us nothing."""
    def mean_soft(make):
        env = _env(10, 10, 4)
        return float(np.mean([global_graph_report(e)["global_soft_shd"]
                              for e in _play(env, make, episodes=6)]))
    greedy = mean_soft(lambda a: UncertaintyGreedyAgent(a, 0, bar=1.0))
    random = mean_soft(lambda a: RandomAgent(a, 0, allow_clamp=False))
    assert greedy < random / 2, (greedy, random)


def test_a_shared_pair_counts_once_not_once_per_agent():
    """The defect in `scripts/shd.py`'s per-window average, which this metric exists to avoid."""
    env = _env(10, 10, 4)
    next(_play(env, lambda a: RandomAgent(a, 0, allow_clamp=False), episodes=1))
    pooled = pooled_global_belief(env)
    shared = tuple(sorted(env.topology.exposed[:2]))
    assert pooled[shared]["sites"] == len(env.topology.agents)   # seen by all
    per_window_slots = sum(w.k * (w.k - 1) // 2 for w in env.windows.values())
    assert len(pooled) < per_window_slots                        # but counted once
