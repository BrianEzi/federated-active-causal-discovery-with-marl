"""Is an agent's local view of the world a DAG at all?

**This is the risk that could force a redesign, and it is measured before anything is built
on top of it.**

Agent A observes `{0, 1, 4, 5}` and would naturally model a DAG over them. But the true
generative model also contains `{2, 3}`, which can be parents of the exposed nodes. From
A's perspective those are *unobserved common causes*. A DAG model over A's view is then
**misspecified**: under latent confounding the correct object is a maximal ancestral graph
(MAG), not a DAG, and A's local BGe posterior is not a correct posterior over anything.

That is not automatically bad news -- it may be the most interesting thing in the two-agent
setting, because it is a precise structural reason why coordination is *necessary* rather
than merely helpful. But it changes what can be claimed, so it has to be a measurement.

The test
--------
A node `u` hidden from an agent confounds that agent's view when two or more of the agent's
observed nodes are reachable from `u` by directed paths whose **intermediate nodes are all
hidden**. Intermediate nodes matter: with `2 -> 3 -> 4` and `2 -> 5`, node 2 confounds 4 and
5 through the hidden node 3, even though 2 has only one observed child. Checking children
alone would miss it and understate the problem.

This is the standard latent-projection construction (Verma & Pearl 1990; Richardson &
Spirtes 2002 for MAGs): project the hidden nodes out, and a bidirected edge appears exactly
where such a common source exists.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ma.topology import Topology, edge_class, masked_indices


def latent_projection_pairs(adjacency: np.ndarray, observed: Sequence[int],
                            hidden: Sequence[int]) -> list:
    """Pairs of observed nodes sharing a hidden common source.

    Each pair is one bidirected edge in the latent projection -- one place where the
    agent's DAG model is wrong.
    """
    adjacency = np.asarray(adjacency) > 0.5
    observed_set = set(int(x) for x in observed)
    hidden_set = set(int(x) for x in hidden)

    pairs = set()
    for source in hidden_set:
        # Observed nodes reachable from `source` through hidden intermediates only.
        reached = set()
        stack = [source]
        seen = {source}
        while stack:
            node = stack.pop()
            for child in np.flatnonzero(adjacency[node]).tolist():
                if child in observed_set:
                    reached.add(child)
                elif child in hidden_set and child not in seen:
                    seen.add(child)
                    stack.append(child)
        ordered = sorted(reached)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs.add((ordered[i], ordered[j]))
    return sorted(pairs)


def is_confounded(adjacency: np.ndarray, observed: Sequence[int],
                  hidden: Sequence[int]) -> bool:
    """Does this agent's view contain at least one latent confounder?"""
    return len(latent_projection_pairs(adjacency, observed, hidden)) > 0


def measure_topology(space, topology: Topology, max_graphs: int = None) -> Dict:
    """GATE-M3's first output: how misspecified is each agent's local DAG model?

    Enumerated over the masked space rather than sampled, so this is a computation and not
    an estimate. Reports, over all graphs the topology permits:

      `confounded_a` / `confounded_b`  -- fraction of graphs giving that agent a
                                          latently-confounded view.
      `confounded_either`              -- fraction where at least one agent's model is
                                          misspecified. This is the number that decides
                                          whether local DAG posteriors are defensible.
      `mean_bidirected`                -- average number of bidirected edges induced, i.e.
                                          how badly wrong, not just how often.
    """
    indices = masked_indices(space, topology)
    if max_graphs is not None and len(indices) > max_graphs:
        indices = np.random.default_rng(0).choice(indices, size=max_graphs, replace=False)

    observed_a, hidden_a = topology.observed_by(0), topology.hidden_from(0)
    observed_b, hidden_b = topology.observed_by(1), topology.hidden_from(1)

    n_a = n_b = n_either = 0
    total_pairs = 0
    for index in indices:
        adjacency = space.dags[index]
        pairs_a = latent_projection_pairs(adjacency, observed_a, hidden_a)
        pairs_b = latent_projection_pairs(adjacency, observed_b, hidden_b)
        n_a += bool(pairs_a)
        n_b += bool(pairs_b)
        n_either += bool(pairs_a or pairs_b)
        total_pairs += len(pairs_a) + len(pairs_b)

    n = len(indices)
    return {
        "topology": topology.name,
        "n_graphs": int(n),
        "n_graphs_in_space": int(len(masked_indices(space, topology))),
        "confounded_a": n_a / n,
        "confounded_b": n_b / n,
        "confounded_either": n_either / n,
        "mean_bidirected": total_pairs / n,
    }


def ambiguity_location(space, topology: Topology, max_graphs: int = None) -> Dict:
    """GATE-M3's second output: WHERE does residual ambiguity sit?

    For each graph, the edges whose orientation is not determined by its Markov equivalence
    class are the ones that differ among class members. Classified by position relative to
    the federation boundary. The design is only interesting if a real share of the
    difficulty is at the boundary -- if all of it is interior, each agent can solve its own
    half alone and there is nothing to coordinate about.
    """
    indices = masked_indices(space, topology)
    allowed = topology.allowed_edges()
    rng = np.random.default_rng(0)
    if max_graphs is not None and len(indices) > max_graphs:
        indices = rng.choice(indices, size=max_graphs, replace=False)

    # Group the masked graphs by Markov equivalence class.
    mec_of = space.mec_id[indices]
    order = np.argsort(mec_of, kind="stable")
    indices, mec_of = indices[order], mec_of[order]
    boundaries = np.flatnonzero(np.diff(mec_of)) + 1
    groups = np.split(np.arange(len(indices)), boundaries)

    counts = {"interior": 0, "private_exposed": 0, "exposed_exposed": 0}
    singletons = 0
    d = topology.d
    for group in groups:
        if len(group) == 1:
            singletons += 1
            continue
        members = np.asarray(space.dags[indices[group]]) > 0.5
        # An edge is ambiguous within the class when it is not present in every member.
        varies = members.any(axis=0) & ~members.all(axis=0)
        for u in range(d):
            for v in range(d):
                if varies[u, v] and allowed[u, v]:
                    label = edge_class(topology, u, v)
                    if label in counts:
                        counts[label] += 1

    total = sum(counts.values())
    return {
        "topology": topology.name,
        "n_classes": len(groups),
        "singleton_classes": singletons,
        "singleton_fraction": singletons / max(len(groups), 1),
        "ambiguous_edge_counts": counts,
        "ambiguous_edge_shares": {k: (v / total if total else 0.0)
                                  for k, v in counts.items()},
    }
