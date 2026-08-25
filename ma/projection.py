"""Latent projection of a DAG onto one agent's observable set.

What an agent can learn about its own window, in the limit of infinite observational
data, is not a DAG -- it is a MAG (maximal ancestral graph). The other agent's private
nodes are marginalised out, and where one of them is a common cause of two visible
variables, the projection carries a *bidirected* edge meaning "these two share an
unobserved common cause", with no claim about what or where that cause is.

This module builds that projection so we can ask a structural question the whole
two-agent belief representation depends on:

    can a bidirected edge ever touch an agent's PRIVATE node?

If it cannot -- if confounding is confined to the shared set `X` -- then an agent's
belief is a DAG over its own window plus one flag per shared *pair*, the DAG part stays
decomposable, and the subset DP carries over untouched. If it can, the belief needs full
MAG machinery and the score stops decomposing. See docs/MA_DESIGN.md sections 3 and 12.

Definitions used here are the textbook ones (Richardson & Spirtes 2002):

  adjacency    u and v are adjacent in the MAG iff NO subset of the remaining observed
               variables d-separates them. (Equivalently: an inducing path exists. The
               separation form is used because it is directly checkable.)
  orientation  u -> v if u is an ancestor of v in the underlying DAG; v -> u if the
               reverse; u <-> v if neither is an ancestor of the other.

Brute force over all separating subsets: correct by construction and fast enough at the
sizes we enumerate. It is a verification tool, not an inference engine.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence, Tuple

import numpy as np

# Edge codes in the returned projection matrix.
NO_EDGE = 0
DIRECTED = 1      # proj[u, v] == DIRECTED means u -> v
BIDIRECTED = 2    # symmetric: proj[u, v] == proj[v, u] == BIDIRECTED


def ancestor_matrix(adjacency: np.ndarray) -> np.ndarray:
    """`[d, d]` bool, `anc[u, v]` iff there is a directed path u -> ... -> v.

    Reflexive closure is excluded: a node is not its own ancestor here, which matches the
    way ancestry is used below (`u -> v` requires a genuine path).
    """
    reach = np.asarray(adjacency, dtype=bool).copy()
    d = reach.shape[0]
    for k in range(d):                       # Floyd-Warshall transitive closure
        reach |= np.outer(reach[:, k], reach[k, :])
    return reach


def d_separated(adjacency: np.ndarray, u: int, v: int, cond: Sequence[int]) -> bool:
    """Is `u` d-separated from `v` given `cond` in the DAG `adjacency`?

    Standard moralisation test: restrict to the ancestral subgraph of the nodes involved,
    moralise (marry co-parents), drop the conditioning set, and ask whether u and v are
    still connected.
    """
    adj = np.asarray(adjacency, dtype=bool)
    d = adj.shape[0]
    cond = list(cond)

    # Ancestral subgraph of {u, v} u cond, inclusive of the nodes themselves.
    anc = ancestor_matrix(adj)
    keep = np.zeros(d, dtype=bool)
    for node in [u, v] + cond:
        keep[node] = True
        keep |= anc[:, node]

    # Moralise: undirected skeleton plus an edge between every pair sharing a child.
    sub = adj & np.outer(keep, keep)
    moral = sub | sub.T
    for child in range(d):
        if not keep[child]:
            continue
        parents = np.flatnonzero(sub[:, child])
        for a, b in combinations(parents, 2):
            moral[a, b] = moral[b, a] = True

    # Remove the conditioning set, then test reachability.
    open_nodes = keep.copy()
    for node in cond:
        open_nodes[node] = False
    if not (open_nodes[u] and open_nodes[v]):
        return True    # u or v conditioned on -- not a case we generate, but be safe

    moral = moral & np.outer(open_nodes, open_nodes)
    seen = {u}
    stack = [u]
    while stack:
        node = stack.pop()
        if node == v:
            return False
        for nxt in np.flatnonzero(moral[node]):
            nxt = int(nxt)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return True


def latent_projection(adjacency: np.ndarray, observed: Sequence[int]) -> np.ndarray:
    """MAG over `observed`, as a `[k, k]` matrix of edge codes indexed by position in
    `observed` (not by global node id).
    """
    observed = list(observed)
    k = len(observed)
    anc = ancestor_matrix(adjacency)
    proj = np.zeros((k, k), dtype=np.int8)

    for i, j in combinations(range(k), 2):
        u, v = observed[i], observed[j]
        rest = [w for w in observed if w not in (u, v)]

        separable = False
        for size in range(len(rest) + 1):
            for cond in combinations(rest, size):
                if d_separated(adjacency, u, v, cond):
                    separable = True
                    break
            if separable:
                break
        if separable:
            continue                      # non-adjacent in the MAG

        if anc[u, v]:
            proj[i, j] = DIRECTED
        elif anc[v, u]:
            proj[j, i] = DIRECTED
        else:
            proj[i, j] = proj[j, i] = BIDIRECTED
    return proj


def observational_skeleton(adjacency: np.ndarray, observed: Sequence[int]):
    """(adjacency [k, k] bool, sepsets {(i, j): frozenset}) -- the infinite-data limit of
    what OBSERVATION alone can know about the window: which pairs are connected, and for
    each unconnected pair one witnessing separating set (window positions).

    Added 2026-08-25 for the oracle warm start ("start the agents at the equivalence
    class"): the same search `latent_projection` runs, but keeping the separating set,
    which is what collider orientation consumes. Conditioning sets range over OBSERVED
    nodes only, so nothing an observational method could not know leaks through --
    in particular, a hidden confounder's pair stays ADJACENT here, and detecting the
    confounding remains entirely the interventions' job.
    """
    observed = list(observed)
    k = len(observed)
    adj = np.zeros((k, k), dtype=bool)
    sepsets = {}
    for i, j in combinations(range(k), 2):
        u, v = observed[i], observed[j]
        rest = [w for w in observed if w not in (u, v)]
        found = None
        for size in range(len(rest) + 1):
            for cond in combinations(rest, size):
                if d_separated(adjacency, u, v, cond):
                    found = frozenset(observed.index(w) for w in cond)
                    break
            if found is not None:
                break
        if found is None:
            adj[i, j] = adj[j, i] = True
        else:
            sepsets[(i, j)] = found
    return adj, sepsets


def bidirected_pairs(adjacency: np.ndarray, observed: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """Global-node-id pairs carrying a bidirected edge in the projection onto `observed`."""
    observed = list(observed)
    proj = latent_projection(adjacency, observed)
    out = []
    for i, j in combinations(range(len(observed)), 2):
        if proj[i, j] == BIDIRECTED:
            out.append((observed[i], observed[j]))
    return tuple(out)


# =======================================================================================
# MERGED FROM ma/confounding.py, 2026-08-23.
#
# The two modules both answered "which observed pairs are confounded", by DIFFERENT
# criteria, and each had its own graph-walking code. They are now one module -- but the
# criteria are NOT unified, because they genuinely disagree and one of them backs a
# reported number.
#
#   bidirected_pairs      the MAG definition (Richardson & Spirtes 2002). A pair is
#                         bidirected only if no observed subset d-separates it AND neither
#                         node is an ancestor of the other. AUTHORITATIVE -- this is what
#                         ma/env._confounded_positions scores identification against, and
#                         what produced the 2.3% structural-ceiling figure.
#
#   common_source_pairs   "these two share a hidden common source, reachable through
#                         hidden intermediates". A SUFFICIENT condition for confounding,
#                         but it OVER-REPORTS relative to the MAG: where a real edge
#                         u -> v coexists with a hidden common cause, u IS an ancestor of
#                         v, so the MAG carries u -> v and this criterion still returns the
#                         pair. Retained because measure_topology's published numbers were
#                         computed with it; see test_the_two_confounding_criteria_diverge.
#
# Renamed from `latent_projection_pairs` on the merge: the old name implied it computed the
# latent projection, which is what `latent_projection` above actually does.
# =======================================================================================

from typing import Dict

from ma.topology import Topology, edge_class, masked_indices


def common_source_pairs(adjacency: np.ndarray, observed: Sequence[int],
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
    return len(common_source_pairs(adjacency, observed, hidden)) > 0


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
        pairs_a = common_source_pairs(adjacency, observed_a, hidden_a)
        pairs_b = common_source_pairs(adjacency, observed_b, hidden_b)
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
