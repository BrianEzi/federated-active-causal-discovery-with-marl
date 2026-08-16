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


def bidirected_pairs(adjacency: np.ndarray, observed: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """Global-node-id pairs carrying a bidirected edge in the projection onto `observed`."""
    observed = list(observed)
    proj = latent_projection(adjacency, observed)
    out = []
    for i, j in combinations(range(len(observed)), 2):
        if proj[i, j] == BIDIRECTED:
            out.append((observed[i], observed[j]))
    return tuple(out)
