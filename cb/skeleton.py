"""Adjacency search -- the PC/FCI phase that decides which pairs are connected at all.

Measured to dominate a constraint-based run (`scripts/cb_feasibility.py`): 3,379 CI tests
and 0.47 s at d=30, against 106 s for the exact DAG posterior at k=15. Orientation is
O(k^3) bookkeeping by comparison.

THE ALGORITHM. Start fully connected. For each still-connected pair, look for a set of
their neighbours that renders them independent; find one and the edge is removed, with that
set recorded as the pair's SEPARATING SET. Conditioning-set size grows one at a time, so
the cheapest explanations are tried first and most edges die at level 0 or 1.

The separating sets are not a by-product -- `cb/orient.py` needs them, and they are the
only reason collider orientation is possible at all.

`max_cond` caps conditioning-set size. Uncapped, the search is exponential in the maximum
degree. The cap is what the sparse-graph literature relies on, it is a REAL approximation,
and it is surfaced in the return value rather than hidden: `truncated` says whether the cap
was ever actually hit, so a caller can tell an exact run from an approximate one.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np


class Skeleton:
    """Undirected adjacency plus the separating set for every removed edge."""

    def __init__(self, adjacency: np.ndarray, sepsets: Dict[Tuple[int, int], FrozenSet[int]],
                 ci_tests: int, truncated: bool):
        self.adjacency = adjacency
        self.sepsets = sepsets
        self.ci_tests = int(ci_tests)
        self.truncated = bool(truncated)

    @property
    def k(self) -> int:
        return int(self.adjacency.shape[0])

    def neighbours(self, node: int):
        return [int(w) for w in np.flatnonzero(self.adjacency[node])]

    def sepset(self, u: int, v: int) -> Optional[FrozenSet[int]]:
        return self.sepsets.get((u, v) if u <= v else (v, u))


def estimate_skeleton(test, k: int, max_cond: int = 3) -> Skeleton:
    """Run the adjacency search over a `k`-node window using `test`.

    `test` needs one method, `independent(x, y, cond) -> bool`, which is what keeps the
    engine indifferent to whether that is a partial correlation or a kernel statistic.
    """
    adjacency = np.ones((k, k), dtype=bool)
    np.fill_diagonal(adjacency, False)
    sepsets: Dict[Tuple[int, int], FrozenSet[int]] = {}
    truncated = False

    for level in range(max_cond + 1):
        # Snapshot the pair list per level: removing an edge mid-level must not change
        # which pairs are examined at THIS level, only at the next. Iterating over a live
        # adjacency makes the result depend on pair ordering, which is not a property the
        # algorithm is supposed to have.
        pairs = [(u, v) for u, v in combinations(range(k), 2) if adjacency[u, v]]
        for u, v in pairs:
            if not adjacency[u, v]:
                continue
            # Candidates are u's OTHER neighbours -- conditioning on v itself is meaningless
            # and conditioning on non-neighbours is unnecessary (PC's key economy).
            candidates = [w for w in range(k) if adjacency[u, w] and w != v]
            if len(candidates) < level:
                continue
            for cond in combinations(candidates, level):
                if test.independent(u, v, cond):
                    adjacency[u, v] = adjacency[v, u] = False
                    sepsets[(u, v)] = frozenset(cond)
                    break

        degrees = adjacency.sum(axis=1)
        if degrees.max(initial=0) <= level:
            break                       # nothing left that could be separated at level+1
        if level == max_cond and degrees.max(initial=0) > level:
            truncated = True            # the cap bound, so this run is approximate

    return Skeleton(adjacency, sepsets, getattr(test, "calls", 0), truncated)
