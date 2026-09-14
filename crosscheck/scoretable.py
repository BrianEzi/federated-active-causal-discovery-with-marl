"""Local scores for every (node, parent set), independent of any enumerated graph space.

This is the half of `PosteriorEngine` that never needed the DAG list. The score of a DAG
decomposes per node, so the expensive part -- `d * 2^(d-1)` marginal likelihoods, each a
pair of determinants -- depends only on `d`, the scorer and the data. Enumeration was only
ever needed afterwards, to gather those numbers into one score per graph.

Splitting it out is what lets the subset-DP posterior (`sa/dp.py`) run at d = 7 and beyond,
where the DAG list does not exist: 1.14 billion graphs at d=7 cannot be materialised, but
448 local scores can.

`PosteriorEngine` delegates here rather than keeping its own copy, so the enumerated and
DP paths are guaranteed to be scoring the same quantities -- if they drifted, the block 1
acceptance test would be comparing two different models and would pass or fail for reasons
that have nothing to do with the DP.
"""
from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np


class LocalScorer:
    """Builds the `[d, 2^(d-1)]` table of local scores.

    Parent sets are enumerated in a fixed order -- by size, then lexicographically within
    a size -- and that order is part of the interface: `parent_masks` is indexed the same
    way, and the DP relies on the correspondence.
    """

    def __init__(self, d: int, score):
        self.d = d
        self.score = score

        self.parent_sets: List[List[Tuple[int, ...]]] = []
        self.lookup: List[dict] = []
        for node in range(d):
            others = [k for k in range(d) if k != node]
            sets = [
                combo
                for r in range(len(others) + 1)
                for combo in itertools.combinations(others, r)
            ]
            self.parent_sets.append(sets)
            self.lookup.append({s: i for i, s in enumerate(sets)})

        self.n_parent_sets = max(len(s) for s in self.parent_sets)

        # Bitmask of each parent set, over the FULL node index (not the compressed
        # "others" index). The DP works in full-node masks throughout, so paying the
        # translation once here keeps the recurrence free of index arithmetic.
        self.parent_masks = np.zeros((d, self.n_parent_sets), dtype=np.int64)
        self.parent_sizes = np.zeros((d, self.n_parent_sets), dtype=np.int64)
        for node in range(d):
            for i, parents in enumerate(self.parent_sets[node]):
                mask = 0
                for p in parents:
                    mask |= 1 << p
                self.parent_masks[node, i] = mask
                self.parent_sizes[node, i] = len(parents)

        self._table_plan = [self._plan_node(node) for node in range(d)]

    def _plan_node(self, node: int) -> dict:
        """Where each of `node`'s marginals lives in the batched output.

        A local score is `marginal(parents + node) - marginal(parents)`, so the subsets
        needed are every parent set and every parent set with the node added. They are
        grouped by size, since only same-sized subsets can share a batched determinant,
        and the position of each one is recorded so the table can be assembled by gather
        rather than by lookup.
        """
        parent_sets = self.parent_sets[node]
        needed = set()
        for parents in parent_sets:
            needed.add(tuple(parents))
            needed.add(tuple(sorted(parents + (node,))))

        by_size: dict = {}
        for subset in sorted(needed, key=lambda t: (len(t), t)):
            by_size.setdefault(len(subset), []).append(subset)
        position = {s: i for size in by_size for i, s in enumerate(by_size[size])}

        return {
            "index_by_size": {p: np.array([list(s) for s in subsets], dtype=int)
                              for p, subsets in by_size.items() if p > 0},
            "with_size": np.array([len(s) + 1 for s in parent_sets]),
            "with_pos": np.array([position[tuple(sorted(s + (node,)))]
                                  for s in parent_sets]),
            "without_size": np.array([len(s) for s in parent_sets]),
            "without_pos": np.array([position[tuple(s)] for s in parent_sets]),
        }

    def table(self, samples: np.ndarray, intervened: np.ndarray) -> np.ndarray:
        """[d, n_parent_sets] local scores, computed once per node/parent-set pair.

        `intervened[i, j]` is truthy when node j was set by intervention in sample i.
        """
        d = self.d
        samples = np.asarray(samples)
        intervened = np.asarray(intervened)
        table = np.zeros((d, self.n_parent_sets))
        # Scorers may expose sufficient statistics, in which case the n rows are read once
        # per node rather than once per (node, parent set) -- 2^(d-1) times fewer passes.
        # Scorers that do not (BIC, KnownVariance) take the original path unchanged.
        use_stats = hasattr(self.score, "sufficient_stats") and hasattr(
            self.score, "local_score_from_stats")
        use_batched = use_stats and hasattr(self.score, "log_marginals_batched")

        for node in range(d):
            usable = intervened[:, node] < 0.5
            # Skip the copy entirely when nothing was intervened on this node, which is
            # every node at reset and most nodes for most of an episode.
            subset = samples if usable.all() else samples[usable]

            if use_batched:
                stats = self.score.sufficient_stats(subset)
                plan = self._table_plan[node]
                marginals = self.score.log_marginals_batched(
                    stats, plan["index_by_size"])
                # Assembled by gather: for each parent set, the marginal WITH the node
                # minus the marginal without. Size 0 is the empty subset, whose marginal
                # is zero by definition.
                with_values = np.array(
                    [marginals[p][i] for p, i in zip(plan["with_size"], plan["with_pos"])])
                without_values = np.array(
                    [0.0 if p == 0 else marginals[p][i]
                     for p, i in zip(plan["without_size"], plan["without_pos"])])
                table[node] = with_values - without_values
            elif use_stats:
                stats = self.score.sufficient_stats(subset)
                for i, parents in enumerate(self.parent_sets[node]):
                    table[node, i] = self.score.local_score_from_stats(node, parents, stats)
            else:
                for i, parents in enumerate(self.parent_sets[node]):
                    table[node, i] = self.score.local_score(node, parents, subset)
        return table
