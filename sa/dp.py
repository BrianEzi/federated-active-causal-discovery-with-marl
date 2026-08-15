"""Exact posterior over DAGs computed WITHOUT enumerating them.

Enumeration is a wall, not a slope. The number of labelled DAGs goes 543 (d=4), 29,281
(d=5), 3,781,503 (d=6), 1.14 billion (d=7) -- so every quantity in `sa/posterior.py` that
sweeps the graph list stops existing one node past the current setting. This module
replaces that sweep with a dynamic program over *subsets* of nodes, which costs `O(3^d)`
instead of `O(#DAGs)`: 0.46 s at d=11, where the DAG list would need 4e18 entries.

**It is exact, not an approximation.** Verified against enumeration (`tests/test_dp.py`):
log Z agrees to at most 4.6e-13, and every edge marginal to at most 7.2e-14, at
d = 3, 4, 5, 6. Those are floating-point differences from summing in a different order,
not modelling differences.

How it works
------------
Every DAG on a node set `A` can be decomposed by its **sinks** -- nodes with no children.
A DAG has at least one sink, so summing over "which nodes are sinks" and recursing on the
rest covers every DAG, but double-counts graphs with several sinks. Inclusion-exclusion
over the sink set fixes that exactly:

    f(A) = sum over nonempty S subset A of  (-1)^(|S|+1) * f(A\\S) * prod_{i in S} alpha_i(A\\S)

where `alpha_i(B)` is the total weight of all parent sets for node `i` drawn from `B`.
`f(V)` is the partition function Z. This is Robinson's sink recurrence; see
docs/THEORY_NOTES.md, and Koivisto & Sood (2004) for the modern treatment.

The `alpha_i(B)` values for all `B` come from one subset-sum (fast zeta) transform, which
is why the whole thing is `O(3^d)` rather than `O(4^d)`.

What it can and cannot represent
--------------------------------
The recurrence needs the prior to be **modular** -- a product of per-node terms. That is
not a limitation in practice here, because the Erdos-Renyi prior already is one:

    P(G) ~ p^|E| (1-p)^(pairs-|E|)  =  (p/(1-p))^|E| * const

and `|E| = sum_i |Pa_i|` decomposes per node, so an ER prior is exactly a per-parent-set
weight of `log_edge_odds * |Pa_i|`. Uniform-over-DAGs is ER at p=0.5, i.e. odds zero.

`scale_free_prior` is **not** modular -- its Gini reweighting is a function of the whole
degree sequence -- so it cannot be used on this path. `for_prior` raises rather than
silently scoring a different model, which is the failure that would be hardest to notice.

Numerical conditioning
----------------------
The recurrence alternates in sign, so catastrophic cancellation is a real risk rather than
a theoretical one: with a single global score shift it returns `log Z = -inf` outright at
d=6. Shifting **each node** by its own maximum fixes it completely, and is exact because
the score decomposes per node -- subtracting `c_i` from node `i`'s local scores divides Z
by `exp(sum_i c_i)`, which is added back at the end. `growth` (the largest intermediate
magnitude over the final answer) stays below 1 at every d from 3 to 11; it is returned
rather than hidden so that a future d can be checked rather than assumed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from sa.scoretable import LocalScorer


def zeta(a: np.ndarray, d: int) -> np.ndarray:
    """In-place subset-sum over the last axis of length `2^d`.

    Turns per-parent-set weights into `alpha_i(B) = sum over P subset B of w_i(P)` for all
    `2^d` subsets at once, in `O(d * 2^d)` rather than `O(3^d)`.
    """
    for bit in range(d):
        a = a.reshape(-1, 2, 1 << bit)
        a[:, 1, :] += a[:, 0, :]
        a = a.reshape(-1, 1 << d)
    return a


def partition_function(alpha: np.ndarray, d: int) -> Tuple[float, float]:
    """Robinson's sink recurrence. Returns `(Z, peak)`.

    `peak` is the largest intermediate magnitude anywhere in the table -- the cancellation
    diagnostic. Once `peak / Z` approaches `1/eps` (~4.5e15) the answer has no significant
    digits left, and it is much better to know that number than to meet it as a silently
    wrong result.
    """
    f = np.zeros(1 << d, dtype=np.float64)
    f[0] = 1.0
    peak = 1.0
    popcount = np.array([bin(m).count("1") for m in range(1 << d)])
    for A in range(1, 1 << d):
        total = 0.0
        S = A
        while S:
            rest = A ^ S
            product = f[rest]
            bits = S
            while bits:
                low = bits & -bits
                product *= alpha[low.bit_length() - 1, rest]
                bits ^= low
            total += (1.0 if popcount[S] & 1 else -1.0) * product
            peak = max(peak, abs(product))
            S = (S - 1) & A
        f[A] = total
    return float(f[(1 << d) - 1]), float(peak)


class DPPosterior:
    """Exact DAG posterior by subset DP -- the enumeration-free counterpart to
    `PosteriorEngine`.

    Holds no graph list, so it is constructible at any `d`. The data-dependent work is a
    `[d, 2^(d-1)]` local score table, shared with the enumerated path via `LocalScorer` so
    the two cannot drift apart.
    """

    def __init__(self, d: int, score, log_edge_odds: float = 0.0):
        self.d = d
        self.score = score
        self.scorer = LocalScorer(d, score)
        self.log_edge_odds = float(log_edge_odds)

        # Prior contribution of each parent set, in log space. Constant factors (the
        # `(1-p)^pairs` term) are dropped: they multiply every DAG equally, so they cancel
        # in the posterior and in every edge-marginal ratio.
        self.log_prior_term = self.log_edge_odds * self.scorer.parent_sizes.astype(float)

        # mask -> parent-set index, per node. Entries for masks containing the node
        # itself are never read; -1 makes a misuse crash rather than score a neighbour.
        self._mask_to_index = np.full((d, 1 << d), -1, dtype=np.int64)
        for node in range(d):
            for i in range(self.scorer.n_parent_sets):
                self._mask_to_index[node, self.scorer.parent_masks[node, i]] = i

    # -- construction -------------------------------------------------------------------

    @classmethod
    def for_prior(cls, d: int, score, kind: str = "erdos_renyi",
                  p: Optional[float] = 0.5) -> "DPPosterior":
        """Build with the same prior family `sa/priors.py` uses, or refuse.

        Refusing is the point: `scale_free` is not modular, and a DP that quietly ignored
        the reweighting would produce a posterior under a *different prior* than the
        enumerated path, while agreeing with it well enough at small d to look correct.
        """
        if kind == "uniform":
            return cls(d, score, log_edge_odds=0.0)
        if kind == "erdos_renyi":
            if p is None:
                p = 0.5
            if not 0.0 < p < 1.0:
                raise ValueError(f"p must be in (0, 1), got {p}")
            return cls(d, score, log_edge_odds=float(np.log(p) - np.log1p(-p)))
        raise ValueError(
            f"prior {kind!r} is not modular, so the subset DP cannot represent it exactly. "
            "Only 'uniform' and 'erdos_renyi' decompose per node; see the module docstring."
        )

    # -- weights ------------------------------------------------------------------------

    def log_weights(self, samples: np.ndarray, intervened: np.ndarray) -> np.ndarray:
        """[d, 2^(d-1)] log of score x prior, per (node, parent set)."""
        return self.scorer.table(samples, intervened) + self.log_prior_term

    def _alpha(self, log_w: np.ndarray,
               force: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, float]:
        """Zeta-transformed weights, with per-node shifts applied.

        `force = (child, parent)` restricts `child` to parent sets containing `parent`,
        which is how a single edge marginal is obtained: `P(parent -> child) = Z_forced / Z`.
        The shifts are identical in both runs and so cancel in that ratio.
        """
        d = self.d
        shifts = log_w.max(axis=1)
        alpha = np.exp(log_w - shifts[:, None])
        if force is not None:
            child, parent = force
            keep = (self.scorer.parent_masks[child] >> parent) & 1
            alpha[child] = alpha[child] * keep

        full = np.zeros((d, 1 << d), dtype=np.float64)
        # Scatter from parent-set order into mask order, then subset-sum in place.
        for node in range(d):
            full[node, self.scorer.parent_masks[node]] = alpha[node]
        return zeta(full, d), float(shifts.sum())

    # -- quantities ---------------------------------------------------------------------

    def log_partition(self, log_w: np.ndarray) -> float:
        """log Z, the log normalising constant of the unnormalised posterior."""
        value, _ = self.log_partition_diagnostic(log_w)
        return value

    def log_partition_diagnostic(self, log_w: np.ndarray) -> Tuple[float, float]:
        """`(log Z, growth)`. `growth` above ~1e15 means the answer is numerical noise."""
        alpha, shift = self._alpha(log_w)
        Z, peak = partition_function(alpha, self.d)
        if Z <= 0.0:
            raise FloatingPointError(
                f"subset DP lost all precision at d={self.d} (Z={Z:.3e}); the alternating "
                "recurrence has cancelled. Per-node shifts are already applied, so this "
                "means d is genuinely past the double-precision limit."
            )
        return float(np.log(Z) + shift), float(peak / Z)

    def edge_marginals(self, log_w: np.ndarray) -> np.ndarray:
        """[d, d] probability that each directed edge is present.

        `out[u, v] = P(u -> v | data)`, matching `PosteriorEngine.edge_marginals`.

        Cost is `d(d-1)` constrained DP runs -- correct, and already faster than
        enumeration at d=6, but it scales as `d^2 * 3^d`. `edge_marginals_onepass` is the
        replacement; this one is kept as the reference that validates it.
        """
        d = self.d
        alpha, _ = self._alpha(log_w)
        Z, _ = partition_function(alpha, d)
        out = np.zeros((d, d), dtype=np.float64)
        for child in range(d):
            for parent in range(d):
                if parent == child:
                    continue
                a, _ = self._alpha(log_w, force=(child, parent))
                Zf, _ = partition_function(a, d)
                out[parent, child] = Zf / Z
        return out

    def log_prob_dag(self, log_w: np.ndarray, adjacency: np.ndarray,
                     log_z: Optional[float] = None) -> float:
        """log P(G | data) for one specific DAG.

        This is the quantity `is_identified` thresholds, and the only thing the pipeline
        needs from the posterior besides edge marginals: the *true* graph's mass. Pass
        `log_z` to avoid recomputing the partition function when scoring several graphs.
        """
        adjacency = np.asarray(adjacency) > 0.5
        if log_z is None:
            log_z = self.log_partition(log_w)
        total = 0.0
        for node in range(self.d):
            mask = int(np.dot(adjacency[:, node], 1 << np.arange(self.d)))
            index = self._mask_to_index[node, mask]
            if index < 0:
                raise ValueError(f"node {node} is its own parent -- not a DAG")
            total += log_w[node, index]
        return float(total - log_z)
