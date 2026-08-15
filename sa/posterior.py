"""Exact Bayesian posterior over all DAGs.

Exact, not approximate: at d <= 4 the whole space is enumerable (543 DAGs), so there is
no need for MCMC or a variational approximation, and no approximation error to argue
about later. This is deliberately the *ceiling* condition -- it tells us what an agent
could achieve with a perfect belief state, which is the reference the scalable
edge-marginal representation gets compared against.

Cost is much lower than "543 DAGs x 4 nodes" suggests, because the score decomposes per
node: the score of a DAG is the sum of local terms depending only on a node and its
parents. So we compute each distinct `(node, parent set)` term ONCE and reuse it. That is
`d * 2^(d-1)` terms -- 12 at d=3, 32 at d=4, 80 at d=5 -- after which each DAG's score is
a gather-and-sum over d cached numbers. The expensive part (matrix determinants) scales
with parent sets, not with the number of graphs.

Interventional data: a hard intervention on node j replaces j's structural equation, so
samples where j was intervened say nothing about j's parents and are dropped from j's
local term. They remain valid parent values for every other node (Cooper & Yoo 1999).
This is what breaks Markov equivalence -- observationally, class members are tied
exactly; once a node is scored on a different sample subset from its class-mates, the tie
separates. See docs/THEORY_NOTES.md #3.
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sa.graphs import GraphSpace
from sa.scoretable import LocalScorer


# Rows of the DAG array converted to float64 at once when accumulating edge marginals.
# 250k x d x d float64 is ~72 MB at d=6, which is the memory/overhead trade-off.
_MARGINAL_CHUNK = 250_000


class PosteriorEngine:
    """Computes exact posteriors over a `GraphSpace`, reusing local scores.

    The mapping from (DAG, node) to parent-set index is precomputed once at construction,
    since it depends only on the graph space and never on the data.
    """

    def __init__(self, space: GraphSpace, score):
        self.space = space
        self.score = score
        d = space.d

        # All parent sets for each node, and a lookup from set -> index. Owned by
        # `LocalScorer`, which is shared with the enumeration-free DP path so the two
        # cannot drift apart; the attributes below stay as aliases because they are part
        # of this class's interface and are read directly by tests and scripts.
        self.scorer = LocalScorer(d, score)
        self.parent_sets: List[List[Tuple[int, ...]]] = self.scorer.parent_sets
        lookup: List[dict] = self.scorer.lookup

        # [N, d] index of each DAG's parent set for each node.
        #
        # Built by bitmask rather than by looping over graphs: at d=6 there are 3.78
        # million DAGs, and a Python loop over them costs minutes on every job that
        # constructs the engine. Each node's parent set is a subset of the other d-1
        # nodes, so it encodes as a (d-1)-bit integer, and a lookup table of size 2^(d-1)
        # -- 32 entries at d=6 -- turns the whole thing into one gather per node.
        self.parent_set_ids = np.empty((space.n_dags, d), dtype=np.int32)
        adjacency = np.asarray(space.dags) > 0.5
        for node in range(d):
            others = [k for k in range(d) if k != node]
            mask_to_index = np.empty(1 << len(others), dtype=np.int32)
            for parents, index in lookup[node].items():
                bits = sum(1 << others.index(p) for p in parents)
                mask_to_index[bits] = index

            masks = np.zeros(space.n_dags, dtype=np.int64)
            for position, parent in enumerate(others):
                masks |= adjacency[:, parent, node].astype(np.int64) << position
            self.parent_set_ids[:, node] = mask_to_index[masks]

        self.n_parent_sets = self.scorer.n_parent_sets

        # Index into the FLATTENED score table. The obvious `table[rows, parent_set_ids]`
        # broadcasts a [1, d] row index against [N, d] and casts the int32 ids to intp on
        # every call; precomputing the flat intp index halves the gather (384 -> 193 ms at
        # d=6, bit-identical). Costs 2x8 bytes per DAG-node, ~180 MB at d=6, against a
        # 12 GB request.
        self._flat_ids = np.ascontiguousarray(
            self.parent_set_ids.astype(np.intp)
            + (np.arange(d) * self.n_parent_sets)[None, :].astype(np.intp)
        )

        # Layout for the batched score table. Depends only on the graph space, never on
        # the data, so it is built once here rather than on every posterior update.
        self._table_plan = self.scorer._table_plan

        # [d, n_parent_sets, d] indicator: does parent set `i` of `node` contain `parent`?
        # Used to turn edge marginals into d small bincounts -- see `edge_marginals`.
        self._parent_membership = np.zeros((d, self.n_parent_sets, d), dtype=np.float64)
        for node in range(d):
            for i, parents in enumerate(self.parent_sets[node]):
                for parent in parents:
                    self._parent_membership[node, i, parent] = 1.0

    def local_score_table(self, samples: np.ndarray, intervened: np.ndarray) -> np.ndarray:
        """[d, n_parent_sets] local scores, computed once per node/parent-set pair.

        `intervened[i, j]` is truthy when node j was set by intervention in sample i.
        """
        return self.scorer.table(samples, intervened)

    def log_scores(self, samples: np.ndarray, intervened: np.ndarray) -> np.ndarray:
        """[N] unnormalised log posterior for every DAG."""
        table = self.local_score_table(samples, intervened)
        return table.ravel()[self._flat_ids].sum(axis=1)

    def edge_marginals(self, posterior: np.ndarray) -> np.ndarray:
        """[d, d] probability that each directed edge is present.

        Same quantity as the module-level `edge_marginals`, computed without ever
        materialising the DAG array as float64. Edge i->j exists exactly when i is in j's
        parent set, so the mass on each of node j's 2^(d-1) parent sets -- one bincount --
        determines the whole j-th column. That is d bincounts over N, rather than a
        weighted sum over N x d x d floats: 517 -> 245 ms at d=6, agreeing to 2.5e-15.
        """
        d = self.space.d
        posterior = np.asarray(posterior, dtype=np.float64)
        out = np.zeros((d, d), dtype=np.float64)
        for node in range(d):
            mass = np.bincount(self.parent_set_ids[:, node], weights=posterior,
                               minlength=self.n_parent_sets)
            out[:, node] = self._parent_membership[node].T @ mass
        return out

    def _log_prior(self, prior: np.ndarray) -> np.ndarray:
        """log of the prior, memoised on the identity of the array passed in.

        The environment holds one fixed prior for its whole lifetime and hands the very
        same array to every call, so this turns a 3.8-million-element log (plus a clamp,
        plus two temporaries) per step into one at reset. Keyed on identity deliberately:
        priors are built once in `sa/priors.py` and never mutated in place. If that ever
        changes, rebind the array rather than editing it -- an in-place edit would be
        silently ignored here.
        """
        cached = getattr(self, "_log_prior_cache", None)
        if cached is not None and cached[0] is prior:
            return cached[1]
        value = np.log(np.maximum(np.asarray(prior, dtype=float), 1e-300))
        self._log_prior_cache = (prior, value)
        return value

    def posterior(self, samples: np.ndarray, intervened: np.ndarray,
                  prior: Optional[np.ndarray] = None) -> np.ndarray:
        """[N] posterior over DAGs.

        `prior` defaults to uniform. It should normally be the SAME distribution the true
        graph is drawn from -- a sparse generator paired with a uniform prior is a
        misspecification that would show up as systematic over-confidence in dense graphs,
        and it would be easy to mistake for an estimator bug later.

        With no usable data the posterior is exactly the prior, which is the honest state
        rather than an arbitrary default.
        """
        if prior is None:
            prior = np.full(self.space.n_dags, 1.0 / self.space.n_dags)
        if np.asarray(samples).shape[0] == 0:
            return np.asarray(prior, dtype=float).copy()
        log_p = self.log_scores(samples, intervened) + self._log_prior(prior)
        log_p = log_p - log_p.max()
        p = np.exp(log_p)
        return p / p.sum()


def edge_marginals(space: GraphSpace, posterior: np.ndarray) -> np.ndarray:
    """[d, d] probability that each directed edge is present.

    This is the *scalable* belief representation: d(d-1) numbers regardless of how many
    DAGs exist. It is a lossy summary -- it discards correlations between edges, so two
    very different posteriors can share edge marginals -- which is exactly the cost the
    experiment is designed to measure.
    """
    # Accumulated in chunks because the obvious one-liner upcasts the whole int8 DAG array
    # to float64 first: 3.78 million graphs at d=6 is a 1.1 GB temporary allocated on every
    # single call. Chunking bounds it to a few tens of megabytes at no cost in accuracy.
    d = space.d
    total = np.zeros((d, d), dtype=np.float64)
    for start in range(0, space.n_dags, _MARGINAL_CHUNK):
        block = slice(start, start + _MARGINAL_CHUNK)
        total += np.tensordot(posterior[block],
                              space.dags[block].astype(np.float64), axes=(0, 0))
    return total


def mec_posterior(space: GraphSpace, posterior: np.ndarray) -> np.ndarray:
    """[n_mecs] posterior mass on each Markov equivalence class.

    Useful for separating the two things an agent can fail at: finding the right class
    (which observation alone can do) versus orienting within it (which needs
    interventions).
    """
    return np.bincount(space.mec_id, weights=posterior, minlength=space.n_mecs)


def is_identified(posterior: np.ndarray, true_index: int, threshold: float = 0.9) -> bool:
    """Has the true DAG been pinned down?

    Deliberately a posterior-mass test rather than `argmax == truth`. Markov-equivalent
    DAGs score identically on observational data, so their posterior entries tie to
    machine precision and `argmax` is decided by floating-point ordering -- it would
    report success or failure essentially at random. Mass above a threshold is well
    defined in exactly the cases argmax is not, and it cannot be reached while a tie
    remains unbroken (a class of size k caps every member at 1/k).
    """
    return bool(posterior[true_index] >= threshold)
