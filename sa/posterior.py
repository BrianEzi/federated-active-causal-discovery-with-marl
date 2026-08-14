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
from typing import List, Sequence, Tuple

import numpy as np

from sa.graphs import GraphSpace


class PosteriorEngine:
    """Computes exact posteriors over a `GraphSpace`, reusing local scores.

    The mapping from (DAG, node) to parent-set index is precomputed once at construction,
    since it depends only on the graph space and never on the data.
    """

    def __init__(self, space: GraphSpace, score):
        self.space = space
        self.score = score
        d = space.d

        # All parent sets for each node, and a lookup from set -> index.
        self.parent_sets: List[List[Tuple[int, ...]]] = []
        lookup: List[dict] = []
        for node in range(d):
            others = [k for k in range(d) if k != node]
            sets = [
                combo
                for r in range(len(others) + 1)
                for combo in itertools.combinations(others, r)
            ]
            self.parent_sets.append(sets)
            lookup.append({s: i for i, s in enumerate(sets)})

        # [N, d] index of each DAG's parent set for each node.
        self.parent_set_ids = np.empty((space.n_dags, d), dtype=np.int32)
        for g, dag in enumerate(space.dags):
            for node in range(d):
                parents = tuple(int(p) for p in np.flatnonzero(dag[:, node] > 0.5))
                self.parent_set_ids[g, node] = lookup[node][parents]

        self.n_parent_sets = max(len(s) for s in self.parent_sets)

    def local_score_table(self, samples: np.ndarray, intervened: np.ndarray) -> np.ndarray:
        """[d, n_parent_sets] local scores, computed once per node/parent-set pair.

        `intervened[i, j]` is truthy when node j was set by intervention in sample i.
        """
        d = self.space.d
        table = np.zeros((d, self.n_parent_sets))
        for node in range(d):
            usable = np.asarray(intervened)[:, node] < 0.5
            subset = np.asarray(samples)[usable]
            for i, parents in enumerate(self.parent_sets[node]):
                table[node, i] = self.score.local_score(node, parents, subset)
        return table

    def log_scores(self, samples: np.ndarray, intervened: np.ndarray) -> np.ndarray:
        """[N] unnormalised log posterior for every DAG."""
        table = self.local_score_table(samples, intervened)
        rows = np.arange(self.space.d)[None, :]
        return table[rows, self.parent_set_ids].sum(axis=1)

    def posterior(self, samples: np.ndarray, intervened: np.ndarray) -> np.ndarray:
        """[N] posterior over DAGs under a uniform prior.

        With no usable data the posterior is exactly uniform, which is the honest state
        rather than an arbitrary default.
        """
        if np.asarray(samples).shape[0] == 0:
            return np.full(self.space.n_dags, 1.0 / self.space.n_dags)
        log_p = self.log_scores(samples, intervened)
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
    return np.tensordot(posterior, space.dags.astype(np.float64), axes=(0, 0))


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
