"""Information-optimal intervention selection: the opponent the agent has to beat.

A good intervention is one the plausible graphs *disagree* about. Under a hard
intervention on node i, the distribution of exactly i's descendants shifts, so two
hypotheses are distinguishable by do(X_i) precisely when their descendant sets from i
differ. Group the hypotheses by that descendant set and the experiment's outcome tells
you which group you are in -- so the value of intervening on i is how uncertain that
outcome is.

**Why this is exactly expected information gain, not a proxy.** The descendant set is a
deterministic function of the graph, so `H(outcome | graph) = 0` and therefore

    I(graph ; outcome) = H(outcome) - H(outcome | graph) = H(outcome)

Maximising the entropy of the outcome partition *is* maximising expected information gain
(Lindley 1956). The previous implementation used a Gini/Simpson index, `1 - sum_g P(g)^2`,
which is the Tsallis-2 analogue and required a defence via generalised uncertainty
measures (DeGroot 1962). Using Shannon entropy instead removes the approximation for the
cost of one line. See docs/THEORY_NOTES.md #5 and #6.

Two honest limitations, stated rather than buried:

- **Myopic.** This is the best *single next* experiment, not the best sequence. Greedy is
  the standard tractable choice, and the `(1-1/e)` guarantee of Golovin & Krause (2011)
  needs adaptive submodularity, which expected information gain does not satisfy in
  general. That gap is deliberate: it is the headroom the learned agent is meant to find.

- **Optimistic.** The derivation assumes the experiment reveals the descendant set
  perfectly. With finite noisy samples it does not, so the oracle credits distinctions
  the agent may be unable to make in practice.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from sa.graphs import GraphSpace, descendants


# Graphs whose transitive closure is computed in one block. 250k x d x d booleans is
# ~9 MB at d=6.
_CLOSURE_CHUNK = 250_000


class InterventionOracle:
    """Scores candidate intervention targets by expected information gain.

    Descendant signatures depend only on the graph space, so they are computed once at
    construction and reused for every posterior.
    """

    def __init__(self, space: GraphSpace):
        self.space = space
        d = space.d
        # [n_dags, d] group label: which descendant-set signature each DAG has from each
        # node. Two DAGs sharing a label at node i are indistinguishable by do(X_i).
        # Computed for every graph at once, in chunks. The transitive closure is the same
        # Floyd-Warshall as `graphs.descendants`, just run on a block of graphs
        # simultaneously; a Python loop over graphs costs minutes once d reaches 6, where
        # there are 3.78 million of them. Chunked because the intermediate is [chunk, d, d]
        # booleans and the full array would be held twice over.
        codes = np.empty((space.n_dags, d), dtype=np.int64)
        adjacency = np.asarray(space.dags) > 0.5
        bit = (1 << np.arange(d)).astype(np.int64)
        for start in range(0, space.n_dags, _CLOSURE_CHUNK):
            block = slice(start, start + _CLOSURE_CHUNK)
            reach = adjacency[block].copy()
            for k in range(d):
                reach |= reach[:, :, k][:, :, None] & reach[:, k, :][:, None, :]
            # Pack each row of the reachability matrix into one integer, so two graphs
            # share a descendant set from `node` exactly when their codes are equal.
            codes[block] = reach.astype(np.int64) @ bit

        signatures = np.empty((space.n_dags, d), dtype=np.int32)
        self.n_groups = []
        for node in range(d):
            groups, inverse = np.unique(codes[:, node], return_inverse=True)
            signatures[:, node] = inverse.reshape(-1)
            self.n_groups.append(len(groups))
        self.signatures = signatures

    def scores(self, posterior: np.ndarray) -> np.ndarray:
        """[d] expected information gain from intervening on each node, in nats.

        Zero when every plausible graph agrees about the node's descendants -- the
        experiment cannot discriminate, however unexplored that node looks.
        """
        posterior = np.asarray(posterior, dtype=np.float64)
        out = np.zeros(self.space.d)
        for node in range(self.space.d):
            mass = np.bincount(
                self.signatures[:, node], weights=posterior, minlength=self.n_groups[node]
            )
            mass = mass[mass > 0]
            out[node] = float(-np.sum(mass * np.log(mass)))
        return out

    def best_targets(self, posterior: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """Returns `(scores, best_mask)`. Ties are returned as a set, not an argmax --
        tied targets are genuinely equivalent, and marking one arbitrarily correct would
        make the metric measure floating-point ordering."""
        scores = self.scores(posterior)
        return scores, scores >= scores.max() - tol

    def best_action(self, posterior: np.ndarray, rng: Optional[np.random.Generator] = None) -> int:
        """A single greedy choice, breaking ties uniformly at random.

        Random tie-breaking rather than lowest-index: the DAG space is enumerated in a
        fixed order, so always taking the first tied node would give the oracle a
        systematic and entirely arbitrary preference.
        """
        _, best = self.best_targets(posterior)
        candidates = np.flatnonzero(best)
        if rng is None:
            return int(candidates[0])
        return int(rng.choice(candidates))

    def score_choice(self, chosen: int, posterior: np.ndarray) -> Dict[str, float]:
        """Score an agent's chosen target against the oracle.

        `informative` is the field that matters and the one whose absence caused a
        retracted result. When every legal target ties at zero the oracle has no
        preference, so *any* choice is trivially "optimal" and counting it as a success
        measures nothing. Aggregate `is_optimal` only over steps where `informative` is
        true; a rate computed over all steps is the metric that reported 99.4-100% while
        being 93-98% vacuous.
        """
        scores, best = self.best_targets(posterior)
        best_score = float(scores.max())
        chosen_score = float(scores[int(chosen)])
        informative = best_score > 1e-9
        return {
            "informative": float(informative),
            "is_optimal": float(bool(best[int(chosen)])),
            "score": chosen_score,
            "best_score": best_score,
            "regret": max(0.0, best_score - chosen_score),
        }
