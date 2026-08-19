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


def _partition_entropy(labels: np.ndarray, weights: np.ndarray, n_groups: int) -> float:
    """H of the outcome partition, in nats -- the expected information gain of one target.

    `labels[k]` is which descendant-set group hypothesis `k` falls into and `weights[k]` is
    its posterior mass (or `1/n_draws` for sampled hypotheses). Zero when every plausible
    graph agrees, however unexplored the node looks.
    """
    mass = np.bincount(labels, weights=weights, minlength=n_groups)
    mass = mass[mass > 0]
    return float(-np.sum(mass * np.log(mass)))


class _OracleChoices:
    """Turning per-target scores into a choice, and scoring someone else's choice.

    Shared verbatim by the enumerated and sampled oracles so that comparing them measures
    the belief representation and nothing else. Subclasses supply `scores(belief)`; what
    `belief` is differs between them (an enumerated posterior, or a log-weight table) and
    is documented on each.
    """

    def best_targets(self, belief, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """Returns `(scores, best_mask)`. Ties are returned as a set, not an argmax --
        tied targets are genuinely equivalent, and marking one arbitrarily correct would
        make the metric measure floating-point ordering."""
        scores = self.scores(belief)
        return scores, scores >= scores.max() - tol

    def best_action(self, belief, rng: Optional[np.random.Generator] = None) -> int:
        """A single greedy choice, breaking ties uniformly at random.

        Random tie-breaking rather than lowest-index: the DAG space is enumerated in a
        fixed order, so always taking the first tied node would give the oracle a
        systematic and entirely arbitrary preference.
        """
        _, best = self.best_targets(belief)
        candidates = np.flatnonzero(best)
        if rng is None:
            return int(candidates[0])
        return int(rng.choice(candidates))

    def score_choice(self, chosen: int, belief) -> Dict[str, float]:
        """Score an agent's chosen target against the oracle.

        `informative` is the field that matters and the one whose absence caused a
        retracted result. When every legal target ties at zero the oracle has no
        preference, so *any* choice is trivially "optimal" and counting it as a success
        measures nothing. Aggregate `is_optimal` only over steps where `informative` is
        true; a rate computed over all steps is the metric that reported 99.4-100% while
        being 93-98% vacuous.
        """
        scores, best = self.best_targets(belief)
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


class SamplingOracle(_OracleChoices):
    """The same oracle, over sampled DAGs instead of an enumerated list.

    **Why sampling is unavoidable here, when the DP handles everything else.** The oracle
    groups hypotheses by the descendant set of the intervention target. Reachability is a
    property of whole paths, not of any single node's parent set, so it does not decompose
    and the subset DP has no way to express it -- unlike `Z` and the edge marginals, which
    it produces exactly. Drawing DAGs and computing descendants per draw is the way through.

    **Affordable because it is an evaluation-only cost.** The oracle builds the greedy
    reference and scores the agent's actions; it is never in the training loop. So it runs
    on a few hundred evaluation episodes, not six thousand training ones.

    `belief` for this class is the `[d, 2^(d-1)]` log-weight table from
    `DPPosterior.log_weights`, not a posterior over graphs -- there is no such array at the
    sizes this exists for.

    Accuracy is a *measured* property, not an assumption: see `tests/test_sampling_oracle.py`,
    which pins its choices against the exact oracle at d=4,5,6.
    """

    # Defaults raised 2026-08-19 after measuring the shipped ones against exact DP
    # marginals: burn_in=5000/thin=10 gave errors up to 0.10 and picked a DIFFERENT target
    # from a well-mixed chain in 38% of episodes, giving up 0.065 nats on average. Since
    # this oracle is the opponent every d=7 result is scored against, that made the
    # baseline itself unreliable.
    #
    # Measured max |MH - exact| marginal error:
    #     burn 20000 thin 20 -> 0.100
    #     burn 50000 thin 50 -> 0.016
    #     burn 100000 thin 20, 50000 draws -> 0.006
    #
    # These settings are a STOPGAP, not the principled fix. Acceptance is 5.8% regardless,
    # because structure-MCMC is the wrong tool for a posterior whose effective support is
    # ~172 graphs. The real fix is partition MCMC (Kuipers & Moffa 2017) or exact sampling
    # from the DP; both are pending review.
    def __init__(self, dp, n_draws: int = 4000, burn_in: int = 50_000, thin: int = 50,
                 seed: int = 0, method: str = "exact"):
        """`method="exact"` uses `LayeredExactSampler`; `"mh"` keeps structure MCMC.

        DEFAULT CHANGED TO EXACT on 2026-08-19, and the reason is a measurement about
        DECISIONS rather than about marginals. Raising burn-in 5k->50k cut the max
        edge-marginal error from 0.100 to 0.016, which looked like a fix. It was not: the
        oracle still picked a DIFFERENT target from a well-mixed reference chain in 35% of
        d=7 episodes (agreement 0.650), giving up 0.113 nats on average and up to 1.82.
        The old settings disagreed ~38%. Ten times the compute bought a better-looking
        number and left the behaviour essentially unchanged.

        Exact draws are independent by construction, so there is no mixing floor to
        discover later and no burn-in to tune. See `sa/dag_samplers.py`.
        """
        self.dp = dp
        self.d = dp.d
        self.n_draws = n_draws
        self.burn_in = burn_in
        self.thin = thin
        self.seed = seed
        self.method = method
        if method not in ("exact", "mh"):
            raise ValueError(f"method must be 'exact' or 'mh', got {method!r}")
        self._calls = 0

    def scores(self, belief: np.ndarray) -> np.ndarray:
        """[d] expected information gain from intervening on each node, in nats."""
        from sa.sampler import descendant_codes, mh_sample

        # A fresh stream per call, derived from the run seed, so a sequence of oracle
        # queries is reproducible from `seed` alone rather than depending on how many
        # times the oracle happened to be consulted earlier in the episode.
        rng = np.random.default_rng([self.seed, self._calls])
        self._calls += 1

        if self.method == "exact":
            from sa.dag_samplers import LayeredExactSampler
            draws = LayeredExactSampler(self.dp, belief).sample(self.n_draws, rng=rng)
        else:
            draws, _ = mh_sample(belief, self.dp._mask_to_index, self.d, self.n_draws,
                                 burn_in=self.burn_in, thin=self.thin, rng=rng)
        codes = descendant_codes(draws)
        weights = np.full(draws.shape[0], 1.0 / draws.shape[0])

        out = np.zeros(self.d)
        for node in range(self.d):
            _, inverse = np.unique(codes[:, node], return_inverse=True)
            inverse = inverse.reshape(-1)
            out[node] = _partition_entropy(inverse, weights, inverse.max() + 1)
        return out


class InterventionOracle(_OracleChoices):
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
        return np.array([
            _partition_entropy(self.signatures[:, node], posterior, self.n_groups[node])
            for node in range(self.space.d)
        ])
