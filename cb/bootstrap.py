"""Bootstrap: turn one graph estimate into a distribution the policy can act on.

THIS IS THE LOAD-BEARING PIECE, and it is what makes a constraint-based engine usable for
active learning at all. FCI returns ONE equivalence class. Expected information gain needs
a distribution over hypotheses, and an equivalence class is not one. Resampling the rows and
re-running the whole pipeline gives an empirical distribution over graphs, and the frequency
with which an edge appears is a usable stand-in for its posterior probability.

It is NOT a posterior. It is a sampling distribution of an estimator, and the two coincide
only asymptotically and under conditions we do not check. That distinction belongs in the
write-up; operationally what matters is that it lives in [0,1], moves with evidence, and
concentrates as data accumulates -- which is everything the policy consumes it for.

OUTPUT SHAPE IS DELIBERATELY `[k, k]`, matching `WindowBeliefDP.edge_marginals` exactly, so
the policy, the observation vector and every downstream metric are untouched by the swap.
`bidirected` is returned as a SEPARATE `[k, k]` channel rather than folded in, because a
confounded pair and a directed edge are different claims and averaging them would destroy
the one the federation design is about.

COST, measured (`scripts/cb_feasibility.py`): B=50 at k=9 is 1.17 s serial. Embarrassingly
parallel, so that is an upper bound, and B is the knob to turn if episodes are too slow.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from cb.citest import FisherZ
from cb.orient import CODE_BIDIRECTED, CODE_DIRECTED, orient
from cb.skeleton import estimate_skeleton


class BootstrapBelief:
    """Edge-appearance frequencies over `B` resampled runs of the pipeline."""

    def __init__(self, directed: np.ndarray, bidirected: np.ndarray, adjacency: np.ndarray,
                 n_boot: int, ci_tests: int, truncated_fraction: float):
        self.directed = directed
        self.bidirected = bidirected
        self.adjacency = adjacency
        self.n_boot = int(n_boot)
        self.ci_tests = int(ci_tests)
        self.truncated_fraction = float(truncated_fraction)

    @property
    def k(self) -> int:
        return int(self.directed.shape[0])

    def edge_marginals(self) -> np.ndarray:
        """`[k, k]`, `out[u, v] = P(u -> v)`. The drop-in for the exact engine's output."""
        return self.directed

    def confounded_pairs(self, threshold: float = 0.5) -> tuple:
        from itertools import combinations
        return tuple((u, v) for u, v in combinations(range(self.k), 2)
                     if self.bidirected[u, v] >= threshold)


def bootstrap_belief(data: np.ndarray, intervened: Optional[np.ndarray] = None,
                     n_boot: int = 50, alpha: float = 0.01, max_cond: int = 3,
                     seed: int = 0, use_interventions: bool = True,
                     require_power: bool = True) -> BootstrapBelief:
    """Resample rows `n_boot` times; run skeleton + orientation on each; count edges.

    Rows are resampled with replacement, NOT columns: the variables are fixed by the
    window, and only which observations were drawn is uncertain.

    `n_boot = 0` is legal and runs the pipeline once on the real data, giving a hard 0/1
    belief. Useful for debugging the pipeline in isolation from the resampling.
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    intervened = (np.zeros_like(data, dtype=bool) if intervened is None
                  else np.asarray(intervened) > 0.5)
    rng = np.random.default_rng(seed)

    directed = np.zeros((k, k), dtype=float)
    bidirected = np.zeros((k, k), dtype=float)
    adjacency = np.zeros((k, k), dtype=float)
    total_tests = 0
    truncations = 0
    runs = max(int(n_boot), 1)

    for b in range(runs):
        if n_boot and b > 0:
            rows = rng.integers(0, n, n)
        elif n_boot:
            rows = np.arange(n)         # first replicate is the real data, unresampled
        else:
            rows = np.arange(n)
        sub, sub_int = data[rows], intervened[rows]

        test = FisherZ(sub, sub_int, alpha=alpha)
        skel = estimate_skeleton(test, k, max_cond=max_cond)
        ancestral = test.ancestral_evidence() if use_interventions else None
        clamped = test.clamped_enough() if use_interventions else None
        powered = test.pair_power() if use_interventions else None
        result = orient(skel, ancestral, clamped, require_power=require_power,
                        powered=powered)

        directed += (result.codes == CODE_DIRECTED)
        bidirected += (result.codes == CODE_BIDIRECTED)
        adjacency += skel.adjacency
        total_tests += skel.ci_tests
        truncations += int(skel.truncated)

    return BootstrapBelief(directed / runs, bidirected / runs, adjacency / runs,
                           runs, total_tests, truncations / runs)
