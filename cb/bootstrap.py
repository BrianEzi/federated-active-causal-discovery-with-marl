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

from concurrent.futures import ProcessPoolExecutor

from cb.citest import FisherZ
from cb.orient import CODE_BIDIRECTED, CODE_DIRECTED, orient
from cb.skeleton import Skeleton, estimate_skeleton

# One persistent pool per process, created lazily. Replicates are embarrassingly
# parallel and DETERMINISTIC BY CONSTRUCTION regardless of scheduling: every replicate's
# row indices are drawn serially from the single rng BEFORE any work is dispatched, each
# task is a pure function of its inputs, and accumulation happens in replicate order.
# The parallel path must be bit-identical to the serial one -- pinned by
# tests/cb/test_fast_stats.py.
_POOL: Optional[ProcessPoolExecutor] = None
_POOL_SIZE = 0


def _pool(n_jobs: int) -> ProcessPoolExecutor:
    global _POOL, _POOL_SIZE
    if _POOL is None or _POOL_SIZE != n_jobs:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ProcessPoolExecutor(max_workers=n_jobs)
        _POOL_SIZE = n_jobs
    return _POOL


def _replicate(task):
    """One bootstrap replicate, a pure function of its inputs -- the unit of parallelism."""
    (data, intervened, foreign, rows, alpha, max_cond, use_interventions,
     require_power, oracle_skeleton, skeleton_alpha) = task
    sub, sub_int = data[rows], intervened[rows]
    test = FisherZ(sub, sub_int, alpha=alpha, foreign=foreign[rows],
                   skeleton_alpha=skeleton_alpha)
    if oracle_skeleton is not None:
        # The oracle warm start: adjacency and separating sets are the TRUE
        # infinite-observational-data limit (ma.projection.observational_skeleton), so
        # the replicates differ only through the interventional channels. ci_tests=0
        # is honest -- no test ran.
        adj, sepsets = oracle_skeleton
        skel = Skeleton(np.asarray(adj, dtype=bool).copy(), dict(sepsets), 0, False)
    else:
        skel = estimate_skeleton(test, data.shape[1], max_cond=max_cond)
    ancestral = test.ancestral_evidence() if use_interventions else None
    clamped = test.clamped_enough() if use_interventions else None
    powered = test.pair_power() if use_interventions else None
    result = orient(skel, ancestral, clamped, require_power=require_power,
                    powered=powered)
    return result.codes, skel.adjacency, skel.ci_tests, skel.truncated


class BootstrapBelief:
    """Edge-appearance frequencies over `B` resampled runs of the pipeline."""

    def __init__(self, directed: np.ndarray, bidirected: np.ndarray, adjacency: np.ndarray,
                 n_boot: int, ci_tests: int, truncated_fraction: float,
                 replicates: Optional[np.ndarray] = None):
        self.directed = directed
        self.bidirected = bidirected
        self.adjacency = adjacency
        self.n_boot = int(n_boot)
        self.ci_tests = int(ci_tests)
        self.truncated_fraction = float(truncated_fraction)
        # `[runs, k, k]` of `cb.orient` edge codes, one graph per replicate. Kept because
        # identification under this engine is a PER-REPLICATE question -- "what fraction of
        # replicates recovered the true structure" -- and frequencies alone cannot answer
        # it: two replicates each half-right average to the same marginals as one right and
        # one wrong. Tiny (50 x 15 x 15 int8), so it is always kept.
        self.replicates = replicates

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
                     require_power: bool = True,
                     foreign: Optional[np.ndarray] = None,
                     blocks: Optional[np.ndarray] = None,
                     n_jobs: int = 1, oracle_skeleton=None,
                     skeleton_alpha: Optional[float] = None) -> BootstrapBelief:
    """Resample rows `n_boot` times; run skeleton + orientation on each; count edges.

    Rows are resampled with replacement, NOT columns: the variables are fixed by the
    window, and only which observations were drawn is uncertain.

    `blocks` (optional, `[n]` int labels) makes the resampling STRATIFIED: rows are
    resampled within their experiment block, block sizes fixed. Without it, a resample
    can draw 340 rows from one intervention block and 60 from another -- which simulates
    running a DIFFERENT EXPERIMENT, not seeing different data, and the wobble it
    manufactures is pure artefact (found 2026-08-24 during the criterion review). The
    environment always passes blocks; the bare path remains for single-regime data.

    `n_boot = 0` is legal and runs the pipeline once on the real data, giving a hard 0/1
    belief. Useful for debugging the pipeline in isolation from the resampling.
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    intervened = (np.zeros_like(data, dtype=bool) if intervened is None
                  else np.asarray(intervened) > 0.5)
    # Per-row "a variable outside the window was intervened" -- see FisherZ.__init__.
    foreign = (np.zeros(n, dtype=bool) if foreign is None
               else np.asarray(foreign) > 0)
    rng = np.random.default_rng(seed)

    directed = np.zeros((k, k), dtype=float)
    bidirected = np.zeros((k, k), dtype=float)
    adjacency = np.zeros((k, k), dtype=float)
    total_tests = 0
    truncations = 0
    runs = max(int(n_boot), 1)
    replicates = np.zeros((runs, k, k), dtype=np.int8)

    block_rows = None
    if blocks is not None:
        labels = np.asarray(blocks)
        block_rows = [np.flatnonzero(labels == lab) for lab in np.unique(labels)]

    # All resample indices are drawn FIRST, serially, so the rng stream is identical
    # whether the replicates then run serially or on the pool.
    row_sets = []
    for b in range(runs):
        if n_boot and b > 0:
            if block_rows is not None:
                rows = np.concatenate([members[rng.integers(0, len(members), len(members))]
                                       for members in block_rows])
            else:
                rows = rng.integers(0, n, n)
        else:
            rows = np.arange(n)         # first replicate is the real data, unresampled
        row_sets.append(rows)

    tasks = [(data, intervened, foreign, rows, alpha, max_cond,
              use_interventions, require_power, oracle_skeleton, skeleton_alpha)
             for rows in row_sets]
    if n_jobs > 1 and runs > 1:
        # One chunk per worker: at small k a replicate is milliseconds of work, and
        # per-task IPC would swamp it (measured 2026-08-25: unchunked 4-way was 2.7x
        # SLOWER than serial).
        chunk = max(1, (len(tasks) + int(n_jobs) - 1) // int(n_jobs))
        outputs = list(_pool(int(n_jobs)).map(_replicate, tasks, chunksize=chunk))
    else:
        outputs = [_replicate(task) for task in tasks]

    for b, (codes, skel_adjacency, ci_tests, truncated) in enumerate(outputs):
        directed += (codes == CODE_DIRECTED)
        bidirected += (codes == CODE_BIDIRECTED)
        adjacency += skel_adjacency
        total_tests += ci_tests
        truncations += int(truncated)
        replicates[b] = codes

    return BootstrapBelief(directed / runs, bidirected / runs, adjacency / runs,
                           runs, total_tests, truncations / runs, replicates)
