"""Can the DP produce what the PIPELINE needs, not just the normalising constant?

Three things are consumed downstream, and they are not equally easy:

  1. p(true DAG | data) -- needs only Z. Trivial once Z exists.
  2. edge marginals    -- the agent's observation. Obtained by re-running the DP with one
                          node's parent sets restricted to those containing the parent in
                          question: p(u->v) = Z[u->v forced] / Z. That is d(d-1) DP runs.
  3. the EIG oracle    -- needs the distribution over each node's DESCENDANT set.
                          Reachability is NOT decomposable per node, so it does not come
                          out of this machinery at all. Measured here only for whether the
                          first two hold up; the third is a separate problem.
"""
import time
from itertools import combinations

import numpy as np

from sa.graphs import build_graph_space
from sa.posterior import PosteriorEngine
from sa.score import BGeScore
from subset_dp import partition_function, zeta


def alpha_rows(table, parent_sets, d, force=None):
    """force = (child, parent): restrict `child` to parent sets containing `parent`."""
    shifts = table.max(axis=1)
    alpha = np.zeros((d, 1 << d), dtype=np.float64)
    for node in range(d):
        for index, parents in enumerate(parent_sets[node]):
            mask = 0
            for p in parents:
                mask |= 1 << p
            if force is not None and node == force[0] and not (mask >> force[1]) & 1:
                continue
            alpha[node, mask] = np.exp(table[node, index] - shifts[node])
    return zeta(alpha, d), float(shifts.sum())


def dp_edge_marginals(table, parent_sets, d):
    alpha, shift = alpha_rows(table, parent_sets, d)
    Z, _ = partition_function(alpha, d)
    out = np.zeros((d, d))
    for child in range(d):
        for parent in range(d):
            if parent == child:
                continue
            a, s = alpha_rows(table, parent_sets, d, force=(child, parent))
            Zf, _ = partition_function(a, d)
            # Shifts are identical in both runs, so they cancel in the ratio.
            out[parent, child] = Zf / Z
    return out


def build_table(d, n_obs=500, seed=0):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n_obs, d))
    score = BGeScore(d)
    stats = score.sufficient_stats(samples)
    parent_sets, table = [], np.zeros((d, 1 << (d - 1)))
    for node in range(d):
        others = [k for k in range(d) if k != node]
        sets = [c for r in range(len(others) + 1) for c in combinations(others, r)]
        parent_sets.append(sets)
        for i, parents in enumerate(sets):
            table[node, i] = score.local_score_from_stats(node, parents, stats)
    return samples, parent_sets, table


for d in (4, 5, 6):
    samples, parent_sets, table = build_table(d)
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    intervened = np.zeros((samples.shape[0], d))

    t0 = time.perf_counter()
    post = engine.posterior(samples, intervened,
                            np.full(space.n_dags, 1.0 / space.n_dags))
    exact = engine.edge_marginals(post)
    t_exact = time.perf_counter() - t0

    t0 = time.perf_counter()
    approx = dp_edge_marginals(table, parent_sets, d)
    t_dp = time.perf_counter() - t0

    print(f"d={d}  max|edge marginal diff| = {np.abs(exact - approx).max():.3e}   "
          f"enum {t_exact*1e3:7.1f} ms   dp {t_dp*1e3:8.1f} ms   "
          f"({d*(d-1)} constrained runs)")
