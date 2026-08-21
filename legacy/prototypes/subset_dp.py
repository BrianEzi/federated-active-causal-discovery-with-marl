"""Subset DP, take two: per-node scaling, vectorised zeta, and a cancellation diagnostic.

Two changes from take one.

1. PER-NODE shift instead of one global shift. score(G) = sum_i local(i, Pa_i), so
   subtracting c_i from node i's local scores divides Z by exp(sum_i c_i) exactly. Using
   each node's own maximum keeps every alpha near 1 instead of letting one node's scale
   swamp the sum. This is what killed d=6 in take one.

2. The zeta transform is vectorised with numpy rather than a Python loop over 2^d masks.

The recurrence itself alternates in sign, so it can suffer catastrophic cancellation
however well scaled: intermediate terms may be far larger than the answer. `growth` below
measures exactly that -- the largest magnitude seen anywhere in the table, divided by the
final value. Once that approaches 1/eps ~ 4.5e15, the answer has no significant digits
left, and it is better to know the number than to discover it as a wrong result.
"""
import time

import numpy as np

from sa.graphs import build_graph_space
from sa.posterior import PosteriorEngine
from sa.score import BGeScore


def zeta(a, d):
    """In-place subset-sum (fast zeta) over the last axis of length 2^d, vectorised."""
    for bit in range(d):
        a = a.reshape(-1, 2, 1 << bit)
        a[:, 1, :] += a[:, 0, :]
        a = a.reshape(-1, 1 << d)
    return a


def build_alpha(engine, table, d):
    shifts = table.max(axis=1)                       # per node
    alpha = np.zeros((d, 1 << d), dtype=np.float64)
    for node in range(d):
        for index, parents in enumerate(engine.parent_sets[node]):
            mask = 0
            for p in parents:
                mask |= 1 << p
            alpha[node, mask] = np.exp(table[node, index] - shifts[node])
    return zeta(alpha, d), float(shifts.sum())


def partition_function(alpha, d):
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
    return f[(1 << d) - 1], peak


def check(d, n_obs=500, seed=0, enumerate_truth=True):
    space = build_graph_space(d, fast=True) if enumerate_truth else None
    if space is None:
        return
    engine = PosteriorEngine(space, BGeScore(d))
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n_obs, d))
    intervened = np.zeros((n_obs, d))
    intervened[: n_obs // 5, 1] = 1.0

    table = engine.local_score_table(samples, intervened)

    log_scores = engine.log_scores(samples, intervened)
    m = log_scores.max()
    logZ_enum = m + np.log(np.exp(log_scores - m).sum())

    t0 = time.perf_counter()
    alpha, shift_sum = build_alpha(engine, table, d)
    Z, peak = partition_function(alpha, d)
    t_dp = time.perf_counter() - t0

    if Z <= 0:
        print(f"d={d}  DP FAILED (Z={Z:.3e}) -- cancellation, growth>={peak/max(abs(Z),1e-300):.1e}")
        return
    logZ_dp = np.log(Z) + shift_sum
    growth = peak / Z
    print(f"d={d}  logZ enum={logZ_enum:.9f}  dp={logZ_dp:.9f}  "
          f"diff={abs(logZ_enum-logZ_dp):.2e}  growth={growth:.2e}  dp {t_dp*1e3:8.1f} ms")


def scale_only(d, n_obs=500, seed=0):
    """Timing and conditioning beyond where enumeration is possible.

    The local score table is the only thing needing the graph space, and it does not:
    it is d * 2^(d-1) calls to BGeScore. Built directly here so d can exceed 6.
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n_obs, d))
    score = BGeScore(d)
    stats = score.sufficient_stats(samples)

    from itertools import combinations
    parent_sets = []
    n_ps = 1 << (d - 1)
    table = np.zeros((d, n_ps))
    for node in range(d):
        others = [k for k in range(d) if k != node]
        sets = [c for r in range(len(others) + 1) for c in combinations(others, r)]
        parent_sets.append(sets)
        for i, parents in enumerate(sets):
            table[node, i] = score.local_score_from_stats(node, parents, stats)

    class Shim:
        pass
    shim = Shim()
    shim.parent_sets = parent_sets

    t0 = time.perf_counter()
    alpha, shift_sum = build_alpha(shim, table, d)
    Z, peak = partition_function(alpha, d)
    t = time.perf_counter() - t0
    ok = Z > 0
    growth = (peak / Z) if ok else float("inf")
    print(f"d={d:>3}  {'ok ' if ok else 'FAIL'}  growth={growth:.2e}  "
          f"time {t:7.3f}s   (2^d={1<<d}, 3^d={3**d:,})")


if __name__ == "__main__":
    print("--- correctness against enumeration ---")
    for d in (3, 4, 5, 6):
        check(d)
    print("\n--- conditioning and cost beyond enumeration ---")
    for d in (6, 7, 8, 9, 10, 11):
        scale_only(d)
