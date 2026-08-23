"""Is a constraint-based belief update cheap enough to train on, and to DEBUG on?

The question is NOT asymptotic. It is: how many seconds does one agent's belief update take
at 30 nodes, because that number decides whether a failed run costs ten minutes or ten hours,
and therefore whether the project can iterate at all in the time left.

Three things measured, all on data from the project's own SCM rather than `rng.normal`,
because CI-test COUNT depends on graph density and density is what the prior controls:

  1. skeleton search   the PC/FCI adjacency phase. This DOMINATES a constraint-based run;
                       the orientation rules are O(d^3) bookkeeping by comparison. Reported
                       as both seconds and NUMBER OF CI TESTS -- the test count is a property
                       of the algorithm and the graph, so it transfers to any implementation,
                       whereas seconds are a property of this one.
  2. bootstrap         B independent skeleton searches on resampled data. This is what turns
                       one equivalence class into a distribution, which is what the active
                       learning loop needs. Embarrassingly parallel; measured serial, so the
                       number is an upper bound.
  3. exact Bayesian    DPPosterior partition + edge marginals at the same sizes, WITHOUT the
                       confounding layer. The honest baseline: this is what we pay today for
                       the DAG posterior alone.

Deliberately NOT a correctness check. It says whether the route is affordable, nothing more.
Correctness is a separate gate and must not be inferred from a timing script.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from itertools import combinations

import numpy as np
from scipy import stats

from crosscheck.dp import DPPosterior
from ma.priors import connectivity_prior_p
from ma.scm import sample, sample_scm_params
from crosscheck.score import BGeScore


def random_dag(d: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Erdos-Renyi over a random topological order, so acyclicity is free."""
    order = rng.permutation(d)
    adj = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                adj[order[i], order[j]] = 1
    return adj


class FisherZ:
    """Partial-correlation independence test, counting its own calls.

    Standard Fisher z transform of the partial correlation. Linear-Gaussian, i.e. exactly
    the assumption BGe already makes -- swapping in a kernel test is the generalisation
    argument, and it changes the CONSTANT here, not the test count.
    """

    def __init__(self, data: np.ndarray, alpha: float = 0.01):
        self.n = data.shape[0]
        self.corr = np.corrcoef(data, rowvar=False)
        self.alpha = alpha
        self.calls = 0

    def independent(self, x: int, y: int, cond) -> bool:
        self.calls += 1
        cond = list(cond)
        idx = [x, y] + cond
        sub = self.corr[np.ix_(idx, idx)]
        try:
            precision = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            return False
        denom = np.sqrt(precision[0, 0] * precision[1, 1])
        if denom <= 0:
            return False
        r = -precision[0, 1] / denom
        r = float(np.clip(r, -0.999999, 0.999999))
        dof = self.n - len(cond) - 3
        if dof <= 0:
            return True
        z = 0.5 * np.log1p(2 * r / (1 - r)) if abs(r) < 1 else 0.0
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(np.sqrt(dof) * z)))
        return p_value > self.alpha


def skeleton(d: int, test: FisherZ, max_cond: int = 3):
    """PC/FCI adjacency phase: drop an edge as soon as SOME conditioning set separates it.

    `max_cond` caps the conditioning-set size. Uncapped, the search is exponential in the
    maximum degree; capping is what the sparse-graph literature relies on and what makes the
    cost polynomial in practice. The cap is a real approximation and is reported, not hidden.
    """
    adj = np.ones((d, d), dtype=bool)
    np.fill_diagonal(adj, False)
    for level in range(max_cond + 1):
        for x, y in list(combinations(range(d), 2)):
            if not adj[x, y]:
                continue
            neighbours = [w for w in range(d) if adj[x, w] and w != y]
            if len(neighbours) < level:
                continue
            for cond in combinations(neighbours, level):
                if test.independent(x, y, cond):
                    adj[x, y] = adj[y, x] = False
                    break
        if max(adj.sum(axis=1)) <= level:
            break
    return adj


def bench_constraint(d: int, n_rows: int, seed: int, boots: int) -> dict:
    rng = np.random.default_rng(seed)
    adjacency = random_dag(d, connectivity_prior_p(d), rng)
    params = sample_scm_params(adjacency, rng)
    data, _ = sample(params, n_rows, rng)

    start = time.perf_counter()
    test = FisherZ(data)
    skeleton(d, test)
    single_s = time.perf_counter() - start
    single_calls = test.calls

    start = time.perf_counter()
    for _ in range(boots):
        rows = rng.integers(0, n_rows, n_rows)
        boot_test = FisherZ(data[rows])
        skeleton(d, boot_test)
    boot_s = time.perf_counter() - start

    return {"d": d, "n_rows": n_rows, "ci_tests": int(single_calls),
            "single_s": single_s, "boots": boots, "bootstrap_s": boot_s,
            "edges_true": int(adjacency.sum())}


def bench_bayes(k: int, n_rows: int, seed: int) -> dict:
    """Exact DAG posterior, NO confounding layer -- today's cost for the easy half."""
    rng = np.random.default_rng(seed)
    adjacency = random_dag(k, connectivity_prior_p(k), rng)
    params = sample_scm_params(adjacency, rng)
    data, intervened = sample(params, n_rows, rng)

    start = time.perf_counter()
    dp = DPPosterior.for_prior(k, BGeScore(k), kind="uniform")
    log_w = dp.log_weights(data, intervened)
    dp.edge_marginals_onepass(log_w)
    return {"k": k, "n_rows": n_rows, "update_s": time.perf_counter() - start}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-rows", type=int, default=500)
    ap.add_argument("--boots", type=int, default=50)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 9, 15, 20, 30])
    ap.add_argument("--bayes-sizes", type=int, nargs="+", default=[5, 9, 12, 15])
    ap.add_argument("--out", default="results/cb_feasibility.json")
    args = ap.parse_args(argv)

    print("CONSTRAINT-BASED: skeleton search, %d rows, bootstrap B=%d\n"
          % (args.n_rows, args.boots))
    header = "%3s %7s %10s %11s %14s" % ("d", "edges", "CI tests", "single s", "bootstrap s")
    print(header); print("-" * len(header))
    cb_rows = []
    for d in args.sizes:
        row = bench_constraint(d, args.n_rows, seed=0, boots=args.boots)
        cb_rows.append(row)
        print("%3d %7d %10d %11.4f %14.3f" % (
            row["d"], row["edges_true"], row["ci_tests"],
            row["single_s"], row["bootstrap_s"]))

    print("\nEXACT BAYESIAN: DAG posterior only, no confounding layer\n")
    header = "%3s %11s" % ("k", "update s")
    print(header); print("-" * len(header))
    bayes_rows = []
    for k in args.bayes_sizes:
        row = bench_bayes(k, args.n_rows, seed=0)
        bayes_rows.append(row)
        print("%3d %11.4f" % (row["k"], row["update_s"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"constraint": cb_rows, "bayes": bayes_rows}, indent=1))
    print("\nwrote %s" % out)
    return cb_rows, bayes_rows


if __name__ == "__main__":
    main()
