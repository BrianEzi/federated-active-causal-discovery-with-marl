"""Accuracy and cost of three DAG samplers against the exact DP.

Arms:
  mh_old        structure MCMC at the settings shipped before 2026-08-19 (burn 5k, thin 10)
  mh_shipped    structure MCMC at the stopgap settings currently in use (burn 50k, thin 50)
  partition     Kuipers & Moffa partition MCMC (arXiv:1504.05006)
  exact         layered exact sampling (Talvitie, Vuoksenmaa & Koivisto, UAI 2019)

GROUND TRUTH is `DPPosterior.edge_marginals_onepass`, which is exact and needs no sampling
at all. That is what makes this a clean accuracy measurement rather than two approximations
arguing: every arm is scored against a number known to be right.

WHAT TO WATCH. The exact sampler should show error falling as 1/sqrt(n) with NO floor,
because its draws are independent. The MCMC arms should flatten at a floor set by their
mixing -- that floor is the thing the burn-in stopgap was buying down with compute, and it
is why it was a stopgap.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

import numpy as np

from sa.dag_samplers import LayeredExactSampler, PartitionMCMC
from sa.dp import DPPosterior
from sa.sampler import mh_sample
from sa.score import BGeScore


def make_problem(d: int, n: int, seed: int):
    """A chain-plus-noise SCM, so the posterior has real structure rather than being flat."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    for j in range(1, d):
        x[:, j] += 0.9 * x[:, j - 1]
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    log_w = dp.log_weights(x, np.zeros((n, d), dtype=bool))
    return dp, log_w


def marginals_from_draws(draws: np.ndarray) -> np.ndarray:
    return draws.astype(float).mean(axis=0)


def run(d: int, n_obs: int, n_draws: int, seed: int) -> Dict[str, dict]:
    dp, log_w = make_problem(d, n_obs, seed)
    exact_marginals = dp.edge_marginals_onepass(log_w)
    results: Dict[str, dict] = {}

    def record(label: str, draws: np.ndarray, setup: float, sampling: float,
               acceptance: float = float("nan")) -> None:
        err = float(np.abs(marginals_from_draws(draws) - exact_marginals).max())
        results[label] = {"max_error": err, "setup_s": setup, "sample_s": sampling,
                          "acceptance": acceptance, "draws": int(len(draws))}
        print(f"  {label:12s} err {err:.5f}  setup {setup:6.2f}s  "
              f"sample {sampling:6.2f}s  acc {acceptance:.3f}", flush=True)

    for label, burn, thin in (("mh_old", 5_000, 10), ("mh_shipped", 50_000, 50)):
        t0 = time.time()
        draws, acceptance = mh_sample(log_w, dp._mask_to_index, d, n_draws,
                                      burn_in=burn, thin=thin,
                                      rng=np.random.default_rng(seed))
        record(label, draws, 0.0, time.time() - t0, acceptance)

    t0 = time.time()
    chain = PartitionMCMC(dp, log_w, seed=seed)
    setup = time.time() - t0
    t0 = time.time()
    draws, acceptance = chain.sample(n_draws, burn_in=2_000, thin=5)
    record("partition", draws, setup, time.time() - t0, acceptance)

    t0 = time.time()
    sampler = LayeredExactSampler(dp, log_w)
    sampler.log_partition()                     # forces the full DP table
    setup = time.time() - t0
    t0 = time.time()
    draws = sampler.sample(n_draws, rng=np.random.default_rng(seed))
    record("exact", draws, setup, time.time() - t0)

    # Independent check that the layered recurrence and the DP's signed sink recurrence
    # agree on the partition function. They share no code path.
    results["logz_agreement"] = {
        "layered": sampler.log_partition(),
        "dp": float(dp.log_partition(log_w)),
        "abs_diff": abs(sampler.log_partition() - float(dp.log_partition(log_w))),
    }
    return results


def convergence(d: int, n_obs: int, seed: int, sizes: List[int]) -> Dict[str, list]:
    """Error against draw count -- separates 'noisy' from 'biased'."""
    dp, log_w = make_problem(d, n_obs, seed)
    exact_marginals = dp.edge_marginals_onepass(log_w)
    out: Dict[str, list] = {"sizes": sizes, "exact": [], "mh_shipped": [], "partition": []}
    sampler = LayeredExactSampler(dp, log_w)
    for n in sizes:
        draws = sampler.sample(n, rng=np.random.default_rng(seed))
        out["exact"].append(float(np.abs(
            marginals_from_draws(draws) - exact_marginals).max()))
        draws, _ = mh_sample(log_w, dp._mask_to_index, d, n, burn_in=50_000, thin=50,
                             rng=np.random.default_rng(seed))
        out["mh_shipped"].append(float(np.abs(
            marginals_from_draws(draws) - exact_marginals).max()))
        chain = PartitionMCMC(dp, log_w, seed=seed)
        draws, _ = chain.sample(n, burn_in=2_000, thin=5)
        out["partition"].append(float(np.abs(
            marginals_from_draws(draws) - exact_marginals).max()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--n_obs", type=int, default=300)
    ap.add_argument("--n_draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sampler/comparison.json")
    args = ap.parse_args()

    report: Dict[str, object] = {"n_obs": args.n_obs, "n_draws": args.n_draws, "by_d": {}}
    for d in args.dims:
        print(f"\n=== d = {d} ===", flush=True)
        report["by_d"][str(d)] = run(d, args.n_obs, args.n_draws, args.seed)
        agree = report["by_d"][str(d)]["logz_agreement"]
        print(f"  logZ layered vs DP: |diff| {agree['abs_diff']:.3e}", flush=True)

    print("\n=== convergence at d=6 ===", flush=True)
    report["convergence_d6"] = convergence(6, args.n_obs, args.seed,
                                           [200, 500, 1000, 2000, 4000])
    for key in ("exact", "mh_shipped", "partition"):
        row = "  ".join(f"{v:.5f}" for v in report["convergence_d6"][key])
        print(f"  {key:12s} {row}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
