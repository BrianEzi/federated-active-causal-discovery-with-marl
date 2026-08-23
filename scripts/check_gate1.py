"""GATE 1: does the observational-only identification rate match the theory?

The fraction of DAGs alone in their Markov equivalence class is computable exactly, and it
is precisely the fraction of problems solvable WITHOUT intervening. Measuring above it means
information is leaking; measuring below it means the estimator is not extracting what the
data contains -- usually because there are not enough samples to pin the class down.

This matters beyond the gate itself: if the observational rate is below target, the agent
starts every episode from a blurrier belief than the design intends, and its scores are not
comparable across d.

    python -m scripts.check_gate1 --d 6 --n_obs 1000 5000 20000
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sa.baselines import no_intervention_policy
from sa.env import EnvConfig
from ma.stats import bootstrap_ci, run_policy
from ma.graphs import build_graph_space


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=6)
    parser.add_argument("--n_obs", type=int, nargs="+", default=[1000, 5000, 20000])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    space = build_graph_space(args.d)
    target = space.singleton_fraction
    print(f"d={args.d}  {space.n_dags} DAGs / {space.n_mecs} classes")
    print(f"  singleton fraction (GATE 1 target): {target:.4f}\n")

    results = {"d": args.d, "target": target, "n_dags": space.n_dags, "measured": {}}
    for n_obs in args.n_obs:
        config = EnvConfig(d=args.d, n_obs=n_obs, budget=20)
        outcome = run_policy(config, no_intervention_policy, args.episodes, seed=7,
                             space=space)
        rate = float(np.mean(outcome["identified"]))
        low, high = bootstrap_ci(outcome["identified"], seed=7)
        covers = low <= target <= high
        results["measured"][str(n_obs)] = {"rate": rate, "ci": [low, high],
                                           "covers_target": covers}
        print(f"  n_obs={n_obs:>6}  rate {rate:.4f}  CI {low:.4f}-{high:.4f}  "
              f"{'OK' if covers else 'MISSES TARGET'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
