"""Stage 6: re-run the headline result in a GATE-1-valid environment.

GATE 1 says the observational-only identification rate must equal the fraction of DAGs
alone in their Markov equivalence class -- a number computable exactly from the graph
space. It was pinned at d=3 with n_obs=1000 and passed there, and it silently stopped
passing as d grew:

    d=4  target 0.1087   n_obs=1000 -> 0.085  OK
    d=5  target 0.0893   n_obs=1000 -> 0.040  MISSES
    d=6  target 0.0810   n_obs=1000 -> 0.025  MISSES  (5000 also misses; 20000 OK)

So every d=5 run in this project -- including the headline result -- used an observational
phase too short to pin down the equivalence class. The agent began each episode from a
blurrier belief than the design intended.

**This does not invalidate the comparison.** gap-closed is measured against random and
greedy baselines evaluated in the SAME environment, so the ranking stands. What it
invalidates is the claim that the environment matches its specification, and with it any
cross-d comparison of absolute difficulty.

This stage re-runs the winning configuration at n_obs where GATE 1 actually passes, so the
headline result rests on an environment that is what it says it is.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from scripts.sweep_configs import BASELINE, to_cli

BEST = {"arch": "pernode", "lr": 1e-3, "hidden": 256, "episodes_per_update": 16,
        "include_counts": True}
FLAGS = ("no_pass", "include_counts")
SEEDS = [0, 1, 2]

ARMS = [
    # d=5 at the two n_obs settings that pass GATE 1.
    ("d5_nobs5000", {**BEST, "n_obs": 5000}),
    ("d5_nobs20000", {**BEST, "n_obs": 20000}),
    # The flat control, so the architecture comparison also holds in the valid environment.
    ("d5_nobs5000_flat", {**BEST, "arch": "flat", "n_obs": 5000}),
    # d=4 was already valid at n_obs=1000; included at 5000 to check nothing else moved.
    ("d4_nobs5000", {**BEST, "d": 4, "n_obs": 5000}),
]


def build_matrix() -> List[Dict]:
    return [{**BASELINE, **overrides, "seeds": SEEDS, "arm": "stage6", "tag": f"s6_{name}"}
            for name, overrides in ARMS]


def _to_cli(config: Dict) -> str:
    set_flags = [f for f in FLAGS if config.pop(f, False)]
    return to_cli(config) + "".join(f" --{f}" for f in set_flags)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--cli", type=int, default=None)
    args = parser.parse_args()
    matrix = build_matrix()
    if args.count:
        print(len(matrix))
    elif args.cli is not None:
        print(_to_cli(matrix[args.cli - 1]))
    elif args.json:
        print(json.dumps(matrix, indent=2))
    else:
        print(f"{len(matrix)} configurations, {len(matrix) * len(SEEDS)} runs")
        for i, c in enumerate(matrix, 1):
            print(f"  {i:>3}  {c['tag']}")


if __name__ == "__main__":
    main()
