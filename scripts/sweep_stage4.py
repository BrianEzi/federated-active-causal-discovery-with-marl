"""Stage 4: give the agent memory of its own actions, plus the best settings found so far.

Stage 1 finished with every one of its 34 configurations failing, and with a diagnosis that
none of them address. The deterministic solve rate sits at 0.25-0.59 while greedy solves
0.99 -- the agent identifies the graph LESS often than random does, which no amount of
inefficiency explains. The `optimal_rate` of 0.02-0.10 against a chance level near 0.29 is
the same story: the agent is not merely unhelpful, it is systematically anti-correlated
with the oracle. Both are what a policy that has stopped reading its observation looks
like, because such a policy picks the same node every step, re-intervening where it already
has and exhausting the budget.

The observation is missing the one thing that would let a policy break that tie: **which
nodes it has already intervened on**. In principle it does not need them -- the posterior
is a sufficient statistic, and if an intervention taught nothing then the same target
really is still best. But that argument is about the OPTIMAL policy. A deterministic
network whose output barely varies with its input has no way out of the loop at all.
`include_counts` appends the per-node counts, and whether it helps is now measurable
(`repeat_rate`, `distinct_targets`).

Stage 1's other lesson is which knobs moved anything: `lr=1e-3` (min gap -5.35 against
-8.60 at 1e-4, and the lowest final entropy of any arm at 1.495), `hidden=256` (-5.08
against -8.98 at 64), and `episodes_per_update=16` (-5.04). None rescued the run alone;
they are combined here because OFAT could not test them together.

Arm 6 is the control that matters: the same best-of-stage-1 settings WITHOUT the counts.
Without it, any improvement could be credited to the observation change when it was really
the learning rate.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from scripts.sweep_configs import BASELINE, to_cli

SEEDS = [0, 1, 2]

# The settings stage 1 showed actually move the needle, applied together.
BEST_OF_STAGE1 = {"lr": 1e-3, "hidden": 256, "episodes_per_update": 16}

ARMS = [
    ("counts", {"include_counts": True}),
    ("counts_best", {"include_counts": True, **BEST_OF_STAGE1}),
    ("counts_shape", {"include_counts": True, "shaping_coef": 1.0}),
    ("counts_shape_best", {"include_counts": True, "shaping_coef": 1.0, **BEST_OF_STAGE1}),
    ("counts_shape_best_nopass", {"include_counts": True, "shaping_coef": 1.0,
                                  "no_pass": True, **BEST_OF_STAGE1}),
    # CONTROL: best settings, no counts. Without this the counts would get the credit.
    ("best_nocounts", dict(BEST_OF_STAGE1)),
    # Does the combination simply need longer?
    ("counts_shape_best_long", {"include_counts": True, "shaping_coef": 1.0,
                                "train_episodes": 15000, **BEST_OF_STAGE1}),
    # Everything at once, including letting the policy sharpen freely.
    ("everything", {"include_counts": True, "shaping_coef": 1.0, "no_pass": True,
                    "entropy_coef": 0.0, "train_episodes": 15000, **BEST_OF_STAGE1}),
]

FLAGS = ("no_pass", "include_counts")   # store_true: rendered without a value


def build_matrix() -> List[Dict]:
    return [{**BASELINE, **overrides, "seeds": SEEDS, "arm": "stage4",
             "tag": f"s4_{suffix}"} for suffix, overrides in ARMS]


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
