"""Stage 5: the permutation-equivariant architecture, which the probe says is the fix.

Stages 1-4 varied the reward, the exploration, the observation and the optimiser. All of
them failed, and a supervised probe explains why none of them could have worked: the flat
MLP cannot express the mapping it was being asked to learn.

Probe accuracy predicting the oracle's tied-best target at d=4 (chance 0.279):

    edge marginals, flat network      0.528
    edge marginals, per-node scorer   0.814
    exact posterior, flat network     0.618

The per-node scorer reading the LOSSY summary beats the flat network reading the EXACT
sufficient statistic. That localises the failure precisely: not the reward, not the
exploration, not the representation's information content, but the architecture's ability
to express "score every node the same way".

`PerNodeActorCritic` embeds each (i->j, j->i) neighbour pair, pools over neighbours, and
scores node i from its own pooled summary -- one shared scorer for all d nodes. The result
is permutation-equivariant, which the oracle is and the flat network cannot be, and its
parameter count does not grow with d, so the same model form carries to d=6 unchanged.

Arm `flat_control` repeats the best settings with the OLD architecture. Without it, an
improvement here could be credited to the architecture when it came from the learning rate.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from scripts.sweep_configs import BASELINE, to_cli

BEST = {"lr": 1e-3, "hidden": 256, "episodes_per_update": 16}
FLAGS = ("no_pass", "include_counts")

ARMS = [
    # The headline run, 5 seeds: architecture change alone on top of the baseline.
    ("pernode", {"arch": "pernode"}, [0, 1, 2, 3, 4]),
    # Architecture plus the settings stage 1 showed actually move the needle.
    ("pernode_best", {"arch": "pernode", **BEST}, [0, 1, 2, 3, 4]),
    # Plus the observation fix from stage 4.
    ("pernode_best_counts", {"arch": "pernode", "include_counts": True, **BEST},
     [0, 1, 2, 3, 4]),
    # Plus dense credit assignment.
    ("pernode_best_counts_shape",
     {"arch": "pernode", "include_counts": True, "shaping_coef": 1.0, **BEST}, [0, 1, 2]),
    # Does it need longer?
    ("pernode_best_counts_long",
     {"arch": "pernode", "include_counts": True, "train_episodes": 15000, **BEST},
     [0, 1, 2]),
    # CONTROL: same settings, old architecture. Isolates the architecture's contribution.
    ("flat_control", {"arch": "flat", "include_counts": True, **BEST}, [0, 1, 2]),
    # The other problem size, for the d comparison.
    ("pernode_best_d4", {"arch": "pernode", "d": 4, **BEST}, [0, 1, 2, 3, 4]),
]


def build_matrix() -> List[Dict]:
    return [{**BASELINE, **overrides, "seeds": seeds, "arm": "stage5", "tag": f"s5_{name}"}
            for name, overrides, seeds in ARMS]


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
        print(f"{len(matrix)} configurations, "
              f"{sum(len(c['seeds']) for c in matrix)} runs")
        for i, c in enumerate(matrix, 1):
            print(f"  {i:>3}  {c['tag']:<32} seeds={len(c['seeds'])}")


if __name__ == "__main__":
    main()
