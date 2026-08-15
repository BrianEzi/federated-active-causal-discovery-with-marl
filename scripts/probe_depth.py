"""Phase 1 depth probe: does multi-hop aggregation lift the probe's 0.89 ceiling?

The per-node scorer does ONE round of neighbour aggregation, so a node's logit sees only
its own edges. The oracle's score depends on each node's descendants -- reachability,
which is inherently multi-hop -- and the supervised probe topped out near 0.89 rather than
1.0. Depth is the leading explanation. This measures it supervised, in minutes, rather
than by launching RL runs and inferring the cause afterwards.

Grid: d in {4, 5} x episodes in {300, 1000, 3000, 9000} x 3 seeds = 24 tasks. Each task
collects its data ONCE and trains depth 1, 2 and 3 on exactly that data, so "at matched
data size" in the decision rule is literal rather than approximate.

Three seeds per cell, not one, because the decision threshold is 0.03 and a single-seed
pilot already showed 0.014 of spread between depths on identical data. A rule that fine
needs to be applied to a mean, or it just reports noise.

    python -m scripts.probe_depth --count
    python -m scripts.probe_depth --cli 7

DECISION RULE, fixed before the numbers exist: if depth 2 or 3 beats depth 1 by more than
0.03 mean accuracy at matched data size on BOTH d=4 and d=5, carry the best depth into
Phase 2. Otherwise keep depth 1 and record that the ceiling is not about multi-hop
reachability -- which is itself worth knowing, since it redirects the next round away from
architecture and towards the belief representation.
"""
from __future__ import annotations

import argparse

D_VALUES = (4, 5)
EPISODE_COUNTS = (300, 1000, 3000, 9000)
SEEDS = (0, 1, 2)
LAYERS = (1, 2, 3)

# Held fixed across the grid so depth is the only thing varying. These match the settings
# used for the probe that produced the 0.89 ceiling, so the numbers are comparable to it.
HIDDEN = 128
EPOCHS = 60
LR = 1e-3


def build_matrix() -> list:
    return [
        {"d": d, "episodes": episodes, "seed": seed}
        for d in D_VALUES
        for episodes in EPISODE_COUNTS
        for seed in SEEDS
    ]


def to_cli(cfg: dict, out_dir: str = "results/probe_depth") -> str:
    name = f"d{cfg['d']}_ep{cfg['episodes']}_s{cfg['seed']}"
    return (
        f"--d {cfg['d']} --episodes {cfg['episodes']} --seed {cfg['seed']} "
        f"--hidden {HIDDEN} --epochs {EPOCHS} --lr {LR} "
        f"--arch both --layers {' '.join(str(k) for k in LAYERS)} "
        f"--out {out_dir}/{name}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=int, default=None,
                        help="print the CLI for this 1-based task index")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results/probe_depth")
    args = parser.parse_args()

    matrix = build_matrix()
    if args.count:
        print(len(matrix))
        return
    if args.cli is not None:
        print(to_cli(matrix[args.cli - 1], args.out_dir))
        return
    for i, cfg in enumerate(matrix, 1):
        print(f"{i:3d}  {to_cli(cfg, args.out_dir)}")


if __name__ == "__main__":
    main()
