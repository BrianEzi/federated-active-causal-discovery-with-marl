"""Stage 2: a focused grid on the three levers that control the learning SIGNAL.

Stage 1 (`sweep_configs.py`) is one-factor-at-a-time, which by construction cannot see
interactions. The first stage-1 result made that limitation concrete, so this grid exists
to cover exactly the gap.

**What stage 1's first configuration showed.** At d=4 with the baseline settings the agent
trains to solve rate 1.00 with mean episode length ~2.3 -- but random costs 2.44, so the
sampled policy is no better than random, and the deterministic policy is far worse
(gap-closed -4.9 to -9.9). Entropy stalls at 1.36 against a 1.609 maximum after ~1500
episodes and policy loss sits near 0.005, while the value loss falls cleanly from 0.38 to
0.01. The critic learns; the actor does not.

**Why.** Nearly every episode identifies eventually, so the +1 terminal bonus is almost
constant across actions and carries no discriminating signal. The whole learnable
difference between a good and a bad intervention is the step cost -- 0.05 against a reward
scale of 1.0, about 5%. Discounting adds little: at gamma=0.99, one extra step costs ~1%.
So the policy gradient is tiny, and at entropy_coef=0.003 the entropy bonus is of
comparable magnitude, which holds the policy near uniform. Near-uniform logits then make
argmax arbitrary, which is precisely the deterministic collapse observed.

**The three levers that change that ratio, and why a grid rather than OFAT.** Raising
`step_cost` scales the signal directly; lowering `gamma` makes finishing sooner worth more
(0.9^1 vs 0.9^3 is a 17% difference against 2% at 0.99); lowering `entropy_coef` stops the
exploration bonus from swamping it. These act on the SAME quantity -- the signal-to-noise
of the policy gradient -- so their effects are not separable and one-at-a-time would
mislead. 3 x 2 x 2 = 12 configurations.

Note `step_cost` and the success bonus are two ends of one ratio: raising the cost is
equivalent to lowering the bonus. Only the cost is swept, because the bonus cannot go to
zero -- with no reward for identifying, passing immediately would be optimal.

Usage mirrors sweep_configs.py:
    python -m legacy.scripts.sweep_stage2 --count | --cli N | --json
"""
from __future__ import annotations

import argparse
import itertools
import json
from typing import Dict, List

from scripts.sweep_configs import BASELINE, to_cli

SEEDS = [0, 1, 2]

STEP_COSTS = [0.05, 0.15, 0.30]     # 0.05 is the stage-1 baseline, kept as the control
GAMMAS = [0.99, 0.9]
ENTROPY_COEFS = [0.003, 0.0]


def build_matrix() -> List[Dict]:
    configs: List[Dict] = []
    for step_cost, gamma, entropy in itertools.product(STEP_COSTS, GAMMAS, ENTROPY_COEFS):
        configs.append({
            **BASELINE,
            "step_cost": step_cost, "gamma": gamma, "entropy_coef": entropy,
            "seeds": SEEDS, "arm": "signal_grid",
            "tag": f"s2_cost{step_cost}_g{gamma}_e{entropy}",
        })
    return configs


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
        print(to_cli(matrix[args.cli - 1]))
    elif args.json:
        print(json.dumps(matrix, indent=2))
    else:
        print(f"{len(matrix)} configurations, {len(matrix) * len(SEEDS)} runs")
        for i, c in enumerate(matrix, 1):
            print(f"  {i:>3}  {c['tag']}")


if __name__ == "__main__":
    main()
