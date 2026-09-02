"""Where did a run ACTUALLY peak on window_rate, as opposed to where MI said it peaked?

WHY. `_best.pt` is selected by highest `best_mi_ratio` during training. MI measures whether
the action depends on the state, not whether the action is GOOD -- and 1 Sep those two came
apart hard: seed 0 of `p85_b70_k8_channels_reprobe_long` carries MI 0.340 (nearly the winning
seed's 0.357) while its window_rate sits at 0.804 against greedy's 0.950. Agent A found the
same split from the other direction at k=20/k=30, where best-vs-final was worth 2.3x and 16x
on SHD.

So the MI-selected checkpoint may simply be the wrong checkpoint, and the "1 of 3 seeds
reaches greedy" reading would then be a selection artefact rather than instability. This
plays EVERY saved eval checkpoint of a run through the same paired window_rate measurement
`scripts/power_window_rate.py` uses, and prints the trajectory.

HONESTY REQUIREMENT, stated here so it travels with the numbers: picking the argmax over
this sweep and then quoting it is selection on the evaluation set. The peak is diagnostic --
it answers "did this policy ever reach greedy" -- and any headline built on it needs either a
held-out episode split or explicit framing as an oracle-selected upper bound. Do not quote
the peak as if it were the run's score.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from ma.baselines import UncertaintyGreedyAgent                      # noqa: E402
from ma.policy import IndependentPPO                                 # noqa: E402
from scripts.power_window_rate import build_env                      # noqa: E402
from scripts.transfer_eval import window_rates                       # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", help="the run's .json")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    d = json.loads(open(args.result).read())
    cfg = d["config"]
    seed = d.get("seed", 0)
    env = build_env(cfg)
    agents = env.topology.agents

    greedy = {a: UncertaintyGreedyAgent(a, seed, bar=1.0) for a in agents}
    gwr = float(np.mean(window_rates(env, greedy, args.episodes, seed_base=seed * 100_000)))

    stem = args.result[:-5]
    checkpoints = sorted(glob.glob(f"{stem}_u*.pt"))
    # `_best.pt` too, so the MI-selected choice appears in the same table as the alternatives
    # rather than having to be compared across two runs of this script.
    if os.path.exists(f"{stem}_best.pt"):
        checkpoints.append(f"{stem}_best.pt")

    print(f"{args.result}   greedy window_rate = {gwr:.3f}")
    print(f"{'checkpoint':16s} {'learned wr':>11s} {'gap':>8s}")
    rows = []
    for path in checkpoints:
        try:
            ppo = IndependentPPO.load(path, env)
        except Exception as exc:                       # a checkpoint from another obs layout
            print(f"{os.path.basename(path):16s}  skipped ({type(exc).__name__})")
            continue
        lwr = float(np.mean(window_rates(env, ppo.policies(deterministic=False),
                                         args.episodes, seed_base=seed * 100_000)))
        label = os.path.basename(path).replace(os.path.basename(stem) + "_", "")
        print(f"{label:16s} {lwr:11.3f} {lwr - gwr:+8.3f}", flush=True)
        rows.append({"checkpoint": label, "learned_wr": lwr, "greedy_wr": gwr,
                     "gap": lwr - gwr})

    if rows:
        peak = max(rows, key=lambda r: r["learned_wr"])
        print(f"\nPEAK: {peak['checkpoint']} at {peak['learned_wr']:.3f} "
              f"(gap {peak['gap']:+.3f}) -- diagnostic only, see module docstring")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"result": args.result, "greedy_wr": gwr, "rows": rows}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
