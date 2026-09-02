"""Is the paired delta carried by the whole episode set, or by a handful of episodes?

A mean of 200 paired differences with a standard error can be honest and still be fragile: if
most of the sum comes from three episodes where one arm collapsed, the number describes those
three episodes rather than the policy. The standard error does not distinguish the two cases
well at this sample size, and nothing else in the pipeline looks.

Two free checks, both computed from the stored per-episode rows, no compute:

  SPLIT-HALF. Score the first 100 episodes and the last 100 separately. These are disjoint
  worlds drawn from the same generator, so the two halves are an independent replication of
  each other. A delta that reverses between halves is noise being read as an effect.

  CONCENTRATION. What share of the total paired difference comes from the five per cent of
  episodes with the largest absolute difference? For a broadly distributed effect this sits
  near five per cent and rises with skew. Reported beside the count of episodes where the
  learned arm is ahead, behind and level, which says whether the arm wins often or wins big.

  A NEGATIVE share is not an error. It means the largest individual episodes push AGAINST the
  overall mean, which the mean survives because many small differences outweigh them -- a sign
  of a robust effect rather than a fragile one.

  Both checks are suppressed for a cell whose delta is inside its own paired standard error.
  Split-half sign agreement is meaningless when the quantity is zero, and a concentration
  share divides by that near-zero sum and returns a number in the quadrillions. The first
  version of this script printed 6.1e15 for such a cell and flagged it as failing to
  replicate, which is true of nothing except the arithmetic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def load(path):
    d = json.loads(open(path).read())
    return d[0] if isinstance(d, list) else d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/power/rho/deterministic")
    ap.add_argument("--top_frac", type=float, default=0.05)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.dir, "xfer_rho*_s?.json")))
    if not files:
        print(f"no cells in {args.dir}")
        return 0

    print(f"{'cell':22s} {'delta':>10s} {'1st half':>10s} {'2nd half':>10s} {'agree':>6s} "
          f"{'top5% share':>11s} {'W/L/D':>12s}")
    disagree = 0
    for f in files:
        e = load(f)
        rows = e.get("rows")
        if not rows or rows.get("learned") is None:
            print(f"{os.path.basename(f)[:22]:22s}  no per-episode rows")
            continue
        d = np.asarray(rows["learned"]["hard"]) - np.asarray(rows["greedy"]["hard"])
        h = len(d) // 2
        a, b = d[:h].mean(), d[h:].mean()
        se = e["paired"]["learned-greedy"]["se"]
        # A cell whose delta is inside its own SE has no sign to replicate and no sum to take
        # a share of. Say so, rather than dividing by it.
        if abs(d.mean()) <= se:
            print(f"{os.path.basename(f).replace('xfer_', '').replace('.json', ''):22s} "
                  f"{d.mean():+10.6f} {a:+10.6f} {b:+10.6f} "
                  f"{'--':>6s} {'--':>11s}   inside 1 SE, no effect to test")
            continue
        agree = np.sign(a) == np.sign(b)
        disagree += not agree
        k = max(1, int(round(args.top_frac * len(d))))
        top = np.abs(d).argsort()[::-1][:k]
        share = d[top].sum() / d.sum()
        w, l, t = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
        print(f"{os.path.basename(f).replace('xfer_', '').replace('.json', ''):22s} "
              f"{d.mean():+10.6f} {a:+10.6f} {b:+10.6f} {'yes' if agree else 'NO':>6s} "
              f"{share:11.2f} {w:>4d}/{l:d}/{t:d}")

    print()
    if disagree:
        print(f"!! {disagree} cell(s) reverse sign between halves. For those cells the paired "
              f"mean is not replicated by its own data and should not be quoted per seed.")
    else:
        print("every cell keeps its sign across both halves of its episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
