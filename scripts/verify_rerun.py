"""Does the re-emitted transfer cell reproduce the original delta exactly?

The rerun exists to add per-episode rows that were not saved the first time, so the numbers
themselves must not move: same checkpoint, same seed, same 200 episode seeds, same baseline
vectors. Any difference means the evaluation path is not deterministic, which would be a far
more serious finding than the missing rows -- every paired comparison in the answer-rate result
assumes that replaying `seed * 100_000 + episode` gives the same worlds to every arm.

Also re-derives the paired SE from the newly stored rows and checks it against the SE the
script reported, which is the specific thing a reader could not previously do.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np


def load(path):
    d = json.loads(open(path).read())
    return d[0] if isinstance(d, list) else d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rerun_dir", default="results/power/rho/rerun")
    ap.add_argument("--orig_dir", default="results/power/rho")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.rerun_dir, "xfer_rho*.json")))
    if not files:
        print(f"nothing in {args.rerun_dir} yet")
        return 0

    print(f"{'cell':22s} {'orig delta':>12s} {'rerun delta':>12s} {'match':>7s} "
          f"{'SE reported':>12s} {'SE from rows':>13s} {'match':>7s}")
    bad = 0
    for f in files:
        name = os.path.basename(f)
        orig_path = os.path.join(args.orig_dir, name)
        if not os.path.exists(orig_path):
            print(f"{name:22s}  no original to compare")
            continue
        new, old = load(f), load(orig_path)
        dn = new["paired"]["learned-greedy"]["delta"]
        do = old["paired"]["learned-greedy"]["delta"]
        ok = abs(dn - do) <= args.tol

        # Re-derive the paired SE from the rows the rerun now stores -- the check a reader
        # could not perform before, and the whole point of shipping the working.
        rows = new.get("rows")
        se_rep = new["paired"]["learned-greedy"]["se"]
        if rows and "learned" in rows and "greedy" in rows:
            d = np.asarray(rows["learned"]["hard"]) - np.asarray(rows["greedy"]["hard"])
            se_calc = float(d.std(ddof=1) / np.sqrt(len(d)))
            se_ok = abs(se_calc - se_rep) <= 1e-9
        else:
            se_calc, se_ok = float("nan"), False

        bad += (not ok) or (not se_ok)
        print(f"{name[:22]:22s} {do:+12.6f} {dn:+12.6f} {'OK' if ok else 'DIFFER':>7s} "
              f"{se_rep:12.6f} {se_calc:13.6f} {'OK' if se_ok else 'DIFFER':>7s}")

    print()
    if bad:
        print(f"!! {bad} cell(s) failed. A delta mismatch means the evaluation path is NOT "
              f"deterministic and every paired comparison in the result needs re-examining.")
        return 1
    print(f"all {len(files)} cells reproduce their original delta exactly, and the reported "
          f"paired SE is recoverable from the stored rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
