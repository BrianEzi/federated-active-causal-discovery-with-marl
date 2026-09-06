"""Where does the answer rate stop producing a competent policy, and does that point move with
window size?

THE QUESTION CHANGED ON 6 SEP AND THIS IS THE TOOL FOR THE NEW ONE. The k=8 grid established
that training under a partial oracle transfers to sampled evidence, with rho=0.50 its strongest
point. Moving to the k=12 principal cell, rho=0.50 does not produce a competent policy at all:
three of three seeds land at window rate 0.120-0.214 against a floor of 0.70, and two of three
are beaten by the myopic rule inside their own training regime. The twelve-seed control at the
identical cell with only the dial changed sits at 0.975-1.000.

So the object of study is no longer "does the answer rate help" but "over what RANGE of the
answer rate is there a policy to speak of, and does that range shrink with the window". This
reports competence against rate for every window size on disk, marks the floor, and refuses to
interpolate a crossing from fewer than two bracketing rates.

DELIBERATELY TRAINING-SIDE ONLY. It reads `window_rate` from each run's own history, which is a
diagnostic, not a thesis number. Nothing here is a transfer result and nothing here should be
quoted as one; its job is to say which cells are eligible to BE transfer results.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

import numpy as np

FLOOR, TAIL = 0.70, 10


def wr(path: str) -> float:
    h = (json.loads(open(path).read()).get("history") or [])[-TAIL:]
    return float(np.mean([x.get("window_rate", 0.0) for x in h])) if h else float("nan")


def collect(patterns) -> dict:
    out = collections.defaultdict(list)
    for pattern, rate_from in patterns:
        for f in sorted(glob.glob(pattern)):
            m = re.search(rate_from, os.path.basename(f))
            if not m:
                continue
            out[float(m.group(1))].append(wr(f))
    return out


def report(label: str, rows: dict) -> None:
    if not rows:
        print(f"\n{label}: nothing on disk")
        return
    print(f"\n{label}")
    print(f"  {'rho':>5s} {'n':>2s} {'window rate (per seed)':<34s} {'mean':>6s} {'verdict':>9s}")
    for rate in sorted(rows, reverse=True):
        v = np.array(rows[rate])
        seeds = " ".join(f"{x:.3f}" for x in v)
        n_pass = int((v >= FLOOR).sum())
        verdict = "PASS" if n_pass == len(v) else ("FAIL" if n_pass == 0 else f"{n_pass}/{len(v)}")
        print(f"  {rate:5.2f} {len(v):2d} {seeds:<34s} {v.mean():6.3f} {verdict:>9s}")

    # The crossing is only meaningful if a passing rate and a failing rate BRACKET it. Reporting
    # "the floor is at 0.6" from an all-pass or all-fail column would be inventing a number.
    ok = sorted([r for r, v in rows.items() if np.mean(v) >= FLOOR])
    bad = sorted([r for r, v in rows.items() if np.mean(v) < FLOOR])
    if ok and bad and min(ok) > max(bad):
        print(f"  -> competence lost between rho={max(bad):.2f} and rho={min(ok):.2f}")
    elif not bad:
        print(f"  -> every measured rate is competent; the floor is BELOW {min(ok):.2f}, "
              f"unmeasured")
    elif not ok:
        print(f"  -> no measured rate is competent; the floor is ABOVE {max(bad):.2f}, "
              f"unmeasured")
    else:
        print(f"  -> passes and fails are interleaved ({ok} pass, {bad} fail); no single "
              f"crossing to quote")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k8_dir", default="results/power/rho")
    ap.add_argument("--k12_dirs", nargs="*",
                    default=["results/rho12", "results/rho12_myriad"])
    args = ap.parse_args(argv)

    report("k_v = 8   (budget 70, 8k episodes, channels+reprobe ON)",
           collect([(os.path.join(args.k8_dir, "rho[01].[0-9][0-9]_s?.json"),
                     r"rho([0-9.]+)_s\d\.json$")]))

    # Both k=12 paths pooled by rate: a cell is a cell whichever machine produced it, and
    # `scripts/diff_dual_path.py` is what checks they agree before either is trusted. Duplicates
    # across paths show up as a larger n, which is visible rather than hidden.
    report("k_v = 12  (budget 50, 12k episodes, channels OFF -- the principal cell)",
           collect([(os.path.join(d, "rho[01].[0-9][0-9]_s?.json"),
                     r"rho([0-9.]+)_s\d\.json$") for d in args.k12_dirs]))

    # The rho=1.00 control at k=12 is the ladder's arm A -- the same cell with the dial at its
    # default, twelve seeds instead of three. Reported separately so nobody mistakes it for part
    # of the swept fleet.
    ladder = sorted(glob.glob("results/central12k/v2_k12_A_s*.json"))
    if ladder:
        v = np.array([wr(f) for f in ladder])
        print(f"\n  k=12 rho=1.00 control (ladder arm A, {len(v)} seeds): "
              f"{v.min():.3f}-{v.max():.3f}, mean {v.mean():.3f}  "
              f"{'PASS' if (v >= FLOOR).all() else 'MIXED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
