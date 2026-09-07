"""Read the channels test against the criterion registered before it landed.

The criterion is from `docs/AGENT_B_INBOX.md`, 7 Sep 01:1x, written with the ON arm at 350 of
750 updates and no result visible. It is reproduced here in code so the verdict is applied
rather than argued, and so an ambiguous result is NAMED as ambiguous instead of being resolved
by whoever reads the table first.

    EXPLAINS      ON mean >= 0.70 AND at least 2 of 3 seeds >= 0.70
    DOES NOT      ON mean <  0.62 AND at most 1 of 3 seeds >= 0.70
    AMBIGUOUS     anything else, including a mean in the 0.62-0.70 band --
                  ask for three more seeds, do not pick a side

Both arms are k_v=12, rho=0.95, budget 53 (effective beta 1.52), 12,000 episodes, seeds 0/1/2,
identical in every other field. The OFF arm is the compensated sweep's own rho=0.95 cells, so
this costs three runs rather than six.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

FLOOR, TAIL = 0.70, 10


def wr(path: str) -> float:
    h = (json.loads(open(path).read()).get("history") or [])[-TAIL:]
    return float(np.mean([x.get("window_rate", 0.0) for x in h])) if h else float("nan")


def arm(paths):
    return {os.path.basename(p)[-8:-5]: wr(p) for p in sorted(paths)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--on", default="results/channels12/on_rho0.95_s?.json")
    ap.add_argument("--off", nargs="*",
                    default=["results/rho12b/rho0.95_s?.json",
                             "results/rho12b_myriad/rho0.95_s?.json"])
    args = ap.parse_args(argv)

    on = arm(glob.glob(args.on))
    off = arm([p for pat in args.off for p in glob.glob(pat)])
    if not on:
        print("channels-ON arm has not landed yet")
        return 0

    print(f"{'seed':>6s} {'channels OFF':>13s} {'channels ON':>12s} {'delta':>8s}")
    for s in sorted(set(on) | set(off)):
        a, b = off.get(s, float("nan")), on.get(s, float("nan"))
        print(f"{s:>6s} {a:13.3f} {b:12.3f} {b - a:+8.3f}")
    o = np.array([v for v in off.values()])
    n = np.array([v for v in on.values()])
    print(f"\n  OFF mean {o.mean():.3f}  ({int((o >= FLOOR).sum())}/{len(o)} seeds clear {FLOOR})")
    print(f"  ON  mean {n.mean():.3f}  ({int((n >= FLOOR).sum())}/{len(n)} seeds clear {FLOOR})")

    if len(n) < 3:
        print(f"\nINCOMPLETE -- {len(n)} of 3 ON seeds. No verdict.")
        return 0
    # The ambiguous BAND, from the registered criterion: "if the mean lands between roughly
    # 0.62 and 0.70". My first version of this script omitted it and classified a synthetic
    # [0.72, 0.60, 0.65] arm -- mean 0.657, one seed clearing -- as DOES NOT EXPLAIN, which
    # contradicts what I registered. The code is corrected to match the criterion rather than
    # the criterion softened to match the code.
    AMBIGUOUS_FLOOR = 0.62
    clear = int((n >= FLOOR).sum())
    if n.mean() >= FLOOR and clear >= 2:
        print("\nVERDICT: CHANNELS EXPLAIN THE STEP. RQ2 reports a stated precondition -- "
              "partial-oracle training works at the principal cell GIVEN belief channels -- and "
              "the flags are adopted into the principal-cell configuration.")
    elif n.mean() < AMBIGUOUS_FLOOR and clear <= 1:
        print("\nVERDICT: CHANNELS DO NOT EXPLAIN IT. Both observation features are ruled out "
              "together; window size is what remains between the k=8 and k=12 grids; RQ2's "
              "honest result is that the answer-rate finding does not survive the move to the "
              "principal cell.")
    else:
        print("\nVERDICT: AMBIGUOUS by the registered criterion. Ask for three more seeds; do "
              "NOT pick a side. The OFF arm's own seeds span 0.489-0.595 and the rho=0.85 row "
              "threw a 0.717 beside a 0.444, so this plateau carries seed noise of that order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
