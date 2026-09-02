"""Does the answer-rate result survive the deterministic evaluation path?

WHAT CHANGED AND WHY A COMPARISON IS NEEDED. Until 2026-09-02 21:15 `global_shd_paired.play`
did not seed the torch RNG, so a learned arm evaluated with `--sample` drew its actions from
the global generator and a re-run of the same checkpoint returned different numbers. The
greedy and random arms, which carry their own seeded generators, reproduced exactly. The
published grid is therefore honest but not reproducible, and the rebuilt grid is expected to
differ from it -- by roughly one paired standard error, per agent A's 24-run measurement.

So the question this script asks is NOT "do the numbers match". They should not. It is:

    1. Did any cell move MORE than the known cost of the fix? A shift of several paired SE
       is not the RNG; it is a mismatch of checkpoint, seed, episode count or baseline, and
       it invalidates the cell rather than merely renumbering it.
    2. Did the HEADLINE survive? The claim is a count -- all fifteen (rate, seed) cells at
       rho <= 0.90 have the learned arm ahead of greedy, and none of the six at rho >= 0.95.
       A count is exactly the kind of claim a one-SE shift can break, because the winning
       margins are -0.009 to -0.018 against per-cell SE of 0.0005 to 0.008 and a marginal
       cell can cross zero.

The count is reported per cell, not only in aggregate. A mean that stays negative while one
of its three seeds flips sign is a different and weaker claim than the one on record, and
printing only the mean would hide precisely the thing that needs checking.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

RATES = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50]
SEEDS = [0, 1, 2]
# The rate at or below which the result claims the learned policy wins. Fixed before the
# rebuild; not to be adjusted to whatever the rebuilt grid happens to show.
WIN_AT_OR_BELOW = 0.90


def load(path: pathlib.Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d[0] if isinstance(d, list) else d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new_dir", default="results/power/rho/deterministic")
    ap.add_argument("--old_dir", default="results/power/rho")
    # Two SE is the flag threshold: agent A measured the fix's own effect at a median 0.4 SE
    # with a maximum of 2.22, so anything past 2 SE is at the edge of what the fix explains
    # and worth looking at by hand rather than absorbing into the total.
    ap.add_argument("--flag_se", type=float, default=2.0)
    ap.add_argument("--out", default="results/power/rho/DETERMINISTIC_COMPARE.json")
    args = ap.parse_args(argv)

    new_dir, old_dir = pathlib.Path(args.new_dir), pathlib.Path(args.old_dir)
    cells, missing = [], []
    for rho in RATES:
        for s in SEEDS:
            name = f"xfer_rho{rho:.2f}_s{s}.json"
            new, old = load(new_dir / name), load(old_dir / name)
            if new is None:
                missing.append(name)
                continue
            n = new["paired"]["learned-greedy"]
            row = {"rho": rho, "seed": s, "new_delta": n["delta"], "new_se": n["se"],
                   "new_learned": new["means"]["learned"]["hard"],
                   "new_greedy": new["means"]["greedy"]["hard"],
                   "has_rows": new.get("rows") is not None}
            if old is not None:
                o = old["paired"]["learned-greedy"]
                row["old_delta"] = o["delta"]
                row["old_se"] = o["se"]
                # Measure the move against the OLD SE, which is the interval the published
                # claim was made inside.
                row["shift_se"] = ((n["delta"] - o["delta"]) / o["se"]) if o["se"] > 0 else float("nan")
                row["sign_flip"] = bool(np.sign(n["delta"]) != np.sign(o["delta"]))
                # Did greedy reproduce? It has its own generator and was never affected, so
                # a moved greedy mean means the two grids are not paired over the same
                # episodes and nothing below it can be trusted.
                row["greedy_moved"] = abs(new["means"]["greedy"]["hard"]
                                          - old["means"]["greedy"]["hard"]) > 1e-12
            cells.append(row)

    if missing:
        print(f"INCOMPLETE -- {len(missing)} of {len(RATES) * len(SEEDS)} cells not yet built:")
        for m in missing[:8]:
            print(f"    {m}")
        print("Nothing below is a verdict until the grid is complete.\n")

    print(f"{'rho':>5s} {'s':>2s} {'old delta':>11s} {'new delta':>11s} {'shift/SE':>9s} "
          f"{'new SE':>8s} {'win?':>5s} {'flip':>5s} {'greedy':>7s} {'rows':>5s}")
    for c in cells:
        win = "yes" if c["new_delta"] < 0 else "NO"
        flip = "FLIP" if c.get("sign_flip") else ""
        gm = "MOVED" if c.get("greedy_moved") else "same"
        od = f"{c['old_delta']:+11.6f}" if "old_delta" in c else " " * 11
        sh = f"{c['shift_se']:+9.2f}" if "shift_se" in c else " " * 9
        print(f"{c['rho']:5.2f} {c['seed']:2d} {od} {c['new_delta']:+11.6f} {sh} "
              f"{c['new_se']:8.6f} {win:>5s} {flip:>5s} {gm:>7s} "
              f"{'yes' if c['has_rows'] else 'NO':>5s}")

    print()
    big = [c for c in cells if abs(c.get("shift_se", 0.0)) > args.flag_se]
    flips = [c for c in cells if c.get("sign_flip")]
    gmoved = [c for c in cells if c.get("greedy_moved")]
    norows = [c for c in cells if not c["has_rows"]]

    if gmoved:
        print(f"!! {len(gmoved)} cell(s) have a MOVED greedy mean. Greedy carries its own seeded "
              f"generator and cannot be affected by the torch fix, so this is a pairing fault "
              f"-- wrong baseline, seed or episode count -- not a renumbering. Stop here.")
    if norows:
        print(f"!! {len(norows)} cell(s) still carry no per-episode rows.")
    print(f"{len(big)} cell(s) moved more than {args.flag_se:.1f} old SE"
          + (f": {[(c['rho'], c['seed'], round(c['shift_se'], 2)) for c in big]}" if big else ""))
    print(f"{len(flips)} cell(s) changed the SIGN of the learned-greedy delta"
          + (f": {[(c['rho'], c['seed']) for c in flips]}" if flips else ""))

    # THE HEADLINE, as a count, under BOTH readings of "beat".
    #
    # The findings note says "15 of 15 seeds at rho <= 0.90 beat greedy, 0 of 6 at rho >= 0.95
    # do", and does not say what "beat" means. It matters, because on the published grid the
    # two halves need different readings to both be true: at rho <= 0.90 all fifteen cells are
    # negative AND all fifteen clear 2 SE, but at rho >= 0.95 two of the six ARE numerically
    # negative (-1.57 and -0.29 SE) and neither is significant. So "0 of 6" is true of
    # significance and false of sign. Both counts are printed here so the ambiguity cannot
    # survive into the rebuilt claim.
    lo = [c for c in cells if c["rho"] <= WIN_AT_OR_BELOW]
    hi = [c for c in cells if c["rho"] > WIN_AT_OR_BELOW]

    def counts(group):
        neg = sum(c["new_delta"] < 0 for c in group)
        sig = sum(c["new_delta"] < 0 and abs(c["new_delta"]) > 2 * c["new_se"] for c in group)
        return neg, sig

    lo_neg, lo_sig = counts(lo)
    hi_neg, hi_sig = counts(hi)
    print(f"\nHEADLINE  rho <= {WIN_AT_OR_BELOW}: {lo_neg}/{len(lo)} ahead by sign, "
          f"{lo_sig}/{len(lo)} ahead beyond 2 SE   (published 15/15 and 15/15)")
    print(f"HEADLINE  rho >  {WIN_AT_OR_BELOW}: {hi_neg}/{len(hi)} ahead by sign, "
          f"{hi_sig}/{len(hi)} ahead beyond 2 SE   (published 2/6 and 0/6)")
    complete = not missing
    if complete and lo_neg == len(lo) and lo_sig == len(lo) and hi_sig == 0 and not gmoved:
        print("\nSURVIVES: separation is unchanged on the deterministic path -- every low-rate "
              "cell ahead beyond 2 SE, no high-rate cell ahead beyond 2 SE.")
    elif not complete:
        print("\nNO VERDICT: grid incomplete.")
    else:
        print("\nDOES NOT SURVIVE UNCHANGED: the count moved. Report the new count, and the "
              "cells that flipped, rather than the aggregate mean.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cells": cells, "missing": missing,
                               "headline": {"low_rate_n": len(lo), "low_rate_neg": lo_neg,
                                            "low_rate_sig": lo_sig, "high_rate_n": len(hi),
                                            "high_rate_neg": hi_neg, "high_rate_sig": hi_sig,
                                            "complete": complete}}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
