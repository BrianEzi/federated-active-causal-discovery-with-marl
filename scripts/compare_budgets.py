"""Side-by-side view of every sweep axis at 4,000 and 12,000 episodes.

EXISTS FOR ONE DECISION. The 4,000-episode sweep is the design every table, figure and claim in
Chapter 4 currently rests on. `FINDINGS_UNDERTRAINING_2026_09_02.md` and
`FINDINGS_AGENT_COUNT_2026_09_02.md` show that budget was short at k=12, so the question is
whether to promote the 12,000-episode re-run to the primary tables or keep it beside them as a
budget comparison. Promoting means regenerating every figure, table and claim; keeping means
the training budget stays an explicit finding rather than a hidden confound.

That decision should be a glance, not a project. This prints both designs on the same axes.

READ THE `n` COLUMNS. The competence floor removes different runs at each budget, so a cell can
rest on two seeds at one budget and three at the other. A ratio computed over different seed
sets is not a like-for-like comparison and the counts are printed so that is visible.

THE SHD COLUMNS ARE FINAL-POLICY AND ARE NOT WHAT THE CHAPTER QUOTES. Each run records
`global_hard_shd` from its own evaluation pass, which scores the policy at the last update.
`FINDINGS_CHECKPOINT_2026_09_01.md` shows that policy degrades on long runs, so comparing a
4,000-episode final policy against a 12,000-episode one confounds training budget with that
degradation. At K=5 this file reads 20.79 where the same cell measured from the selected
checkpoint reads 0.06, a factor of 300 -- entirely one seed whose final policy collapsed.

So these columns are a PROGRESS VIEW, not evidence. The chapter's numbers come from
`scripts/global_shd_paired.py --checkpoint best`, and the re-run must be measured that way
before any table is promoted. Run it on the completed cells and compare those outputs, not
these.
"""
from __future__ import annotations
import argparse, glob, json, pathlib, re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELL = re.compile(r"k(\d+)s(\d+)n(\d+)b(\d+)")
FLOOR, TAIL = 0.70, 10


def rows(directory: str):
    out = []
    for p in sorted((ROOT / directory).glob("k*_s*.json")):
        m = CELL.match(p.stem)
        if not m:
            continue
        d = json.loads(p.read_text())
        if "arms" not in d:
            continue
        tail = [h.get("window_rate", 0.0) for h in (d.get("history") or [])[-TAIL:]]
        out.append(dict(cell=p.stem.rsplit("_s", 1)[0], seed=d.get("seed"),
                        k=int(m[1]), sigma=int(m[2]) / 100, n=int(m[3]), beta=int(m[4]) / 100,
                        wr=(sum(tail) / len(tail) if tail else 0.0),
                        eps=d["config"]["ppo_total_episodes"], arms=d["arms"]))
    return out


AXES = [("window size k", "k", lambda r: r["sigma"] == .5 and r["n"] == 4 and r["beta"] == 1.5),
        ("agents K", "n", lambda r: r["k"] == 12 and r["sigma"] == .5 and r["beta"] == 1.5),
        ("contended sigma", "sigma", lambda r: r["k"] == 12 and r["n"] == 4 and r["beta"] == 1.5),
        ("budget beta", "beta", lambda r: r["k"] == 12 and r["sigma"] == .5 and r["n"] == 4)]


def summarise(sel, key, x):
    cell = [r for r in sel if r[key] == x and r["wr"] >= FLOOR]
    if not cell:
        return None
    L = np.mean([r["arms"]["learned"]["global_hard_shd"] for r in cell])
    G = np.mean([r["arms"]["greedy_uncertainty"]["global_hard_shd"] for r in cell])
    S = np.mean([r["arms"]["learned"]["success"] for r in cell])
    return L, G, (L / G if G else float("nan")), S, len(cell)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default="results/sweep12k")
    args = ap.parse_args(argv)

    old, new = rows("results/sweep/oracle"), rows(args.new)
    done = len({(r["cell"], r["seed"]) for r in new})
    print(f"4,000-episode design: {len(old)} runs | re-run: {done} runs present")
    print("SHD columns are FINAL-POLICY, from each run's own eval pass. Not what the chapter")
    print("quotes and not evidence -- see the module docstring. Progress view only.\n")

    for label, key, keep in AXES:
        so, sn = [r for r in old if keep(r)], [r for r in new if keep(r)]
        xs = sorted({r[key] for r in so})
        print(f"--- {label}")
        print(f"{'x':>6} | {'4k L':>9} {'4k M':>9} {'ratio':>6} {'succ':>6} {'n':>2}"
              f" | {'12k L':>9} {'12k M':>9} {'ratio':>6} {'succ':>6} {'n':>2}  verdict")
        for x in xs:
            a, b = summarise(so, key, x), summarise(sn, key, x)
            left = (f"{a[0]:9.5f} {a[1]:9.5f} {a[2]:6.2f} {a[3]:6.3f} {a[4]:2d}"
                    if a else " " * 36)
            if b is None:
                print(f"{x:>6g} | {left} | {'pending':>36}")
                continue
            # A verdict on one seed is not a verdict. Seed counts differing between budgets
            # means the two ratios range over different runs, which is not a comparison.
            flip = ""
            if a and b[4] >= 2 and a[4] == b[4]:
                if a[2] > 1 and b[2] <= 1:
                    flip = "reversal gone (final-policy)"
                elif a[2] <= 1 and b[2] > 1:
                    flip = "reversal appears (final-policy)"
            elif a and b[4] < 2:
                flip = f"n={b[4]}, no verdict"
            elif a and a[4] != b[4]:
                flip = f"seed count {a[4]} -> {b[4]}, not comparable"
            print(f"{x:>6g} | {left} | {b[0]:9.5f} {b[1]:9.5f} {b[2]:6.2f} {b[3]:6.3f} "
                  f"{b[4]:2d}  {flip}")
        print()

    gated_old = [r for r in old if r["wr"] < FLOOR]
    gated_new = [r for r in new if r["wr"] < FLOOR]
    print(f"below the competence floor: {len(gated_old)} at 4,000 episodes, "
          f"{len(gated_new)} in the re-run")
    for r in gated_new:
        print(f"   still failing at 12,000: {r['cell']} s{r['seed']} wr={r['wr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
