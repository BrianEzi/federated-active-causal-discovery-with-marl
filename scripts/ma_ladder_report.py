"""Read the scale ladder: does the learned policy beat its baselines, rung by rung?

The ladder's question is not "what is the success rate" but "does the gap between learned
and the best reference survive as the problem grows". So the headline is the PAIRED
difference per seed, not two independent means -- seeds differ in the graphs they draw, and
comparing pooled averages throws away the pairing that makes the comparison sensitive.

SCORED AGAINST THE BEST BASELINE, not against random. `greedy` conditions on the belief and
is the real opponent; beating `random_clamp` while losing to `greedy` is not a result. The
per-baseline columns are kept so a rung where greedy collapses is visible rather than
hidden inside a max().

A rung with fewer than three seeds reporting is printed but marked, because a two-seed
interval is not an interval.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

import numpy as np

from ma.evaluate import bootstrap_ci

LEARNED = "learned"
RUNG_RE = re.compile(r"^(rung(\d+)_[0-9a-z_]+_d(\d+))_s(\d+)_([a-z_]+)\.json$")


def collect(directory: pathlib.Path) -> dict:
    """`{rung_name: {seed: {arm: row}}}`, from one file per (rung, seed, arm)."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    meta: dict = {}
    for path in sorted(directory.glob("*.json")):
        match = RUNG_RE.match(path.name)
        if not match:
            continue
        rung, index, d, seed, arm = match.groups()
        out[rung][int(seed)][arm] = json.loads(path.read_text())
        meta[rung] = (int(index), int(d))
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ladder_eval")
    ap.add_argument("--metric", default="success")
    ap.add_argument("--out", default="results/ladder_summary.json")
    args = ap.parse_args()

    grouped, meta = collect(pathlib.Path(args.dir))
    if not grouped:
        raise SystemExit(f"no ladder results in {args.dir}")

    rows = []
    order = sorted(grouped, key=lambda r: meta[r][0])
    print(f"{'rung':26s} {'d':>3} {'n':>2} {'learned':>15s} {'greedy':>8s} "
          f"{'random':>8s} {'pass':>7s} {'paired vs best':>22s}")
    print("-" * 104)
    for rung in order:
        seeds = grouped[rung]
        per_arm = defaultdict(list)
        paired = []
        for seed in sorted(seeds):
            arms = seeds[seed]
            if LEARNED not in arms:
                continue
            learned = float(arms[LEARNED][args.metric])
            references = {a: float(r[args.metric]) for a, r in arms.items() if a != LEARNED}
            if not references:
                continue
            for arm, value in arms.items():
                per_arm[arm].append(float(value[args.metric]))
            paired.append(learned - max(references.values()))

        if not paired:
            continue
        n = len(paired)
        mean = float(np.mean(paired))
        low, high = bootstrap_ci(paired, seed=0)
        verdict = "BEATS" if low > 0 else ("loses" if high < 0 else "ties")
        flag = "" if n >= 3 else "  (underpowered)"

        def col(arm):
            values = per_arm.get(arm, [])
            return f"{np.mean(values):8.3f}" if values else f"{'--':>8s}"

        print(f"{rung:26s} {meta[rung][1]:3d} {n:2d} "
              f"{np.mean(per_arm[LEARNED]):15.3f} {col('greedy')} {col('random_clamp')} "
              f"{col('pass'):>7s} "
              f"{mean:+7.3f} [{low:+.3f},{high:+.3f}] {verdict}{flag}")
        rows.append({"rung": rung, "d": meta[rung][1], "seeds": n,
                     "metric": args.metric,
                     "means": {a: float(np.mean(v)) for a, v in per_arm.items()},
                     "paired_vs_best": mean, "ci": [low, high], "verdict": verdict})

    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
