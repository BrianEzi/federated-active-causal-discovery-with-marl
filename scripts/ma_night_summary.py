"""Summarise every completed two-agent arm, whatever finished.

Built to be robust to a partially-finished night: it reads whatever seeds exist, reports `n`
alongside every figure, and never silently averages over a different number of seeds than the
arm it is being compared with. Paired comparisons use only the seeds present in BOTH arms.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
from typing import Dict, List

import numpy as np

AGENT_KEYS = ("A", "B")


def load(arm: str) -> Dict[int, dict]:
    out = {}
    for f in glob.glob(f"results/ma_fixed/{arm}_s*.json"):
        seed = int(pathlib.Path(f).stem.rsplit("_s", 1)[1])
        try:
            out[seed] = json.load(open(f))
        except json.JSONDecodeError:            # a seed still being written
            continue
    return dict(sorted(out.items()))


def _sum_over_agents(block: dict, key: str) -> float:
    return float(sum(block[key].values())) if key in block else float("nan")


def arm_row(arm: str, runs: Dict[int, dict]) -> dict:
    learned = [r["arms"]["learned"] for r in runs.values()]
    if not learned:
        return {}
    succ = np.array([a["success"] for a in learned])
    rnd = np.array([r["arms"]["random_clamp"]["success"] for r in runs.values()])
    greedy = np.array([r["arms"]["greedy"]["success"] for r in runs.values()])
    passes = np.array([r["arms"]["pass"]["success"] for r in runs.values()])
    private = np.array([_sum_over_agents(a, "clamps_private_per_agent") for a in learned])
    shared = np.array([_sum_over_agents(a, "clamps_shared_per_agent") for a in learned])
    total = private + shared
    return {
        "arm": arm, "n": len(succ),
        "learned_median": float(np.median(succ)), "learned_mean": float(succ.mean()),
        "learned_sd": float(succ.std(ddof=1)) if len(succ) > 1 else float("nan"),
        "random": float(np.median(rnd)), "greedy": float(np.median(greedy)),
        "pass": float(np.median(passes)),
        "ahead_of_random": int((succ > rnd).sum()),
        "collapsed": int(sum(bool(r.get("collapsed")) for r in runs.values())),
        "private_clamp_pct": float(np.nanmean(np.where(total > 0, private / total, np.nan)) * 100),
        "free_rider": float(np.nanmean([a.get("free_rider_index", np.nan) for a in learned])),
        "interventions": float(np.nanmean(
            [_sum_over_agents(a, "interventions_per_agent") for a in learned])),
        "success_connected": float(np.nanmean(
            [a.get("success_connected", np.nan) for a in learned])),
        "success_disconnected": float(np.nanmean(
            [a.get("success_disconnected", np.nan) for a in learned])),
    }


def paired(a: str, b: str, runs_a: Dict[int, dict], runs_b: Dict[int, dict],
           seed: int = 0) -> dict:
    """Bootstrap CI on the per-seed difference, over seeds present in BOTH arms."""
    shared_seeds = sorted(set(runs_a) & set(runs_b))
    if len(shared_seeds) < 3:
        return {"pair": f"{a} - {b}", "n": len(shared_seeds), "note": "too few shared seeds"}
    diff = np.array([runs_a[s]["arms"]["learned"]["success"]
                     - runs_b[s]["arms"]["learned"]["success"] for s in shared_seeds])
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(20000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"pair": f"{a} - {b}", "n": len(shared_seeds),
            "mean_diff": float(diff.mean()), "ci": [float(lo), float(hi)],
            "a_ahead_on": int((diff > 0).sum()),
            "significant": bool(lo > 0 or hi < 0)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="tb_both,tb_clamp,nobit_clamp,randturn_clamp")
    ap.add_argument("--pairs", default="tb_both:tb_clamp,tb_clamp:nobit_clamp,"
                                       "tb_clamp:randturn_clamp")
    ap.add_argument("--out", default="results/night_summary.json")
    args = ap.parse_args(argv)

    runs = {arm: load(arm) for arm in args.arms.split(",")}
    rows = [arm_row(arm, r) for arm, r in runs.items() if r]

    head = (f"{'arm':16s}{'n':>3}{'learned':>9}{'sd':>7}{'random':>8}{'greedy':>8}"
            f"{'pass':>7}{'ahead':>7}{'coll':>6}{'priv%':>7}{'free':>7}{'conn':>7}{'disc':>7}")
    print(head); print("-" * len(head))
    for r in rows:
        print(f"{r['arm']:16s}{r['n']:>3}{r['learned_median']:>9.3f}{r['learned_sd']:>7.3f}"
              f"{r['random']:>8.3f}{r['greedy']:>8.3f}{r['pass']:>7.3f}"
              f"{r['ahead_of_random']:>4}/{r['n']:<2}{r['collapsed']:>6}"
              f"{r['private_clamp_pct']:>7.1f}{r['free_rider']:>7.2f}"
              f"{r['success_connected']:>7.3f}{r['success_disconnected']:>7.3f}")

    print("\npaired comparisons (same seeds only)")
    pairs = []
    for spec in args.pairs.split(","):
        a, b = spec.split(":")
        if runs.get(a) and runs.get(b):
            p = paired(a, b, runs[a], runs[b])
            pairs.append(p)
            if "note" in p:
                print(f"  {p['pair']:28s} n={p['n']}  {p['note']}")
            else:
                mark = "SIGNIFICANT" if p["significant"] else "not significant"
                print(f"  {p['pair']:28s} n={p['n']}  mean {p['mean_diff']:+.4f}  "
                      f"CI [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]  "
                      f"first ahead on {p['a_ahead_on']}/{p['n']}  {mark}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": rows, "pairs": pairs}, indent=1))
    print(f"\nwrote {out}")
    return {"arms": rows, "pairs": pairs}


if __name__ == "__main__":
    main()
