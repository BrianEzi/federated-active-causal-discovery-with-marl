"""Aggregate results/attr_scale/*_scored.json into the tables the four jobs are judged on.

Two rules this enforces, because both have already cost this project a result:

  * ACROSS-ARM COMPARISONS ARE PAIRED PER EPISODE. Every arm in a file played
    `seed * 100_000 + episode`, so two files at the same seed saw the same worlds. The
    reported error on a difference is the standard error of the per-episode difference,
    never the quadrature sum of two independent standard errors.
  * THE EPISODES ARE CHECKED, NOT ASSUMED. `random_vary` reads nothing and is seeded
    identically, so its per-episode rows MUST agree between two files at the same seed. If
    they do not, the files are not comparable and this refuses to pair them rather than
    printing a number that looks fine.

Across SEEDS the spread is reported as a plain mean and standard error over seed means:
pairing controls for which worlds were drawn, and it does not control for training-seed
variance, which is a separate and larger source of spread.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np

METRICS = ("identified", "attribution", "structure", "private_share")


def load(directory: pathlib.Path) -> Dict[str, Dict[int, dict]]:
    """{arm: {seed: report}} from every *_scored.json in the directory."""
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for path in sorted(directory.glob("*_scored.json")):
        match = re.match(r"(.+)_s(\d+)_scored\.json$", path.name)
        if not match:
            continue
        arm, seed = match.group(1), int(match.group(2))
        out[arm][seed] = json.loads(path.read_text())
    return out


def _seed_spread(reports: Dict[int, dict], arm_key: str, metric: str):
    values = [r["arms"][arm_key][metric] for r in reports.values()
              if arm_key in r["arms"]]
    if not values:
        return float("nan"), float("nan"), 0
    arr = np.array(values, dtype=float)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), se, len(arr)


def per_arm_table(data) -> None:
    print("\n=== per arm, mean over seeds (se over SEEDS, not episodes) ===")
    header = f"{'arm':22s} {'n':>2s} " + " ".join(f"{m:>18s}" for m in METRICS)
    print(header)
    for arm in sorted(data):
        reports = data[arm]
        row = [f"{arm:22s} {len(reports):2d} "]
        for metric in METRICS:
            mean, se, n = _seed_spread(reports, "learned", metric)
            row.append(f"{mean:11.3f}+/-{se:5.3f}" if n else f"{'--':>18s}")
        print(" ".join(row))

    print("\n=== baselines, from the first seed of each arm (identical across seeds "
          "by construction; disagreement here is a bug) ===")
    for arm in sorted(data):
        first = data[arm][min(data[arm])]
        for label in ("probe_then_work", "greedy_uncertainty", "random_vary"):
            if label in first["arms"]:
                v = first["arms"][label]
                print(f"{arm:22s} {label:20s} " + " ".join(
                    f"{v[m]:11.3f}" for m in METRICS))
        break


def _rows(report, label) -> np.ndarray:
    return np.array([r["identified"] for r in report["rows"][label]], dtype=float)


def _episodes_match(a: dict, b: dict) -> bool:
    """random_vary reads nothing and is seeded identically -- its rows pin the episodes."""
    try:
        return np.array_equal(_rows(a, "random_vary"), _rows(b, "random_vary"))
    except KeyError:
        return False


def paired_within(data) -> None:
    print("\n=== learned vs each baseline, PAIRED per episode, pooled over seeds ===")
    for arm in sorted(data):
        deltas = defaultdict(list)
        for report in data[arm].values():
            if "learned" not in report["rows"]:
                continue
            learned = _rows(report, "learned")
            for label in ("probe_then_work", "greedy_uncertainty", "random_vary"):
                if label in report["rows"]:
                    deltas[label].extend(learned - _rows(report, label))
        for label, values in deltas.items():
            arr = np.array(values)
            se = arr.std(ddof=1) / np.sqrt(len(arr))
            flag = "" if abs(arr.mean()) > 2 * se else "   (inside 2 se)"
            print(f"{arm:22s} learned - {label:20s} {arr.mean():+.3f} +/- {se:.3f}{flag}")


def paired_between(data, left: str, right: str) -> None:
    """The job-2 and job-1-control comparisons: two ARMS, same seeds, same episodes."""
    if left not in data or right not in data:
        return
    seeds = sorted(set(data[left]) & set(data[right]))
    if not seeds:
        return
    print(f"\n=== {left} vs {right}, PAIRED per episode, seeds {seeds} ===")
    pooled = []
    for seed in seeds:
        a, b = data[left][seed], data[right][seed]
        if not _episodes_match(a, b):
            print(f"  seed {seed}: REFUSED -- random_vary rows differ, so these two files "
                  f"did not see the same episodes")
            continue
        delta = _rows(a, "learned") - _rows(b, "learned")
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        print(f"  seed {seed}: {delta.mean():+.3f} +/- {se:.3f}")
        pooled.extend(delta)
    if pooled:
        arr = np.array(pooled)
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        print(f"  POOLED : {arr.mean():+.3f} +/- {se:.3f}   "
              f"({'outside' if abs(arr.mean()) > 2 * se else 'INSIDE'} 2 se)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/attr_scale")
    args = ap.parse_args(argv)
    data = load(pathlib.Path(args.dir))
    if not data:
        print("no scored results yet")
        return
    print(f"arms present: {', '.join(f'{a}(seeds {sorted(data[a])})' for a in sorted(data))}")
    per_arm_table(data)
    paired_within(data)
    # Job 2: does paying for the group beat paying for your own window?
    paired_between(data, "attr3a_peragent", "attr3a_shared")
    # Job 1's control: what does the density guard cost, at a size cheap enough to ask?
    paired_between(data, "attr3a_peragent", "attr3a_guarded")


if __name__ == "__main__":
    main()
