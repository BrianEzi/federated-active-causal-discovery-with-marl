"""Apply Phase 1's decision rule to the depth probe.

THE RULE, fixed in docs/SA_NEXT_PHASE_PLAN.md before any of these numbers existed:

    If depth 2 or 3 beats depth 1 by more than 0.03 mean accuracy at matched data size on
    BOTH d=4 and d=5, carry the best depth into Phase 2's RL runs. Otherwise keep depth 1
    and record that the ceiling is not about multi-hop reachability.

It is applied here mechanically rather than by eye, because the entire value of fixing a
rule in advance is lost if the reading of it is negotiable afterwards. "Beats at matched
data size on both d" is deliberately demanding: a lift that appears at one d, or only at
one data size, is the shape of noise, and the probe's own seed spread was 0.014 in a pilot
against a 0.03 threshold.

    python -m scripts.analyse_depth --dir results/probe_depth
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

THRESHOLD = 0.03


def load(directory: Path) -> dict:
    """(d, episodes, layers) -> list of accuracies across seeds."""
    table = defaultdict(list)
    flat = defaultdict(list)
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        d, episodes = payload["d"], payload["episodes"]
        for key, stats in payload["conditions"].items():
            if not key.startswith("edge_marginals/"):
                continue
            if "/pernode/" in key:
                table[(d, episodes, stats["layers"])].append(stats["probe_accuracy"])
            elif key.endswith("/flat"):
                flat[(d, episodes)].append(stats["probe_accuracy"])
    return {"pernode": table, "flat": flat}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="results/probe_depth")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    directory = Path(args.dir)
    data = load(directory)
    table, flat = data["pernode"], data["flat"]
    if not table:
        print(f"no per-node probe results under {directory.resolve()}")
        return

    depths = sorted({k[2] for k in table})
    ds = sorted({k[0] for k in table})
    sizes = sorted({k[1] for k in table})

    print(f"Depth probe -- mean probe accuracy over seeds "
          f"(threshold for carrying depth forward: +{THRESHOLD})\n")
    header = "  d  episodes " + "".join(f"  L{k:<10}" for k in depths) + "  flat"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Per (d, size): does any deeper network clear the threshold over depth 1?
    beats = {d: [] for d in ds}
    rows = []
    for d in ds:
        for size in sizes:
            means = {}
            for k in depths:
                values = table.get((d, size, k), [])
                means[k] = float(np.mean(values)) if values else float("nan")
            flat_values = flat.get((d, size), [])
            flat_mean = float(np.mean(flat_values)) if flat_values else float("nan")

            cells = "".join(
                f"  {means[k]:.3f}({len(table.get((d, size, k), []))})  "
                for k in depths)
            print(f"  {d}  {size:>8} {cells}  {flat_mean:.3f}")

            base = means.get(1, float("nan"))
            best_deep = max((means[k] for k in depths if k > 1), default=float("nan"))
            lift = best_deep - base
            beats[d].append(bool(np.isfinite(lift) and lift > THRESHOLD))
            rows.append({"d": d, "episodes": size, "means": means,
                         "flat": flat_mean, "lift": lift})

    print("\nLift of the best deeper network over depth 1, per cell:")
    for row in rows:
        marker = "BEATS" if row["lift"] > THRESHOLD else "     "
        print(f"  d={row['d']} episodes={row['episodes']:>5}  "
              f"lift {row['lift']:+.3f}  {marker}")

    # The rule: depth must win at matched data size on BOTH d values.
    per_d = {d: any(beats[d]) for d in ds}
    fires = all(per_d.values()) and len(ds) > 1
    print("\n=== DECISION ===")
    for d in ds:
        cleared = sum(beats[d])
        print(f"  d={d}: depth clears +{THRESHOLD} at {cleared}/{len(beats[d])} "
              f"data sizes")

    if fires:
        best = max(depths, key=lambda k: np.nanmean(
            [r["means"][k] for r in rows if np.isfinite(r["means"][k])]))
        print(f"  RULE FIRES -> carry layers={best} into Phase 2 (run E3).")
    else:
        missing = [d for d in ds if not per_d[d]]
        print(f"  RULE DOES NOT FIRE -> keep layers=1; E3 is skipped and recorded as "
              f"skipped.")
        print(f"  Depth failed to clear the threshold at d={missing}.")
        print("  Finding: the probe's ceiling is NOT about multi-hop reachability. That "
              "redirects the next round away from architecture depth and towards the "
              "belief representation itself.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"threshold": THRESHOLD, "rows": rows, "rule_fires": fires,
             "per_d_beats": per_d}, indent=2, default=float), encoding="utf-8")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
