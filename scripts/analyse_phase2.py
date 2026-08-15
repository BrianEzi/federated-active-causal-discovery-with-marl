"""E1 vs E2: which levers matter for the task, and which only mattered for a broken net.

The overnight sweep measured 13 levers around a flat network that could not express the
task whatever the lever was set to. So a lever that "mattered" there may have been
compensating for the architecture rather than acting on the problem. Running both
architectures over identical configurations makes that separable, and this script does the
separation mechanically.

For each lever value, the effect is its median gap closed MINUS its own architecture's
baseline. Comparing deltas rather than raw numbers is the point: per-node and flat sit at
completely different absolute levels (+1.23 against -1.86 overnight), so raw values would
only re-measure the architecture gap that is already known.

Classification, using a threshold of 0.5 gap-closed -- the same figure G4 uses for an
unstable seed spread, on the reasoning that an effect smaller than the noise a
configuration shows across seeds is not an effect:

  task       moves under BOTH architectures -- a real property of the problem
  artefact   moves under FLAT only -- was compensating for the broken network, and the
             overnight conclusion about it does not carry over
  unlocked   moves under PER-NODE only -- interacts with the fix; only reachable once the
             network can express the mapping at all
  dead       moves under neither

    python -m scripts.analyse_phase2 --results results/phase2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.analyse_sweep import load_rows
from scripts.sweep_phase2 import ARCHES, build_matrix

# An effect below this is not distinguishable from the spread a configuration already
# shows across its own seeds. Stated here rather than chosen per lever after the fact.
EFFECT_THRESHOLD = 0.5


def tag_to_arm() -> Dict[str, str]:
    """Recovered from the matrix definition, not parsed out of the tag string.

    The old `_arm_of` splits a tag on underscores, which now yields "pernode" for every
    configuration -- the architecture prefix sits where the lever name used to. Reading the
    matrix directly cannot drift from what was actually run.
    """
    return {c["tag"]: c["arm"] for c in build_matrix()}


def load_canaries(results_dir: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        tag = payload.get("tag") or path.stem
        out[tag] = payload.get("canaries", [])
    return out


def summarise_by_tag(rows: List[Dict], arms: Dict[str, str]) -> Dict[str, dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["tag"]].append(row)

    out = {}
    for tag, group in grouped.items():
        gaps = np.array([r["gap_closed"] for r in group], dtype=float)
        finite = gaps[np.isfinite(gaps)]
        if not len(finite):
            continue
        out[tag] = {
            "arch": group[0]["arch"],
            "arm": arms.get(tag, "?"),
            "n_seeds": len(group),
            # Median for the effect estimate, min reported alongside: a configuration is
            # only as good as its worst seed, and this project has already mistaken one
            # lucky run for a working method.
            "median_gap": float(np.median(finite)),
            "min_gap": float(finite.min()),
            "max_gap": float(finite.max()),
            "spread": float(finite.max() - finite.min()),
            "mean_entropy": float(np.nanmean([r["final_entropy"] for r in group])),
        }
    return out


def classify(delta_pernode: float, delta_flat: float) -> str:
    moves_pernode = abs(delta_pernode) > EFFECT_THRESHOLD
    moves_flat = abs(delta_flat) > EFFECT_THRESHOLD
    if moves_pernode and moves_flat:
        return "task"
    if moves_flat:
        return "artefact"
    if moves_pernode:
        return "unlocked"
    return "dead"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/phase2")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    directory = Path(args.results)
    rows = load_rows(str(directory))
    if not rows:
        print(f"no results under {directory.resolve()}")
        return

    arms = tag_to_arm()
    summary = summarise_by_tag(rows, arms)
    canaries = load_canaries(directory)

    baselines = {}
    for arch in ARCHES:
        entry = summary.get(f"{arch}_baseline")
        baselines[arch] = entry["median_gap"] if entry else float("nan")

    print(f"=== BASELINES (median gap closed over seeds) ===")
    for arch in ARCHES:
        entry = summary.get(f"{arch}_baseline")
        if entry:
            print(f"  {arch:<8} {entry['median_gap']:+.3f}  "
                  f"(seeds {entry['min_gap']:+.3f} to {entry['max_gap']:+.3f}, "
                  f"n={entry['n_seeds']})")
        else:
            print(f"  {arch:<8} MISSING -- every delta below is undefined")

    # Pair configurations by the part of the tag after the architecture prefix.
    paired = defaultdict(dict)
    for tag, entry in summary.items():
        arch = entry["arch"]
        if not tag.startswith(f"{arch}_"):
            continue
        paired[tag[len(arch) + 1:]][arch] = entry

    print(f"\n=== LEVER EFFECTS (delta from the SAME architecture's baseline; "
          f"|effect| > {EFFECT_THRESHOLD} counts) ===")
    header = (f"  {'configuration':<34} {'pernode':>9} {'flat':>9}   verdict")
    print(header)
    print("  " + "-" * (len(header) - 2))

    verdicts = defaultdict(list)
    records = []
    for key in sorted(paired):
        if key == "baseline":
            continue
        pair = paired[key]
        if not all(arch in pair for arch in ARCHES):
            missing = [a for a in ARCHES if a not in pair]
            print(f"  {key:<34} {'':>9} {'':>9}   INCOMPLETE (missing {missing})")
            continue
        deltas = {arch: pair[arch]["median_gap"] - baselines[arch] for arch in ARCHES}
        verdict = classify(deltas["pernode"], deltas["flat"])
        verdicts[verdict].append(key)
        records.append({"configuration": key, "arm": pair["pernode"]["arm"],
                        "delta_pernode": deltas["pernode"],
                        "delta_flat": deltas["flat"], "verdict": verdict,
                        "pernode_median": pair["pernode"]["median_gap"],
                        "flat_median": pair["flat"]["median_gap"],
                        "pernode_spread": pair["pernode"]["spread"]})
        print(f"  {key:<34} {deltas['pernode']:>+9.3f} {deltas['flat']:>+9.3f}   {verdict}")

    print("\n=== SUMMARY ===")
    for verdict in ("task", "artefact", "unlocked", "dead"):
        names = verdicts.get(verdict, [])
        print(f"  {verdict:<9} {len(names):>2}  {', '.join(names) if names else '-'}")
    if verdicts.get("artefact"):
        print("\n  'artefact' levers moved the number under the flat network only. The "
              "overnight conclusions about them describe compensation for an architecture "
              "that could not express the task, and do not carry over.")

    # Canaries are the reason a number can be trusted, so they are reported with the
    # numbers rather than in a separate pass nobody runs.
    fired = [(tag, c) for tag, recs in canaries.items() for c in recs if not c["ok"]]
    print(f"\n=== CANARIES: {len(fired)} fired across "
          f"{len(canaries)} configurations ===")
    by_name = defaultdict(list)
    for tag, record in fired:
        by_name[record["name"]].append(tag)
    for name, tags in sorted(by_name.items()):
        expected = [t for t in tags if "NEGCONTROL" in t]
        print(f"  {name:<26} {len(tags):>3}"
              + (f"   ({len(expected)} on the negative control, as designed)"
                 if expected else ""))
    if not fired:
        print("  none -- which for the NEGCONTROL arms would itself be a bug, since G5 "
              "is supposed to fire there.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"threshold": EFFECT_THRESHOLD, "baselines": baselines,
             "records": records,
             "verdict_counts": {k: len(v) for k, v in verdicts.items()}},
            indent=2, default=float), encoding="utf-8")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
