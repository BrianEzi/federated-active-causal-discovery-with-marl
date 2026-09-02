"""Measure every complete 12,000-episode cell from the SELECTED checkpoint.

WHY THIS EXISTS SEPARATELY FROM THE TRAINING RUN. Each result file records
`global_hard_shd` from its own evaluation pass, which scores the policy at its last update.
That is not what Chapter 4 quotes, and on a long run the two differ by up to a factor of 300
on the same seed (`FINDINGS_CHECKPOINT_2026_09_01.md`, and the K=5 case in
`FINDINGS_AGENT_COUNT_2026_09_02.md`). Promoting the re-run to the primary tables therefore
needs a fresh measurement, not a read of a field.

Resumable and idempotent: a cell is measured once all three seeds exist and skipped once its
output is on disk, so this can be run on every tick while the sweep fills in.

    python scripts/measure_sweep12k.py            # measure what is ready
    python scripts/measure_sweep12k.py --report   # print what has been measured
"""
from __future__ import annotations
import argparse, collections, glob, json, pathlib, re, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results/sweep12k"
OUT = SRC / "shd"
CELL = re.compile(r"(k\d+s\d+n\d+b\d+)_s(\d)\.json$")
# Cells already measured this way during the undertraining work; do not repeat them.
PREMEASURED = {"k12s50n05b150": "results/longcheck/shd_n05_12k.json",
               "k12s50n08b150": "results/longcheck/shd_n08_12k.json",
               "k12s50n10b150": "results/longcheck/shd_n10_12k.json",
               "k12s75n04b150": "results/longcheck/shd_s75_12k.json"}


def complete_cells():
    seen = collections.defaultdict(set)
    for p in SRC.glob("k*_s*.json"):
        m = CELL.search(p.name)
        if m:
            seen[m.group(1)].add(int(m.group(2)))
    return sorted(c for c, s in seen.items() if s >= {0, 1, 2})


def measured_path(cell: str) -> pathlib.Path:
    if cell in PREMEASURED:
        return ROOT / PREMEASURED[cell]
    return OUT / f"{cell}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    cells = complete_cells()
    if args.report:
        print(f"{'cell':18s} {'learned':>9} {'myopic':>9} {'ratio':>7} {'favour':>7} {'sig':>4}")
        for cell in cells:
            path = measured_path(cell)
            if not path.exists():
                print(f"{cell:18s} {'not measured yet':>40}")
                continue
            d = json.loads(path.read_text())
            L = np.mean([e["means"]["learned"]["hard"] for e in d])
            G = np.mean([e["means"]["greedy"]["hard"] for e in d])
            fav = sum(1 for e in d if e["paired"]["learned-greedy"]["delta"] < 0)
            sig = sum(1 for e in d if e["paired"]["learned-greedy"]["significant"])
            print(f"{cell:18s} {L:9.5f} {G:9.5f} {L/G if G else float('nan'):7.2f} "
                  f"{fav:5d}/3 {sig:2d}/3")
        return 0

    todo = [c for c in cells if not measured_path(c).exists()]
    print(f"{len(cells)} complete cells, {len(todo)} to measure: {todo}")
    running = []
    for cell in todo:
        cmd = [".venv/bin/python", "-u", "scripts/global_shd_paired.py",
               *[f"results/sweep12k/{cell}_s{s}.json" for s in (0, 1, 2)],
               "--episodes", "200", "--sample", "--checkpoint", "best",
               "--out", f"results/sweep12k/shd/{cell}.json"]
        log = open(OUT / f"{cell}.log", "w")
        running.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        while len([p for p in running if p.poll() is None]) >= args.workers:
            running[0].wait()
            running = [p for p in running if p.poll() is None]
    for p in running:
        p.wait()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
