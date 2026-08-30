"""MEASURE what each sweep cell costs, instead of extrapolating a fit across it.

WHY. The schedule in docs/SESSION_STATE_2026_08_30.md rests on `t ~ 8.38 * k^1.81 *
n^0.56`, fitted on the window ladder -- five points that never varied sigma, never varied
beta, and were all at four agents. The sweep varies all four axes, and beta moves the
BUDGET, which moves the number of rounds in an episode directly. Extrapolating a fit past
the range it was fitted on is how a schedule quietly becomes a wish.

So: run every cell briefly, measure seconds per episode, and multiply. The short run is
also a SMOKE TEST -- a cell whose topology, budget or observation layout is malformed
fails here, in minutes, rather than four hours into an overnight launch.

WHAT IT DOES NOT MEASURE. Seconds per episode falls as a policy learns, because solved
episodes end early. Measuring at the start therefore OVER-estimates, which is the safe
direction for a schedule. Quoted as such.

    .venv/bin/python scripts/calibrate_sweep.py --probe 24 --workers 8

Writes a manifest with the per-cell estimate and a longest-job-first schedule.
"""
from __future__ import annotations

import argparse
import heapq
import json
import pathlib
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.sweep import build_cells, command                       # noqa: E402


def _run(cell, episodes, *, evidence, arch, seed, eval_episodes=1):
    """One short training run. `train_seconds` comes from the result file rather than a
    stopwatch around the process, so interpreter start-up does not inflate a per-episode
    figure that then gets multiplied by four thousand."""
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/probe.json"
        argv = command(cell, seed, tmp, evidence=evidence, arch=arch, episodes=episodes)
        argv[argv.index("--out") + 1] = out
        argv[argv.index("--eval_episodes") + 1] = str(eval_episodes)
        started = time.perf_counter()
        done = subprocess.run(argv, capture_output=True, text=True,
                              env={"PYTHONPATH": ".", "OMP_NUM_THREADS": "1",
                                   "MKL_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
        wall = time.perf_counter() - started
        if done.returncode != 0:
            return {"ok": False, "error": (done.stderr or done.stdout)[-600:]}
        report = json.loads(pathlib.Path(out).read_text())
        evaluation = sum(arm["seconds"] for arm in report["arms"].values())
        return {"ok": True, "train": float(report["train_seconds"]),
                "eval": evaluation, "arms": len(report["arms"]),
                "overhead": wall - float(report["train_seconds"]) - evaluation,
                "entropy": (report["history"][-1]["entropy"]
                            if report.get("history") else None)}


def probe(cell, *, probe_episodes: int, evidence: str, arch: str, seed: int = 0):
    """Seconds per episode, measured as a SLOPE between two run lengths.

    A single short run does not give it. Training carries a fixed start-up cost -- torch
    warming up on the first backward pass, mostly -- that a per-episode figure divides by
    the episode count and so inflates. Measured at the baseline cell: 0.778 s/ep over 16
    episodes, 0.555 over 48, 0.539 over 96, against a true asymptote of 0.527. The short
    probe over-estimated by 48%.

    Two points at `probe` and `3 * probe` episodes cancel the constant exactly:
    (t2 - t1) / (e2 - e1). Anchored at 48/96 this recovered 0.5235 against the 0.527
    measured over 120 episodes -- within 1%.
    """
    small = _run(cell, probe_episodes, evidence=evidence, arch=arch, seed=seed)
    if not small["ok"]:
        return small
    large = _run(cell, 3 * probe_episodes, evidence=evidence, arch=arch, seed=seed)
    if not large["ok"]:
        return large
    slope = (large["train"] - small["train"]) / (2 * probe_episodes)
    return {"ok": True, "per_episode": slope,
            "warmup": small["train"] - slope * probe_episodes,
            "per_eval_episode": large["eval"] / max(large["arms"], 1),
            "arms": large["arms"], "overhead": max(large["overhead"], 0.0),
            "entropy": large["entropy"]}


def schedule(durations, workers: int):
    """Longest-processing-time-first: the standard 4/3-optimal greedy list schedule.

    Returned as the makespan, because that -- not the sum -- is what "when will this be
    finished" means when the jobs run on `workers` cores at once.
    """
    heap = [0.0] * workers
    heapq.heapify(heap)
    for duration in sorted(durations, reverse=True):
        heapq.heappush(heap, heapq.heappop(heap) + duration)
    return max(heap)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=int, default=48,
                    help="episodes in the SHORT probe; the long one is 3x this")
    ap.add_argument("--episodes", type=int, default=4000, help="episodes in the real run")
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--evidence", default="oracle", choices=["oracle", "sampled"])
    ap.add_argument("--arch", default="gnn_portable")
    ap.add_argument("--only", default=None, help="comma-separated cell names")
    ap.add_argument("--manifest", default="results/sweep/calibration.json")
    args = ap.parse_args(argv)

    cells = build_cells()
    if args.only:
        wanted = set(args.only.split(","))
        cells = [c for c in cells if c.name in wanted]

    print(f"{'cell':16s} {'k':>3s} {'n':>3s} {'budget':>7s} {'s/ep':>7s} "
          f"{'train':>8s} {'eval':>7s} {'per run':>9s}")
    rows, failed = [], []
    for cell in cells:
        result = probe(cell, probe_episodes=args.probe, evidence=args.evidence,
                       arch=args.arch)
        if not result["ok"]:
            failed.append((cell.name, result["error"]))
            print(f"{cell.name:16s} {cell.k:3d} {cell.n:3d} {cell.budget:7d}    FAILED")
            continue
        train = result["per_episode"] * args.episodes
        evaluation = result["per_eval_episode"] * args.eval_episodes * result["arms"]
        total = train + evaluation + max(result["overhead"], 0.0)
        rows.append({**cell.as_dict(), "seconds_per_episode": result["per_episode"],
                     "train_seconds": train, "eval_seconds": evaluation,
                     "run_seconds": total, "probe_entropy": result["entropy"]})
        print(f"{cell.name:16s} {cell.k:3d} {cell.n:3d} {cell.budget:7d} "
              f"{result['per_episode']:7.3f} {train/60:7.1f}m {evaluation/60:6.1f}m "
              f"{total/60:8.1f}m")

    if failed:
        print("\nCELLS THAT DID NOT RUN -- fix before launching anything:")
        for name, error in failed:
            print(f"  {name}: {error.strip().splitlines()[-1] if error.strip() else '?'}")

    durations = [row["run_seconds"] for row in rows for _ in range(args.seeds)]
    core_hours = sum(durations) / 3600.0
    makespan = schedule(durations, args.workers) / 3600.0
    print(f"\n{len(rows)} cells x {args.seeds} seeds = {len(durations)} runs, "
          f"{args.evidence} evidence")
    print(f"  core-hours      {core_hours:6.1f}")
    print(f"  wall on {args.workers:2d}     {makespan:6.1f} h   (longest-job-first; the "
          f"longest single run is {max(durations)/3600:.1f} h)")
    print("  NOTE: measured on an UNTRAINED policy, so episodes run to the budget. A "
          "policy that learns ends episodes early, so this over-estimates.")

    path = pathlib.Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"evidence": args.evidence, "arch": args.arch, "probe_episodes": args.probe,
         "episodes": args.episodes, "eval_episodes": args.eval_episodes,
         "seeds": args.seeds, "workers": args.workers, "core_hours": core_hours,
         "wall_hours": makespan, "failed": [name for name, _ in failed],
         "cells": rows}, indent=1))
    print(f"\nwrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
