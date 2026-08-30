"""Is the sweep's training actually federated? A cheap head-to-head.

THE PROBLEM THIS MEASURES. The sweep leaves `local_epochs=0`, which selects the POOLED
path, and pooling concatenates every site's raw trajectories into one buffer. That is data
pooling -- strictly more centralised than gradient sharing, and the thing a federated
setting forbids. A thesis whose federation lives in the structure learning can defend it,
but not silently, and not while describing the training as FedAvg.

WHAT THE ARMS ARE.

  pooled (E=0)   the current sweep. Raw trajectories merged, `cfg.epochs` passes over the
                 merged buffer. Nothing federated about it.
  E=1            FedAvg, one local epoch per site per round. Note this is roughly a QUARTER
                 of pooled's compute: pooled does cfg.epochs (4) passes over N rows, FedAvg
                 does n sites x E passes over N/n rows each.
  E=4            FedAvg at E = cfg.epochs, which is the step-matched comparison to pooled.
                 This is the arm that isolates FEDERATION from step count.

E=1 IS NOT THE EQUIVALENCE CHECK, and it would be convenient but wrong to read it as one.
Two differences survive: the step count above, and advantage normalisation, which pooled
computes over every site's experience together while FedAvg computes within a site --
because a site sharing its advantage statistics leaks exactly what FedAvg exists to keep
local. The clean equivalence check is the single-site case, pinned in tests/ma/test_fedavg.py.

WHY IT IS CHEAP. At k=8 a run is ~13 minutes. Three arms x two seeds is under an hour of
core time, and it answers whether the federated training story costs anything measurable
before 60 runs are committed to the pooled one.

    .venv/bin/python scripts/fedavg_compare.py --cell k08s50n04b150 --seeds 2 --workers 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.sweep import build_cells, command                       # noqa: E402

ENV = {"PYTHONPATH": ".", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "PATH": "/usr/bin:/bin"}


def run(cell, seed, local_epochs, out_dir, episodes, eval_episodes):
    label = f"E{local_epochs}" if local_epochs else "pooled"
    out = f"{out_dir}/{cell.name}_{label}_s{seed}.json"
    if pathlib.Path(out).exists():
        return label, seed, json.loads(pathlib.Path(out).read_text()), 0.0
    argv = command(cell, seed, out_dir, episodes=episodes)
    argv[argv.index("--out") + 1] = out
    argv[argv.index("--eval_episodes") + 1] = str(eval_episodes)
    if local_epochs:
        argv += ["--local_epochs", str(local_epochs)]
    started = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=ENV)
    if done.returncode != 0:
        print(f"  {label} s{seed} FAILED\n{(done.stderr or done.stdout)[-700:]}")
        return label, seed, None, 0.0
    return label, seed, json.loads(pathlib.Path(out).read_text()), time.perf_counter() - started


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="k08s50n04b150")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--local_epochs", default="0,1,4")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out_dir", default="results/fedavg")
    args = ap.parse_args(argv)

    cell = next(c for c in build_cells() if c.name == args.cell)
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    arms = [int(x) for x in args.local_epochs.split(",")]

    jobs = [(cell, seed, E) for E in arms for seed in range(args.seeds)]
    print(f"{len(jobs)} runs at {cell.name} (k={cell.k}, n={cell.n}, budget={cell.budget}), "
          f"{args.workers} workers")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda job: run(*job, args.out_dir, args.episodes, args.eval_episodes), jobs))

    rows = {}
    for label, seed, report, seconds in results:
        if report is None:
            continue
        learned = report["arms"]["learned"]
        rows.setdefault(label, []).append({
            "seed": seed, "success": learned["success"],
            "window_rate": report["history"][-1].get("window_rate"),
            "entropy": report["history"][-1]["entropy"],
            "mi": (report.get("checkpoints") or {}).get("best_mi_ratio"),
            "greedy": report["arms"].get("greedy_uncertainty", {}).get("success"),
            "ceiling": report["arms"].get("oracle_cover", {}).get("success"),
            "train_seconds": report["train_seconds"],
        })

    print(f"\n{'arm':8s} {'seeds':>5s} {'success':>18s} {'vs greedy':>10s} {'vs ceiling':>11s} "
          f"{'entropy':>8s} {'best MI':>8s} {'train':>8s}")
    for label in ("pooled", "E1", "E4"):
        if label not in rows:
            continue
        got = rows[label]
        mean = lambda key: sum(r[key] for r in got if r[key] is not None) / max(
            len([r for r in got if r[key] is not None]), 1)
        successes = ", ".join(f"{r['success']:.3f}" for r in got)
        print(f"{label:8s} {len(got):5d} {successes:>18s} "
              f"{mean('success') - mean('greedy'):+10.3f} "
              f"{mean('success') - mean('ceiling'):+11.3f} "
              f"{mean('entropy'):8.3f} {mean('mi'):8.3f} {mean('train_seconds')/60:7.1f}m")

    out = pathlib.Path(args.out_dir) / f"{cell.name}_comparison.json"
    out.write_text(json.dumps({"cell": cell.as_dict(), "episodes": args.episodes,
                               "arms": rows}, indent=1))
    print(f"\nwrote {out}")
    print("READ IT AS: does federated training cost success, and how much? The step-matched "
          "arm is E4; E1 is the cheap one, not the equivalent one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
