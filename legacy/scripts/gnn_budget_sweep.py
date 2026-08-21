"""Does the LEARNED (per-node / GNN) agent behave like greedy as the budget tightens?

Single agent only. Two questions the baseline sweep raised but could not answer:

  Q1  Budget 2-3 is where greedy and random are most separated. Does the learned agent
      hold its advantage there, or is its gain an artefact of slack budgets where
      everything converges?

  Q2  At d=7 with scarce data, greedy PLATEAUS at 0.905 while random climbs past it to
      0.960 -- roughly 9% of episodes the myopic oracle never solves at any budget. That
      is the clearest headroom above greedy found so far. Does the learned agent capture
      any of it, or does it inherit greedy's blind spot?

Q2 is the interesting one. Q1 is the control that stops a Q2 result being explained by
"the learned agent is just better everywhere".

References (greedy, random, no-intervention) cost ~8.5s/episode at d=7 and dominate
everything else, so they are cached per configuration and shared across seeds. Without
this the sweep is references-bound rather than training-bound.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

# (d, n_obs, budget). Budgets chosen from results/budget/budget_sweep.json:
#   d=5: 2 and 3 straddle the peak-discrimination point, 5 is the new default, 8 is slack.
#   d=7: 3 is peak discrimination, 5 the default, 16 is where greedy's plateau is visible
#        and random has already overtaken it.
CONFIGS = [
    (5, 100, 2), (5, 100, 3), (5, 100, 5), (5, 100, 8),
    (7, 100, 3), (7, 100, 5), (7, 100, 16),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train_episodes", type=int, default=3000)
    ap.add_argument("--eval_episodes", type=int, default=150)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--outdir", default="results/budget/gnn")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    index = []

    for d, n_obs, budget in CONFIGS:
        tag = f"d{d}_nobs{n_obs}_b{budget}"
        cache = outdir / f"refs_{tag}.json"
        out = outdir / f"{tag}.json"
        cmd = [
            sys.executable, "-u", "scripts/run_experiment.py",
            "--d", str(d), "--n_obs", str(n_obs), "--budget", str(budget),
            "--arch", "pernode", "--layers", str(args.layers),
            "--observation", "edge_marginals",
            "--seeds", *[str(s) for s in args.seeds],
            "--train_episodes", str(args.train_episodes),
            "--eval_episodes", str(args.eval_episodes),
            "--ref_cache", str(cache), "--out", str(out), "--tag", tag,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        print(f"\n===== {tag}  rc={proc.returncode}  [{elapsed:.0f}s, "
              f"total {time.time() - started:.0f}s] =====", flush=True)
        # Keep the summary and canary blocks; the per-update training log is noise here.
        tail = proc.stdout.strip().splitlines()
        keep, seen = [], False
        for line in tail:
            if line.startswith(("=== SUMMARY", "=== CANARIES", "  seeds passing",
                                "  gap_closed", "  [WARN", "  [FAIL", "  OVERALL")):
                seen = True
            if seen:
                keep.append(line)
        print("\n".join(keep[-40:]), flush=True)
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-1200:], flush=True)
        index.append({"tag": tag, "d": d, "n_obs": n_obs, "budget": budget,
                      "seconds": elapsed, "returncode": proc.returncode,
                      "out": str(out)})
        (outdir / "index.json").write_text(json.dumps(index, indent=1))

    print(f"\nALL DONE in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
