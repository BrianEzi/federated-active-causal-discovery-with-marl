"""Sync finished two-agent results to Weights & Biases.

Deliberately a POST-HOC sync from the result JSONs rather than live logging inside the
training loop. Two reasons: runs already in flight cannot be retrofitted, and a sync that
reads committed artefacts is reproducible -- anyone can re-run it against the same files and
get the same dashboard, which is not true of a live logger whose output depends on when it
happened to be running.

One W&B run per (arm, seed). Learning curves are replayed step by step from the saved
history so the dashboard shows the same curves as the report, and the summary carries the
evaluation for every arm so the baselines are comparable inside W&B rather than only here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

ARMS = ("learned", "random_clamp", "random_vary", "greedy", "pass")


def load(patterns: List[str]) -> List[tuple]:
    out = []
    for pattern in patterns:
        for path in sorted(pathlib.Path().glob(pattern)):
            try:
                out.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except Exception as exc:
                print(f"  skipping {path}: {exc}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="ma-two-agent")
    ap.add_argument("--patterns", nargs="+",
                    default=["results/ma_night/*_s*.json", "results/ma_train/*_s*.json"])
    ap.add_argument("--offline", action="store_true",
                    help="write runs locally without contacting the server")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    reports = load(args.patterns)
    if not reports:
        print("no result files found -- nothing to sync")
        return
    print(f"found {len(reports)} result files")
    if args.dry_run:
        for path, report in reports:
            print(f"  {path}  arm={report.get('arm')} seed={report.get('seed')} "
                  f"success={report.get('arms', {}).get('learned', {}).get('success')}")
        return

    import os
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    import wandb

    for path, report in reports:
        arm, seed = report.get("arm", "unknown"), report.get("seed", 0)
        run = wandb.init(project=args.project, name=f"{arm}_s{seed}",
                         group=arm, job_type="train_eval",
                         config={**report.get("config", {}), "arm": arm, "seed": seed},
                         reinit=True)
        for point in report.get("history", []):
            wandb.log({"train/solve_rate": point.get("solve_rate"),
                       "train/entropy": point.get("entropy"),
                       "train/update": point.get("update")},
                      step=int(point.get("update", 0)))

        summary = {
            "train_seconds": report.get("train_seconds"),
            "first_success_episode": report.get("first_success_episode"),
            "final_entropy": report.get("final_entropy"),
            # The canary. A collapsed seed must be visible in the dashboard as a flag, not
            # inferred from a suspiciously low success rate.
            "collapsed": report.get("collapsed"),
        }
        for name in ARMS:
            row = report.get("arms", {}).get(name)
            if not row:
                continue
            summary[f"eval/{name}/success"] = row.get("success")
            summary[f"eval/{name}/mean_steps"] = row.get("mean_steps")
            summary[f"eval/{name}/clamp_fraction"] = row.get("clamp_fraction")
            summary[f"eval/{name}/union_acyclic"] = row.get("union_acyclic")
            summary[f"eval/{name}/union_equivalent"] = row.get("union_equivalent")
        run.summary.update({k: v for k, v in summary.items() if v is not None})
        run.finish()
        print(f"  synced {arm} seed {seed}")

    print("done")


if __name__ == "__main__":
    main()
