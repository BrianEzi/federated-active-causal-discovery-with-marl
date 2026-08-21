"""Upload offline WandB runs. Run this on the LOGIN node, never in a job.

Compute nodes on Myriad have no outbound internet, so runs are written offline and sit as
directories under `wandb/` until something with a network connection pushes them. That is
this script.

    python -m legacy.scripts.sync_wandb                # sync everything not yet synced
    python -m legacy.scripts.sync_wandb --dry_run      # list what would be synced
    python -m legacy.scripts.sync_wandb --dir wandb    # non-default location

Deliberately not part of `run_experiment`: a job that tried to sync would hang on the
network call rather than fail, spending its remaining walltime doing nothing. Keeping the
upload in a separate, manually-run step is what makes that impossible.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# `wandb sync` writes this marker inside a run directory once it has been uploaded, which
# is what makes re-running the script safe and incremental.
SYNCED_MARKER = ".synced"


def find_runs(root: Path) -> list:
    """Offline run directories, oldest first."""
    if not root.exists():
        return []
    runs = [p for p in root.iterdir()
            if p.is_dir() and p.name.startswith(("offline-run-", "run-"))]
    return sorted(runs, key=lambda p: p.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="wandb",
                        help="directory holding offline runs (default: ./wandb)")
    parser.add_argument("--dry_run", action="store_true",
                        help="list what would be synced and exit")
    parser.add_argument("--include_synced", action="store_true",
                        help="re-sync runs already marked as uploaded")
    args = parser.parse_args()

    root = Path(args.dir)
    runs = find_runs(root)
    if not runs:
        print(f"no offline runs under {root.resolve()}")
        return

    pending = [r for r in runs
               if args.include_synced or not (r / SYNCED_MARKER).exists()]
    print(f"{len(runs)} run(s) under {root.resolve()}, {len(pending)} not yet synced")

    if args.dry_run:
        for run in pending:
            print(f"  would sync {run.name}")
        return
    if not pending:
        print("nothing to do")
        return

    failed = []
    for i, run in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] syncing {run.name}", flush=True)
        result = subprocess.run([sys.executable, "-m", "wandb", "sync", str(run)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            # One bad run must not abort the batch -- the whole point is to recover as
            # many as possible from a sweep that may have had a few malformed writers.
            failed.append(run.name)
            print(f"  FAILED: {result.stderr.strip().splitlines()[-1:] or ['(no output)']}")

    print(f"\nsynced {len(pending) - len(failed)}/{len(pending)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
        print("These runs are still on disk; the JSON result files are unaffected.")


if __name__ == "__main__":
    main()
