"""Run a queue of two-agent seeds with a concurrency cap.

Three seeds is a sanity check; sizing an effect needs many more, and the local machine has
cores to spare between cluster submissions. A cap rather than a free-for-all because these
runs are CPU-bound and oversubscription makes every one of them slower without finishing any
sooner.

Each seed is a separate process writing its own JSON, so a crash costs one seed rather than
the batch, and the report generator picks up whatever has landed.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="withbit_fixed")
    ap.add_argument("--seeds", default="3-9")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--disclose_regime", action="store_true")
    ap.add_argument("--step_cost", type=float, default=0.05)
    ap.add_argument("--train_episodes", type=int, default=2000)
    ap.add_argument("--eval_episodes", type=int, default=150)
    # ROUNDS for the whole system, not interventions per agent -- semantics changed on
    # 2026-08-21, see docs/TURN_BUDGET_SPEC.md section 2. Was hardcoded to 8, which under
    # the new meaning is 4 turns each and thin enough to risk a null result.
    ap.add_argument("--budget", type=int, default=10)
    # Protocol and action space are passed through so that an arm name and its settings
    # travel together. Defaults reproduce the pre-2026-08-20 batches exactly.
    ap.add_argument("--turn_order", default="simultaneous",
                    choices=["simultaneous", "round_robin", "random"])
    ap.add_argument("--clamp_only", action="store_true")
    args = ap.parse_args(argv)

    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    pathlib.Path("results/ma_fixed").mkdir(parents=True, exist_ok=True)
    pathlib.Path("results/batch_logs").mkdir(parents=True, exist_ok=True)

    pending, running = list(seeds), []
    started = time.time()
    while pending or running:
        while pending and len(running) < args.jobs:
            seed = pending.pop(0)
            out = "results/ma_fixed/%s_s%d.json" % (args.arm, seed)
            if pathlib.Path(out).exists():
                print("seed %d already done, skipping" % seed, flush=True)
                continue
            cmd = [sys.executable, "-u", "-m", "scripts.ma_train",
                   "--seed", str(seed), "--arm", args.arm,
                   "--n_obs", "1000", "--n_int", "100",
                   "--budget", str(args.budget),
                   "--train_episodes", str(args.train_episodes),
                   "--eval_episodes", str(args.eval_episodes),
                   "--step_cost", str(args.step_cost),
                   "--turn_order", args.turn_order, "--out", out]
            if args.disclose_regime:
                cmd.append("--disclose_regime")
            if args.clamp_only:
                cmd.append("--clamp_only")
            log = open("results/batch_logs/%s_s%d.log" % (args.arm, seed), "w")
            running.append((seed, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT),
                            log, time.time()))
            print("started seed %d  (%d running, %d queued)"
                  % (seed, len(running), len(pending)), flush=True)
        time.sleep(5)
        for entry in list(running):
            seed, proc, log, t0 = entry
            if proc.poll() is not None:
                running.remove(entry)
                log.close()
                print("seed %d finished rc=%d in %.0fs  (%d running, %d queued)"
                      % (seed, proc.returncode, time.time() - t0, len(running),
                         len(pending)), flush=True)
    print("batch done in %.0fs" % (time.time() - started), flush=True)


if __name__ == "__main__":
    main()
