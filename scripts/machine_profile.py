"""One comparable number per machine, so work is allocated on measurement rather than belief.

WHY. Three machines are in play -- this laptop, a second laptop, and Myriad -- and the
obvious ranking (fastest CPU first) is wrong, because what matters is TIME TO RESULT:

    time_to_result  =  queue_wait  +  runtime / effective_parallelism

A cluster with a two-hour queue is the worst place for a twenty-minute job and the best place
for sixty three-hour jobs. A laptop with no queue is the opposite. Neither fact is visible in
a CPU benchmark, and both change which machine should get which tier.

It also matters that effective parallelism is MEASURED. On this laptop it plateaus at ~2.8x
and EIGHT workers is worse than six -- so dividing core-hours by the core count, which is what
every estimate did until it was checked, overstates throughput by roughly 3x.

    .venv/bin/python scripts/machine_profile.py --label "brian-laptop" --workers 1,2,4,6

Emits a JSON profile. Collect one per machine, then `--compare` prints the allocation table.
For the cluster, pass `--queue_wait_minutes` from a real `qstat` observation: it is the one
term that cannot be measured from inside a job.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.sweep import build_cells, command                       # noqa: E402

# Extends the real environment rather than replacing it -- a bare {"PATH": "/usr/bin:/bin"}
# loses Windows' PATH entirely (no python.exe, no DLL search path), the same portability bug
# fixed in scripts/credit_probe.py on 31 Aug.
ENV = {**os.environ, "PYTHONPATH": ".", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
REFERENCE = "k08s50n04b150"          # small, fast, and present on every machine's grid


def _run_batch(cell, count, episodes):
    """Launch `count` copies of the reference run at once; return the wall time for all."""
    tmp = tempfile.mkdtemp()
    procs = []
    started = time.perf_counter()
    for index in range(count):
        argv = command(cell, index, tmp, episodes=episodes)
        argv[0] = sys.executable         # ".venv/bin/python" is a POSIX shim, not directly
                                          # executable by Windows' subprocess.Popen
        argv[argv.index("--out") + 1] = f"{tmp}/w{index}.json"
        argv[argv.index("--eval_episodes") + 1] = "1"
        procs.append(subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, env=ENV))
    codes = [p.wait() for p in procs]
    return time.perf_counter() - started, codes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default=platform.node())
    ap.add_argument("--workers", default="1,2,4,6")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--queue_wait_minutes", type=float, default=0.0,
                    help="typical wait between submitting and starting. Zero for a laptop; "
                         "for a cluster take it from a real qstat observation.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="*", default=None,
                    help="profile JSONs to compare instead of measuring")
    args = ap.parse_args(argv)

    if args.compare:
        profiles = [json.loads(pathlib.Path(p).read_text()) for p in args.compare]
        print(f"{'machine':22s} {'solo s':>8s} {'best W':>7s} {'speedup':>8s} "
              f"{'throughput':>11s} {'queue min':>10s}")
        for p in sorted(profiles, key=lambda x: -x["throughput"]):
            print(f"{p['label']:22s} {p['solo_seconds']:8.1f} {p['best_workers']:7d} "
                  f"{p['best_speedup']:8.2f} {p['throughput']:11.3f} "
                  f"{p['queue_wait_minutes']:10.1f}")
        print("\nthroughput = best_speedup / solo_seconds  (runs per second at best width)")
        print("ALLOCATE ON TIME TO RESULT, not throughput alone:")
        print("  short + urgent   -> highest throughput with ZERO queue wait")
        print("  long + parallel  -> highest throughput even with a queue; the wait amortises")
        print("  low priority     -> whatever is left, since its latency does not matter")
        return 0

    cell = next(c for c in build_cells() if c.name == REFERENCE)
    widths = [int(x) for x in args.workers.split(",")]
    solo, codes = _run_batch(cell, 1, args.episodes)
    if any(codes):
        print(f"FAILED: the reference run exited {codes}")
        return 1
    print(f"{args.label}: reference cell {REFERENCE}, {args.episodes} episodes")
    print(f"{'workers':>8s} {'wall s':>9s} {'speedup':>8s} {'efficiency':>11s}")
    print(f"{1:8d} {solo:9.1f} {1.0:8.2f} {'100%':>11s}")
    best_w, best_s = 1, 1.0
    rows = [{"workers": 1, "seconds": solo, "speedup": 1.0}]
    for w in widths:
        if w == 1:
            continue
        wall, codes = _run_batch(cell, w, args.episodes)
        if any(codes):
            print(f"{w:8d}    FAILED")
            continue
        speedup = w * solo / wall
        rows.append({"workers": w, "seconds": wall, "speedup": speedup})
        print(f"{w:8d} {wall:9.1f} {speedup:8.2f} {speedup/w:10.0%}")
        if speedup > best_s:
            best_w, best_s = w, speedup

    profile = {"label": args.label, "platform": platform.platform(),
               "solo_seconds": solo, "best_workers": best_w, "best_speedup": best_s,
               "throughput": best_s / solo, "queue_wait_minutes": args.queue_wait_minutes,
               "episodes": args.episodes, "reference_cell": REFERENCE, "rows": rows}
    out = pathlib.Path(args.out or f"results/machines/{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1))
    print(f"\nbest: {best_w} workers at {best_s:.2f}x -> throughput {best_s/solo:.4f} runs/s")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
