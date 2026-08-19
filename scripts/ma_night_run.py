"""Local overnight driver: train and evaluate the with-regime-bit arm, seed by seed.

Insurance against cluster queue time, not a replacement for it. Myriad job 176251 runs the
full 40-task array (20 no-bit + 20 with-bit); this runs a smaller with-bit set locally so
there is definitely something to report by morning.

Writes each seed's result as soon as it finishes, so a partial night still yields a usable
set rather than nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from scripts.ma_train2 import main as train_main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--train_episodes", type=int, default=3000)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--outdir", default="results/ma_night")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    index = []

    for seed in args.seeds:
        out = outdir / f"withbit_s{seed}.json"
        if out.exists():
            print(f"== seed {seed} already done, skipping ==", flush=True)
            continue
        print(f"\n===== with-bit seed {seed} "
              f"[elapsed {time.time() - started:.0f}s] =====", flush=True)
        t0 = time.time()
        report = train_main([
            "--seed", str(seed), "--arm", "withbit", "--disclose_regime",
            "--n_obs", "1000", "--n_int", "100", "--budget", str(args.budget),
            "--train_episodes", str(args.train_episodes),
            "--eval_episodes", str(args.eval_episodes),
            "--out", str(out),
        ])
        index.append({"seed": seed, "seconds": time.time() - t0,
                      "success": report["arms"]["learned"]["success"],
                      "collapsed": report["collapsed"]})
        (outdir / "index.json").write_text(json.dumps(index, indent=1))
        print(f"== seed {seed} done in {time.time() - t0:.0f}s, "
              f"success {report['arms']['learned']['success']:.3f}, "
              f"collapsed={report['collapsed']} ==", flush=True)

    print(f"\nALL DONE in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
