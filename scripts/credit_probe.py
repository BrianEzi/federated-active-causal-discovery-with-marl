"""Is the FedAvg gap a CREDIT problem? Cross turn-aware credit with the training mode.

WHAT IS WRONG. Under round-robin only the ACTIVE agent's action is applied, but with
`turn_aware_credit=False` -- the default, and what the sweep uses -- every agent stores a
transition every round. Measured at four agents: 75% of stored rows are actions that were
DISCARDED, and the reward on them (+0.188, sd 0.387) is statistically indistinguishable
from the reward on rows that were applied (+0.197, sd 0.391). The observation carries 173
features and none of them say whose turn it is, so the policy cannot separate the two even
in principle.

WHY IT SHOULD HURT FEDAVG MORE. The signal-to-noise ratio per row is the same either way,
but pooled averages GRADIENTS over 4N rows in one batch, so the phantom noise falls as
sqrt(4N). FedAvg takes local nonlinear Adam steps on N rows each and then averages the
resulting WEIGHTS, which does not recover that variance reduction. So the same noise is
amplified.

THE CROSS. Four arms, because either factor alone is uninterpretable:

    pooled  credit=off     the sweep as configured
    pooled  credit=on      does fixing credit help even without federation?
    E4      credit=off     the FedAvg gap as measured
    E4      credit=on      does fixing credit CLOSE the gap?

If the gap closes, it was credit and not federation, and the sweep should carry
`--turn_aware_credit` regardless of what it does about FedAvg.

    .venv/bin/python scripts/credit_probe.py --cell k08s50n04b150 --seeds 3 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.sweep import build_cells, command                       # noqa: E402

# COPY the real environment rather than replacing it. `PATH: "/usr/bin:/bin"` assumed a
# POSIX host; on Windows it deletes SystemRoot, TEMP and everything else CreateProcess
# needs to load a DLL, and the child fails before it can even print an error. Overriding
# only the three variables this probe actually cares about is portable either way.
ENV = {**os.environ, "PYTHONPATH": ".", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def run(cell, seed, local_epochs, credit, out_dir, episodes, eval_episodes):
    label = f"{'E%d' % local_epochs if local_epochs else 'pooled'}_{'credit' if credit else 'nocredit'}"
    out = f"{out_dir}/{cell.name}_{label}_s{seed}.json"
    if pathlib.Path(out).exists():
        return label, seed, json.loads(pathlib.Path(out).read_text())
    argv = command(cell, seed, out_dir, episodes=episodes)
    # `command()` hardcodes ".venv/bin/python" -- correct when a POSIX shell resolves the
    # shebang for it (launch.sh's xargs+sh), wrong here: `subprocess.run` without a shell
    # calls CreateProcess/execve directly on argv[0], which cannot run a shim script or a
    # relative path missing a platform extension. `sys.executable` is the interpreter this
    # probe is ALREADY running under, which is what should be training the child anyway.
    argv[0] = sys.executable
    argv[argv.index("--out") + 1] = out
    argv[argv.index("--eval_episodes") + 1] = str(eval_episodes)
    if local_epochs:
        argv += ["--local_epochs", str(local_epochs)]
    if credit:
        argv += ["--turn_aware_credit"]
    done = subprocess.run(argv, capture_output=True, text=True, env=ENV)
    if done.returncode != 0:
        print(f"  {label} s{seed} FAILED\n{(done.stderr or done.stdout)[-600:]}")
        return label, seed, None
    return label, seed, json.loads(pathlib.Path(out).read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="k08s50n04b150")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", default="results/credit")
    args = ap.parse_args(argv)

    cell = next(c for c in build_cells() if c.name == args.cell)
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(cell, s, E, c) for E in (0, 4) for c in (False, True)
            for s in range(args.seeds)]
    print(f"{len(jobs)} runs at {cell.name} (k={cell.k}, n={cell.n}), {args.workers} workers")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda j: run(*j, args.out_dir, args.episodes, args.eval_episodes), jobs))

    rows = {}
    for label, seed, report in results:
        if report is None:
            continue
        learned = report["arms"]["learned"]
        rows.setdefault(label, []).append({
            "seed": seed, "success": learned["success"],
            "entropy": report["history"][-1]["entropy"],
            "mi": (report.get("checkpoints") or {}).get("best_mi_ratio"),
            "greedy": report["arms"].get("greedy_uncertainty", {}).get("success"),
            "ceiling": report["arms"].get("oracle_cover", {}).get("success")})

    print(f"\n{'arm':18s} {'success per seed':>26s} {'mean':>7s} {'entropy':>8s} {'best MI':>8s}")
    for label in ("pooled_nocredit", "pooled_credit", "E4_nocredit", "E4_credit"):
        if label not in rows:
            continue
        got = sorted(rows[label], key=lambda r: r["seed"])
        mean = lambda k: sum(r[k] for r in got if r[k] is not None) / max(
            len([r for r in got if r[k] is not None]), 1)
        per_seed = ", ".join("%.3f" % r["success"] for r in got)
        print(f"{label:18s} {per_seed:>26s} "
              f"{mean('success'):7.3f} {mean('entropy'):8.3f} {mean('mi'):8.3f}")

    out = pathlib.Path(args.out_dir) / f"{cell.name}_credit_probe.json"
    out.write_text(json.dumps({"cell": cell.as_dict(), "arms": rows}, indent=1))
    print(f"\nwrote {out}")
    print("READ IT AS: if turn-aware credit closes the E4 gap, the gap was CREDIT, not "
          "federation -- and the sweep needs --turn_aware_credit whatever it does about FedAvg.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
