"""Pooled vs FedAvg vs FedAdam vs fully decentralised, all with the same care taken.

WHY THESE FOUR. The pooled path concatenates every site's raw trajectories, which is data
pooling and is not federated at all. Plain FedAvg was measured worse -- but a single FedAvg
update matches a single pooled update to 0.9971 cosine and 0.99x displacement, so the gap
is NOT in the update rule. The only thing FedAvg discards that pooling keeps is optimiser
state: local Adam moments are rebuilt from zero every round. FedAdam (Reddi et al., ICLR
2021) keeps the adaptivity on the SERVER, where it persists across rounds and is never
computed across a weight average it does not belong to. `gnn_solo` is the other end: one
network per agent, nothing shared at all.

EVERY ARM RUNS WITH --turn_aware_credit. Without it 75% of each agent's rows are actions
that were discarded, carrying reward produced by someone else's move, and FedAvg amplifies
that noise more than pooling does -- so a comparison without credit measures the phantom
rows rather than the training mode.

A FAIRNESS CAVEAT, stated rather than buried: FedAdam's two learning rates were picked by
matching the first update's displacement to pooled's, while pooled and plain FedAvg run at
the defaults they have always used. That favours FedAdam. It is the honest way round --
FedAdam has two extra knobs and would be indefensible untuned -- but the comparison is
"FedAdam tuned lightly" against "the others untuned", not a level field.

    .venv/bin/python scripts/training_modes.py --cell k08s50n04b150 --seeds 3 --workers 4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.sweep import build_cells, command                       # noqa: E402

ENV = {"PYTHONPATH": ".", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "PATH": "/usr/bin:/bin"}

ARMS = {
    "pooled":   [],
    "fedavg":   ["--local_epochs", "4"],
    "fedadam":  ["--local_epochs", "4", "--server_optimiser", "adam",
                 "--client_optimiser", "sgd", "--lr", "0.1", "--server_lr", "0.03"],
    "solo":     ["--policy_arch", "gnn_solo"],
}


def run(cell, seed, arm, out_dir, episodes, eval_episodes):
    out = f"{out_dir}/{cell.name}_{arm}_s{seed}.json"
    if pathlib.Path(out).exists():
        return arm, seed, json.loads(pathlib.Path(out).read_text())
    argv = command(cell, seed, out_dir, episodes=episodes)
    argv[argv.index("--out") + 1] = out
    argv[argv.index("--eval_episodes") + 1] = str(eval_episodes)
    argv += ["--turn_aware_credit"]
    extra = list(ARMS[arm])
    if "--policy_arch" in extra:                      # override rather than duplicate
        argv[argv.index("--policy_arch") + 1] = extra[extra.index("--policy_arch") + 1]
        extra = [x for i, x in enumerate(extra)
                 if i not in (extra.index("--policy_arch"), extra.index("--policy_arch") + 1)]
    argv += extra
    done = subprocess.run(argv, capture_output=True, text=True, env=ENV)
    if done.returncode != 0:
        print(f"  {arm} s{seed} FAILED\n{(done.stderr or done.stdout)[-700:]}")
        return arm, seed, None
    return arm, seed, json.loads(pathlib.Path(out).read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="k08s50n04b150")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--arms", default="pooled,fedavg,fedadam,solo")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", default="results/training_modes")
    args = ap.parse_args(argv)

    cell = next(c for c in build_cells() if c.name == args.cell)
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a in ARMS]
    jobs = [(cell, s, a) for a in arms for s in range(args.seeds)]
    print(f"{len(jobs)} runs at {cell.name} (k={cell.k}, n={cell.n}), {args.workers} workers")
    print("all arms carry --turn_aware_credit\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda j: run(*j, args.out_dir, args.episodes, args.eval_episodes), jobs))

    rows = {}
    for arm, seed, report in results:
        if report is None:
            continue
        learned = report["arms"]["learned"]
        rows.setdefault(arm, []).append({
            "seed": seed, "success": learned["success"],
            "entropy": report["history"][-1]["entropy"],
            "mi": (report.get("checkpoints") or {}).get("best_mi_ratio"),
            "greedy": report["arms"].get("greedy_uncertainty", {}).get("success"),
            "ceiling": report["arms"].get("oracle_cover", {}).get("success")})

    print(f"\n{'arm':10s} {'success per seed':>26s} {'mean':>7s} {'entropy':>8s} {'best MI':>8s}"
          f" {'vs greedy':>10s}")
    for arm in arms:
        if arm not in rows:
            continue
        got = sorted(rows[arm], key=lambda r: r["seed"])
        mean = lambda k: sum(r[k] for r in got if r[k] is not None) / max(
            len([r for r in got if r[k] is not None]), 1)
        per_seed = ", ".join("%.3f" % r["success"] for r in got)
        print(f"{arm:10s} {per_seed:>26s} {mean('success'):7.3f} {mean('entropy'):8.3f} "
              f"{mean('mi'):8.3f} {mean('success') - mean('greedy'):+10.3f}")

    out = pathlib.Path(args.out_dir) / f"{cell.name}_modes.json"
    out.write_text(json.dumps({"cell": cell.as_dict(), "arms": rows,
                               "arm_flags": ARMS}, indent=1))
    print(f"\nwrote {out}")
    print("MI is the reading that matters here: this cell is near-saturated on success, so "
          "entropy and MI separate the arms where success cannot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
