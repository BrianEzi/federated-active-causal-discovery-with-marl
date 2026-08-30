"""THE LAUNCH GATES -- the checks that are about the RUNS, not about the metrics.

`scripts/preflight_metrics.py` asks whether every metric is live in what a run writes out.
This asks the three questions that decide whether a 60-run sweep is worth starting:

  feasibility  is every cell's BUDGET actually sufficient? beta is defined as a multiple of
               the required cover, so beta=1.0 should be exactly enough for the optimal arm
               and beta=1.5 comfortable. If the ceiling arm cannot identify inside the
               budget, beta is mis-normalised and every beta result means nothing. This is
               the one gate that is about the SCIENCE rather than the plumbing.

  determinism  does the same command twice give the same file? Three defects on this
               project were global-state leaks -- a seeded torch stream that made every
               confidence interval replay one fixed path, most recently. They are invisible
               until two runs that should agree do not.

  throughput   what is the REAL parallel speedup on this machine? Every wall-clock estimate
               divides core-hours by a worker count, which assumes the cores are equal.
               They are not: this is a 6 performance + 4 efficiency part, and a schedule
               built on a linear assumption will be wrong in the optimistic direction.

    .venv/bin/python scripts/preflight_runs.py feasibility
    .venv/bin/python scripts/preflight_runs.py determinism
    .venv/bin/python scripts/preflight_runs.py throughput --workers 8

Each exits non-zero on failure, so a launch script can gate on it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ma.baselines import make_baselines                              # noqa: E402
from ma.env import MAConfig, TwoAgentEnv                             # noqa: E402
from ma.evaluate import run_arm                                      # noqa: E402
from ma.topology import federated_topology                           # noqa: E402
from scripts.sweep import (build_cells, command,                     # noqa: E402
                           required_cover_fraction)

ENV = {"PYTHONPATH": ".", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "PATH": "/usr/bin:/bin"}


def _env_for(cell, evidence="oracle"):
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(cell.n, cell.private, cell.shared),
        n_obs=60, n_int=20, budget=cell.budget, turn_order="round_robin",
        belief_backend="factored", action_modes=("vary",), claim_bar=1.0,
        reward_criterion="claims", policy_arch="gnn_portable", graph_model="sf", sf_m=2,
        episode_mix="confounded", vs_evidence=evidence, per_agent_reward=True))


def feasibility(args) -> int:
    """Can the CEILING arm identify inside each cell's budget?

    Read the two columns together. `success` alone conflates two different failures, and
    only one of them is a problem with the design:

      rounds/budget at 1.00  the arm ran out of ROUNDS. beta is too small for this cell, or
                             `required_cover_fraction` is extrapolating badly -- a real
                             defect in the normalisation.
      rounds/budget well below, success still short
                             the arm finished early and still missed. That is DUPLICATION
                             on the shared surface, which OracleCoverAgent does not resolve
                             by construction (it computes its own window's forced set and
                             no more). Expected, and it rises with sigma and n. Not a defect
                             -- it is the headroom a coordinating policy should claim.

    THE UNITS, because getting them wrong made this gate read 0.12 on its first run and
    look like a catastrophically over-generous budget. `mean_steps` is per-agent
    INTERVENTIONS (max over agents); the budget is a pool of ROUNDS shared by all n agents.
    The two are off by a factor of n. `mean_rounds` is the episode length and is the
    quantity the budget bounds, so that is what is compared here. `cover/agent` beside it
    is what the closed form predicted each agent would need, which is the number beta is a
    multiple of -- if the ceiling consistently beats it, the normalisation is loose.
    """
    cells = [c for c in build_cells() if not args.only or c.name in args.only.split(",")]
    print(f"{'cell':16s} {'axis':10s} {'budget':>7s} {'ceiling':>8s} {'rounds':>7s} "
          f"{'rnd/bud':>8s} {'ivn/agent':>10s} {'predicted':>10s} {'greedy':>7s}   verdict")
    starved = []
    for cell in cells:
        env = _env_for(cell)
        rows = {}
        for label in ("oracle_cover", "greedy_uncertainty"):
            policies = {a: make_baselines(env, a, 0)[label] for a in env.topology.agents}
            rows[label] = run_arm(env, policies, args.episodes, seed=0)
        ceiling, greedy = rows["oracle_cover"], rows["greedy_uncertainty"]
        ratio = ceiling["mean_rounds"] / cell.budget
        predicted = required_cover_fraction(cell.k) * cell.k
        # Budget-starved means the arm was still working when the rounds ran out.
        budget_bound = ratio > 0.95 and ceiling["success"] < 0.95
        verdict = "BUDGET-STARVED" if budget_bound else (
            "ok" if ceiling["success"] >= 0.95 else "duplication-limited")
        if budget_bound:
            starved.append(cell.name)
        print(f"{cell.name:16s} {cell.axis:10s} {cell.budget:7d} {ceiling['success']:8.3f} "
              f"{ceiling['mean_rounds']:7.2f} {ratio:8.2f} {ceiling['mean_steps']:10.2f} "
              f"{predicted:10.2f} {greedy['success']:7.3f}   {verdict}")

    if starved:
        print(f"\nFAIL: {len(starved)} cell(s) cannot be solved inside their budget even by "
              f"the optimal arm: {', '.join(starved)}")
        print("beta is a multiple of the required cover, so this means the normalisation is "
              "wrong -- fix it before any beta result is quoted.")
        return 1
    print("\nEvery cell's budget admits the ceiling. beta is normalised as intended.")
    return 0


def determinism(args) -> int:
    """Two identical commands must produce two identical files."""
    cell = next(c for c in build_cells() if c.name == args.cell)
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        for repeat in range(2):
            out = f"{tmp}/run{repeat}.json"
            argv = command(cell, 0, tmp, episodes=args.episodes)
            argv[argv.index("--out") + 1] = out
            argv[argv.index("--eval_episodes") + 1] = str(args.eval_episodes)
            done = subprocess.run(argv, capture_output=True, text=True, env=ENV)
            if done.returncode != 0:
                print(f"FAIL: run {repeat} exited {done.returncode}\n{done.stderr[-800:]}")
                return 1
            reports.append(json.loads(pathlib.Path(out).read_text()))

    def comparable(report):
        # Timings and paths differ by construction and say nothing about determinism.
        drop = {"seconds", "train_seconds", "eval_seconds", "path", "best_path", "out"}
        def walk(node):
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items() if k not in drop}
            if isinstance(node, list):
                return [walk(v) for v in node]
            return node
        return walk(report)

    first, second = comparable(reports[0]), comparable(reports[1])
    if first != second:
        print("FAIL: two identical commands produced different results.")
        for key in sorted(set(first) | set(second)):
            if first.get(key) != second.get(key):
                print(f"  differs at {key!r}")
        return 1
    print(f"ok: {args.cell} reproduces exactly over {args.episodes} episodes and "
          f"{args.eval_episodes} evaluation episodes, arms included.")
    return 0


def throughput(args) -> int:
    """Wall time for ONE run, then for `workers` of them at once.

    The ratio is the real speedup, and it is what core-hours must be divided by. On a
    performance/efficiency-core part it is meaningfully below the worker count, so a
    schedule that assumes linearity finishes late.
    """
    cell = next(c for c in build_cells() if c.name == args.cell)

    def launch(count):
        procs, outs = [], []
        tmp = tempfile.mkdtemp()
        started = time.perf_counter()
        for index in range(count):
            out = f"{tmp}/w{index}.json"
            argv = command(cell, index, tmp, episodes=args.episodes)
            argv[argv.index("--out") + 1] = out
            argv[argv.index("--eval_episodes") + 1] = "1"
            procs.append(subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL, env=ENV))
            outs.append(out)
        codes = [p.wait() for p in procs]
        return time.perf_counter() - started, codes

    solo, codes = launch(1)
    if any(codes):
        print(f"FAIL: the single run exited {codes}")
        return 1
    print(f"  1 worker  {solo:7.1f} s")
    many, codes = launch(args.workers)
    if any(codes):
        print(f"FAIL: a concurrent run exited {codes}")
        return 1
    print(f" {args.workers:2d} workers {many:7.1f} s")

    speedup = args.workers * solo / many
    print(f"\nEffective speedup at {args.workers} workers: {speedup:.2f}x "
          f"(perfect would be {args.workers}.00x, {100 * speedup / args.workers:.0f}% of it)")
    print(f"Divide core-hours by {speedup:.2f}, not by {args.workers}, to get wall hours.")
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"workers": args.workers, "solo_seconds": solo,
                                "parallel_seconds": many, "speedup": speedup,
                                "cell": cell.name, "episodes": args.episodes}, indent=1))
    print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="gate", required=True)

    f = sub.add_parser("feasibility", help="is every cell's budget sufficient?")
    f.add_argument("--episodes", type=int, default=40)
    f.add_argument("--only", default=None)
    f.set_defaults(fn=feasibility)

    d = sub.add_parser("determinism", help="does the same command twice agree?")
    d.add_argument("--cell", default="k12s50n04b150")
    d.add_argument("--episodes", type=int, default=32)
    d.add_argument("--eval_episodes", type=int, default=20)
    d.set_defaults(fn=determinism)

    t = sub.add_parser("throughput", help="real parallel speedup on this machine")
    t.add_argument("--cell", default="k12s50n04b150")
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--episodes", type=int, default=48)
    t.add_argument("--out", default="results/sweep/throughput.json")
    t.set_defaults(fn=throughput)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
