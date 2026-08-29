"""Evaluate an EXISTING checkpoint at a DIFFERENT budget than it was trained on.

WHY THIS IS NOT THE SAME QUESTION `w20iso` ANSWERS. `w20iso_s0` was TRAINED and evaluated at
budget 40 -- in distribution both ways. This script takes a checkpoint trained at one budget
and plays it at another, so the policy's `budget_left` feature -- `(budget - rounds_used) /
budget`, per `ma/env.py::observation` -- is IN RANGE (always [0, 1], not a raw count) but was
shaped by a different decay RATE per round during training. A policy trained at budget 15
learns "this fraction remaining means roughly THIS MANY ROUNDS HAVE PASSED, calibrated to a
15-round episode"; played at budget 40 the same fraction means a different absolute round
count. This is genuine but softer distribution shift, not the raw-count blowup that would
make the result meaningless outright.

WHAT THIS CAN AND CANNOT ANSWER. It CAN answer "does the trained policy's structural
knowledge transfer to more rounds", a weaker and different question from "would a policy
TRAINED at this budget succeed" -- w30iso, this repository's analogue of w20iso, would answer
that one and would cost ~2 hours of training per seed. It CANNOT distinguish a genuinely
capped policy from one that is merely confused by the unfamiliar pacing, so a poor result here
does not confirm the requirement is truly binding the way w30's own-budget zero already does
-- it only says this SPECIFIC transfer failed.

Reports at several budgets so a monotone climb (or its absence) is visible without guessing
one number in advance.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import numpy as np

from ma.baselines import UncertaintyGreedyAgent
from ma.evaluate import evaluate_episode
from ma.policy import IndependentPPO
from scripts.rescore_from_config import env_from_config


def play(env, policies, episodes: int, seed: int) -> List[float]:
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    out = []
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
        out.append(float(evaluate_episode(env)["success"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="the trained run's .json, e.g. results/ladder/w30_s0.json")
    ap.add_argument("--budgets", default=None,
                    help="comma-separated budgets to try; default is the trained budget "
                         "times 1, 1.5, 2, 3")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    path = pathlib.Path(args.result)
    report = json.loads(path.read_text())
    config = dict(report["config"])
    trained_budget = config["budget"]
    use_seed = args.seed if args.seed is not None else report.get("seed", 0)
    checkpoint = path.with_suffix(".pt")
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint beside {path.name}")

    budgets = ([int(round(trained_budget * f)) for f in (1, 1.5, 2, 3)]
              if args.budgets is None else [int(x) for x in args.budgets.split(",")])
    budgets = sorted(set(b for b in budgets if b >= trained_budget))

    print(f"{path.stem}: trained at budget {trained_budget}, "
         f"evaluated OFF-DISTRIBUTION at higher budgets (see this file's docstring)\n")
    print(f"{'budget':>8s} {'ratio-vs-trained':>17s} {'learned':>9s} {'greedy@1.0':>11s}")
    rows = []
    for budget in budgets:
        env = env_from_config({**config, "budget": budget}, seed=use_seed)
        learned_policy = IndependentPPO.load(str(checkpoint), env).policies(
            deterministic=False)
        greedy_policy = {a: UncertaintyGreedyAgent(a, use_seed, bar=1.0)
                         for a in env.topology.agents}
        learned = play(env, learned_policy, args.episodes, use_seed)
        greedy = play(env, greedy_policy, args.episodes, use_seed)
        rows.append({"budget": budget, "ratio_vs_trained": budget / trained_budget,
                    "learned": float(np.mean(learned)), "greedy": float(np.mean(greedy)),
                    "n": args.episodes})
        print(f"{budget:8d} {budget / trained_budget:17.2f} "
             f"{np.mean(learned):9.3f} {np.mean(greedy):11.3f}")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"source": str(path), "trained_budget": trained_budget,
                                   "rows": rows}, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
