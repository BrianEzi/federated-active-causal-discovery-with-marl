"""Train and evaluate the PPO agent against the pinned criteria.

Usage:
    python -m scripts.run_experiment --d 5 --observation edge_marginals --seeds 0 1 2 3 4

Every number printed comes from `sa.evaluate`, so what is reported and what was agreed in
docs/SA_PLAN.md cannot drift apart. References (random, greedy, edge-marginal greedy) are
computed once and shared across seeds so the comparison is like-for-like.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sa.baselines import make_baselines
from sa.env import EnvConfig
from sa.evaluate import (
    check_criteria,
    evaluate,
    run_episodes,
    summarise_seeds,
)
from sa.graphs import build_graph_space
from sa.oracle import InterventionOracle
from sa.policy import PPOAgent, PPOConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--observation", type=str, default="edge_marginals",
                        choices=["posterior", "edge_marginals"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train_episodes", type=int, default=4000)
    parser.add_argument("--eval_episodes", type=int, default=300)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--entropy_coef", type=float, default=0.003)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    space = build_graph_space(args.d)
    oracle = InterventionOracle(space)
    env_config = EnvConfig(d=args.d, budget=args.budget)
    baselines = make_baselines(space, seed=0)

    print(f"d={args.d}  observation={args.observation}  "
          f"{space.n_dags} DAGs / {space.n_mecs} classes")

    # References, computed once and reused by every seed.
    refs = {}
    for name in ("random", "greedy_oracle", "edge_marginal_greedy", "no_intervention"):
        t0 = time.time()
        refs[name] = run_episodes(env_config, baselines[name], args.eval_episodes,
                                  seed=99, space=space, oracle=oracle)
        solved = np.mean([t.identified for t in refs[name]])
        cost = np.mean([t.n_interventions if t.identified else args.budget
                        for t in refs[name]])
        print(f"  {name:<22} solve={solved:.2f} cost={cost:.2f} ({time.time() - t0:.0f}s)")

    random_ref = refs["random"]
    # A condition-B agent is compared against the edge-marginal greedy policy, not the
    # full-posterior oracle -- otherwise the comparison conflates a worse policy with a
    # lossier belief.
    greedy_ref = refs["edge_marginal_greedy" if args.observation == "edge_marginals"
                      else "greedy_oracle"]

    per_seed = []
    for seed in args.seeds:
        t0 = time.time()
        agent = PPOAgent(
            env_config,
            PPOConfig(observation=args.observation, total_episodes=args.train_episodes,
                      entropy_coef=args.entropy_coef, lr=args.lr, seed=seed),
            space=space,
        )
        history = agent.train()

        deterministic = evaluate(env_config, agent.as_policy(True), random_ref, greedy_ref,
                                 args.eval_episodes, seed=99, space=space, oracle=oracle)
        sampled = evaluate(env_config, agent.as_policy(False), random_ref, greedy_ref,
                           args.eval_episodes, seed=99, space=space, oracle=oracle)
        verdict = check_criteria(deterministic, sampled)
        per_seed.append(verdict)

        print(f"\nseed {seed} ({time.time() - t0:.0f}s, final entropy "
              f"{history[-1]['entropy']:.2f})")
        print(f"  gap_closed   {deterministic['gap_closed']:+.3f} "
              f"(sampled {sampled['gap_closed']:+.3f})")
        print(f"  solve_rate   {deterministic['solve_rate']:.2f} "
              f"(greedy {deterministic['greedy_solve_rate']:.2f})")
        print(f"  cost         {deterministic['mean_cost']:.2f} "
              f"CI {deterministic['cost_ci'][0]:.2f}-{deterministic['cost_ci'][1]:.2f}")
        print(f"  under_acting {deterministic['under_acting_rate']:.3f}")
        print(f"  optimal      {deterministic['optimal_rate']:.3f} "
              f"(informative {deterministic['informative_fraction']:.2f}) "
              f"regret {deterministic['mean_regret']:.3f}")
        print(f"  by_mec       " + "  ".join(
            f"{k}:{v['gap_closed']:+.2f}(n={v['n']})"
            for k, v in deterministic["by_mec_size"].items()))
        print(f"  checks       {verdict['checks']}  PASSED={verdict['passed']}")

    summary = summarise_seeds(per_seed)
    print(f"\n=== SUMMARY d={args.d} {args.observation} ===")
    print(f"  seeds passing: {summary['n_passed']}/{summary['n_seeds']}")
    print(f"  gap_closed  min {summary['min_gap_closed']:+.3f}  "
          f"median {summary['median_gap_closed']:+.3f}  "
          f"max {summary['max_gap_closed']:+.3f}")
    print(f"  OVERALL: {'PASS' if summary['passed'] else 'FAIL'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "per_seed": per_seed, "summary": summary},
                      f, indent=2, default=float)
        print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
