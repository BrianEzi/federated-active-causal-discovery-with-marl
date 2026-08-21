"""GATE 1 and GATE 2 on the enumeration-free path.

**These gates are the prerequisite for any d=7 result, not a consolation prize.** The whole
reason the earlier d=6 numbers had to be thrown away is that they were measured on an
environment where GATE 1 failed -- the task did not require intervening, so "the agent
beats greedy" measured nothing. Running d=7 RL before validating the d=7 environment would
repeat that exactly, one size up.

GATE 1 -- the task must require intervening.
    The observational-only identification rate must equal the prior-weighted fraction of
    DAGs alone in their Markov equivalence class. Both sides are now available past
    enumeration: the rate from the DP environment, the target from the covered-edge test
    over prior samples.

GATE 2 -- choices must matter.
    Random must be clearly worse than the greedy oracle, with non-overlapping intervals. If
    they tie, nothing rewards good experiment selection and there is nothing to learn.

Usage:
    python scripts/gates_dp.py --d 7 --n_obs 20000 --episodes 300 --out results/gates_d7.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from sa.baselines import GreedyOracleDPPolicy, RandomPolicy
from sa.env import PASS_ACTION, EnvConfig
from sa.env_dp import DPCausalDiscoveryEnv
from sa.gates import bootstrap_ci, estimate_singleton_fraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=7)
    parser.add_argument("--n_obs", type=int, default=20000)
    parser.add_argument("--n_int", type=int, default=100)
    # GATE 2 asks whether CHOICES matter, so it must run where choice quality still
    # affects the outcome. Discrimination peaks at budget 2-3 and is gone by 16.
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--prior_p", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--gate1_episodes", type=int, default=600)
    parser.add_argument("--oracle_draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    return parser


def run_episodes(env, policy, n_episodes: int, seed: int) -> dict:
    """Steps to identification and whether it happened, per episode."""
    if hasattr(policy, "reset"):
        policy.reset(seed)
    solved, steps = [], []
    for episode in range(n_episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            result = env.step(policy(env, result))
        solved.append(float(result.identified))
        # Unsolved episodes are recorded at the budget, so the mean is a real "cost to
        # identify" rather than an average over only the easy cases.
        steps.append(float(result.n_interventions if result.identified else env.config.budget))
    return {"solved": np.array(solved), "steps": np.array(steps)}


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    config = EnvConfig(d=args.d, n_obs=args.n_obs, n_int=args.n_int, budget=args.budget,
                       prior="erdos_renyi", prior_p=args.prior_p,
                       identify_threshold=args.threshold)
    env = DPCausalDiscoveryEnv(config)
    started = time.time()

    # -- GATE 1 ----------------------------------------------------------------------
    # Measured by resetting and reading identification before any action is taken, which
    # is exactly "what could be learned from observation alone".
    observational = []
    for episode in range(args.gate1_episodes):
        observational.append(float(env.reset(seed=args.seed * 7919 + episode).identified))
    observational = np.array(observational)
    rate = float(observational.mean())
    rate_ci = bootstrap_ci(observational, seed=args.seed)

    target = estimate_singleton_fraction(args.d, p=args.prior_p, seed=args.seed)
    gate1_pass = bool(rate_ci[0] <= target["estimate"] <= rate_ci[1]
                      or target["ci"][0] <= rate <= target["ci"][1])

    # -- GATE 2 ----------------------------------------------------------------------
    random_policy = RandomPolicy(seed=args.seed)
    greedy_policy = GreedyOracleDPPolicy(env.dp, n_draws=args.oracle_draws, seed=args.seed)

    random_out = run_episodes(env, random_policy, args.episodes, args.seed)
    greedy_out = run_episodes(env, greedy_policy, args.episodes, args.seed)

    random_ci = bootstrap_ci(random_out["steps"], seed=args.seed)
    greedy_ci = bootstrap_ci(greedy_out["steps"], seed=args.seed)
    # Greedy must need STRICTLY FEWER steps, with intervals that do not overlap.
    gate2_pass = bool(greedy_ci[1] < random_ci[0])

    payload = {
        "config": vars(args),
        "gate1": {
            "observational_rate": rate,
            "observational_ci": rate_ci,
            "singleton_target": target["estimate"],
            "singleton_ci": target["ci"],
            "pass": gate1_pass,
            "n_episodes": args.gate1_episodes,
        },
        "gate2": {
            "random_steps": float(random_out["steps"].mean()),
            "random_steps_ci": random_ci,
            "random_solve_rate": float(random_out["solved"].mean()),
            "greedy_steps": float(greedy_out["steps"].mean()),
            "greedy_steps_ci": greedy_ci,
            "greedy_solve_rate": float(greedy_out["solved"].mean()),
            "pass": gate2_pass,
            "n_episodes": args.episodes,
        },
        "elapsed_seconds": time.time() - started,
    }

    print(json.dumps(payload, indent=2, default=float))
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
