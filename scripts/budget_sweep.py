"""How much does the intervention budget actually matter -- greedy vs random, SA and MA?

The standing conclusion was "[CORRECTED] budget is largely a metric artifact and must not
be read as a lever": in the Phase 2 sweep, budget 10 -> 40 moved the working architecture
by less than 0.2. But every one of those measurements was taken at n_obs=20000, where the
median episode needs 2 interventions and budget 10 is enormous slack. Budget was a null
because it was never BINDING.

Two things this fixes:

  1. The metric. `episode_costs` charges unsolved episodes at the full budget, so raising
     the budget multiplies the penalty for the same failure -- that is what produced the
     artifact. Here solve-rate-within-budget and steps-among-solved are reported
     SEPARATELY and never combined.

  2. The regime. Swept at low n_obs, where the horizon is real.

Method note -- why one run per (setting, policy) covers every budget. None of these
baselines reads `budget_left`: random draws uniformly over its authority, greedy scores
the current belief. So a smaller budget is EXACTLY a truncation of the same trajectory,
and the whole budget curve is the cumulative distribution of steps-to-identification. This
is exact, not an approximation, but it would silently become wrong for a learned policy
(which does observe remaining budget) -- so this script is for baselines only.

Usage:
    python scripts/budget_sweep.py --episodes 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

import numpy as np

from ma.baselines import GreedyAgentPolicy, RandomAgentPolicy
from ma.env import MAConfig, PASS_ACTION, TwoAgentEnv
from ma.projection import bidirected_pairs
from ma.topology import Topology
from sa.baselines import GreedyOracleDPPolicy, RandomPolicy
from sa.env import EnvConfig
from sa.env_dp import DPCausalDiscoveryEnv

# A budget of 0 is the observational control; the curve is reported from 1 upward.
BUDGETS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20)


def bootstrap_ci(values: np.ndarray, seed: int = 0, draws: int = 2000) -> List[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def curve_from_steps(steps: np.ndarray, budgets=BUDGETS, seed: int = 0) -> List[dict]:
    """Solve rate within each budget, and mean steps among the episodes solved there.

    `steps` holds the intervention count at identification, or np.inf if the episode was
    never solved inside the maximum budget run.
    """
    out = []
    for b in budgets:
        solved = (steps <= b)
        among = steps[solved]
        out.append({
            "budget": int(b),
            "solve_rate": float(solved.mean()),
            "solve_ci": bootstrap_ci(solved.astype(float), seed=seed),
            # Deliberately NOT a cost that mixes the two. Unsolved episodes are absent
            # here and counted in solve_rate instead -- that separation is the whole point.
            "steps_among_solved": float(among.mean()) if len(among) else float("nan"),
        })
    return out


# ---------------------------------------------------------------------------- single agent


def single_agent(d: int, n_obs: int, n_int: int, episodes: int, max_budget: int,
                 seed: int) -> dict:
    config = EnvConfig(d=d, n_obs=n_obs, n_int=n_int, budget=max_budget,
                       prior="erdos_renyi", prior_p=0.5, identify_threshold=0.7)
    env = DPCausalDiscoveryEnv(config)
    arms = {}
    for label in ("greedy", "random"):
        policy = (GreedyOracleDPPolicy(env.dp, seed=seed) if label == "greedy"
                  else RandomPolicy(seed=seed))
        steps = []
        for episode in range(episodes):
            result = env.reset(seed=seed * 100_000 + episode)
            # Identification before acting is a 0-step solve, and must not be attributed
            # to the policy.
            if result.identified:
                steps.append(0.0)
                continue
            while not result.done:
                result = env.step(policy(env, result))
            steps.append(float(result.n_interventions) if result.identified else np.inf)
        arms[label] = {
            "curve": curve_from_steps(np.array(steps), seed=seed),
            "never_solved": float(np.isinf(steps).mean()),
        }
    return {"d": d, "n_obs": n_obs, "n_int": n_int, "episodes": episodes, "arms": arms}


# ----------------------------------------------------------------------------- two agent


def two_agent(topology: Topology, n_obs: int, n_int: int, episodes: int, max_budget: int,
              seed: int, rule: str) -> dict:
    config = MAConfig(topology=topology, n_obs=n_obs, n_int=n_int, budget=max_budget,
                      identify_threshold=0.7, score_rule=rule)
    env = TwoAgentEnv(config)
    arms = {}
    for label in ("greedy", "random"):
        policies = {
            n: (GreedyAgentPolicy(n, env, seed=seed) if label == "greedy"
                else RandomAgentPolicy(n, seed=seed + 1))
            for n in ("A", "B")
        }
        steps, confounded, clamps, moves = [], [], 0, 0
        for episode in range(episodes):
            result = env.reset(seed=seed * 100_000 + episode)
            pairs = bidirected_pairs(env.true_adjacency, env.topology.observed_by("A"))
            confounded.append(bool(pairs))
            if result.info["both_identified"]:
                steps.append(0.0)
                continue
            while not result.done:
                actions = {n: policies[n](env, result) for n in ("A", "B")}
                for n, a in actions.items():
                    if a == PASS_ACTION or a == env.views[n].n_actions - 1:
                        continue
                    moves += 1
                    if env.views[n].actions[a][1] == "clamp":
                        clamps += 1
                result = env.step(actions["A"], actions["B"])
            solved = result.info["both_identified"]
            rounds = max(result.n_interventions.values())
            steps.append(float(rounds) if solved else np.inf)
        steps = np.array(steps)
        confounded = np.array(confounded)
        arms[label] = {
            "curve": curve_from_steps(steps, seed=seed),
            # The coordination question lives entirely on the confounded episodes; the
            # overall curve is dominated by the unconfounded majority.
            "curve_confounded": curve_from_steps(steps[confounded], seed=seed),
            "curve_unconfounded": curve_from_steps(steps[~confounded], seed=seed),
            "confounded_fraction": float(confounded.mean()),
            "clamp_fraction": float(clamps / moves) if moves else float("nan"),
        }
    return {"topology": topology.name, "n_obs": n_obs, "n_int": n_int,
            "episodes": episodes, "rule": rule, "arms": arms}


# --------------------------------------------------------------------------------- report


def show_curve(title: str, arms: dict, key: str = "curve") -> None:
    print(f"\n{title}")
    budgets = [row["budget"] for row in arms["greedy"][key]]
    print("  budget       " + "".join(f"{b:>7d}" for b in budgets))
    for label in ("greedy", "random"):
        rates = [row["solve_rate"] for row in arms[label][key]]
        print(f"  {label:<11}solve" + "".join(f"{r:>7.3f}" for r in rates))
    for label in ("greedy", "random"):
        st = [row["steps_among_solved"] for row in arms[label][key]]
        print(f"  {label:<11}steps" + "".join(f"{s:>7.2f}" for s in st))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_budget", type=int, default=20)
    ap.add_argument("--n_int", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/budget/budget_sweep.json")
    args = ap.parse_args(argv)

    started = time.time()
    report = {"single_agent": [], "two_agent": []}

    # Two n_obs settings so the "budget only binds when data is scarce" claim is tested
    # rather than assumed -- 20000 is the regime every earlier budget measurement used.
    for d, n_obs in ((5, 20000), (5, 100), (7, 20000), (7, 100)):
        entry = single_agent(d, n_obs, args.n_int, args.episodes, args.max_budget,
                             args.seed)
        report["single_agent"].append(entry)
        show_curve(f"SINGLE AGENT  d={d}  n_obs={n_obs}", entry["arms"])
        print(f"  [{time.time() - started:.0f}s]")

    topology = Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,),
                        exposed=(2, 3, 4))
    for n_obs in (20000, 100):
        entry = two_agent(topology, n_obs, args.n_int, args.episodes, args.max_budget,
                          args.seed, rule="joint_conf")
        report["two_agent"].append(entry)
        show_curve(f"TWO AGENT  (1,1,3)  n_obs={n_obs}  ALL", entry["arms"])
        show_curve(f"TWO AGENT  (1,1,3)  n_obs={n_obs}  CONFOUNDED "
                   f"({entry['arms']['greedy']['confounded_fraction']:.2f} of episodes)",
                   entry["arms"], key="curve_confounded")
        for label in ("greedy", "random"):
            print(f"  {label} clamp_fraction "
                  f"{entry['arms'][label]['clamp_fraction']:.3f}")
        print(f"  [{time.time() - started:.0f}s]")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return report


if __name__ == "__main__":
    main()
