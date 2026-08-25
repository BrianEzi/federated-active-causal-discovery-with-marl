"""Episode wall-clock at every rung of the scale ladder. Run BEFORE committing a grid.

The ladder climbs one axis at a time -- agents, then shared, then private -- to 5 agents and
30 nodes. Two costs grow along it and they multiply:

  window k = n_private + n_shared     the subset DP, O(3^k) after vectorisation
  pairs    = C(n_shared, 2)           the confounding assignments, screened to `screen_keep`

A training run is `train_episodes` x `budget` x `n_agents` belief updates, so a rung whose
episode costs 3 minutes cannot be trained at 2000 episodes no matter how many cores the
cluster gives -- parallelism buys throughput ACROSS runs, never latency WITHIN one. The
point of this script is to find where that line falls before jobs are queued behind it,
rather than after.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ma.env import MAConfig, TwoAgentEnv
from scripts.ma_train import build_topology

# agents, private per agent, shared. Agents first, then shared, then private.
LADDER = [
    (2, 1, 3), (3, 1, 3), (5, 1, 3),
    (5, 1, 4), (5, 1, 5),
    (5, 2, 5), (5, 3, 5), (5, 4, 5), (5, 5, 5),
]


def time_rung(n_agents: int, n_private: int, n_shared: int, budget: int,
              n_obs: int, n_int: int, episodes: int = 2) -> dict:
    topology = build_topology(n_agents, n_private, n_shared)
    config = MAConfig(topology=topology, n_obs=n_obs, n_int=n_int, budget=budget)
    env = TwoAgentEnv(config, seed=0)
    rng = np.random.default_rng(0)

    started = time.perf_counter()
    steps = 0
    for episode in range(episodes):
        env.reset(seed=episode)
        for _ in range(budget):
            actions = {a: int(rng.integers(env.n_actions(a))) for a in topology.agents}
            result = env.step(actions)
            steps += 1
            if result.done:
                break
    elapsed = time.perf_counter() - started

    k = n_private + n_shared
    pairs = n_shared * (n_shared - 1) // 2
    belief = env.windows[topology.agents[0]].belief
    return {"name": topology.name, "d": topology.d, "k": k, "n_agents": n_agents,
            "n_private": n_private, "n_shared": n_shared, "pairs": pairs,
            "eager": belief._eager, "n_assignments": belief.n_assignments,
            "steps": steps, "seconds_per_episode": elapsed / episodes,
            "seconds_per_step": elapsed / max(steps, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--train_episodes", type=int, default=2000,
                    help="only used to project a full run's cost")
    ap.add_argument("--out", default="results/ma_rung_timing.json")
    args = ap.parse_args()

    rows = []
    for n_agents, n_private, n_shared in LADDER:
        row = time_rung(n_agents, n_private, n_shared, args.budget,
                        args.n_obs, args.n_int, args.episodes)
        row["projected_train_hours"] = (row["seconds_per_episode"]
                                        * args.train_episodes / 3600.0)
        rows.append(row)
        print(f"{row['name']:14s} d={row['d']:2d} k={row['k']:2d} pairs={row['pairs']:2d} "
              f"assign={row['n_assignments']:5d} {'exact' if row['eager'] else 'screen'}  "
              f"{row['seconds_per_episode']:8.2f}s/ep  "
              f"-> {row['projected_train_hours']:7.1f}h for {args.train_episodes} eps",
              flush=True)
        if row["seconds_per_episode"] > 300:
            print("  (stopping: beyond this a training run is not schedulable)")
            break

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
