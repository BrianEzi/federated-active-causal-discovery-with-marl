"""Where does coordination still have something to win? A sweep over private vs shared.

THE TENSION THIS RESOLVES. Scaling the federation pulls three quantities against each other:

  window size    private + shared. The version space enumerates 3^(edges in a window), so
                 this is the COST axis and it caps out around 6.
  contention     agents per shared variable. Too low and every contested variable gets
                 tested by somebody anyway, so dividing the work wins nothing. Too high and
                 no assignment can cover the surface, so dividing it wins nothing either.
  scale          agents, and total variables.

Growing the shared set in step with the agents holds contention constant but adds those
variables to EVERY window, so cost explodes. The resolution (student's, 2026-08-26) is that
the ratio is the lever, not the absolute sizes: raise shared RELATIVE to private and the
window stays bounded, because the window is private + shared and not agents + shared.

WHAT IS MEASURED, and why it needs no training. Headroom is `ceiling - greedy`: what the
best possible assignment achieves, minus what a myopic rule achieves. Both are computable
directly -- the ceiling by search over intervention sets (exact, because pruning the version
space is commutative and idempotent), greedy by running it. Neither involves a policy, so
the whole frontier is mappable in minutes and tells us WHERE to spend training compute.

The budget at each cell is set so the ceiling is 1.000 where that is reachable, because a
cell where no assignment can win measures the budget rather than the structure -- the
mistake caught at 4 agents this morning, where 25 of 40 episodes were unsolvable.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from ma.baselines import make_baselines
from ma.topology import ER, SF
from scripts.vs_evaluate import build_env, ceiling, run_policy


def budget_for(n_agents: int, private_size: int, n_shared: int, episodes: int,
               graph_model: str, candidates) -> tuple:
    """Smallest budget in `candidates` whose ceiling clears 0.99, and that ceiling.

    Returns the largest candidate if none does, so a cell that is simply out of reach is
    reported with its true (sub-1.0) ceiling rather than silently dropped.
    """
    best = (candidates[-1], 0.0)
    for budget in candidates:
        env = build_env(n_agents, budget, n_shared=n_shared, private_size=private_size,
                        graph_model=graph_model)
        reached = float(np.mean([_ceiling_at(env, seed) for seed in range(episodes)]))
        best = (budget, reached)
        if reached >= 0.99:
            break
    return best


def _ceiling_at(env, seed: int) -> float:
    env.reset(seed=90_000 + seed)
    return ceiling(env)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--private", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--shared", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--max_window", type=int, default=6,
                    help="skip cells past the enumeration's usable range")
    ap.add_argument("--graph_model", default=ER, choices=[ER, SF])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--ceiling_episodes", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    print(f"{args.n_agents} agents, {args.graph_model} graphs, {args.episodes} episodes "
          f"per cell\n")
    print(f"{'private':>7}{'shared':>7}{'k':>3}{'agents/shared':>14}{'budget':>7}"
          f"{'ceiling':>9}{'greedy':>9}{'headroom':>10}{'duplicates':>11}")
    rows = []
    for private_size in args.private:
        for n_shared in args.shared:
            k = private_size + n_shared
            if k > args.max_window:
                continue
            budget, reached = budget_for(args.n_agents, private_size, n_shared,
                                         args.ceiling_episodes, args.graph_model,
                                         [args.n_agents, 2 * args.n_agents,
                                          3 * args.n_agents, 4 * args.n_agents])
            env = build_env(args.n_agents, budget, n_shared=n_shared,
                            private_size=private_size, graph_model=args.graph_model)
            agents = list(env.topology.agents)
            reference = {a: make_baselines(env, a, seed=0) for a in agents}
            rates, _rounds, duplicates = run_policy(
                env, {a: reference[a]["greedy_uncertainty"] for a in agents},
                args.episodes)
            row = {"private": private_size, "shared": n_shared, "k": k,
                   "contention": args.n_agents / n_shared, "budget": budget,
                   "ceiling": reached, "greedy": float(rates.mean()),
                   "greedy_stderr": float(rates.std(ddof=1) / np.sqrt(len(rates))),
                   "headroom": reached - float(rates.mean()),
                   "duplicate_coverage": float(duplicates.mean())}
            rows.append(row)
            print(f"{private_size:7d}{n_shared:7d}{k:3d}{row['contention']:14.2f}"
                  f"{budget:7d}{reached:9.3f}{row['greedy']:9.3f}"
                  f"{row['headroom']:10.3f}{row['duplicate_coverage']:11.3f}", flush=True)

    if rows:
        best = max(rows, key=lambda r: r["headroom"])
        print(f"\nlargest headroom: private {best['private']}, shared {best['shared']} "
              f"(k={best['k']}, {best['contention']:.2f} agents per shared variable) "
              f"-- {best['headroom']:.3f}")
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"n_agents": args.n_agents,
                                    "graph_model": args.graph_model, "cells": rows},
                                   indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
