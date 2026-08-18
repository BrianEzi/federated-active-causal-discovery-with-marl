"""Train two independent PPO agents at `(1,1,3)` and compare against the baselines.

The headline question is NOT "do they solve it" -- greedy already solves the unconfounded
episodes. It is whether the learned policies do the thing the myopic oracle structurally
cannot: clamp a private variable to break a partner's confounding, at a cost to themselves.

That behaviour has a measurable signature. Greedy never clamps (expected information gain
over an agent's own window cannot see any value in it). So the test is:

    does the learned policy clamp MORE on confounded episodes than on unconfounded ones?

A policy that clamps uniformly has learned that clamping is sometimes good. A policy that
clamps selectively, when its partner is confounded, has learned to coordinate -- without
ever seeing its partner's belief, observation, or reward decomposition.

PRE-REGISTERED PREDICTIONS, before the numbers exist:

  P1  Learned beats greedy-greedy on overall solve rate, driven entirely by the confounded
      episodes, where greedy is pinned near 0 by construction.
  P2  Clamp rate is higher on confounded episodes than unconfounded ones.
  P3  On unconfounded episodes learned is roughly level with greedy, not better -- there is
      nothing to coordinate about there, and the myopic oracle is strong.

  P2 is the one that matters. P1 could be satisfied by a policy that clamps indiscriminately
  and simply pays the cost, which is a weaker and less interesting result. If P1 holds and
  P2 fails, I will report it as "learned to clamp, did not learn when", which is not
  coordination.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ma.baselines import GreedyAgentPolicy, RandomAgentPolicy
from ma.env import MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO, MAPPOConfig
from ma.projection import bidirected_pairs
from ma.topology import Topology


def evaluate_baseline(config: MAConfig, kind: str, episodes: int, seed: int) -> dict:
    env = TwoAgentEnv(config, seed=seed)
    make = (GreedyAgentPolicy if kind == "greedy" else RandomAgentPolicy)
    policies = {name: (make("A", env, seed=seed) if kind == "greedy" and name == "A" else
                       make(name, env, seed=seed) if kind == "greedy" else
                       RandomAgentPolicy(name, seed=seed + (name == "B")))
                for name in ("A", "B")}
    return _evaluate(env, config, lambda n, e, r: policies[n](e, r), episodes, seed)


def evaluate_learned(agent: IndependentPPO, config: MAConfig,
                     episodes: int, seed: int) -> dict:
    env = TwoAgentEnv(config, seed=seed)

    def choose(name, e, result):
        action, _, _ = agent._act(name, e.observation(name), deterministic=True)
        return action

    return _evaluate(env, config, choose, episodes, seed)


def _evaluate(env, config, choose, episodes, seed) -> dict:
    rows = []
    for ep in range(episodes):
        result = env.reset(seed=seed * 500_000 + ep)
        confounded = {
            name: len(bidirected_pairs(env.true_adjacency, env.views[name].nodes)) > 0
            for name in ("A", "B")}
        clamps = acts = 0
        steps = 0
        while not result.done and steps < config.budget:
            actions = {name: choose(name, env, result) for name in ("A", "B")}
            for name, index in actions.items():
                target, mode = env.views[name].actions[index]
                if target != -1:
                    acts += 1
                    clamps += (mode == "clamp")
            result = env.step(actions["A"], actions["B"])
            steps += 1
        rows.append({
            "solved": bool(result.info["both_identified"]),
            "solved_A": bool(result.identified["A"]),
            "solved_B": bool(result.identified["B"]),
            "confounded": bool(confounded["A"] or confounded["B"]),
            "steps": steps,
            "clamp_fraction": clamps / max(acts, 1),
        })

    def slice_stats(subset):
        if not subset:
            return None
        n = len(subset)
        p = float(np.mean([r["solved"] for r in subset]))
        z = 1.96
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return {
            "n": n,
            "solve_rate": p,
            "ci": [float(max(0.0, centre - half)), float(min(1.0, centre + half))],
            "mean_steps": float(np.mean([r["steps"] for r in subset])),
            "clamp_fraction": float(np.mean([r["clamp_fraction"] for r in subset])),
        }

    return {
        "all": slice_stats(rows),
        "confounded": slice_stats([r for r in rows if r["confounded"]]),
        "unconfounded": slice_stats([r for r in rows if not r["confounded"]]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_episodes", type=int, default=6000)
    ap.add_argument("--eval_episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_obs", type=int, default=2000)
    ap.add_argument("--n_int", type=int, default=200)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--score_rule", default=None,
                    help="Belief rule to TRAIN under. Defaults to MAConfig's default.")
    ap.add_argument("--clamp_cost", type=float, default=0.0)
    ap.add_argument("--tag", default=None,
                    help="Checkpoint filename tag; defaults to the score rule.")
    ap.add_argument("--out", default="results/ma/train.json")
    ap.add_argument("--checkpoint_dir", default="results/ma/checkpoints",
                    help="Where to persist each seed's trained pair, so the cross-rule "
                         "evaluation does not require retraining.")
    args = ap.parse_args()

    topology = Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                      budget=args.budget,
                      **({"score_rule": args.score_rule} if args.score_rule else {}))
    tag = args.tag or config.score_rule
    print(f"[config] rule={config.score_rule} clamp_cost={args.clamp_cost} tag={tag}",
          flush=True)

    references = {}
    for kind in ("random", "greedy"):
        references[kind] = evaluate_baseline(config, kind, args.eval_episodes, seed=99)
        r = references[kind]["all"]
        print(f"[ref] {kind:>7}: solve {r['solve_rate']:.3f} "
              f"steps {r['mean_steps']:.2f} clamp {r['clamp_fraction']:.3f}", flush=True)
        for part in ("confounded", "unconfounded"):
            s = references[kind][part]
            if s:
                print(f"          {part:>13}: solve {s['solve_rate']:.3f} "
                      f"clamp {s['clamp_fraction']:.3f} n={s['n']}", flush=True)

    per_seed = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        ppo = MAPPOConfig(total_episodes=args.train_episodes, seed=seed,
                          clamp_cost=args.clamp_cost)
        agent = IndependentPPO(config, ppo)
        history = agent.train(verbose=True)
        checkpoint = Path(args.checkpoint_dir) / f"{tag}_seed{seed}.pt"
        agent.save(checkpoint)
        evaluation = evaluate_learned(agent, config, args.eval_episodes, seed=99)
        per_seed.append({
            "seed": seed,
            "train_seconds": time.perf_counter() - t0,
            "eval": evaluation,
            "checkpoint": str(checkpoint),
            "history_tail": history[-10:],
        })
        e = evaluation["all"]
        print(f"[seed {seed}] learned: solve {e['solve_rate']:.3f} "
              f"steps {e['mean_steps']:.2f} clamp {e['clamp_fraction']:.3f}", flush=True)
        for part in ("confounded", "unconfounded"):
            s = evaluation[part]
            if s:
                print(f"            {part:>13}: solve {s['solve_rate']:.3f} "
                      f"clamp {s['clamp_fraction']:.3f} n={s['n']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "references": references,
                               "per_seed": per_seed}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
