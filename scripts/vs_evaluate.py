"""Evaluate policies in the deterministic environment against a COMPUTABLE OPTIMUM.

The point of the deterministic backend is that the best achievable result is not a mystery:
with one intervention per agent, the outcome depends only on which nodes get intervened on,
so the optimum can be enumerated exactly. That converts the headline from "beats greedy by
N points, CIs overlapping" into "closed X% of the achievable headroom", which is a claim
with a denominator.

Reported per WINDOW, not as the all-agents conjunction: the conjunction falls exponentially
in the number of agents whatever the policy does (measured 2026-08-25), so it hides exactly
the effect this environment exists to measure.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np

from cb.claims import score_window
from cb.versionspace import VersionSpaceBelief, reveal
from ma.baselines import make_baselines
from ma.env import ROUND_ROBIN, VARY, MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO
from ma.topology import Topology


def build_env(n_agents: int, budget: int, n_shared: int = 3, seed: int = 0,
              channels: bool = False) -> TwoAgentEnv:
    topology = Topology(name=f"T_{n_agents}agent_1each",
                        private=tuple((i,) for i in range(n_agents)),
                        exposed=tuple(range(n_agents, n_agents + n_shared)))
    config = MAConfig(topology=topology, n_obs=60, n_int=20, budget=budget,
                      disclose_regime=True, turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend="version_space", policy_arch="gnn",
                      episode_mix="confounded", reward_criterion="claims",
                      claim_bar=1.0, per_agent_reward=True,
                      observe_belief_channels=channels)
    return TwoAgentEnv(config, seed=seed)


def _identified(env, agent, space) -> bool:
    window = env.windows[agent]
    belief = VersionSpaceBelief(space, window.k)
    score = score_window(belief, env._true_mag(agent),
                         [window.pos[n] for n in window.private], bar=1.0)
    return bool(score.identified)


def window_rate(env) -> float:
    """Fraction of windows identified in the CURRENT state."""
    return float(np.mean([_identified(env, a, env.windows[a].belief.last.space)
                          for a in env.topology.agents]))


def ceiling(env) -> float:
    """The best per-window identification any assignment could have reached.

    With one intervention each, a window's fate depends only on which of ITS nodes anyone
    intervened on, so this enumerates assignments of one authority node per agent and prunes
    directly -- no replaying of episodes.
    """
    agents = list(env.topology.agents)
    options = {a: list(env.windows[a].authority) for a in agents}
    initial = {a: env.windows[a].belief._space for a in agents}
    truth = {a: env.windows[a].belief.truth for a in agents}

    best = 0.0
    for assignment in itertools.product(*(options[a] for a in agents)):
        touched = set(assignment)
        identified = 0
        for a in agents:
            window = env.windows[a]
            space = initial[a]
            for node in touched:
                if node not in window.nodes:
                    continue
                position = window.nodes.index(node)
                target = reveal(truth[a], window.k, position)
                space = tuple(m for m in space if reveal(m, window.k, position) == target)
            identified += _identified(env, a, space)
        best = max(best, identified / len(agents))
        if best == 1.0:
            break
    return best


def run_policy(env, policies, episodes: int, seed_base: int = 90_000):
    rates = []
    for episode in range(episodes):
        result = env.reset(seed=seed_base + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result)
                               for a in env.topology.agents})
        rates.append(window_rate(env))
    return np.array(rates, float)


def run_ceiling(env, episodes: int, seed_base: int = 90_000):
    out = []
    for episode in range(episodes):
        env.reset(seed=seed_base + episode)
        out.append(ceiling(env))
    return np.array(out, float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--budget", type=int, default=None, help="default: one per agent")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--observe_belief_channels", action="store_true")
    ap.add_argument("--policy", default=None, help="a trained .pt to include as 'learned'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    budget = args.budget if args.budget else args.n_agents
    env = build_env(args.n_agents, budget, channels=args.observe_belief_channels)
    agents = list(env.topology.agents)

    reference = {a: make_baselines(env, a, seed=0) for a in agents}
    arms = {"greedy_uncertainty": {a: reference[a]["greedy_uncertainty"] for a in agents},
            "random_vary": {a: reference[a]["random_vary"] for a in agents}}
    if args.policy:
        ppo = IndependentPPO.load(args.policy, env)
        arms["learned"] = ppo.policies(deterministic=False)

    results = {}
    ceil = run_ceiling(env, args.episodes)
    for name, policies in arms.items():
        rates = run_policy(env, policies, args.episodes)
        results[name] = {"mean": float(rates.mean()),
                         "stderr": float(rates.std(ddof=1) / np.sqrt(len(rates)))}

    greedy = results["greedy_uncertainty"]["mean"]
    print(f"deterministic env: {args.n_agents} agents, budget {budget} "
          f"({budget / args.n_agents:.2f}/agent), {args.episodes} episodes")
    print(f"  {'ceiling':20s} {ceil.mean():.3f}")
    for name, row in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        closed = ((row["mean"] - greedy) / (ceil.mean() - greedy)
                  if ceil.mean() > greedy else float("nan"))
        extra = f"   closed {closed:+.1%} of headroom" if name != "greedy_uncertainty" else ""
        print(f"  {name:20s} {row['mean']:.3f} +/- {row['stderr']:.3f}{extra}")

    payload = {"n_agents": args.n_agents, "budget": budget, "episodes": args.episodes,
               "ceiling": float(ceil.mean()), "arms": results}
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
