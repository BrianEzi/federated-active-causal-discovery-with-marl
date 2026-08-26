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
from ma.topology import federated_topology


def build_env(n_agents: int, budget: int, n_shared: int = 3, seed: int = 0,
              channels: bool = False, private_size: int = 1,
              partner_counts: bool = False, mode_by_role: bool = False,
              require_all_types: bool = True, graph_model: str = "er",
              sf_m: int = 2) -> TwoAgentEnv:
    topology = federated_topology(n_agents, private_size, n_shared)
    config = MAConfig(topology=topology, n_obs=60, n_int=20, budget=budget,
                      disclose_regime=True, turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend="version_space", policy_arch="gnn",
                      episode_mix="confounded", reward_criterion="claims",
                      claim_bar=1.0, per_agent_reward=True,
                      observe_belief_channels=channels,
                      observe_partner_counts=partner_counts,
                      mode_by_role=mode_by_role,
                      claims_require_all_types=require_all_types,
                      graph_model=graph_model, sf_m=sf_m)
    return TwoAgentEnv(config, seed=seed)


def _identified(env, agent, space) -> bool:
    window = env.windows[agent]
    belief = VersionSpaceBelief(space, window.k)
    score = score_window(belief, env._true_mag(agent),
                         [window.pos[n] for n in window.private], bar=1.0,
                         require_all_types=env.config.claims_require_all_types)
    return bool(score.identified)


def window_rate(env) -> float:
    """Fraction of windows identified in the CURRENT state."""
    return float(np.mean([_identified(env, a, env.windows[a].belief.last.space)
                          for a in env.topology.agents]))


def _round_capacity(env, rounds: int) -> dict:
    """How many moves each agent gets in the first `rounds` rounds of the rotation.

    Round-robin hands round `i` to agent `i mod n`, so after `r` rounds agent `i` has had
    `r // n` moves plus one more if `i < r % n`. Under any other turn order this is only an
    upper bound and the number reported is a bound rather than an attained optimum.
    """
    agents = list(env.topology.agents)
    n = len(agents)
    return {a: rounds // n + (1 if index < rounds % n else 0)
            for index, a in enumerate(agents)}


def _feasible(subset, capacity, authority) -> bool:
    """Can every node in `subset` be assigned to an agent with authority and spare capacity?

    Bipartite matching with capacities, by augmenting paths. `subset` is at most the round
    budget, so nothing cleverer than this is warranted.
    """
    assigned = {a: [] for a in capacity}

    def augment(node, seen):
        for agent, cap in capacity.items():
            if agent in seen or cap == 0 or node not in authority[agent]:
                continue
            seen.add(agent)
            if len(assigned[agent]) < cap:
                assigned[agent].append(node)
                return True
            for slot, held in enumerate(assigned[agent]):
                if augment(held, seen):
                    assigned[agent][slot] = node
                    return True
        return False

    return all(augment(node, set()) for node in subset)


def _prune(env, agent, space, touched):
    """The version space after everything in `touched` that this window can see."""
    window = env.windows[agent]
    truth = window.belief.truth
    for node in touched:
        if node not in window.nodes:
            continue
        position = window.nodes.index(node)
        target = reveal(truth, window.k, position)
        space = tuple(m for m in space if reveal(m, window.k, position) == target)
    return space


def _reachable_sets(env, size: int):
    """Every set of `size` distinct nodes reachable within `size` rounds of the rotation.

    Distinct because pruning is IDEMPOTENT: intervening twice on the same node tells you
    nothing the first one did not, so a repeat is a wasted round and can never be part of
    an optimum.
    """
    authority = {a: set(env.windows[a].authority) for a in env.topology.agents}
    universe = sorted(set().union(*authority.values()))
    capacity = _round_capacity(env, size)
    for subset in itertools.combinations(universe, size):
        if _feasible(subset, capacity, authority):
            yield subset


def ceiling(env) -> float:
    """The best per-window identification rate any budget-feasible assignment reaches.

    Exact, and cheap, because pruning the version space by an intervention is COMMUTATIVE
    and IDEMPOTENT: the outcome depends only on the SET of nodes intervened on, never on
    the order or on who did it. So this searches SETS of size `budget` -- C(d, budget),
    filtered by whether the rotation can actually deliver them -- rather than replaying
    episodes or enumerating |authority|^n assignments.
    """
    agents = list(env.topology.agents)
    initial = {a: env.windows[a].belief._space for a in agents}
    size = min(env.config.budget, len(set().union(
        *(set(env.windows[a].authority) for a in agents))))

    best = 0.0
    for subset in _reachable_sets(env, size):
        identified = sum(_identified(env, a, _prune(env, a, initial[a], subset))
                         for a in agents)
        best = max(best, identified / len(agents))
        if best == 1.0:
            break
    return best


def optimal_rounds(env, limit: int) -> float:
    """Fewest rounds in which SOME assignment identifies every window; `limit + 1` if none.

    Same search as `ceiling`, stopping at the first size that identifies EVERY window.
    Right-censored at `limit + 1`, matching `env.rounds_to_identification`, so learned and
    optimal are on one scale and their difference is the regret in rounds.
    """
    agents = list(env.topology.agents)
    initial = {a: env.windows[a].belief._space for a in agents}

    def all_identified(touched) -> bool:
        return all(_identified(env, a, _prune(env, a, initial[a], touched))
                   for a in agents)

    if all_identified(()):                       # already settled by observation alone
        return 0.0
    universe = len(set().union(*(set(env.windows[a].authority) for a in agents)))
    for size in range(1, min(limit, universe) + 1):
        for subset in _reachable_sets(env, size):
            if all_identified(subset):
                return float(size)
    return float(limit + 1)


def run_policy(env, policies, episodes: int, seed_base: int = 90_000):
    """Per-episode identification rate, rounds-to-identification, and duplicate coverage.

    Three arrays, all per episode, so every one of them carries its own standard error
    over the SAME episodes -- the arms are paired by seed.
    """
    rates, rounds, duplicates = [], [], []
    for episode in range(episodes):
        result = env.reset(seed=seed_base + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result)
                               for a in env.topology.agents})
        rates.append(window_rate(env))
        rounds.append(float(np.mean(list(env.rounds_to_identification().values()))))
        duplicates.append(env.duplicate_coverage())
    return (np.array(rates, float), np.array(rounds, float), np.array(duplicates, float))


def run_ceiling(env, episodes: int, seed_base: int = 90_000, rounds: bool = False):
    out, best_rounds = [], []
    for episode in range(episodes):
        env.reset(seed=seed_base + episode)
        out.append(ceiling(env))
        if rounds:
            best_rounds.append(optimal_rounds(env, env.config.budget))
    return np.array(out, float), np.array(best_rounds, float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--private_size", type=int, default=1)
    ap.add_argument("--n_shared", type=int, default=3)
    ap.add_argument("--budget", type=int, default=None, help="default: one per agent")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--observe_belief_channels", action="store_true")
    ap.add_argument("--observe_partner_counts", action="store_true")
    ap.add_argument("--mode_by_role", action="store_true")
    ap.add_argument("--legacy_claim_exemption", action="store_true")
    ap.add_argument("--optimal_rounds", action="store_true",
                    help="also compute the exact fewest-rounds baseline (slower)")
    ap.add_argument("--policy", default=None, help="a trained .pt to include as 'learned'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    budget = args.budget if args.budget else args.n_agents
    env = build_env(args.n_agents, budget, n_shared=args.n_shared,
                    channels=args.observe_belief_channels,
                    private_size=args.private_size,
                    partner_counts=args.observe_partner_counts,
                    mode_by_role=args.mode_by_role,
                    require_all_types=not args.legacy_claim_exemption)
    agents = list(env.topology.agents)

    reference = {a: make_baselines(env, a, seed=0) for a in agents}
    arms = {"greedy_uncertainty": {a: reference[a]["greedy_uncertainty"] for a in agents},
            "random_vary": {a: reference[a]["random_vary"] for a in agents}}
    if args.policy:
        ppo = IndependentPPO.load(args.policy, env)
        arms["learned"] = ppo.policies(deterministic=False)

    results = {}
    ceil, best_rounds = run_ceiling(env, args.episodes, rounds=args.optimal_rounds)
    for name, policies in arms.items():
        rates, rounds, duplicates = run_policy(env, policies, args.episodes)
        results[name] = {"mean": float(rates.mean()),
                         "stderr": float(rates.std(ddof=1) / np.sqrt(len(rates))),
                         "rounds": float(rounds.mean()),
                         "rounds_stderr": float(rounds.std(ddof=1) / np.sqrt(len(rounds))),
                         "duplicate_coverage": float(duplicates.mean()),
                         "duplicate_stderr": float(
                             duplicates.std(ddof=1) / np.sqrt(len(duplicates)))}

    greedy = results["greedy_uncertainty"]["mean"]
    print(f"deterministic env: {args.n_agents} agents x {args.private_size} private "
          f"+ {args.n_shared} shared (k={args.private_size + args.n_shared}), "
          f"budget {budget} ({budget / args.n_agents:.2f}/agent), "
          f"{args.episodes} episodes")
    print(f"  {'ceiling':20s} {ceil.mean():.3f}")
    if args.optimal_rounds:
        print(f"  {'optimal rounds':20s} {best_rounds.mean():.3f}")
    for name, row in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        closed = ((row["mean"] - greedy) / (ceil.mean() - greedy)
                  if ceil.mean() > greedy else float("nan"))
        extra = f"   closed {closed:+.1%} of headroom" if name != "greedy_uncertainty" else ""
        print(f"  {name:20s} {row['mean']:.3f} +/- {row['stderr']:.3f}{extra}")
        print(f"  {'':20s}   rounds {row['rounds']:.2f} +/- {row['rounds_stderr']:.2f}"
              f"   duplicate coverage {row['duplicate_coverage']:.3f}"
              f" +/- {row['duplicate_stderr']:.3f}")

    payload = {"n_agents": args.n_agents, "private_size": args.private_size,
               "n_shared": args.n_shared, "budget": budget, "episodes": args.episodes,
               "ceiling": float(ceil.mean()),
               "optimal_rounds": float(best_rounds.mean()) if args.optimal_rounds else None,
               "arms": results}
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
