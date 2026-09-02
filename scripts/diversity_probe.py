"""STEP 0 of the all-night power-limited-evidence task (see docs/AGENT_B_INBOX.md, 1 Sep 00:40).

Measures private_coverage and private_repeat_rate for a checkpoint under a given evidence
regime, so a power-limited-training run has a behavioural target to be checked against before
anyone reads its headline number. Mirrors scripts/attr_score.py's tally logic (moves counted
from env.last_chosen, AFTER the turn-taking protocol discards inactive agents' submissions --
see that file's own comment on why counting submitted actions inflates the denominator) but
drops everything attribution-specific, since these checkpoints are trained on plain `factored`,
not an attribution backend, and attr_score.py's --backend only accepts the three attribution
variants.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from ma.baselines import RandomAgent, UncertaintyGreedyAgent
from ma.env import MAConfig, PASS_ACTION, ROUND_ROBIN, VARY, TwoAgentEnv
from ma.policy import IndependentPPO
from ma.topology import federated_topology


def build_env(args) -> TwoAgentEnv:
    topology = federated_topology(args.n_agents, args.private_size, args.n_shared)
    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                      budget=args.budget, turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend="factored", policy_arch="gnn_portable",
                      episode_mix="confounded", reward_criterion="claims", claim_bar=1.0,
                      per_agent_reward=True, graph_model=args.graph_model, sf_m=args.sf_m,
                      vs_evidence=args.vs_evidence, vs_evidence_power=args.evidence_power,
                      # THE OBSERVATION FLAGS CHANGE obs_size, so a checkpoint trained with
                      # them refuses to load into an env built without them -- the error is a
                      # torch shape mismatch (79 against 77), not anything about evidence.
                      # Exposed as flags rather than hardcoded because this script is also
                      # used on older checkpoints that predate both.
                      observe_belief_channels=args.observe_belief_channels,
                      observe_reprobe_signal=args.observe_reprobe_signal)
    return TwoAgentEnv(config)


def play(env: TwoAgentEnv, policies, episodes: int, seed: int) -> List[Dict[str, float]]:
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    rows = []
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        moves = private_moves = 0
        touched = {a: set() for a in env.topology.agents}
        while not result.done:
            actions = {a: policies[a](env, result) for a in env.topology.agents}
            result = env.step(actions)
            for agent, (node, _mode) in env.last_chosen.items():
                if node == PASS_ACTION:
                    continue
                moves += 1
                if node not in env.windows[agent].shared:
                    private_moves += 1
                    touched[agent].add(node)
        distinct = sum(len(v) for v in touched.values())
        available = sum(len(env.topology.private[a]) for a in env.topology.agents)
        rows.append({"moves": moves, "private_moves": private_moves,
                     "distinct_private": distinct, "available_private": available,
                     "repeat_private": private_moves - distinct})
    return rows


def summarise(rows: List[Dict[str, float]]) -> Dict[str, float]:
    def ratio(num, den):
        bottom = sum(r[den] for r in rows)
        return float(sum(r[num] for r in rows) / bottom) if bottom else float("nan")
    return {"episodes": len(rows),
            "private_coverage": ratio("distinct_private", "available_private"),
            "private_repeat_rate": ratio("repeat_private", "private_moves"),
            "moves_per_episode": sum(r["moves"] for r in rows) / len(rows)}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_agents", type=int, default=4)
    ap.add_argument("--private_size", type=int, default=4)
    ap.add_argument("--n_shared", type=int, default=4)
    ap.add_argument("--budget", type=int, default=35)
    ap.add_argument("--n_obs", type=int, default=60)
    ap.add_argument("--n_int", type=int, default=20)
    ap.add_argument("--graph_model", default="sf", choices=["er", "sf"])
    ap.add_argument("--sf_m", type=int, default=2)
    ap.add_argument("--vs_evidence", default="sampled", choices=["oracle", "sampled"])
    ap.add_argument("--evidence_power", type=float, default=1.0)
    ap.add_argument("--observe_belief_channels", action="store_true",
                    help="must match the checkpoint's training config or load fails on shape")
    ap.add_argument("--observe_reprobe_signal", action="store_true",
                    help="must match the checkpoint's training config or load fails on shape")
    ap.add_argument("--policy", default=None, help="a .pt from scripts/ma_train.py; omit for baselines only")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    env = build_env(args)
    agents = env.topology.agents

    arms = {}
    if args.policy:
        ppo = IndependentPPO.load(args.policy, env)
        arms["learned"] = ppo.policies(deterministic=False)
    arms["greedy_uncertainty"] = {a: UncertaintyGreedyAgent(a, args.seed, bar=1.0) for a in agents}
    arms["random_vary"] = {a: RandomAgent(a, args.seed, allow_clamp=False) for a in agents}

    report = {"config": vars(args), "arms": {}}
    for label, policies in arms.items():
        rows = play(env, policies, args.episodes, args.seed)
        report["arms"][label] = summarise(rows)
        row = report["arms"][label]
        print(f"  {label:19s} coverage {row['private_coverage']:.3f}  "
              f"repeat {row['private_repeat_rate']:.3f}  "
              f"moves/ep {row['moves_per_episode']:.1f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}")
    return report


if __name__ == "__main__":
    main()
