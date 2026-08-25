"""Record everything that happens in an episode, step by step, as JSON.

The diagnostic layer that was missing: aggregate success rates say an episode failed, not
WHERE it went wrong. This dumps, for every step, who intervened on what, and what that did
to every claim in every window -- the frequencies, not just the verdicts, so a claim that
missed the bar by 0.02 is distinguishable from one that was never close.

Truth is recorded alongside belief because this is an oracle-side diagnostic, exactly as
the reward is. Nothing here is visible to a policy.

Companion viewer: scripts/trace_view.py renders these traces as a step-through page.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from cb.claims import enumerate_claims, score_window
from ma.baselines import make_baselines
from ma.env import VARY, CLAMP, MAConfig, ROUND_ROBIN, TwoAgentEnv
from ma.policy import IndependentPPO
from ma.projection import BIDIRECTED as MAG_BIDIRECTED
from ma.projection import DIRECTED as MAG_DIRECTED
from ma.topology import Topology, two_agent


def _true_edges(mag: np.ndarray) -> list:
    """The true MAG as a list of edges, in window positions."""
    mag = np.asarray(mag)
    k = mag.shape[0]
    out = []
    for u in range(k):
        for v in range(u + 1, k):
            if mag[u, v] == MAG_BIDIRECTED:
                out.append({"u": u, "v": v, "kind": "bidirected"})
            elif mag[u, v] == MAG_DIRECTED:
                out.append({"u": u, "v": v, "kind": "directed"})
            elif mag[v, u] == MAG_DIRECTED:
                out.append({"u": v, "v": u, "kind": "directed"})
    return out


def _window_state(env, agent, bar):
    """One agent's window: truth, belief frequencies, and every claim's outcome."""
    window = env.windows[agent]
    belief = window.belief.last
    mag = env._true_mag(agent)
    private = [window.pos[n] for n in env.topology.private[agent] if n in window.pos]
    claims = enumerate_claims(belief, mag, private, bar=bar)
    score = score_window(belief, mag, private, bar=bar)
    return {
        "nodes": [int(n) for n in window.nodes],
        "private_positions": [int(p) for p in private],
        "true_edges": _true_edges(mag),
        "claims": [{"kind": c.kind, "u": c.u, "v": c.v, "required": bool(c.required),
                    "outcome": c.outcome, "truth": c.truth,
                    "freq_correct": round(float(c.freq_correct), 4),
                    "freq_wrong": round(float(c.freq_wrong), 4)} for c in claims],
        "score": {"right": score.n_right, "wrong": score.n_wrong,
                  "unsure": score.n_unsure,
                  "required_right": score.required_right,
                  "required_total": score.required_total,
                  "identified": bool(score.identified),
                  "fraction": round(score.fraction(), 4)},
        "adjacency": np.round(np.asarray(belief.adjacency), 3).tolist(),
        "directed": np.round(np.asarray(belief.directed), 3).tolist(),
        "bidirected": np.round(np.asarray(belief.bidirected), 3).tolist(),
    }


def trace_episode(env, policies, seed, bar) -> dict:
    agents = list(env.topology.agents)
    result = env.reset(seed=seed)
    steps = [{"step": 0, "actions": [], "reward": 0.0, "done": False,
              "windows": {str(a): _window_state(env, a, bar) for a in agents}}]

    while not result.done:
        chosen = {a: policies[a](env, result) for a in agents}
        result = env.step(chosen)
        # Read what the ENVIRONMENT applied, not what was submitted: under turn-taking the
        # inactive agent's move is discarded, and recording the submission would show
        # interventions that never happened.
        actions = []
        for agent, (node, mode) in env.last_chosen.items():
            actions.append({"agent": int(agent), "node": int(node), "mode": str(mode),
                            "passed": bool(node == -1)})
        steps.append({
            "step": len(steps),
            "actions": actions,
            "reward": round(float(result.reward), 4),
            "done": bool(result.done),
            "windows": {str(a): _window_state(env, a, bar) for a in agents},
        })

    final = steps[-1]["windows"]
    return {
        "seed": int(seed),
        "identified": {a: final[str(a)]["score"]["identified"] for a in agents},
        "joint_identified": all(final[str(a)]["score"]["identified"] for a in agents),
        "steps": steps,
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--three_agents", action="store_true")
    ap.add_argument("--budget", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=250)
    ap.add_argument("--cb_n_boot", type=int, default=12)
    ap.add_argument("--episode_mix", default="confounded")
    ap.add_argument("--claim_bar", type=float, default=0.7)
    ap.add_argument("--policy", default="greedy_uncertainty",
                    help="a baseline name, or the path to a saved policy pair (.pt)")
    ap.add_argument("--oracle_obs", action="store_true")
    ap.add_argument("--backend", default="constraint",
                    choices=["constraint", "version_space"])
    ap.add_argument("--n_agents", type=int, default=None)
    ap.add_argument("--per_agent_reward", action="store_true")
    ap.add_argument("--out", default="results/traces/trace.json")
    args = ap.parse_args(argv)

    if args.n_agents:
        topology = Topology(name=f"T_{args.n_agents}agent_1each",
                            private=tuple((i,) for i in range(args.n_agents)),
                            exposed=tuple(range(args.n_agents, args.n_agents + 3)))
    elif args.three_agents:
        topology = Topology(name="T_3agent_1each",
                            private=((0,), (1,), (2,)), exposed=(3, 4, 5))
    else:
        topology = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,),
                             exposed=(2, 3, 4))

    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                      budget=args.budget, disclose_regime=True, turn_order=ROUND_ROBIN,
                      action_modes=(VARY,), belief_backend=args.backend,
                      cb_n_boot=args.cb_n_boot, policy_arch="gnn",
                      episode_mix=args.episode_mix, reward_criterion="claims",
                      claim_bar=args.claim_bar,
                      per_agent_reward=args.per_agent_reward,
                      oracle_obs_structure=args.oracle_obs)
    env = TwoAgentEnv(config, seed=args.seed)
    agents = list(topology.agents)

    if args.policy.endswith(".pt"):
        ppo = IndependentPPO.load(args.policy, env)
        policies = ppo.policies(deterministic=False)
        policy_label = f"learned:{pathlib.Path(args.policy).name}"
    else:
        reference = {a: make_baselines(env, a, seed=args.seed) for a in agents}
        policies = {a: reference[a][args.policy] for a in agents}
        policy_label = args.policy

    episodes = []
    for i in range(args.episodes):
        episodes.append(trace_episode(env, policies, seed=1000 + i, bar=args.claim_bar))
        got = episodes[-1]
        print(f"  episode {i} seed {got['seed']}  joint {got['joint_identified']}  "
              f"per-agent {got['identified']}", flush=True)

    trace = {
        "policy": policy_label,
        "topology": {"name": topology.name, "agents": len(agents),
                     "private": [list(p) for p in topology.private],
                     "exposed": list(topology.exposed)},
        "config": {"budget": args.budget, "n_obs": args.n_obs, "n_int": args.n_int,
                   "belief_backend": args.backend,
                   "cb_n_boot": args.cb_n_boot, "claim_bar": args.claim_bar,
                   "episode_mix": args.episode_mix,
                   "oracle_obs_structure": args.oracle_obs,
                   "per_agent_budget": round(args.budget / len(agents), 2)},
        "episodes": episodes,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    return trace


if __name__ == "__main__":
    main()
