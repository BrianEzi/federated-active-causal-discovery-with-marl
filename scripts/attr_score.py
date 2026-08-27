"""Score the attribution arms on IDENTICAL episodes, and report PAIRED differences.

WHY THIS EXISTS. `scripts/ma_train.py` already plays every arm over the same episode
seeds, but `ma/evaluate.py::run_arm` reports only the claims verdict (`success`). The
attribution axis needs three more numbers per handover: attribution accuracy, structure
accuracy, and the share of applied moves spent on an agent's OWN PRIVATE nodes. Those were
computed in a session scratchpad on 26-27 August and the script did not survive, so every
`results/attr_scale/` run would otherwise be unreportable.

TWO THINGS THIS FILE IS CAREFUL ABOUT, both of which have already cost a result set:

  * MOVES, NOT QUERIES. Under turn-taking every policy is queried every round and the
    inactive agent's move is discarded. `env.last_chosen` is rebuilt AFTER the protocol has
    forced the inactive agents to pass, so tallying from it counts what was applied.
    Counting submitted actions inflates the denominator by the agent count.
  * PAIRED DIFFERENCES. Every arm plays `seed * 100_000 + episode`, exactly as `run_arm`
    does, so the episodes are identical across arms. The reported error on a difference is
    therefore the standard error of the PER-EPISODE difference, not the quadrature sum of
    two independent standard errors -- which over-states it, and was the reason three
    "results" on 26 August were smaller than their own noise.

It also refuses to build `make_baselines`, which constructs `GreedyAgent` eagerly; that
agent enumerates the window and raises past size 5, and it has crashed two jobs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np

from cb.attribution import score_groups
from cb.claims import score_window
from ma.baselines import ProbeThenWorkAgent, RandomAgent, UncertaintyGreedyAgent
from ma.density_guard import DensityGuardedEnv
from ma.env import ATTRIBUTED, PASS_ACTION, ROUND_ROBIN, VARY, MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO
from ma.topology import federated_topology


def build_env(args) -> TwoAgentEnv:
    """The training environment, rebuilt from flags.

    Held identical to `scripts/ma_train.py`'s construction on purpose: a checkpoint refuses
    to load into an environment whose observation layout differs, and a silently different
    reward criterion would make the baselines answer a different question from the learner.
    """
    topology = federated_topology(args.n_agents, args.private_size, args.n_shared)
    # The guard has to match the TRAINING arm. A policy trained on draws capped at 7 edges
    # and scored on uncapped ones would be measured out of distribution, and the baselines
    # beside it would be playing a different task -- which is the same error as comparing
    # across belief rules.
    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                      budget=args.budget, disclose_regime=True,
                      turn_order=ROUND_ROBIN, action_modes=(VARY,),
                      belief_backend=ATTRIBUTED, policy_arch=args.policy_arch,
                      episode_mix="confounded", reward_criterion="claims",
                      claim_bar=1.0, per_agent_reward=args.per_agent_reward,
                      observe_belief_channels=True, observe_partner_counts=True,
                      graph_model=args.graph_model, sf_m=args.sf_m)
    return DensityGuardedEnv(config, max_edges=args.max_edges, seed=args.seed)


def _episode_row(env: TwoAgentEnv, private_moves: int, moves: int) -> Dict[str, float]:
    """Every metric for one finished episode, pooled over the agents' windows.

    `identified` is per WINDOW, never the all-agents conjunction: the conjunction falls
    exponentially in the agent count whatever the policy does, so at 4 and 6 agents it
    would hide exactly the effect these jobs exist to measure.
    """
    windows = env.windows
    identified: List[float] = []
    attr_right = attr_total = 0
    req_right = req_total = 0
    for agent, window in windows.items():
        # `confounding_claims=False`: under the attributed backend the bidirected claim is
        # REPLACED by the attribution claim and the two are never scored together.
        structure = score_window(window.belief.last, env._true_mag(agent),
                                 [window.pos[n] for n in window.private], bar=1.0,
                                 require_all_types=env.config.claims_require_all_types,
                                 confounding_claims=False)
        attribution = score_groups(window.belief.last, window.belief.true_groups, bar=1.0)
        identified.append(float(structure.identified and attribution["identified"]))
        attr_right += attribution["right"]
        attr_total += attribution["total"]
        req_right += structure.required_right
        req_total += structure.required_total
    return {
        "identified": float(np.mean(identified)),
        # Pooled over windows within the episode. A window with no latent group contributes
        # nothing to either side rather than a free 1.0 -- averaging per-window rates would
        # let unconfounded windows carry the number.
        "attr_right": float(attr_right), "attr_total": float(attr_total),
        "req_right": float(req_right), "req_total": float(req_total),
        "private_moves": float(private_moves), "moves": float(moves),
    }


def _tally(env: TwoAgentEnv, chosen, private: List[int], total: List[int]) -> None:
    """One round's moves, split private/shared, from a {agent: action_index} mapping."""
    for agent, index in chosen.items():
        node, _mode = env.windows[agent].actions[index]
        if node == PASS_ACTION:
            continue
        total[0] += 1
        private[0] += node not in env.windows[agent].shared


def play(env: TwoAgentEnv, policies, episodes: int, seed: int) -> List[Dict[str, float]]:
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    rows = []
    for episode in range(episodes):
        # The same expression `run_arm` uses, so an arm scored here and an arm scored there
        # see the same worlds.
        result = env.reset(seed=seed * 100_000 + episode)
        private_moves = moves = 0
        # The same split over SUBMITTED actions, kept only to show that it is a different
        # number. Under turn-taking two of every three submissions are discarded, so this
        # denominator is the agent count times too large -- it is the query-counting error,
        # measured rather than asserted.
        submitted_private, submitted = [0], [0]
        while not result.done:
            actions = {a: policies[a](env, result) for a in env.topology.agents}
            _tally(env, actions, submitted_private, submitted)
            result = env.step(actions)
            for agent, (node, _mode) in env.last_chosen.items():
                if node == PASS_ACTION:
                    continue
                moves += 1
                private_moves += node not in env.windows[agent].shared
        row = _episode_row(env, private_moves, moves)
        row["submitted_private"] = float(submitted_private[0])
        row["submitted"] = float(submitted[0])
        rows.append(row)
    return rows


def summarise(rows: List[Dict[str, float]]) -> Dict[str, float]:
    identified = np.array([r["identified"] for r in rows], dtype=float)
    moves = sum(r["moves"] for r in rows)
    return {
        "episodes": len(rows),
        "identified": float(identified.mean()),
        "identified_se": float(identified.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0,
        "attribution": _ratio(rows, "attr_right", "attr_total"),
        "structure": _ratio(rows, "req_right", "req_total"),
        "private_share": float(sum(r["private_moves"] for r in rows) / moves) if moves else float("nan"),
        "submitted_private_share": _ratio(rows, "submitted_private", "submitted"),
        "moves": float(moves),
    }


def _ratio(rows, numerator: str, denominator: str) -> float:
    bottom = sum(r[denominator] for r in rows)
    return float(sum(r[numerator] for r in rows) / bottom) if bottom else float("nan")


def paired(rows_a: List[Dict[str, float]], rows_b: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean and standard error of the PER-EPISODE difference a - b on identification."""
    delta = np.array([a["identified"] - b["identified"] for a, b in zip(rows_a, rows_b)])
    return {"delta": float(delta.mean()),
            "delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_agents", type=int, required=True)
    ap.add_argument("--private_size", type=int, default=2)
    ap.add_argument("--n_shared", type=int, default=3)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--n_obs", type=int, default=60)
    ap.add_argument("--n_int", type=int, default=20)
    ap.add_argument("--graph_model", default="sf", choices=["er", "sf"])
    ap.add_argument("--sf_m", type=int, default=2)
    ap.add_argument("--policy_arch", default="gnn")
    ap.add_argument("--per_agent_reward", action="store_true", default=True)
    ap.add_argument("--shared_reward", dest="per_agent_reward", action="store_false",
                    help="job 2's control arm: the all-agents conjunction pays everyone")
    # MEASURED 2026-08-27: UncertaintyGreedyAgent defaults to bar=0.7 while these
    # backends grade at claim_bar=1.0, so greedy stops scoring claims the task still
    # counts open. Worth +0.233 to greedy on scale-free at 4 agents. Exposed so the
    # baseline can be run at the bar it is actually graded on.
    ap.add_argument("--greedy_bar", type=float, default=0.7,
                    help="confidence bar for greedy_uncertainty; 1.0 matches the grading")
    ap.add_argument("--max_edges", type=int, default=None,
                    help="density guard; must match the training arm")
    ap.add_argument("--policy", default=None, help="a .pt from scripts/ma_train.py")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    env = build_env(args)
    agents = env.topology.agents

    arms = {}
    if args.policy:
        ppo = IndependentPPO.load(args.policy, env)
        arms["learned"] = ppo.policies(deterministic=False)
    # Constructed one at a time. `make_baselines` would also build `GreedyAgent`, whose
    # enumeration refuses past window size 5 -- job 3 runs at 6.
    arms["probe_then_work"] = {a: ProbeThenWorkAgent(a, args.seed) for a in agents}
    arms["greedy_uncertainty"] = {a: UncertaintyGreedyAgent(a, args.seed, bar=args.greedy_bar)
                                  for a in agents}
    arms["random_vary"] = {a: RandomAgent(a, args.seed, allow_clamp=False) for a in agents}

    report = {"config": vars(args), "arms": {}, "rows": {}, "paired": {}}
    for label, policies in arms.items():
        rows = play(env, policies, args.episodes, args.seed)
        report["arms"][label] = summarise(rows)
        report["rows"][label] = rows
        row = report["arms"][label]
        print(f"  {label:19s} identified {row['identified']:.3f} +/- {row['identified_se']:.3f}"
              f"  attribution {row['attribution']:.3f}  structure {row['structure']:.3f}"
              f"  private_share {row['private_share']:.3f}"
              f"  (submitted {row['submitted_private_share']:.3f})", flush=True)

    reference = "learned" if "learned" in report["rows"] else "probe_then_work"
    for label in report["rows"]:
        if label != reference:
            report["paired"][f"{reference}-{label}"] = paired(
                report["rows"][reference], report["rows"][label])
            row = report["paired"][f"{reference}-{label}"]
            print(f"  paired {reference} - {label:19s} {row['delta']:+.3f} +/- {row['delta_se']:.3f}",
                  flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
