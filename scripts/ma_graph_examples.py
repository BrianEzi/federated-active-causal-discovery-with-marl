"""Extract worked examples: the true graph, what each agent concluded, and what it did.

The success rate says how often the agents are right. It does not say what "right" LOOKS
like, which is the first thing anyone asks. This dumps whole episodes in enough detail to
draw them:

    the true global DAG, and which shared pairs are genuinely confounded for each agent
    each agent's window, and the true structure induced on it
    each agent's MAP hypothesis -- the causal graph AND the confounding set it claims
    the posterior mass it put on its credit set, against the 0.7 bar
    the union of the two answers, and whether it is acyclic and globally equivalent
    the actual move sequence, so the clamping behaviour is visible rather than inferred

WHY THE MAP IS OVER (DAG, ASSIGNMENT) PAIRS. Under `joint_conf` a hypothesis is a DAG H
together with a set P of pairs declared confounded, whose edges are present in H. The
agent's CAUSAL claim is `H \\ P`. Reporting H alone would draw confounding edges as if they
were causal ones -- the exact confusion that produced a metric scoring 0.000 on every
confounded episode. So the joint grid is reconstructed and the MAP taken over pairs.

Episodes are selected to be worth looking at rather than uniformly: the output holds
confounded successes, unconfounded successes, and failures, because a page of eight
identical wins teaches nothing about the method.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np

from ma.baselines2 import _PerDagIndex, _Window, enumerated_posterior
from ma.belief_dp import JOINT_CONF
from ma.env2 import AGENTS, MA2Config, TwoAgentEnv2
from ma.evaluate2 import credit_set, evaluate_episode, union_graph
from ma.policy2 import IndependentPPO2, MA2PPOConfig
from ma.projection import bidirected_pairs
from ma.topology import Topology
from sa.graphs import is_acyclic, mec_signature


def joint_grid(env: TwoAgentEnv2, name: str) -> np.ndarray:
    """Normalised posterior over (assignment, DAG), shape [n_assign, n_dags].

    Rebuilds the same grid `enumerated_posterior` marginalises away, because the identity
    of the winning ASSIGNMENT is the whole point here -- it is the agent's claim about what
    is confounded.
    """
    window = env.windows[name]
    belief = window.belief
    space = _Window.get(window.k)
    samples = env.samples[:, window.nodes]
    known = env.known[name]
    clean = (env.clean[name] if env.config.disclose_regime
             else np.zeros(len(env.samples), dtype=bool))
    index = _PerDagIndex.get(window.k, belief.assignments, belief.scorer)
    nodes = np.arange(window.k)

    clean_table = belief.local_table(samples, known, clean)
    dirty_table = belief.local_table(samples, known, ~clean)
    dirty_part = dirty_table[nodes[None, :], index.own].sum(axis=1)

    rows = []
    for a in range(len(belief.assignments)):
        clean_part = clean_table[nodes[None, :], index.stripped[:, a, :]].sum(axis=1)
        row = clean_part + dirty_part
        row[~index.compatible[:, a]] = -np.inf
        rows.append(row)
    grid = np.vstack(rows)
    grid = grid - grid.max()
    weights = np.exp(grid)
    return weights / weights.sum()


def describe_agent(env: TwoAgentEnv2, name: str) -> Dict:
    """One agent's answer, in a form that can be drawn."""
    window = env.windows[name]
    space = _Window.get(window.k)
    truth = window.induced(env.true_adjacency)

    grid = joint_grid(env, name)
    a_star, d_star = np.unravel_index(int(np.argmax(grid)), grid.shape)
    assignment = env.windows[name].belief.assignments[a_star]
    hypothesis = np.asarray(space.dags[d_star], dtype=int)

    # The CAUSAL claim is H minus the confounding edges, never H itself.
    causal = hypothesis.copy()
    claimed = []
    for edge in assignment:
        if edge is not None:
            causal[edge[0], edge[1]] = 0
            claimed.append([int(edge[0]), int(edge[1])])

    posterior = enumerated_posterior(
        window, env.samples[:, window.nodes], env.known[name],
        env.clean[name] if env.config.disclose_regime
        else np.zeros(len(env.samples), dtype=bool),
        env.config.score_rule)
    credit = credit_set(window, truth)
    pairs = env._confounded_positions(name)
    mass = float(window.belief.joint_conf_set_probability(
        env.samples[:, window.nodes], env.known[name],
        env.clean[name] if env.config.disclose_regime
        else np.zeros(len(env.samples), dtype=bool),
        space.dags[credit], pairs))

    return {
        "nodes": [int(n) for n in window.nodes],
        "private": [int(n) for n in window.private],
        "shared": [int(n) for n in window.shared],
        "truth": truth.astype(int).tolist(),
        "true_confounded_pairs": [[int(u), int(v)] for u, v in pairs],
        "map_hypothesis": hypothesis.tolist(),
        "map_causal": causal.tolist(),
        "claimed_confounded": claimed,
        "map_joint_probability": float(grid[a_star, d_star]),
        "credit_mass": mass,
        "credit_ok": bool(mass >= env.config.identify_threshold),
        "equivalent_to_truth": bool(mec_signature(causal > 0.5) == mec_signature(truth > 0.5)),
        "causal_exact": bool(np.array_equal(causal, truth.astype(int))),
        # THE CONFOUNDING CLAIM IS PART OF THE ANSWER, NOT AN ANNOTATION ON IT.
        #
        # Reporting only whether the causal graph matched overstates the agent: it can get
        # H \ P exactly right and still be wrong, because identification requires the
        # claim about WHAT IS CONFOUNDED too. Caught on episode 0 of the first extraction,
        # where B's MAP causal graph was exact and its mass on (true DAG, true confounding)
        # was 5e-12 -- a correct failure that the display was about to render as a success.
        "confounding_exact": bool(
            {tuple(sorted(p)) for p in claimed}
            == {tuple(sorted(map(int, p))) for p in pairs}),
        "window_is_confounded": bool(len(pairs) > 0),
        "map_index": int(d_star),
    }


def run_episode(env: TwoAgentEnv2, policies, seed: int) -> Dict:
    result = env.reset(seed=seed)
    # PER AGENT. The episode-level flag used to be computed from A's observed set alone and
    # then labelled "confounded", which mislabels every episode confounded only for B --
    # the same "a number that means something narrower than its name" failure this project
    # has been chasing all month, resurfacing in the display layer.
    confounded_by = {n: bool(bidirected_pairs(env.true_adjacency,
                                              env.topology.observed_by(n)))
                     for n in AGENTS}
    confounded = any(confounded_by.values())
    moves: List[Dict] = []
    while not result.done:
        actions = {n: policies[n](env, result) for n in AGENTS}
        row = {}
        for name, index in actions.items():
            node, mode = env.windows[name].actions[index]
            row[name] = {"node": None if node == -1 else int(node),
                         "mode": "pass" if node == -1 else mode}
        moves.append(row)
        result = env.step(actions["A"], actions["B"])

    report = evaluate_episode(env)
    agents = {name: describe_agent(env, name) for name in AGENTS}
    union = union_graph(env, {n: agents[n]["map_index"] for n in AGENTS})
    return {
        "seed": seed,
        "confounded": confounded,
        "confounded_by": confounded_by,
        "true_adjacency": np.asarray(env.true_adjacency, dtype=int).tolist(),
        "topology": {"a_private": [int(x) for x in env.topology.a_private],
                     "b_private": [int(x) for x in env.topology.b_private],
                     "exposed": [int(x) for x in env.topology.exposed]},
        "agents": agents,
        "union": np.asarray(union, dtype=int).tolist(),
        "union_acyclic": bool(is_acyclic(union)),
        "union_equivalent": bool(report["union_equivalent"]),
        "success": bool(report["success"]),
        "moves": moves,
        "n_moves": sum(1 for m in moves
                       for name in AGENTS if m[name]["node"] is not None),
        "clamps": sum(1 for m in moves
                      for name in AGENTS if m[name]["mode"] == "clamp"),
    }


def pick(episodes: List[Dict], want: int) -> List[Dict]:
    """A spread worth looking at, not the first n.

    A page of identical wins teaches nothing, so take confounded successes first (the case
    the design exists for), then unconfounded successes, then failures -- and keep at least
    one failure if one exists, because a report with no failures in it invites the question
    of what was left out.
    """
    conf_ok = [e for e in episodes if e["confounded"] and e["success"]]
    unconf_ok = [e for e in episodes if not e["confounded"] and e["success"]]
    failed = [e for e in episodes if not e["success"]]
    out = conf_ok[:max(want // 2, 1)] + unconf_ok[:max(want // 4, 1)]
    out += failed[:max(want - len(out), 1)]
    return out[:want]


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="results/ma_examples/withbit_demo_s0.pt")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=100)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--out", default="results/ma_examples/examples.json")
    args = ap.parse_args(argv)

    topology = Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    env = TwoAgentEnv2(MA2Config(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                                 budget=args.budget, disclose_regime=True))
    learner = IndependentPPO2.load(args.checkpoint, env)
    policies = learner.policies(deterministic=False)

    episodes = []
    for i in range(args.episodes):
        episodes.append(run_episode(env, policies, seed=900_000 + i))
        if (i + 1) % 20 == 0:
            print("  %d/%d episodes" % (i + 1, args.episodes), flush=True)

    chosen = pick(episodes, args.keep)
    summary = {
        "n_episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "confounded_fraction": float(np.mean([e["confounded"] for e in episodes])),
        "success_when_confounded": float(np.mean(
            [e["success"] for e in episodes if e["confounded"]] or [np.nan])),
        "success_when_unconfounded": float(np.mean(
            [e["success"] for e in episodes if not e["confounded"]] or [np.nan])),
        "mean_moves": float(np.mean([e["n_moves"] for e in episodes])),
        "clamp_fraction": float(np.sum([e["clamps"] for e in episodes])
                                / max(np.sum([e["n_moves"] for e in episodes]), 1)),
    }
    out = {"checkpoint": args.checkpoint, "summary": summary, "examples": chosen}
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n%d episodes  success %.3f  (confounded %.3f / unconfounded %.3f)  "
          "clamps %.3f of %.1f moves"
          % (summary["n_episodes"], summary["success_rate"],
             summary["success_when_confounded"], summary["success_when_unconfounded"],
             summary["clamp_fraction"], summary["mean_moves"]))
    print("kept %d examples -> %s" % (len(chosen), path))
    return out


if __name__ == "__main__":
    main()
