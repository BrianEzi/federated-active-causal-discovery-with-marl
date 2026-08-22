"""GATE 2's failure: is it collision, and is collision fixable without communication?

GATE 2 asks that a good experiment selector beat a bad one. At two agents it FAILS -- the
myopic oracle scores no better than random, and it has now been reproduced independently
inside the training evaluation, so it is not a sampling accident.

The live hypothesis is COLLISION: both agents compute the same objective over overlapping
authority, pick the same shared target, and waste the round.

This measures three arms on UNCONFOUNDED episodes at a tight budget, where the gate runs:

    random          the floor
    greedy          tie broken at random, per agent -- the arm that fails the gate
    greedy_split    A takes the lowest-indexed tied action, B the highest

`greedy_split` is a diagnostic, not a proposed method. It uses no communication: each agent
applies a fixed convention to its OWN action list, nothing crosses the federation boundary.
If collision is the cause, its collision rate drops and its success rises.

It does not. Measured 2026-08-20, and the reason is the third measurement here: the argmax
set. A tie-break can only separate two agents when they HAVE a tie, and at the node level
they almost never do -- the argmax node set is a singleton in ~94% of rounds. The 2-element
action set that shows up almost every round is VARY and CLAMP on the SAME node, which is the
already-documented indifference between the two modes, not a choice between targets.

So collisions are not coin flips that happened to land the same way. Both agents
independently compute a UNIQUE best target and it is the same target, because the objective
is identical and the shared variables are visible to both. No local convention can separate
them, and that is what settles GATE 2: myopic design does not fail here by accident, it
fails because a one-step information criterion has no term for what the partner needs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from ma.baselines import GreedyAgent, RandomAgent
from ma.env import AGENTS, MAConfig, TwoAgentEnv
from ma.evaluate import bootstrap_ci
from ma.projection import bidirected_pairs
from ma.topology import Topology, two_agent


def argmax_diagnostic(env, episodes, seed):
    """How often is the greedy agent's best TARGET actually tied, and are collisions ties?

    This is the measurement that decides whether a tie-break convention could ever have
    worked. Reported alongside the arms so the negative result above is falsifiable rather
    than asserted.
    """
    from ma.baselines import _partition_entropy, enumerated_posterior

    agents = {n: GreedyAgent(n, env, seed=seed) for n in AGENTS}

    def best_set(agent):
        window = env.windows[agent.name]
        clean = (env.clean[agent.name] if env.config.disclose_regime
                 else np.zeros(len(env.samples), dtype=bool))
        post = enumerated_posterior(window, env.samples[:, window.nodes],
                                    env.known[agent.name], clean, agent.rule)
        scores = np.full(len(agent.candidates), -np.inf)
        for slot, action in enumerate(agent.candidates):
            node, _mode = window.actions[action]
            position = window.pos[node]
            scores[slot] = _partition_entropy(
                agent.space.signatures[:, position], post,
                int(agent.space.signatures[:, position].max()) + 1)
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        nodes = {window.actions[agent.candidates[int(s)]][0] for s in best}
        return len(nodes), window.actions[agent.candidates[int(best[0])]][0]

    node_tied = rounds = collisions = collisions_tied = 0
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            info = {n: best_set(agents[n]) for n in AGENTS}
            for n in AGENTS:
                node_tied += (info[n][0] > 1)
            rounds += 1
            if info["A"][1] == info["B"][1]:
                collisions += 1
                collisions_tied += (min(info["A"][0], info["B"][0]) > 1)
            actions = {n: agents[n](env, result) for n in AGENTS}
            result = env.step(actions["A"], actions["B"])
    return {
        "rounds": rounds,
        "fraction_with_a_target_tie": float(node_tied / max(2 * rounds, 1)),
        "collisions": collisions,
        "collision_rate": float(collisions / max(rounds, 1)),
        # The number that matters: a tie-break can only ever touch this fraction.
        "collisions_where_both_had_a_tie": collisions_tied,
        "fixable_by_tie_break": float(collisions_tied / max(collisions, 1)),
    }


def play(env, policies, episodes, seed, only=None):
    """Like `scripts.ma_gates2.play`, but also counts how often the two agents collide."""
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    solved, rounds, collisions = [], 0, 0
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        confounded = bool(bidirected_pairs(env.true_adjacency,
                                           env.topology.observed_by(0)))
        if only is not None and confounded != only:
            continue
        while not result.done:
            actions = {n: policies[n](env, result) for n in AGENTS}
            targets = {}
            for name, index in actions.items():
                node, _mode = env.windows[name].actions[index]
                targets[name] = node
            # A collision is two agents spending the round on the SAME variable. Passing
            # (-1) is not a collision -- neither agent spent anything.
            if targets["A"] != -1 and targets["A"] == targets["B"]:
                collisions += 1
            if targets["A"] != -1 or targets["B"] != -1:
                rounds += 1
            result = env.step(actions["A"], actions["B"])
        solved.append(float(result.info["both_identified"]))
    solved = np.asarray(solved)
    return {"n": int(len(solved)),
            "rate": float(solved.mean()) if len(solved) else float("nan"),
            "ci": bootstrap_ci(solved, seed=seed),
            "collision_rate": float(collisions / rounds) if rounds else float("nan"),
            "rounds": rounds}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=100)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disclose_regime", action="store_true")
    ap.add_argument("--out", default="results/ma_fixed/gate2_collision.json")
    args = ap.parse_args(argv)

    topology = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    env = TwoAgentEnv(MAConfig(
        topology=topology, n_obs=args.n_obs, n_int=args.n_int, budget=args.budget,
        disclose_regime=args.disclose_regime))

    arms = {
        "random": {n: RandomAgent(n, seed=args.seed + 1, allow_clamp=True) for n in AGENTS},
        "greedy": {n: GreedyAgent(n, env, seed=args.seed) for n in AGENTS},
        "greedy_split": {"A": GreedyAgent("A", env, seed=args.seed, tie_break="low"),
                         "B": GreedyAgent("B", env, seed=args.seed, tie_break="high")},
    }

    started = time.time()
    report = {"config": {"episodes": args.episodes, "budget": args.budget,
                         "n_obs": args.n_obs, "disclose_regime": args.disclose_regime},
              "arms": {}}
    for label, policies in arms.items():
        row = play(env, policies, args.episodes, args.seed, only=False)
        report["arms"][label] = row
        print("%-13s n=%d  solve %.3f  CI %.3f-%.3f  collisions %.3f of %d rounds"
              % (label, row["n"], row["rate"], row["ci"][0], row["ci"][1],
                 row["collision_rate"], row["rounds"]), flush=True)

    report["argmax"] = argmax_diagnostic(env, min(args.episodes, 80), args.seed)
    a = report["argmax"]
    print("\nargmax: a target-level tie in %.3f of decisions; %d of %d collisions had one "
          "-- a tie-break can reach %.3f of them"
          % (a["fraction_with_a_target_tie"], a["collisions_where_both_had_a_tie"],
             a["collisions"], a["fixable_by_tie_break"]), flush=True)

    g, gs, r = (report["arms"][k] for k in ("greedy", "greedy_split", "random"))
    report["collision_drop"] = g["collision_rate"] - gs["collision_rate"]
    report["solve_gain"] = gs["rate"] - g["rate"]
    # The gate's own bar: beating random means non-overlapping intervals, not a better mean.
    report["split_passes_gate2"] = bool(gs["ci"][0] > r["ci"][1])
    report["seconds"] = time.time() - started

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("\ncollision %.3f -> %.3f (%+.3f)   solve %.3f -> %.3f (%+.3f)   "
          "beats random: %s  [%.0fs] -> %s"
          % (g["collision_rate"], gs["collision_rate"], -report["collision_drop"],
             g["rate"], gs["rate"], report["solve_gain"],
             report["split_passes_gate2"], report["seconds"], out))
    return report


if __name__ == "__main__":
    main()
