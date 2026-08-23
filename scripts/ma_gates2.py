"""PHASE 3 -- the three gates, run before any RL.

A failing gate STOPS the phase. The single-agent rebuild exists because gates were skipped
once and a whole results table had to be thrown away.

  GATE 1  the task must require intervening.
          Observational-only joint identification must equal the prior-weighted fraction of
          window DAGs alone in their equivalence class. Both sides are computable, so this
          is a predicted number rather than a vibe.

          NOTE the two-sidedness. A rate ABOVE target is a leak -- the task is solvable
          without acting. A rate BELOW target means the posterior cannot concentrate even
          where the graph is uniquely identifiable, which is a finite-sample power problem,
          not a leak. Measured 2026-08-19: at n_obs=100 the single-agent rate is 0.000
          against 0.0892 and the best episode of 150 reached 0.579 mass against a 0.7
          threshold. Both failures are reported, and they are NOT the same finding.

  GATE 2  choices must matter. random_clamp clearly worse than greedy on UNCONFOUNDED
          episodes, with non-overlapping intervals. Run at a tight budget: the greedy-random
          gap is entirely a budget-scarcity effect and vanishes by budget ~10, so a slack
          budget would pass this gate while measuring nothing.

  GATE 3  coordination must be necessary AND available. On CONFOUNDED episodes, a pair that
          CANNOT clamp (vary-only) must do worse than a pair that can. The gap is the
          headroom a learned policy competes for. No gap, no coordination problem, and the
          two-agent case collapses into two single-agent ones.

          The upper arm is a MIXED policy, not an always-clamping one. Always clamping your
          own private node destroys your own boundary information -- see the comment at the
          arm itself. That is a finding, not a technicality: pure altruism is dominated, so
          the coordination problem is one of TIMING, not of willingness.

Usage:
    python scripts/ma_gates2.py --episodes 200 --budget 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from ma.baselines import ForcedClampAgent, GreedyAgent, RandomAgent, make_baselines
from ma.env import MAConfig, TwoAgentEnv
from ma.evaluate import bootstrap_ci
from ma.projection import bidirected_pairs
from ma.topology import Topology, two_agent
from ma.graphs import mec_signature


def singleton_fraction(env: TwoAgentEnv, draws: int, seed: int) -> dict:
    """Prior-weighted fraction of episodes whose BOTH induced windows are alone in their
    equivalence class -- the asymptotic ceiling on observational-only identification."""
    rng = np.random.default_rng(seed)
    from ma.baselines import _Window

    hits = []
    for _ in range(draws):
        adjacency = env.topology.sample_dag(rng, p=env.config.prior_p)
        alone = True
        for agent in env.topology.agents:
            window = env.windows[agent]
            space = _Window.get(window.k)
            # Class SIZE from the precomputed partition. This was a full 543-graph
            # signature pass per draw per agent -- 10,000 passes at --draws 5000.
            class_id = space.id_of(window.induced(adjacency))
            if class_id < 0 or space.mec_size[class_id] != 1:
                alone = False
                break
        hits.append(float(alone))
    hits = np.asarray(hits)
    return {"estimate": float(hits.mean()), "ci": bootstrap_ci(hits, seed=seed)}


def play(env: TwoAgentEnv, policies, episodes: int, seed: int, only=None):
    """Run episodes, optionally restricted to confounded / unconfounded ones."""
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    solved, clamps, moves = [], 0, 0
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        confounded = any(bool(bidirected_pairs(env.true_adjacency,
                                               env.topology.observed_by(agent)))
                         for agent in env.topology.agents)
        if only is not None and confounded != only:
            continue
        while not result.done:
            actions = {a: policies[a](env, result) for a in env.topology.agents}
            for agent, index in actions.items():
                node, mode = env.windows[agent].actions[index]
                if node != -1:
                    moves += 1
                    clamps += (mode == "clamp")
            result = env.step(actions)
        solved.append(float(result.info["both_identified"]))
    solved = np.asarray(solved)
    return {"n": int(len(solved)),
            "rate": float(solved.mean()) if len(solved) else float("nan"),
            "ci": bootstrap_ci(solved, seed=seed),
            "clamp_fraction": float(clamps / moves) if moves else float("nan")}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=100)
    # Tight on purpose: discrimination peaks at budget 2-3 and is gone by 16.
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--draws", type=int, default=3000)
    # GATE 3 runs at a LARGER budget than GATE 2, and that is a structural finding rather
    # than a fudge. The two gates ask opposite questions of the budget: GATE 2 needs it
    # TIGHT, because greedy-vs-random discrimination peaks at 2-3 and is gone by 16;
    # GATE 3 needs it LOOSE, because a confounded episode requires an agent to spend moves
    # clamping for its partner AND moves experimenting on itself. Measured: on confounded
    # episodes nothing solves at all below budget 5, and random reaches 0.444 only by
    # budget 16. Running both gates at one budget guarantees one of them is uninformative.
    ap.add_argument("--gate3_budget", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disclose_regime", action="store_true",
                    help="with-bit arm; default is the no-bit BASELINE")
    ap.add_argument("--out", default="results/ma2/gates.json")
    args = ap.parse_args(argv)

    topology = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                       budget=args.budget, disclose_regime=args.disclose_regime)
    env = TwoAgentEnv(config)
    started = time.time()
    report = {"config": {"n_obs": args.n_obs, "n_int": args.n_int,
                         "budget": args.budget, "episodes": args.episodes,
                         "disclose_regime": args.disclose_regime}}

    # -- GATE 1 --------------------------------------------------------------------------
    # RUN UNDER A DAG-ONLY RULE, and this is the resolution of three failed attempts rather
    # than a convenience.
    #
    # GATE 1 asks a question about the ENVIRONMENT -- is the task solvable without acting?
    # Its target is the prior-weighted fraction of windows alone in their Markov equivalence
    # class, which is derived under a DAG model. `joint_conf` does not use a DAG model: its
    # hypotheses are (DAG, confounding set) pairs, and WITHOUT CLEAN ROWS THE CONFOUNDING
    # LABEL IS UNFALSIFIABLE -- any extra edge can be added and called confounding for the
    # price of a BGe penalty. So observational-only identification under joint_conf is near
    # zero BY DESIGN, and comparing it to a DAG-derived target measures the rule, not the
    # environment.
    #
    # Measured, all three criteria tried, unconfounded episodes against a target of 0.0402:
    #   P(H == truth)                    0.0256   too harsh   (confounded agent always 0)
    #   P(H \ P == truth)                0.2387   leak        (unfalsifiable labels credited)
    #   P(H \ P == truth AND P correct)  0.0000   too harsh   (must rule out 24 rivals blind)
    # Under `pooled`, a genuine DAG model, the same environment gives 0.0547 against 0.0442.
    #
    # So the gate runs on `pooled`. What joint_conf does to identification is a separate
    # question, measured separately, and not a property of the environment.
    gate1_env = TwoAgentEnv(MAConfig(
        topology=topology, n_obs=args.n_obs, n_int=args.n_int, budget=args.budget,
        disclose_regime=args.disclose_regime, score_rule="pooled"))
    # CONDITIONED ON UNCONFOUNDED EPISODES, and this is a correction to the gate rather
    # than a convenience. A confounded window is a latent projection, not a DAG, so its
    # true DAG is NOT identifiable from observational data at any sample size -- measured
    # 0.0000 at n_obs=30000, unchanged from n_obs=3000. The singleton-MEC target is
    # computed under a DAG model and therefore only predicts the unconfounded subset.
    # Comparing it to the pooled rate guarantees a spurious failure of exactly the
    # confounded fraction.
    observational, unconfounded_only = [], []
    for episode in range(args.episodes):
        result = gate1_env.reset(seed=args.seed * 7919 + episode)
        confounded = any(bool(bidirected_pairs(gate1_env.true_adjacency,
                                               gate1_env.topology.observed_by(agent)))
                         for agent in gate1_env.topology.agents)
        observational.append(float(result.info["both_identified"]))
        if not confounded:
            unconfounded_only.append(float(result.info["both_identified"]))
    pooled_rate = float(np.mean(observational))
    observational = np.asarray(unconfounded_only)
    rate, ci = float(observational.mean()), bootstrap_ci(observational, seed=args.seed)
    target = singleton_fraction(gate1_env, args.draws, args.seed)
    passed1 = bool(ci[0] <= target["estimate"] <= ci[1]
                   or target["ci"][0] <= rate <= target["ci"][1])
    side = "leak (too easy)" if rate > target["estimate"] else "power (cannot concentrate)"
    report["gate1"] = {"rate": rate, "ci": ci, "target": target,
                       "pooled_rate": pooled_rate, "n_unconfounded": len(observational),
                       "passed": passed1, "failure_side": None if passed1 else side}
    print(f"GATE 1  observational (unconfounded, n={len(observational)}) {rate:.4f} "
          f"CI {ci[0]:.4f}-{ci[1]:.4f}  target {target['estimate']:.4f}  "
          f"[pooled incl. confounded {pooled_rate:.4f}]  "
          f"-> {'PASS' if passed1 else 'FAIL: ' + side}", flush=True)

    # -- GATE 2 --------------------------------------------------------------------------
    greedy = {a: GreedyAgent(a, env, seed=args.seed) for a in env.topology.agents}
    rand = {a: RandomAgent(a, seed=args.seed + 1, allow_clamp=True) for a in env.topology.agents}
    g2_greedy = play(env, greedy, args.episodes, args.seed, only=False)
    g2_random = play(env, rand, args.episodes, args.seed, only=False)
    passed2 = bool(g2_greedy["ci"][0] > g2_random["ci"][1])
    report["gate2"] = {"greedy": g2_greedy, "random_clamp": g2_random, "passed": passed2}
    print(f"GATE 2  unconfounded: greedy {g2_greedy['rate']:.3f} "
          f"CI {g2_greedy['ci'][0]:.3f}-{g2_greedy['ci'][1]:.3f}  vs random "
          f"{g2_random['rate']:.3f} CI {g2_random['ci'][0]:.3f}-{g2_random['ci'][1]:.3f}"
          f"  -> {'PASS' if passed2 else 'FAIL'}", flush=True)

    # -- GATE 3 --------------------------------------------------------------------------
    # The upper arm is random_clamp, NOT forced_clamp. Measured 2026-08-19: an agent that
    # clamps its own private node EVERY round never learns that node's parents, because a
    # constant carries no information about what drives it -- and those boundary edges are
    # part of its OWN success criterion. Traced directly: A rises 0.368 -> 0.814 while B,
    # clamping every round, stays at 0.04 and never identifies. So "always clamp" is
    # self-defeating and cannot be the coordination ceiling. What is needed is a MIX --
    # clamp some rounds, experiment on others -- which is what random_clamp does and what
    # a learned policy would have to discover.
    g3_env = TwoAgentEnv(MAConfig(
        topology=topology, n_obs=args.n_obs, n_int=args.n_int,
        budget=args.gate3_budget, disclose_regime=args.disclose_regime))
    never = {a: RandomAgent(a, seed=args.seed + 2, allow_clamp=False) for a in env.topology.agents}
    forced = {a: RandomAgent(a, seed=args.seed + 3, allow_clamp=True) for a in env.topology.agents}
    g3_never = play(g3_env, never, args.episodes, args.seed, only=True)
    g3_forced = play(g3_env, forced, args.episodes, args.seed, only=True)
    headroom = g3_forced["rate"] - g3_never["rate"]
    passed3 = bool(headroom > 0 and g3_forced["ci"][0] > g3_never["ci"][1])
    report["gate3"] = {"budget": args.gate3_budget,
                       "never_clamp": g3_never, "forced_clamp": g3_forced,
                       "headroom": headroom, "passed": passed3}
    print(f"GATE 3  confounded (n={g3_never['n']}): never-clamp {g3_never['rate']:.3f} "
          f"vs forced-clamp {g3_forced['rate']:.3f}  headroom {headroom:+.3f}"
          f"  -> {'PASS' if passed3 else 'FAIL'}", flush=True)

    report["all_passed"] = bool(passed1 and passed2 and passed3)
    report["seconds"] = time.time() - started
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nALL GATES: {'PASS' if report['all_passed'] else 'FAIL'}  "
          f"[{report['seconds']:.0f}s] -> {out}")
    return report


if __name__ == "__main__":
    main()
