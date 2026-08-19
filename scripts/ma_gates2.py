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

  GATE 3  coordination must be necessary AND available. On CONFOUNDED episodes, a
          never-clamping pair must fail and a forced-clamping pair must succeed. The gap
          between them is the headroom a learned policy competes for. No gap, no
          coordination problem, and the two-agent case collapses into two single-agent ones.

Usage:
    python scripts/ma_gates2.py --episodes 200 --budget 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from ma.baselines2 import ForcedClampAgent, GreedyAgent, RandomAgent, make_baselines
from ma.env2 import AGENTS, MA2Config, TwoAgentEnv2
from ma.evaluate2 import bootstrap_ci
from ma.projection import bidirected_pairs
from ma.topology import Topology
from sa.graphs import mec_signature


def singleton_fraction(env: TwoAgentEnv2, draws: int, seed: int) -> dict:
    """Prior-weighted fraction of episodes whose BOTH induced windows are alone in their
    equivalence class -- the asymptotic ceiling on observational-only identification."""
    rng = np.random.default_rng(seed)
    from ma.baselines2 import _Window

    hits = []
    for _ in range(draws):
        adjacency = env.topology.sample_dag(rng, p=env.config.prior_p)
        alone = True
        for name in AGENTS:
            window = env.windows[name]
            truth = window.induced(adjacency)
            target = mec_signature(truth)
            count = sum(1 for dag in _Window.get(window.k).dags
                        if mec_signature(dag) == target)
            if count != 1:
                alone = False
                break
        hits.append(float(alone))
    hits = np.asarray(hits)
    return {"estimate": float(hits.mean()), "ci": bootstrap_ci(hits, seed=seed)}


def play(env: TwoAgentEnv2, policies, episodes: int, seed: int, only=None):
    """Run episodes, optionally restricted to confounded / unconfounded ones."""
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    solved, clamps, moves = [], 0, 0
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        confounded = bool(bidirected_pairs(env.true_adjacency,
                                           env.topology.observed_by("A")))
        if only is not None and confounded != only:
            continue
        while not result.done:
            actions = {n: policies[n](env, result) for n in AGENTS}
            for name, index in actions.items():
                node, mode = env.windows[name].actions[index]
                if node != -1:
                    moves += 1
                    clamps += (mode == "clamp")
            result = env.step(actions["A"], actions["B"])
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disclose_regime", action="store_true",
                    help="with-bit arm; default is the no-bit BASELINE")
    ap.add_argument("--out", default="results/ma2/gates.json")
    args = ap.parse_args(argv)

    topology = Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    config = MA2Config(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                       budget=args.budget, disclose_regime=args.disclose_regime)
    env = TwoAgentEnv2(config)
    started = time.time()
    report = {"config": {"n_obs": args.n_obs, "n_int": args.n_int,
                         "budget": args.budget, "episodes": args.episodes,
                         "disclose_regime": args.disclose_regime}}

    # -- GATE 1 --------------------------------------------------------------------------
    observational = []
    for episode in range(args.episodes):
        result = env.reset(seed=args.seed * 7919 + episode)
        observational.append(float(result.info["both_identified"]))
    observational = np.asarray(observational)
    rate, ci = float(observational.mean()), bootstrap_ci(observational, seed=args.seed)
    target = singleton_fraction(env, args.draws, args.seed)
    passed1 = bool(ci[0] <= target["estimate"] <= ci[1]
                   or target["ci"][0] <= rate <= target["ci"][1])
    side = "leak (too easy)" if rate > target["estimate"] else "power (cannot concentrate)"
    report["gate1"] = {"rate": rate, "ci": ci, "target": target,
                       "passed": passed1, "failure_side": None if passed1 else side}
    print(f"GATE 1  observational {rate:.4f} CI {ci[0]:.4f}-{ci[1]:.4f}  "
          f"target {target['estimate']:.4f}  -> {'PASS' if passed1 else 'FAIL: ' + side}",
          flush=True)

    # -- GATE 2 --------------------------------------------------------------------------
    greedy = {n: GreedyAgent(n, env, seed=args.seed) for n in AGENTS}
    rand = {n: RandomAgent(n, seed=args.seed + 1, allow_clamp=True) for n in AGENTS}
    g2_greedy = play(env, greedy, args.episodes, args.seed, only=False)
    g2_random = play(env, rand, args.episodes, args.seed, only=False)
    passed2 = bool(g2_greedy["ci"][0] > g2_random["ci"][1])
    report["gate2"] = {"greedy": g2_greedy, "random_clamp": g2_random, "passed": passed2}
    print(f"GATE 2  unconfounded: greedy {g2_greedy['rate']:.3f} "
          f"CI {g2_greedy['ci'][0]:.3f}-{g2_greedy['ci'][1]:.3f}  vs random "
          f"{g2_random['rate']:.3f} CI {g2_random['ci'][0]:.3f}-{g2_random['ci'][1]:.3f}"
          f"  -> {'PASS' if passed2 else 'FAIL'}", flush=True)

    # -- GATE 3 --------------------------------------------------------------------------
    never = {n: RandomAgent(n, seed=args.seed + 2, allow_clamp=False) for n in AGENTS}
    forced = {n: ForcedClampAgent(n, seed=args.seed + 3) for n in AGENTS}
    g3_never = play(env, never, args.episodes, args.seed, only=True)
    g3_forced = play(env, forced, args.episodes, args.seed, only=True)
    headroom = g3_forced["rate"] - g3_never["rate"]
    passed3 = bool(headroom > 0 and g3_forced["ci"][0] > g3_never["ci"][1])
    report["gate3"] = {"never_clamp": g3_never, "forced_clamp": g3_forced,
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
