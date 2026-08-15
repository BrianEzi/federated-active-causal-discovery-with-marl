"""Train and evaluate the PPO agent against the pinned criteria.

Usage:
    python -m scripts.run_experiment --d 5 --observation edge_marginals --seeds 0 1 2 3 4

Every number printed comes from `sa.evaluate`, so what is reported and what was agreed in
docs/SA_PLAN.md cannot drift apart. References (random, greedy, edge-marginal greedy) are
computed once and shared across seeds so the comparison is like-for-like.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from sa.baselines import make_baselines
from sa.env import EnvConfig
from sa.evaluate import (
    check_criteria,
    evaluate,
    run_episodes,
    summarise_seeds,
)
from sa.graphs import build_graph_space
from sa.oracle import InterventionOracle
from sa.policy import PPOAgent, PPOConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--observation", type=str, default="edge_marginals",
                        choices=["posterior", "edge_marginals"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train_episodes", type=int, default=4000)
    parser.add_argument("--eval_episodes", type=int, default=300)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--tag", type=str, default="",
                        help="free-text label recorded in the output, e.g. the sweep arm")
    parser.add_argument("--ref_cache", type=str, default=None,
                        help="reuse reference-policy traces from this file, computing and "
                             "saving them if it does not exist")
    parser.add_argument("--refs_only", action="store_true",
                        help="compute and cache the references, then stop (no training)")

    # -- environment levers -----------------------------------------------------------
    env_group = parser.add_argument_group("environment")
    env_group.add_argument("--budget", type=int, default=20)
    env_group.add_argument("--n_obs", type=int, default=1000)
    env_group.add_argument("--n_int", type=int, default=100)
    env_group.add_argument("--identify_threshold", type=float, default=0.7)
    env_group.add_argument("--prior", type=str, default="erdos_renyi",
                           choices=["uniform", "erdos_renyi", "scale_free"])
    env_group.add_argument("--prior_p", type=float, default=0.5)
    env_group.add_argument("--intervene_scale", type=float, default=2.0)

    # -- agent levers -----------------------------------------------------------------
    ppo_group = parser.add_argument_group("agent")
    ppo_group.add_argument("--entropy_coef", type=float, default=0.003)
    ppo_group.add_argument("--lr", type=float, default=3e-4)
    ppo_group.add_argument("--step_cost", type=float, default=0.05)
    ppo_group.add_argument("--hidden", type=int, default=128)
    ppo_group.add_argument("--gamma", type=float, default=0.99)
    ppo_group.add_argument("--episodes_per_update", type=int, default=32)
    ppo_group.add_argument("--no_pass", action="store_true",
                           help="remove `pass` from the action space (makes the "
                                "under-acting criterion vacuous -- see sa/policy.py)")
    ppo_group.add_argument("--shaping_coef", type=float, default=0.0,
                           help="potential-based shaping on posterior entropy; "
                                "policy-invariant (Ng, Harada & Russell 1999)")
    args = parser.parse_args()

    space = build_graph_space(args.d)
    oracle = InterventionOracle(space)
    env_config = EnvConfig(
        d=args.d, budget=args.budget, n_obs=args.n_obs, n_int=args.n_int,
        identify_threshold=args.identify_threshold, prior=args.prior,
        prior_p=args.prior_p, intervene_scale=args.intervene_scale,
    )
    baselines = make_baselines(space, seed=0)

    print(f"d={args.d}  observation={args.observation}  "
          f"{space.n_dags} DAGs / {space.n_mecs} classes")

    # References, computed once and reused by every seed. They are recomputed for every
    # configuration rather than cached across the sweep, because environment levers
    # (budget, n_obs, prior, ...) move the baselines too -- gap-closed is only meaningful
    # against baselines measured in the SAME environment.
    #
    # At d=6 a single posterior update takes ~0.7s, which puts the four references at
    # roughly two hours -- far too much to repeat for every seed. `--ref_cache` computes
    # them once and shares them, which is safe precisely because they are deterministic
    # given the environment config and the fixed seed 99.
    refs = _load_ref_cache(args.ref_cache, env_config, args.eval_episodes)
    if refs is None:
        refs = {}
        for name in ("random", "greedy_oracle", "edge_marginal_greedy", "no_intervention"):
            t0 = time.time()
            refs[name] = run_episodes(env_config, baselines[name], args.eval_episodes,
                                      seed=99, space=space, oracle=oracle)
            print(f"  {name:<22} computed ({time.time() - t0:.0f}s)")
        _save_ref_cache(args.ref_cache, env_config, args.eval_episodes, refs)
    else:
        print(f"  references loaded from {args.ref_cache}")

    reference_metrics = {}
    for name, traces in refs.items():
        solved = float(np.mean([t.identified for t in traces]))
        cost = float(np.mean([t.n_interventions if t.identified else args.budget
                              for t in traces]))
        reference_metrics[name] = {"solve_rate": solved, "mean_cost": cost}
        print(f"  {name:<22} solve={solved:.2f} cost={cost:.2f}")

    if args.refs_only:
        print("--refs_only: references cached, stopping before training")
        return

    random_ref = refs["random"]
    # A condition-B agent is compared against the edge-marginal greedy policy, not the
    # full-posterior oracle -- otherwise the comparison conflates a worse policy with a
    # lossier belief.
    greedy_ref = refs["edge_marginal_greedy" if args.observation == "edge_marginals"
                      else "greedy_oracle"]

    per_seed = []
    histories = {}
    for seed in args.seeds:
        t0 = time.time()
        agent = PPOAgent(
            env_config,
            PPOConfig(observation=args.observation, total_episodes=args.train_episodes,
                      entropy_coef=args.entropy_coef, lr=args.lr,
                      step_cost=args.step_cost, hidden=args.hidden, gamma=args.gamma,
                      episodes_per_update=args.episodes_per_update,
                      allow_pass=not args.no_pass, shaping_coef=args.shaping_coef,
                      seed=seed),
            space=space,
        )
        history = agent.train()
        # Kept in full: the entropy and solve-rate trajectories are how a collapse is
        # diagnosed after the fact, and re-running to recover them costs a whole night.
        histories[str(seed)] = history
        per_seed_time = time.time() - t0

        deterministic = evaluate(env_config, agent.as_policy(True), random_ref, greedy_ref,
                                 args.eval_episodes, seed=99, space=space, oracle=oracle)
        sampled = evaluate(env_config, agent.as_policy(False), random_ref, greedy_ref,
                           args.eval_episodes, seed=99, space=space, oracle=oracle)
        verdict = check_criteria(deterministic, sampled)
        verdict["seed"] = seed
        verdict["train_seconds"] = per_seed_time
        verdict["final_entropy"] = float(history[-1]["entropy"]) if history else float("nan")
        verdict["deterministic"] = _serialisable(deterministic)
        verdict["sampled"] = _serialisable(sampled)
        per_seed.append(verdict)

        print(f"\nseed {seed} ({time.time() - t0:.0f}s, final entropy "
              f"{history[-1]['entropy']:.2f})")
        print(f"  gap_closed   {deterministic['gap_closed']:+.3f} "
              f"(sampled {sampled['gap_closed']:+.3f})")
        print(f"  solve_rate   {deterministic['solve_rate']:.2f} "
              f"(greedy {deterministic['greedy_solve_rate']:.2f})")
        print(f"  cost         {deterministic['mean_cost']:.2f} "
              f"CI {deterministic['cost_ci'][0]:.2f}-{deterministic['cost_ci'][1]:.2f}")
        print(f"  under_acting {deterministic['under_acting_rate']:.3f}")
        print(f"  optimal      {deterministic['optimal_rate']:.3f} "
              f"(informative {deterministic['informative_fraction']:.2f}) "
              f"regret {deterministic['mean_regret']:.3f}")
        print(f"  by_mec       " + "  ".join(
            f"{k}:{v['gap_closed']:+.2f}(n={v['n']})"
            for k, v in deterministic["by_mec_size"].items()))
        print(f"  checks       {verdict['checks']}  PASSED={verdict['passed']}")

    summary = summarise_seeds(per_seed)
    print(f"\n=== SUMMARY d={args.d} {args.observation} ===")
    print(f"  seeds passing: {summary['n_passed']}/{summary['n_seeds']}")
    print(f"  gap_closed  min {summary['min_gap_closed']:+.3f}  "
          f"median {summary['median_gap_closed']:+.3f}  "
          f"max {summary['max_gap_closed']:+.3f}")
    print(f"  OVERALL: {'PASS' if summary['passed'] else 'FAIL'}")

    if args.out:
        payload = {
            "args": vars(args),
            "tag": args.tag,
            "provenance": _provenance(),
            "space": {"d": args.d, "n_dags": space.n_dags, "n_mecs": space.n_mecs,
                      "singleton_fraction": space.singleton_fraction},
            "references": reference_metrics,
            "per_seed": per_seed,
            "training_history": histories,
            "summary": summary,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"  written to {args.out}")


def _ref_fingerprint(env_config, eval_episodes: int) -> str:
    """Everything the reference traces depend on.

    The references are baseline policies run in a specific environment, so reusing a cache
    built under different settings would silently compare an agent against the wrong
    opponent -- and gap-closed would look like a result rather than a bug. The cache
    therefore refuses to load unless this fingerprint matches exactly.
    """
    from dataclasses import asdict
    return json.dumps({"env": asdict(env_config), "eval_episodes": eval_episodes},
                      sort_keys=True, default=str)


def _load_ref_cache(path, env_config, eval_episodes):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        cached = pickle.load(f)
    if cached.get("fingerprint") != _ref_fingerprint(env_config, eval_episodes):
        raise SystemExit(
            f"reference cache {path} was built for a different configuration.\n"
            f"Delete it or point --ref_cache elsewhere; reusing it would compare the "
            f"agent against baselines measured in another environment."
        )
    return cached["refs"]


def _save_ref_cache(path, env_config, eval_episodes, refs) -> None:
    if not path:
        return
    payload = {"fingerprint": _ref_fingerprint(env_config, eval_episodes), "refs": refs}
    # Written to a unique temporary name and renamed, so a task that reads the cache while
    # another is still writing it sees either the old file or the complete new one, never
    # a half-written pickle. Array tasks start together and would otherwise race.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)
    print(f"  references cached to {path}")


def _serialisable(metrics: dict) -> dict:
    """Drop the raw `EpisodeTrace` objects, keeping every summary number.

    The traces are per-episode Python objects carried through `evaluate` for the
    stratified breakdowns; they are not JSON and re-deriving them is not needed, since
    every statistic computed from them is already in the dict alongside.
    """
    return {k: v for k, v in metrics.items() if not _has_traces(v)}


def _has_traces(value) -> bool:
    if isinstance(value, (list, tuple)):
        return any(hasattr(v, "n_interventions") for v in value)
    return hasattr(value, "n_interventions")


def _provenance() -> dict:
    """Enough to re-run this exact configuration months later."""
    import platform
    import subprocess

    import torch
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    # torch is recorded because the cluster and the laptop are NOT on the same version
    # (the cluster's package index tops out at 2.6.0+cpu, the laptop has 2.10.0+cpu), and
    # a numerical difference between the two would otherwise be invisible in the results.
    return {"git_commit": commit, "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "host": platform.node(),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


if __name__ == "__main__":
    main()
