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
from sa.gates import collect_canaries
from ma.graphs import build_graph_space
from sa.backend import Backend
from sa.oracle import InterventionOracle
from sa.policy import PPOAgent, PPOConfig
from sa.tracking import start_run


def build_parser() -> argparse.ArgumentParser:
    """Separated from `main` so a sweep definition can be checked against the REAL parser.

    Sweeps render command lines that nobody reads and that are then executed dozens of
    times; a mis-rendered flag does not crash, it silently runs a different experiment.
    Parsing the generated CLI back through this is what makes that detectable in a test.
    """
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
    parser.add_argument("--require_gate1", action="store_true",
                        help="refuse to train when GATE 1 fails, instead of warning")
    parser.add_argument("--gate1_episodes", type=int, default=200,
                        help="episodes for the GATE 1 precondition check; 0 to skip")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="log to this WandB project; off unless given. Writes offline "
                             "(compute nodes have no internet) -- sync afterwards with "
                             "scripts/sync_wandb.py from the login node")
    parser.add_argument("--wandb_dir", type=str, default=None,
                        help="where offline WandB runs are written (default: ./wandb)")

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
    env_group.add_argument("--include_counts", action="store_true",
                           help="append per-node intervention counts to the observation")

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
    ppo_group.add_argument("--arch", type=str, default="flat",
                           choices=["flat", "pernode"],
                           help="pernode = permutation-equivariant per-node scorer")
    ppo_group.add_argument("--layers", type=int, default=1,
                           help="rounds of neighbour aggregation in the per-node scorer. "
                                "1 reproduces the network behind the d=4/5/6 results "
                                "exactly; higher lets a node see further than one hop")
    ppo_group.add_argument("--shaping_coef", type=float, default=0.0,
                           help="potential-based shaping on posterior entropy; "
                                "policy-invariant (Ng, Harada & Russell 1999)")
    # default=None, NOT the store_true default of False. False would mean "force the
    # ENUMERATED path", which at d=7 means enumerating 1.14 billion DAGs -- a hang, then an
    # out-of-memory kill on a compute node, with no error message pointing at the flag.
    parser.add_argument("--force_dp", action="store_true", default=None,
                        help="use the subset-DP path even where enumeration is possible; "
                             "the d=6 control that validates any d=7 result")
    parser.add_argument("--oracle_draws", type=int, default=4000,
                        help="MH draws per sampled-oracle call (DP path only). 4000 gives "
                             "0.0103 nats regret at d=6 against an ideal-sampling floor")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    env_config_for_backend = EnvConfig(
        d=args.d, budget=args.budget, n_obs=args.n_obs, n_int=args.n_int,
        identify_threshold=args.identify_threshold, prior=args.prior,
        prior_p=args.prior_p, intervene_scale=args.intervene_scale,
        include_counts=args.include_counts,
    )
    # Enumerated below d=7, subset DP at and above it -- see sa/backend.py. `--force_dp`
    # exists so the DP path can be run at d=6, where the enumerated answer also exists and
    # can be compared against it. That control is what makes a d=7 number believable.
    backend = Backend(env_config_for_backend, force_dp=args.force_dp,
                      oracle_draws=args.oracle_draws, seed=args.seeds[0])
    space = backend.space
    oracle = backend.oracle
    env_config = EnvConfig(
        d=args.d, budget=args.budget, n_obs=args.n_obs, n_int=args.n_int,
        identify_threshold=args.identify_threshold, prior=args.prior,
        prior_p=args.prior_p, intervene_scale=args.intervene_scale,
        include_counts=args.include_counts,
    )
    baselines = backend.make_baselines(seed=0)

    print(backend.describe() + f"  observation={args.observation}")
    if args.observation not in backend.observation_kinds:
        raise SystemExit(
            f"observation={args.observation!r} is unavailable on this path; "
            f"choose from {backend.observation_kinds}")

    gate1 = _check_gate1(env_config, space, args.gate1_episodes, backend=backend)
    if gate1 is not None:
        status = "OK" if gate1["passed"] else "FAILS"
        print(f"  GATE 1 {status}: observational-only rate {gate1['rate']:.4f} "
              f"CI {gate1['ci'][0]:.4f}-{gate1['ci'][1]:.4f}, "
              f"target {gate1['target']:.4f}")
        if not gate1["passed"]:
            message = (
                f"GATE 1 FAILS at d={args.d}, n_obs={args.n_obs}: the observational-only "
                f"identification rate is {gate1['rate']:.4f} (CI {gate1['ci'][0]:.4f}-"
                f"{gate1['ci'][1]:.4f}) but the singleton fraction of the graph space is "
                f"{gate1['target']:.4f}. The environment does not match its specification "
                f"-- the agent starts from a blurrier belief than intended, and absolute "
                f"difficulty is not comparable across d. Raise n_obs."
            )
            if args.require_gate1:
                raise SystemExit(message)
            print("")
            print("  *** WARNING ***")
            print("  " + message)
            print("")

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
        # Driven by what the backend actually offers, not a fixed list: the DP path has no
        # `edge_marginal_greedy` yet, and a hardcoded name fails only AFTER the expensive
        # references have already been computed.
        for name in baselines:
            t0 = time.time()
            refs[name] = run_episodes(env_config, baselines[name], args.eval_episodes,
                                      seed=99, space=space, oracle=oracle,
                                      backend=backend)
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
        # Grouped so a 34-config x 3-seed sweep stays readable: seeds of one configuration
        # collapse into one group, and E1 vs E2 split on job_type.
        tracker = start_run(
            args.wandb_project,
            name=f"{args.tag or 'run'}_s{seed}",
            group=args.tag or None,
            job_type=args.arch,
            config={**vars(args), "seed": seed},
            tags=[f"d{args.d}", f"n_obs{args.n_obs}", args.arch, args.observation],
            directory=args.wandb_dir,
        )
        agent = PPOAgent(
            env_config,
            PPOConfig(observation=args.observation, total_episodes=args.train_episodes,
                      entropy_coef=args.entropy_coef, lr=args.lr,
                      step_cost=args.step_cost, hidden=args.hidden, gamma=args.gamma,
                      episodes_per_update=args.episodes_per_update,
                      allow_pass=not args.no_pass, shaping_coef=args.shaping_coef,
                      arch=args.arch, layers=args.layers,
                      seed=seed),
            space=space,
            backend=backend,
        )
        history = agent.train()
        # Kept in full: the entropy and solve-rate trajectories are how a collapse is
        # diagnosed after the fact, and re-running to recover them costs a whole night.
        histories[str(seed)] = history
        # Replayed after training rather than streamed from inside the PPO loop, so that
        # `sa/policy.py` stays free of any tracking dependency. The curves are identical;
        # only their arrival time differs, and nothing watches them live on a batch queue.
        for update, entry in enumerate(history):
            tracker.log({k: v for k, v in entry.items()
                         if isinstance(v, (int, float))}, step=update)
        per_seed_time = time.time() - t0

        deterministic = evaluate(env_config, agent.as_policy(True), random_ref, greedy_ref,
                                 args.eval_episodes, seed=99, space=space, oracle=oracle,
                           backend=backend)
        sampled = evaluate(env_config, agent.as_policy(False), random_ref, greedy_ref,
                           args.eval_episodes, seed=99, space=space, oracle=oracle,
                           backend=backend)
        verdict = check_criteria(deterministic, sampled)
        verdict["seed"] = seed
        verdict["train_seconds"] = per_seed_time
        verdict["final_entropy"] = float(history[-1]["entropy"]) if history else float("nan")
        verdict["deterministic"] = _serialisable(deterministic)
        verdict["sampled"] = _serialisable(sampled)
        per_seed.append(verdict)

        tracker.summarise({
            "gap_closed": deterministic["gap_closed"],
            "sampled_gap_closed": sampled["gap_closed"],
            "solve_rate": deterministic["solve_rate"],
            "greedy_solve_rate": deterministic["greedy_solve_rate"],
            "mean_cost": deterministic["mean_cost"],
            "under_acting_rate": deterministic["under_acting_rate"],
            "optimal_rate": deterministic["optimal_rate"],
            "informative_fraction": deterministic["informative_fraction"],
            "final_entropy": verdict["final_entropy"],
            "passed": verdict["passed"],
            "train_seconds": per_seed_time,
            "gate1_passed": None if gate1 is None else gate1["passed"],
        })
        tracker.finish()

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

    # Attached to every result, not run on demand. In each case the earlier failure was
    # not that a check failed -- it is that nobody thought to run it, so a number was read
    # without the thing that qualified it.
    n_actions = args.d + (0 if args.no_pass else 1)
    canaries = collect_canaries(per_seed, gate1, n_actions,
                                random_ref=random_ref, greedy_ref=greedy_ref,
                                budget=args.budget)
    print("\n=== CANARIES ===")
    for record in canaries:
        marker = "ok  " if record["ok"] else record["severity"].upper()
        print(f"  [{marker}] {record['name']}: {record['detail']}")
    fired = [r["name"] for r in canaries if not r["ok"]]
    if fired:
        print(f"  {len(fired)} fired: {', '.join(fired)} -- results below still stand as "
              f"recorded, but read them with these in view.")

    if args.out:
        payload = {
            "args": vars(args),
            "tag": args.tag,
            "provenance": _provenance(),
            # On the DP path there is no enumerated space, so the counts that describe it
            # do not exist. Recorded as null rather than omitted, so a reader can tell
            # "not applicable on this path" from "forgot to record it".
            "space": ({"d": args.d, "n_dags": space.n_dags, "n_mecs": space.n_mecs,
                       "singleton_fraction": space.singleton_fraction}
                      if space is not None else
                      {"d": args.d, "n_dags": None, "n_mecs": None,
                       "singleton_fraction": None, "path": "subset_dp"}),
            "references": reference_metrics,
            # Recorded on every run so a result can never be read without its validity
            # check alongside. This gate was pinned once at d=3 and silently stopped
            # holding at d>=5, which invalidated a night of environments before anyone
            # noticed.
            "gate1": gate1,
            "canaries": canaries,
            "per_seed": per_seed,
            "training_history": histories,
            "summary": summary,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"  written to {args.out}")


def _check_gate1(env_config, space, n_episodes: int, backend=None):
    """Does the environment satisfy GATE 1 -- is intervening actually necessary?

    The fraction of DAGs alone in their Markov equivalence class is exactly the fraction of
    problems solvable without intervening, and it is computable from the graph space. If
    the measured no-intervention rate sits above it, information is leaking; if below, there
    is not enough observational data to identify even the identifiable graphs.

    Run per training run rather than once per project, because the default n_obs=1000
    passed this at d=3 and d=4 and then silently failed at d=5 and d=6 -- the check had
    been performed once and assumed thereafter.
    """
    if n_episodes <= 0:
        return None
    from sa.baselines import no_intervention_policy
    from ma.stats import bootstrap_ci, run_policy

    outcome = run_policy(env_config, no_intervention_policy, n_episodes, seed=7,
                         space=space, backend=backend)
    rate = float(np.mean(outcome["identified"]))
    low, high = bootstrap_ci(outcome["identified"], seed=7)
    if space is not None:
        target = space.singleton_fraction
        target_ci = None
    else:
        # No DAG list: the target is estimated from prior samples via the covered-edge
        # test. It carries its own interval, so the gate must account for BOTH -- the
        # measured rate's uncertainty and the target's -- rather than treating an
        # estimate as if it were exact.
        from sa.gates import estimate_singleton_fraction
        estimate = estimate_singleton_fraction(
            env_config.d, p=env_config.prior_p, seed=7)
        target = estimate["estimate"]
        target_ci = list(estimate["ci"])
    passed = (bool(low <= target <= high) if target_ci is None
              else bool(low <= target_ci[1] and target_ci[0] <= high))
    return {"rate": rate, "ci": [low, high], "target": target, "target_ci": target_ci,
            "passed": passed, "n_episodes": n_episodes}


def _ref_fingerprint(env_config, eval_episodes: int) -> str:
    """Everything the reference traces depend on.

    The references are baseline policies run in a specific environment, so reusing a cache
    built under different settings would silently compare an agent against the wrong
    opponent -- and gap-closed would look like a result rather than a bug. The cache
    therefore refuses to load unless this fingerprint matches exactly.
    """
    from dataclasses import asdict
    fields = asdict(env_config)
    # `include_counts` only changes what `env.observation()` returns, and NO reference
    # policy calls it -- random draws from its RNG, no_intervention is constant, and both
    # greedy variants read `result.posterior` directly. So it cannot move the references,
    # and excluding it lets one cache serve both settings. Every other field can, so every
    # other field stays in.
    fields.pop("include_counts", None)
    return json.dumps({"env": fields, "eval_episodes": eval_episodes},
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
