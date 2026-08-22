"""Train one two-agent seed and evaluate it against every baseline.

One seed per invocation so the cluster array can be one task per (arm, seed) and a partial
failure is legible. Arms are `nobit` (the baseline, no regime disclosure) and `withbit`.

Every comparison holds the belief rule FIXED. Cross-rule numbers are void: a
joint_conf-trained policy scored under `subset` collapses below random, and greedy drops
0.542 -> 0.190 on the same switch. Performance belongs to the (policy, rule) PAIR.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from ma.baselines import make_baselines
from ma.env import (AGENTS, CLAMP, MODES, SIMULTANEOUS, TURN_ORDERS,
                     MAConfig, TwoAgentEnv)
from ma.evaluate import run_arm
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import Topology, two_agent


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="nobit")
    ap.add_argument("--disclose_regime", action="store_true")
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=100)
    # ROUNDS for the whole system, a shared pool -- NOT interventions per agent. Semantics
    # changed 2026-08-21, see docs/TURN_BUDGET_SPEC.md section 2.
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--train_episodes", type=int, default=4000)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--rule", default="joint_conf")
    ap.add_argument("--potential_shaping", type=float, default=0.0)
    ap.add_argument("--mask_pass_updates", type=int, default=0)
    # DEFAULT ZERO since 2026-08-21. At 0.05 a random-level policy has expected value
    # -0.255 against 0.000 for passing, so PASSING WAS OPTIMAL and every recorded collapse
    # was the agent being correct. Coupled to the absence of voluntary termination -- see
    # docs/TURN_BUDGET_SPEC.md section 5 before changing either.
    ap.add_argument("--step_cost", type=float, default=0.0)
    # Protocol. The default stays `simultaneous` so that re-running an old command
    # reproduces the old number; turn-taking is opted into explicitly, and the choice is
    # recorded in the report so no two numbers can be compared across protocols by accident.
    ap.add_argument("--turn_order", default=SIMULTANEOUS, choices=list(TURN_ORDERS))
    # Clamp-only. Measured 2026-08-21: costs at most ~4pp against keeping both modes
    # (paired, 8/10 seeds favour both, CI [-0.005, +0.041]) and buys a halved action space.
    # A trade, not a demonstration that vary is useless.
    ap.add_argument("--clamp_only", action="store_true",
                    help="restrict the action space to clamps; the vary mode is removed")
    # None (unset) means "let MAConfig resolve it" -- 2 ln(d)/d since 2026-08-22. Exposed
    # explicitly so a run can be PINNED to the old fixed value, which is what rung 0 of the
    # n-agent refactor needs: isolate the topology refactor from the prior change by holding
    # this constant while comparing against pre-refactor results measured at p=0.5.
    ap.add_argument("--prior_p", type=float, default=None,
                    help="override the graph prior; unset resolves to 2 ln(d)/d")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out even if it holds a result from a different config")
    args = ap.parse_args(argv)

    topology = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    modes = (CLAMP,) if args.clamp_only else MODES
    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                       budget=args.budget, disclose_regime=args.disclose_regime,
                       score_rule=args.rule, step_cost=args.step_cost,
                       turn_order=args.turn_order, action_modes=modes,
                       prior_p=args.prior_p)
    env = TwoAgentEnv(config)
    started = time.time()

    ppo = IndependentPPO(env, PPOConfig(
        total_episodes=args.train_episodes, seed=args.seed,
        potential_shaping=args.potential_shaping,
        mask_pass_updates=args.mask_pass_updates))
    history = ppo.train(verbose=True)
    train_seconds = time.time() - started
    # Persist the trained pair. Ten seeds were previously evaluated and discarded because
    # nothing wrote them out, so any question about what an agent LEARNED needed a retrain.
    if args.out:
        checkpoint = pathlib.Path(args.out).with_suffix(".pt")
        ppo.save(checkpoint)
        print(f"  saved policy pair -> {checkpoint}", flush=True)

    report = {
        "arm": args.arm, "seed": args.seed,
        # Read off the RESOLVED config, not off `args`. `prior_p` is derived from `d` when
        # it is left unset, and `identify_threshold` has no CLI flag at all, so logging the
        # arguments would have recorded neither -- and the 2026-08-22 prior change is
        # exactly the kind of thing that later makes a results file unattributable.
        # Same lesson as "log the raw quantity, never the verdict".
        "config": {"n_obs": config.n_obs, "n_int": config.n_int, "budget": config.budget,
                   "rule": config.score_rule,
                   "disclose_regime": config.disclose_regime,
                   "turn_order": config.turn_order,
                   "action_modes": list(config.action_modes),
                   "prior_p": config.prior_p,
                   "identify_threshold": config.identify_threshold,
                   "intervene_scale": config.intervene_scale,
                   "reward_criterion": config.reward_criterion,
                   "topology": {"name": topology.name, "d": topology.d,
                                "a_private": list(topology.private[0]),
                                "b_private": list(topology.private[1]),
                                "exposed": list(topology.exposed)},
                   "train_episodes": args.train_episodes,
                   "potential_shaping": args.potential_shaping,
                   "step_cost": config.step_cost},
        "train_seconds": train_seconds,
        # The collapse diagnostic. A seed that never sampled the terminal reward has a
        # different problem from one that sampled it and could not exploit it.
        "first_success_episode": ppo.first_success_episode,
        "final_entropy": history[-1]["entropy"] if history else None,
        # Full trace, so the report can plot learning curves rather than parsing stdout.
        "history": history,
        "arms": {},
    }

    arms = {"learned": ppo.policies(deterministic=False)}
    reference = {name: make_baselines(env, name, seed=args.seed) for name in AGENTS}
    # `random_vary` has no legal actions in the clamp-only arm, so it is dropped rather
    # than reported as an empty comparison. Its absence is visible in the report's arm list.
    labels = ["random_clamp", "greedy", "pass"]
    if not args.clamp_only:
        labels.insert(1, "random_vary")
    for label in labels:
        arms[label] = {name: reference[name][label] for name in AGENTS}

    for label, policies in arms.items():
        t0 = time.time()
        report["arms"][label] = run_arm(env, policies, args.eval_episodes, seed=args.seed)
        report["arms"][label]["seconds"] = time.time() - t0
        row = report["arms"][label]
        print(f"  {label:13s} success {row['success']:.3f} "
              f"CI {row['success_ci'][0]:.3f}-{row['success_ci'][1]:.3f}  "
              f"steps {row['mean_steps']:.2f}  clamp {row['clamp_fraction']:.3f}",
              flush=True)

    # The canary that caught a dead run before: a policy that never acts reports a clamp
    # fraction of nan and a success rate that says nothing about choice quality.
    learned = report["arms"]["learned"]
    report["collapsed"] = bool(learned["mean_steps"] < 1.5)
    if report["collapsed"]:
        print("  [CANARY] learned policy is under-acting -- mean_steps < 1.5, so this seed "
              "collapsed into passing rather than learning.", flush=True)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        _refuse_to_clobber_a_different_config(out, report["config"], args.force)
        out.write_text(json.dumps(report, indent=1))
        print(f"wrote {out}")
    return report


def _refuse_to_clobber_a_different_config(out: pathlib.Path, config: dict, force: bool):
    """Abort rather than overwrite a result produced under DIFFERENT settings.

    Written after a real incident on 22 August 2026. An overnight local launcher and a
    Myriad array both wrote `results/ma_fixed/tb_both_s15..19.json` -- same arm name, same
    seeds, but `step_cost` 0.05 against 0.0. The local job finished second and silently
    replaced five already-committed results with runs from a different configuration. It
    surfaced only because `git status` showed modified files that nothing had edited.

    The arm name is not a configuration fingerprint, and treating it as one is what made the
    collision silent. Same name plus same seed plus different config is always a mistake:
    either the name is too coarse or the wrong command is being run.

    Re-running the SAME configuration is fine and stays silent -- that is an ordinary
    recompute, and blocking it would make reruns annoying enough to be worked around.
    """
    if force or not out.exists():
        return
    try:
        existing = json.loads(out.read_text()).get("config")
    except (json.JSONDecodeError, OSError):
        return                      # unreadable or half-written: overwriting is the fix
    if existing is None or existing == config:
        return
    differing = {k: (existing.get(k), config.get(k))
                 for k in set(existing) | set(config)
                 if existing.get(k) != config.get(k)}
    lines = ["REFUSING to overwrite %s" % out,
             "  It holds a result from a DIFFERENT configuration:"]
    for k, (a, b) in sorted(differing.items()):
        lines.append("    %s: on disk %r -> would become %r" % (k, a, b))
    lines.append("  Same arm name and seed, different settings. Either give this run its "
                 "own --out,")
    lines.append("  or pass --force if you really mean to replace the existing result.")
    raise SystemExit(chr(10).join(lines))


if __name__ == "__main__":
    main()
