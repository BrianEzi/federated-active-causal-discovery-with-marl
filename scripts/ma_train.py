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
from ma.env import (CLAMP, MODES, SIMULTANEOUS, TURN_ORDERS, VARY,
                    MAConfig, TwoAgentEnv)
from ma.evaluate import run_arm
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import Topology, federated_topology, two_agent


def _config_record(config, topology, args) -> dict:
    """The RESOLVED settings, read off the config object rather than off `args`.

    `prior_p` is derived from `d` when left unset and `identify_threshold` has no flag at
    all, so recording the arguments would record neither -- which is what once made result
    files unattributable. One function so the JSON report and the W&B run config cannot
    drift apart.
    """
    return {"n_obs": config.n_obs, "n_int": config.n_int, "budget": config.budget,
            "rule": config.score_rule,
            "disclose_regime": config.disclose_regime,
            "turn_order": config.turn_order,
            "action_modes": list(config.action_modes),
            "prior_p": config.prior_p,
            "graph_model": config.graph_model,
            "vs_evidence": config.vs_evidence,
            "vs_evidence_alpha": config.vs_evidence_alpha,
            "sf_m": config.sf_m,
            "identify_threshold": config.identify_threshold,
            "intervene_scale": config.intervene_scale,
            "reward_criterion": config.reward_criterion,
            "belief_backend": config.belief_backend,
            "policy_arch": config.policy_arch,
            "cb_n_boot": config.cb_n_boot,
            "cb_alpha": config.cb_alpha,
            "episode_mix": config.episode_mix,
            "oracle_obs_structure": config.oracle_obs_structure,
            "claim_bar": config.claim_bar,
            "claim_penalty": config.claim_penalty,
            "per_agent_reward": config.per_agent_reward,
            "observe_belief_channels": config.observe_belief_channels,
            "observe_partner_counts": config.observe_partner_counts,
            "mode_by_role": config.mode_by_role,
            "claims_require_all_types": config.claims_require_all_types,
            "topology": {"name": topology.name, "d": topology.d,
                         "private": [list(p) for p in topology.private],
                         "exposed": list(topology.exposed)},
            "train_episodes": args.train_episodes,
            "potential_shaping": args.potential_shaping,
            "step_cost": config.step_cost}


def _wandb_run(args, config_record: dict):
    """Open a W&B run, or return None. Telemetry must never be able to kill a training run,
    so every failure here -- missing package, unwritable dir, no credentials -- degrades to
    a printed warning and `None`."""
    if args.no_wandb or args.wandb_mode == "disabled":
        return None
    import os
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, name=f"{args.arm}_s{args.seed}",
                         group=args.arm, job_type="train_eval", mode=args.wandb_mode,
                         config={**config_record, "arm": args.arm, "seed": args.seed},
                         reinit=True)
        print(f"  wandb {args.wandb_mode}: {run.dir}", flush=True)
        return run
    except Exception as exc:                                    # noqa: BLE001
        print(f"  [wandb] disabled -- {type(exc).__name__}: {exc}", flush=True)
        return None


def _wandb_logger(run):
    if run is None:
        return None

    def log(record: dict) -> None:
        try:
            run.log({"train/solve_rate": record["solve_rate"],
                     "train/entropy": record["entropy"],
                     "train/mask_pass": float(record["mask_pass"]),
                     "train/update": record["update"]},
                    step=int(record["update"]))
        except Exception:                                       # noqa: BLE001
            pass
    return log


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
    # Vary-only, ADOPTED 2026-08-24 for constraint arms after the measured comparison
    # (mean credit 0.60/0.51 vs clamp's 0.50/0.32 on 12 known-graph episodes): randomised
    # values give the engine first-order power. See SA_EXPERIMENT_LOG 2026-08-24.
    ap.add_argument("--vary_only", action="store_true",
                    help="restrict the action space to vary; the clamp mode is removed")
    # The two new axes (2026-08-24). Both recorded in the report config and in the
    # checkpoint, and refused on mismatch at load -- performance belongs to the
    # (policy, backend, arch) triple.
    ap.add_argument("--backend", default="exact",
                    choices=["exact", "constraint", "version_space", "attributed", "factored"])
    ap.add_argument("--claim_bar", type=float, default=None,
                    help="confidence bar per claim; version_space requires 1.0")
    ap.add_argument("--per_agent_reward", action="store_true",
                    help="pay each agent for its own window, not the all-agents conjunction")
    ap.add_argument("--observe_belief_channels", action="store_true",
                    help="show the policy its confounding and adjacency beliefs, not just "
                         "directed-edge frequencies (found 2026-08-26: without this the "
                         "learner cannot see the channel its reward is scored on)")
    ap.add_argument("--observe_partner_counts", action="store_true",
                    help="cumulative per-partner intervention counts (shared nodes by "
                         "node, private ones as an unnamed total). Converts per-round "
                         "disclosure the feedforward policy cannot retain into memory")
    ap.add_argument("--mode_by_role", action="store_true",
                    help="clamp on own private nodes, vary on shared ones; one action per "
                         "node. Overrides --clamp_only/--vary_only")
    ap.add_argument("--legacy_claim_exemption", action="store_true",
                    help="restore the pre-2026-08-26 grading in which shared-block "
                         "directions were not required. For reproduction only")
    ap.add_argument("--graph_model", default="er", choices=["er", "sf"],
                    help="er: each allowed pair with prob prior_p. sf: scale-free by "
                         "preferential attachment, density set by --sf_m")
    ap.add_argument("--sf_m", type=int, default=2,
                    help="parents per node under --graph_model sf; prior_p is ignored")
    ap.add_argument("--vs_evidence", default="oracle", choices=["oracle", "sampled"],
                    help="deterministic backends: consult the true graph (oracle) or the "
                         "DATA (sampled). With 'sampled', --n_int is the noise dial")
    ap.add_argument("--vs_evidence_alpha", type=float, default=0.001,
                    help="evidence threshold; 1e-3 measured optimal (2026-08-27) -- "
                         "stricter is NOT safer, power-based pruning takes over")
    ap.add_argument("--policy_arch", default="mlp", choices=["mlp", "gnn", "gnn_portable"])
    ap.add_argument("--cb_n_boot", type=int, default=12,
                    help="bootstrap replicates per refresh (constraint backend only)")
    ap.add_argument("--gnn_layers", type=int, default=2)
    ap.add_argument("--n_agents", type=int, default=None,
                    help="N agents with one private node each and three shared; overrides "
                         "--three_agents. The measured coordination headroom peaks near "
                         "N=4 for three shared nodes (docs/FINDINGS_2026_08_26.md).")
    # PRIVATE SETS >= 2 (2026-08-26). Needed for two independent reasons: at ONE private
    # node per agent the disclosure-privacy claim is empty (agent identity is node
    # identity, so "I intervened privately" names the node), and the diversity story needs
    # windows that are not all k=4. Cost ceiling: the version space enumerates
    # 3^(edges in window), so k <= 6 is the usable range -- private_size 2 with 3 shared
    # nodes is k=5, private_size 3 with 3 shared is k=6 and dense windows there are slow.
    ap.add_argument("--private_size", type=int, default=1,
                    help="private nodes per agent (with --n_agents)")
    ap.add_argument("--n_shared", type=int, default=3,
                    help="shared/exposed nodes (with --n_agents)")
    ap.add_argument("--three_agents", action="store_true",
                    help="rung 1: three agents, one private node each, three shared. "
                         "Runs ONLY on the constraint backend (widest_hidden = 2).")
    # The Day-1 redesign axes (2026-08-24). "claims" is the training criterion for
    # constraint arms; "confounded" episodes are what the thesis is about, with the
    # unconfounded arm kept as the standing sanity check.
    ap.add_argument("--reward_criterion", default=None,
                    choices=["u14", "identified", "claims"],
                    help="unset keeps MAConfig's default (u14)")
    ap.add_argument("--episode_mix", default="any",
                    choices=["any", "confounded", "unconfounded"])
    ap.add_argument("--oracle_obs", action="store_true",
                    help="oracle warm start: agents receive the true observational-limit "
                         "structure of their window; interventions are the whole task")
    # None (unset) means "let MAConfig resolve it" -- 2 ln(d)/d since 2026-08-22. Exposed
    # explicitly so a run can be PINNED to the old fixed value, which is what rung 0 of the
    # n-agent refactor needs: isolate the topology refactor from the prior change by holding
    # this constant while comparing against pre-refactor results measured at p=0.5.
    ap.add_argument("--prior_p", type=float, default=None,
                    help="override the graph prior; unset resolves to 2 ln(d)/d")
    # Both ported from sa/policy.py, 2026-08-22 -- see ma/policy.py::PPOConfig for the
    # measured justification. Off/default until a comparison confirms they help HERE too.
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--orthogonal_init", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out even if it holds a result from a different config")
    # Live telemetry, ON by default since 2026-08-25 and OFFLINE by default: a compute node
    # has no outbound internet, and an online init there blocks the run. `wandb sync` from
    # a machine that does have it, later. `scripts/ma_wandb_sync.py` remains the
    # reproducible post-hoc path -- this one is for watching a run while it is alive.
    ap.add_argument("--wandb_project", default="ma-two-agent")
    ap.add_argument("--wandb_mode", default="offline",
                    choices=["offline", "online", "disabled"])
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.n_agents:
        topology = federated_topology(args.n_agents, args.private_size, args.n_shared)
    elif args.three_agents:
        topology = Topology(name="T_3agent_1each",
                            private=((0,), (1,), (2,)), exposed=(3, 4, 5))
    else:
        topology = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,),
                             exposed=(2, 3, 4))
    if args.clamp_only and args.vary_only:
        raise SystemExit("--clamp_only and --vary_only are mutually exclusive")
    if args.clamp_only:
        modes = (CLAMP,)
    elif args.vary_only:
        modes = (VARY,)
    else:
        modes = MODES
    config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                       budget=args.budget, disclose_regime=args.disclose_regime,
                       score_rule=args.rule, step_cost=args.step_cost,
                       turn_order=args.turn_order, action_modes=modes,
                       prior_p=args.prior_p, graph_model=args.graph_model,
                       sf_m=args.sf_m, vs_evidence=args.vs_evidence,
                       vs_evidence_alpha=args.vs_evidence_alpha,
                       belief_backend=args.backend, cb_n_boot=args.cb_n_boot,
                       policy_arch=args.policy_arch, episode_mix=args.episode_mix,
                       oracle_obs_structure=args.oracle_obs,
                       **({"claim_bar": args.claim_bar} if args.claim_bar else {}),
                       per_agent_reward=args.per_agent_reward,
                       observe_belief_channels=args.observe_belief_channels,
                       observe_partner_counts=args.observe_partner_counts,
                       mode_by_role=args.mode_by_role,
                       claims_require_all_types=not args.legacy_claim_exemption,
                       **({"reward_criterion": args.reward_criterion}
                          if args.reward_criterion else {}))
    env = TwoAgentEnv(config)
    config_record = _config_record(config, topology, args)
    run = _wandb_run(args, config_record)
    started = time.time()

    ppo = IndependentPPO(env, PPOConfig(
        total_episodes=args.train_episodes, seed=args.seed,
        potential_shaping=args.potential_shaping,
        entropy_coef=args.entropy_coef, orthogonal_init=args.orthogonal_init,
        mask_pass_updates=args.mask_pass_updates, gnn_layers=args.gnn_layers))
    history = ppo.train(verbose=True, on_update=_wandb_logger(run))
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
        "config": config_record,
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
    reference = {agent: make_baselines(env, agent, seed=args.seed) for agent in env.topology.agents}
    # A baseline appears only where it has legal moves and a working oracle: the random
    # arms follow the action modes, and `greedy` reads the exact DP's score tables so it
    # exists only on the exact backend (`enumerated_posterior` raises otherwise -- a
    # constraint-side greedy is a separate design, see cb/backend.py).
    labels = ["pass"]
    if CLAMP in modes:
        labels.insert(0, "random_clamp")
    if VARY in modes:
        labels.insert(0, "random_vary")
    if args.backend == "exact":
        labels.insert(-1, "greedy")
    else:
        labels.insert(-1, "greedy_uncertainty")
    if args.backend == "attributed":
        # The attribution-aware reference. Without it the learner is compared only against
        # arms that are not scored on the thing it is trained for.
        labels.insert(-1, "probe_then_work")
    for label in labels:
        arms[label] = {agent: reference[agent][label] for agent in env.topology.agents}

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

    if run is not None:
        try:
            summary = {"train_seconds": train_seconds,
                       "first_success_episode": ppo.first_success_episode,
                       "final_entropy": report["final_entropy"],
                       "collapsed": report["collapsed"]}
            for label, row in report["arms"].items():
                for field in ("success", "mean_steps", "clamp_fraction",
                              "union_acyclic", "union_equivalent"):
                    if row.get(field) is not None:
                        summary[f"eval/{label}/{field}"] = row[field]
                if row.get("success_ci"):
                    summary[f"eval/{label}/success_lo"] = row["success_ci"][0]
                    summary[f"eval/{label}/success_hi"] = row["success_ci"][1]
            # The number every arm exists to answer: how far the learned policy is ahead
            # of the greedy it has to beat.
            greedy = report["arms"].get("greedy_uncertainty") or report["arms"].get("greedy")
            if greedy is not None:
                summary["eval/margin_over_greedy"] = learned["success"] - greedy["success"]
            run.summary.update(summary)
            run.finish()
        except Exception as exc:                                # noqa: BLE001
            print(f"  [wandb] summary skipped -- {type(exc).__name__}: {exc}", flush=True)

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
