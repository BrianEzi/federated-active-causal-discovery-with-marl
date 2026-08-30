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

from ma.baselines import (GreedyAgent, PassAgent, ProbeThenWorkAgent, RandomAgent,
                          UncertaintyGreedyAgent)
from ma.density_guard import DensityGuardedEnv
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

    EVERY FIELD, NOT A HAND-MAINTAINED LIST. The explicit dict below is kept for ordering
    and for the derived entries, but every remaining `MAConfig` field is swept in
    automatically. The list drifted before: 329 of 436 result files on disk record no
    `vs_evidence` at all, so nobody could tell afterwards whether they were oracle or
    sampled -- and that is the single field that decides what a number means. A field added
    to MAConfig now appears here without anyone remembering to add it.
    """
    record = {"n_obs": config.n_obs, "n_int": config.n_int, "budget": config.budget,
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
            "difference_reward": config.difference_reward,
            "difference_reward_mode": config.difference_reward_mode,
            "reward_scale": config.reward_scale,
            "observe_belief_channels": config.observe_belief_channels,
            "observe_partner_counts": config.observe_partner_counts,
            "mode_by_role": config.mode_by_role,
            "claims_require_all_types": config.claims_require_all_types,
            "topology": {"name": topology.name, "d": topology.d,
                         "private": [list(p) for p in topology.private],
                         "exposed": list(topology.exposed)},
            "train_episodes": args.train_episodes,
            "potential_shaping": args.potential_shaping,
            "step_cost": config.step_cost,
            # THE OPTIMISER SETTINGS, added 2026-08-28. Until now none of these were
            # recorded, so a result file could not say what PPO was actually run with --
            # which made adding a seed to an old rung a matter of trusting that the
            # defaults had been used. `scripts/train_from_config.py` reconstructs a
            # training command from this block and can only check what is in it.
            "entropy_coef": args.entropy_coef,
            "orthogonal_init": args.orthogonal_init,
            "gnn_layers": args.gnn_layers,
            "turn_aware_credit": args.turn_aware_credit,
            "normalise_returns": args.normalise_returns,
            "mask_pass_updates": args.mask_pass_updates,
            "max_edges": args.max_edges}
    # EVERY REMAINING MAConfig FIELD, swept in rather than listed. See the docstring.
    from dataclasses import fields as _fields
    for field in _fields(config):
        record.setdefault(field.name, getattr(config, field.name))
    record["topology"] = {"name": topology.name,
                          "private": [list(b) for b in topology.private],
                          "exposed": list(topology.exposed)}
    # The sweep axes, recorded so a result file can be placed on the (k, sigma, n, beta)
    # grid without re-deriving them from the topology -- and so that `sigma` is visible at
    # a glance, which is what would have revealed that w04 sits at 0.75 while every other
    # window rung sits at 0.50.
    k = len(topology.private[0]) + len(topology.exposed)
    record["k"] = k
    record["sigma_contended"] = len(topology.exposed) / k if k else 0.0
    record["n_agents"] = topology.n_agents
    return _jsonable(record)


def _jsonable(value):
    """Config values must survive `json.dumps` -- tuples, sets and numpy scalars do not."""
    import numpy as _np
    if isinstance(value, dict):
        return {key: _jsonable(v) for key, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, _np.generic):
        return value.item()
    return value


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
    ap.add_argument("--difference_reward_mode", default="both",
                    choices=("both", "delta", "bonus"),
                    help="which half of the difference reward to apply: 'delta' gates the "
                         "dense term to whoever moved, 'bonus' replaces the outcome bonus "
                         "with own causal contribution, 'both' does each.")
    ap.add_argument("--reward_scale", type=float, default=1.0,
                    help="uniform multiplier on the per-agent reward; the control that "
                         "separates credit assignment from policy/value loss balance.")
    ap.add_argument("--difference_reward", action="store_true",
                    help="pay each agent for the credit ITS OWN interventions caused, "
                         "instead of for the state its window happens to be in. Under the "
                         "plain reward an agent's return tracks its PARTNERS' contribution "
                         "more closely than its own at every agent count (3.2x at eight) "
                         "and negatively at two and three. Factored backend only.")
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
    ap.add_argument("--policy_arch", default="mlp", choices=["mlp", "gnn", "gnn_portable", "gnn_solo", "gnn_hybrid"])
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
    # Density guard. Measured 2026-08-27: at four agents the attributed episode cost is
    # heavy-TAILED -- median under a second, one draw in five at 48 s, one that did not
    # finish in twenty minutes -- and the cost is exponential in the window's EDGE count.
    # Rejecting dense draws changes the episode distribution, so the rejection rate is
    # reported and no guarded result above three agents may be quoted without the
    # three-agent guarded/unguarded control beside it.
    ap.add_argument("--normalise_returns", action="store_true",
                    help="divide rewards by a running estimate of the discounted-return "
                         "scale, so the critic's MSE target stays O(1) as the return grows "
                         "with agent count. The principled form of --reward_scale: same "
                         "mechanism, no hand-picked constant.")
    ap.add_argument("--turn_aware_credit", action="store_true",
                    help="one transition per agent per TURN, not per round; rewards "
                         "from rounds it did not act in accrue onto its most recent action")
    ap.add_argument("--max_edges", type=int, default=None,
                    help="reject draws whose densest window MAG has more than this many "
                         "edges; unset disables the guard")
    ap.add_argument("--out", default=None)
    # CHECKPOINTING. See `ma/checkpoints.py` for why the schedule is log-spaced rather than
    # uniform and why the best policy is chosen on the MI gate rather than on reward.
    ap.add_argument("--checkpoint_updates", default=None,
                    help="comma-separated update indices for EVAL checkpoints; default is "
                         "the log-spaced schedule, dense early")
    ap.add_argument("--resume_every", type=int, default=50,
                    help="write restartable state this often; 0 disables")
    ap.add_argument("--keep_resume", type=int, default=2)
    ap.add_argument("--mi_episodes", type=int, default=8,
                    help="episodes used to RANK checkpoints by MI. The certifying "
                         "measurement is scripts/mi_gate.py, run afterwards on the winner")
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
                       difference_reward=args.difference_reward,
                       difference_reward_mode=args.difference_reward_mode,
                       reward_scale=args.reward_scale,
                       observe_belief_channels=args.observe_belief_channels,
                       observe_partner_counts=args.observe_partner_counts,
                       mode_by_role=args.mode_by_role,
                       claims_require_all_types=not args.legacy_claim_exemption,
                       **({"reward_criterion": args.reward_criterion}
                          if args.reward_criterion else {}))
    # `--max_edges` routes through the density guard; without it this is the ordinary
    # environment, and DensityGuardedEnv(max_edges=None) is its parent's behaviour exactly,
    # so both halves of the three-agent guarded/unguarded control share one code path.
    env = DensityGuardedEnv(config, max_edges=args.max_edges)
    config_record = _config_record(config, topology, args)
    run = _wandb_run(args, config_record)
    started = time.time()

    ppo = IndependentPPO(env, PPOConfig(
        total_episodes=args.train_episodes, seed=args.seed,
        potential_shaping=args.potential_shaping,
        entropy_coef=args.entropy_coef, orthogonal_init=args.orthogonal_init,
        turn_aware_credit=args.turn_aware_credit,
        normalise_returns=args.normalise_returns,
        mask_pass_updates=args.mask_pass_updates, gnn_layers=args.gnn_layers))
    # Checkpointing rides the existing `on_update` hook, so the training loop is untouched.
    # Both callbacks fire; a checkpointing failure is swallowed inside the writer so it can
    # never take down a run that has already spent hours of compute.
    from ma.checkpoints import CheckpointWriter, default_schedule
    # Mirrors IndependentPPO.train's own derivation; there is no n_updates on the config.
    n_updates = max(1, ppo.config.total_episodes // ppo.config.episodes_per_update)
    schedule = (default_schedule(n_updates) if args.checkpoint_updates is None
                else [int(x) for x in args.checkpoint_updates.split(",") if x.strip()])
    writer = None
    if args.out:
        writer = CheckpointWriter(
            ppo, env, pathlib.Path(args.out), n_updates=n_updates,
            schedule=schedule, resume_every=args.resume_every,
            keep_resume=args.keep_resume, mi_episodes=args.mi_episodes, seed=args.seed,
            log=lambda msg: print(msg, flush=True))
    wandb_hook = _wandb_logger(run)

    def _on_update(record):
        # `_wandb_logger` returns None when tracking is off, which is the common case.
        if wandb_hook is not None:
            wandb_hook(record)
        if writer is not None:
            writer(record)

    history = ppo.train(verbose=True, on_update=_on_update)
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
        # Where every checkpoint went, and which one the MI gate ranked highest. Recorded
        # so a reader can find and rank them without listing the directory or re-deriving
        # the schedule -- and so the FINAL policy is never quoted by default when a better
        # one exists, which measured runs show it often does.
        "checkpoints": writer.manifest() if writer is not None else None,
        "arms": {},
    }

    arms = {"learned": ppo.policies(deterministic=False)}
    # ONE CONSTRUCTOR PER LABEL, and the labels are chosen BEFORE anything is built.
    # `make_baselines` built all of them eagerly, `GreedyAgent` among them, and that
    # constructor enumerates the window and REFUSES past size 5. So at --private_size 3
    # (k=6) a run died before playing a single episode, on a backend where the greedy
    # oracle arm is not in `labels` at all -- measured 2026-08-27 on the attribution
    # timing probe. Building only what will be scored removes the whole class of failure.
    factories = {
        "pass": lambda agent: PassAgent(agent, args.seed),
        "random_vary": lambda agent: RandomAgent(agent, args.seed, allow_clamp=False),
        "random_clamp": lambda agent: RandomAgent(agent, args.seed, allow_clamp=True),
        "greedy": lambda agent: GreedyAgent(agent, env, args.seed),
        "greedy_uncertainty": lambda agent: UncertaintyGreedyAgent(agent, args.seed),
        "probe_then_work": lambda agent: ProbeThenWorkAgent(agent, args.seed),
    }
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
        arms[label] = {agent: factories[label](agent) for agent in env.topology.agents}

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
