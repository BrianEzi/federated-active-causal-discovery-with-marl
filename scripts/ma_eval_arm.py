"""Evaluate ONE arm of ONE trained run, into its own result file.

WHY THIS IS SEPARATE FROM TRAINING. `ma_train` trains and then evaluates four arms inline,
which is fine while an episode is cheap. At the top of the scale ladder it stops being
fine: an episode at 5 agents and 30 nodes runs into the hundreds of seconds, so four arms
at 150 episodes is more wall-clock than the training that produced them, all of it
sequential inside a single job.

The arms are INDEPENDENT -- each replays the same seeded episodes under a different policy
set, and nothing is shared but the environment. So they belong in separate array tasks.
That is the difference between one 50-hour job and five 13-hour ones running at the same
time, and it is what keeps eval_episodes high enough for the confidence intervals to
resolve anything. One episode at 50 eval episodes is worth 2 percentage points; the whole
point of the ladder is comparisons finer than that.

Reads the checkpoint `ma_train --out X.json` wrote at `X.pt`, and reconstructs the
environment from the `config` block of `X.json` -- never from flags passed here, so an arm
cannot silently be evaluated against a different environment than it was trained in.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from ma.baselines import make_baselines
from ma.env import CLAMP, MAConfig, TwoAgentEnv
from ma.evaluate import run_arm
from ma.policy import IndependentPPO
from ma.topology import Topology

LEARNED = "learned"


def env_from_report(report: dict) -> TwoAgentEnv:
    """Rebuild the exact environment a run was trained in, from its own result file."""
    cfg = report["config"]
    topo = cfg["topology"]
    topology = Topology(name=topo["name"],
                        private=tuple(tuple(block) for block in topo["private"]),
                        exposed=tuple(topo["exposed"]))
    config = MAConfig(
        topology=topology, n_obs=cfg["n_obs"], n_int=cfg["n_int"], budget=cfg["budget"],
        disclose_regime=cfg["disclose_regime"], score_rule=cfg["rule"],
        step_cost=cfg["step_cost"], turn_order=cfg["turn_order"],
        action_modes=tuple(cfg["action_modes"]), prior_p=cfg["prior_p"],
        identify_threshold=cfg["identify_threshold"],
        intervene_scale=cfg["intervene_scale"])
    return TwoAgentEnv(config)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the .json ma_train wrote; .pt sits beside it")
    ap.add_argument("--arm", required=True,
                    help=f"{LEARNED}, or a baseline name: random_clamp, greedy, pass, ...")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=None,
                    help="defaults to the seed the run was trained with")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    run_path = pathlib.Path(args.run)
    report = json.loads(run_path.read_text())
    seed = args.seed if args.seed is not None else report["seed"]
    env = env_from_report(report)

    if args.arm == LEARNED:
        checkpoint = run_path.with_suffix(".pt")
        if not checkpoint.exists():
            raise SystemExit(f"no checkpoint at {checkpoint} -- was the run trained with --out?")
        ppo = IndependentPPO.load(checkpoint, env)
        # Stochastic, matching how ma_train evaluates the learned arm. A deterministic
        # policy is a DIFFERENT arm, not a cleaner reading of this one.
        policies = ppo.policies(deterministic=False)
    else:
        reference = {agent: make_baselines(env, agent, seed=seed)
                     for agent in env.topology.agents}
        if args.arm not in reference[env.topology.agents[0]]:
            raise SystemExit(f"unknown arm {args.arm!r}")
        if args.arm == "random_vary" and CLAMP in env.config.action_modes \
                and len(env.config.action_modes) == 1:
            raise SystemExit("random_vary has no legal action in a clamp-only run")
        policies = {agent: reference[agent][args.arm] for agent in env.topology.agents}

    started = time.time()
    row = run_arm(env, policies, args.episodes, seed=seed)
    row["seconds"] = time.time() - started
    row["arm"] = args.arm
    row["seed"] = seed
    row["run"] = str(run_path)
    row["eval_episodes"] = args.episodes

    print(f"{report['arm']} / {args.arm:13s} success {row['success']:.3f} "
          f"CI {row['success_ci'][0]:.3f}-{row['success_ci'][1]:.3f}  "
          f"steps {row['mean_steps']:.2f}  clamp {row['clamp_fraction']:.3f}  "
          f"({row['seconds']:.0f}s)", flush=True)

    if args.out:
        dest = pathlib.Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(row, indent=2))
        print(f"  wrote {dest}", flush=True)
    return row


if __name__ == "__main__":
    main()
