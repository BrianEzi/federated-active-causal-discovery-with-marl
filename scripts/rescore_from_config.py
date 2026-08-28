"""Re-score any saved run against a greedy configured at the bar the task is GRADED on.

WHY. `UncertaintyGreedyAgent` defaults to `bar=0.7` and every construction in the
repository uses that default, while the deterministic, factored and attributed backends all
grade at `claim_bar=1.0`. Greedy therefore stops scoring claims the task still counts open.
Measured 2026-08-27: worth +0.233 to greedy at four agents on scale-free, and enough to
INVERT the attribution headline. Any `learned vs greedy` number produced by
`scripts/ma_train.py`'s own evaluation inherits the handicap, which includes the noise-dial
runs and the scaling ladder.

WHY IT REBUILDS FROM THE RESULT FILE. The environment is reconstructed from the run's OWN
`config` block rather than from flags retyped by hand. Twice in one night a comparison was
built from a neighbouring experiment's settings instead of the one being compared against --
`--n_obs 60` where the dial used 1000, and `gnn` where the ladder used `gnn_portable`. Both
produced numbers that looked plausible. Reading the config back removes the whole class.

Evaluation only: no retraining, and the learned checkpoint beside the result is used as-is.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np

from ma.baselines import RandomAgent, UncertaintyGreedyAgent
from ma.env import MAConfig, TwoAgentEnv
from ma.evaluate import run_arm
from ma.policy import IndependentPPO
from ma.topology import Topology


def env_from_config(config: dict, seed: int = 0) -> TwoAgentEnv:
    """Rebuild the exact environment a result was produced in."""
    t = config["topology"]
    topology = Topology(name=t["name"],
                        private=tuple(tuple(p) for p in t["private"]),
                        exposed=tuple(t["exposed"]))
    kwargs = dict(
        topology=topology, n_obs=config["n_obs"], n_int=config["n_int"],
        budget=config["budget"], disclose_regime=config["disclose_regime"],
        score_rule=config["rule"], step_cost=config.get("step_cost", 0.0),
        turn_order=config["turn_order"], action_modes=tuple(config["action_modes"]),
        prior_p=config.get("prior_p"), graph_model=config.get("graph_model", "er"),
        sf_m=config.get("sf_m", 2), belief_backend=config["belief_backend"],
        cb_n_boot=config.get("cb_n_boot", 12), policy_arch=config["policy_arch"],
        episode_mix=config.get("episode_mix", "any"),
        oracle_obs_structure=config.get("oracle_obs_structure", False),
        claim_bar=config["claim_bar"], per_agent_reward=config.get("per_agent_reward", False),
        observe_belief_channels=config.get("observe_belief_channels", False),
        observe_partner_counts=config.get("observe_partner_counts", False),
        mode_by_role=config.get("mode_by_role", False),
        claims_require_all_types=config.get("claims_require_all_types", True),
        reward_criterion=config.get("reward_criterion", "u14"),
    )
    # Only pass the sampled-evidence knobs when the saved run recorded them, so this stays
    # loadable against results written before those fields existed.
    for key in ("vs_evidence", "vs_evidence_alpha"):
        if key in config:
            kwargs[key] = config[key]
    return TwoAgentEnv(MAConfig(**kwargs), seed=seed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", help="a *.json written by scripts/ma_train.py")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=None, help="default: the run's own seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    path = pathlib.Path(args.result)
    report = json.loads(path.read_text())
    seed = args.seed if args.seed is not None else report.get("seed", 0)
    env = env_from_config(report["config"], seed=seed)
    agents = env.topology.agents

    arms: Dict[str, dict] = {}
    ckpt = path.with_suffix(".pt")
    if ckpt.exists():
        arms["learned"] = IndependentPPO.load(str(ckpt), env).policies(deterministic=False)
    # The bar sweep is the point: 0.7 is what the repository builds, 1.0 is what the task
    # grades on. Both are truth-free -- greedy reads only its own belief frequencies.
    arms["greedy_bar0.7"] = {a: UncertaintyGreedyAgent(a, seed, bar=0.7) for a in agents}
    arms["greedy_bar1.0"] = {a: UncertaintyGreedyAgent(a, seed, bar=1.0) for a in agents}
    arms["random_vary"] = {a: RandomAgent(a, seed, allow_clamp=False) for a in agents}

    out: Dict[str, dict] = {"source": str(path), "seed": seed, "arms": {}}
    print(f"\n=== {path.name}  ({args.episodes} identical episodes, seed {seed}) ===")
    print(f"{'arm':16s} {'joint success':>14s} {'rounds':>8s}")
    for label, policies in arms.items():
        row = run_arm(env, policies, args.episodes, seed=seed)
        out["arms"][label] = {k: v for k, v in row.items() if isinstance(v, (int, float))}
        print(f"{label:16s} {row['success']:14.3f} {row['mean_rounds']:8.2f}")

    if "learned" in out["arms"]:
        for bar in ("greedy_bar0.7", "greedy_bar1.0"):
            d = out["arms"]["learned"]["success"] - out["arms"][bar]["success"]
            print(f"  learned - {bar:14s} {d:+.3f}")
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
