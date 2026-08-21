"""Cross-rule evaluation: score every trained pair under every belief rule.

This is the blocker on reporting anything from the 2026-08-17 runs. Greedy scored 0.568
under SUBSET and 0.190 under JOINT_CONF, so "learned 0.380 beats greedy" means completely
different things depending on which rule the reference was measured under, and comparing a
number from one rule against a number from the other is invalid.

The fix is a matrix. Every policy -- learned pairs from each training rule, plus greedy and
random, which are rule-agnostic since they never see a belief they cannot compute -- is
evaluated under every rule, on the SAME episodes. Then:

  * reading DOWN a column compares policies fairly, because the belief model is held fixed;
  * reading ACROSS a row shows how much of a policy's performance is the belief model rather
    than the policy.

The second is the interesting one and it has never been measured. A policy trained under
JOINT_CONF may be exploiting a belief model that is simply better at confounded episodes,
in which case the credit belongs to the scoring rule and not to anything learned. Separating
those two is the whole point.

PRE-REGISTERED, before the numbers exist:

  X1  Greedy's confounded solve rate is 0.000 under EVERY rule. It never clamps, so no rule
      can give it clean rows to condition on. If this fails, something is wrong with the
      harness rather than interesting.
  X2  A JOINT_CONF-trained policy scored under SUBSET keeps most of its confounded
      advantage, because the behaviour that earns it -- clamping -- produces clean rows that
      SUBSET can also use. The valley is a LEARNING obstacle, not a scoring one.
  X3  A SUBSET-trained policy scored under JOINT_CONF gains little, because it never learned
      to clamp and JOINT_CONF without clean rows is strictly worse than the alternatives
      (0.244 against 0.815 in the scoring sweep).

  X2 and X3 together would establish that the rule change bought a LEARNING SIGNAL rather
  than an inference advantage, which is the honest version of the claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legacy.ma_v1.env import MAConfig
from legacy.ma_v1.policy import IndependentPPO, MAPPOConfig
from ma.score_regimes import RULES
from ma.topology import Topology
from scripts.ma_train import evaluate_baseline, evaluate_learned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="results/ma/checkpoints")
    ap.add_argument("--eval_episodes", type=int, default=400)
    ap.add_argument("--n_obs", type=int, default=2000)
    ap.add_argument("--n_int", type=int, default=200)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--out", default="results/ma/cross_rule.json")
    args = ap.parse_args()

    topology = Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    checkpoints = sorted(Path(args.checkpoint_dir).glob("*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {args.checkpoint_dir} -- train first")
    print(f"found {len(checkpoints)} checkpoints", flush=True)

    rows = []
    for eval_rule in args.rules:
        config = MAConfig(topology=topology, n_obs=args.n_obs, n_int=args.n_int,
                          budget=args.budget, score_rule=eval_rule)

        for kind in ("random", "greedy"):
            stats = evaluate_baseline(config, kind, args.eval_episodes, seed=99)
            rows.append({"policy": kind, "trained_under": None,
                         "eval_rule": eval_rule, "seed": None, "stats": stats})
            print(f"  [{eval_rule:>10}] {kind:>10}: "
                  f"all {stats['all']['solve_rate']:.3f} "
                  f"conf {stats['confounded']['solve_rate']:.3f}", flush=True)

        for path in checkpoints:
            agent = IndependentPPO(config, MAPPOConfig(total_episodes=0, seed=0))
            meta = agent.load(path)
            stats = evaluate_learned(agent, config, args.eval_episodes, seed=99)
            rows.append({
                "policy": path.stem,
                "trained_under": meta["trained_under_rule"],
                "eval_rule": eval_rule,
                "seed": meta["seed"],
                "stats": stats,
            })
            print(f"  [{eval_rule:>10}] {path.stem:>22}: "
                  f"all {stats['all']['solve_rate']:.3f} "
                  f"conf {stats['confounded']['solve_rate']:.3f} "
                  f"clamp {stats['all']['clamp_fraction']:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))

    # Matrix view: median over seeds of each (trained_under, eval_rule) cell.
    print("\nCONFOUNDED solve rate -- rows are what the policy was TRAINED under,")
    print("columns are what it was SCORED under. Read down a column to compare")
    print("policies fairly; read across a row to see how much is the belief model.\n")
    trained = sorted({r["trained_under"] for r in rows if r["trained_under"]})
    header = "".join(f"{rule:>13}" for rule in args.rules)
    print(f"{'trained':>14}{header}")
    for source in ["random", "greedy"] + trained:
        cells = []
        for eval_rule in args.rules:
            if source in ("random", "greedy"):
                subset = [r for r in rows
                          if r["policy"] == source and r["eval_rule"] == eval_rule]
            else:
                subset = [r for r in rows if r["trained_under"] == source
                          and r["eval_rule"] == eval_rule]
            values = [r["stats"]["confounded"]["solve_rate"] for r in subset]
            cells.append(f"{np.median(values):>13.3f}" if values else f"{'--':>13}")
        print(f"{source:>14}{''.join(cells)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
