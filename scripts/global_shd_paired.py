"""Paired standard errors on the HEADLINE metric: hard SHD of the pooled global graph.

WHY A SECOND SHD SCRIPT, AND WHY NOT `scripts/shd.py`. They measure different objects and
the difference is not cosmetic -- on `k12s50n04b500_s0` they disagree in SIGN:

    sweep headline  global_hard_shd    learned 0.0003   greedy 0.0007   learned BETTER
    scripts/shd.py  dedup (soft)       learned - greedy = +0.1099       learned WORSE

`scripts/shd.py` reports the mean per-window SOFT contribution, which gives partial credit
for every unsettled pair, so most of its mass is pairs nobody ever resolved. The headline
number is `ma.evaluate.global_graph_report`: the belief POOLED across agents, each covered
pair counted once, and a pair is wrong only if the pooled belief settled on the wrong mark.
Both are defensible; only one is what the results chapter quotes, and error bars have to be
on the quantity being quoted.

Two further confounds `scripts/shd.py` carries that this does not:
  * it loads `<result>.pt`, the FINAL policy, while the sweep's own arms are evaluated from
    the policy in memory at the end of training;
  * it defaults to ARGMAX while `run_arm` evaluates by SAMPLING, and the argmax/sampling gap
    is worth half the measured effect at some window sizes (docs/FINDINGS_SHD_2026_08_29.md).
Both are exposed here as flags and recorded in the output, rather than left implicit.

WHAT IT ANSWERS. "Learned 0.0000 against greedy 0.0005 at k=20" is a difference of five
ten-thousandths between two numbers with no stated uncertainty and three seeds behind them.
This plays every arm over IDENTICAL episodes and reports the mean and standard error of the
PER-EPISODE difference, which is the paired test that question needs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ma.baselines import RandomAgent, UncertaintyGreedyAgent           # noqa: E402
from ma.evaluate import global_graph_report                            # noqa: E402
from ma.policy import IndependentPPO                                   # noqa: E402
from scripts.rescore_from_config import env_from_config                # noqa: E402


def play(env, policies, episodes: int, seed: int) -> Dict[str, List[float]]:
    """One arm over `episodes` fixed seeds. Returns per-episode hard and soft global SHD."""
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    hard, soft, resolved = [], [], []
    for episode in range(episodes):
        # The same expression `run_arm` uses, so an arm scored here and an arm scored there
        # see the same worlds -- which is the whole point of a PAIRED comparison.
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
        report = global_graph_report(env)
        hard.append(report["global_hard_shd"])
        soft.append(report["global_soft_shd"])
        resolved.append(report["global_resolved_fraction"])
    return {"hard": hard, "soft": soft, "resolved": resolved}


def paired(a: List[float], b: List[float]) -> Dict[str, float]:
    d = np.asarray(a) - np.asarray(b)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    return {"delta": float(d.mean()), "se": se,
            "significant": bool(abs(d.mean()) > 2 * se) if se > 0 else False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "final"],
                    help="_best.pt (highest MI during training) or .pt (final)")
    ap.add_argument("--sample", action="store_true",
                    help="evaluate the learned arm by SAMPLING rather than argmax")
    ap.add_argument("--out", default="results/global_shd_paired.json")
    args = ap.parse_args(argv)

    payload = []
    for path in args.results:
        path = pathlib.Path(path)
        report = json.loads(path.read_text())
        config = report["config"]
        use_seed = args.seed if args.seed is not None else report.get("seed", 0)
        env = env_from_config(config, seed=use_seed)

        suffix = "_best.pt" if args.checkpoint == "best" else ".pt"
        checkpoint = path.with_name(path.stem + suffix) if suffix.startswith("_") \
            else path.with_suffix(suffix)
        if not checkpoint.exists():
            print(f"!! {path.stem}: no {checkpoint.name}, skipped")
            continue

        ppo = IndependentPPO.load(str(checkpoint), env)
        arms = {
            "learned": ppo.policies(deterministic=not args.sample),
            "greedy": {a: UncertaintyGreedyAgent(a, use_seed, bar=1.0)
                       for a in env.topology.agents},
            "random_vary": {a: RandomAgent(a, use_seed, allow_clamp=False)
                            for a in env.topology.agents},
        }
        rows = {label: play(env, policies, args.episodes, use_seed)
                for label, policies in arms.items()}

        print(f"\n=== {path.stem}  ({args.episodes} episodes, "
              f"{args.checkpoint} checkpoint, "
              f"{'sampled' if args.sample else 'argmax'}) ===")
        print(f"{'arm':14s} {'hard SHD':>12s} {'soft SHD':>12s} {'resolved':>9s}")
        for label, r in rows.items():
            print(f"{label:14s} {np.mean(r['hard']):12.5f} {np.mean(r['soft']):12.5f} "
                  f"{np.mean(r['resolved']):9.3f}")
        entry = {"source": str(path), "seed": use_seed, "episodes": args.episodes,
                 "checkpoint": args.checkpoint, "sampled": bool(args.sample),
                 "means": {k: {m: float(np.mean(v[m])) for m in ("hard", "soft", "resolved")}
                           for k, v in rows.items()},
                 "paired": {}}
        for other in ("greedy", "random_vary"):
            p = paired(rows["learned"]["hard"], rows[other]["hard"])
            entry["paired"][f"learned-{other}"] = p
            flag = "" if p["significant"] else "   (INSIDE 2 SE -- not distinguishable)"
            print(f"  PAIRED hard SHD  learned - {other:12s} "
                  f"{p['delta']:+.5f} +/- {p['se']:.5f}{flag}")
        payload.append(entry)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
