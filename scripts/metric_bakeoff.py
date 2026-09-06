"""Candidate replacements for the pooled hard-SHD metric, all computed on the same episodes.

BACKGROUND (Brian, 6 Sep). The reported `global_hard_shd` assigns a hard error when the
pooled intersection holds MORE THAN ONE mark, identically to when it holds a single false
one. Measured over 13,140 pooled pair-instances per arm, the count of false committed marks
is ZERO in every configuration tested, so the entire reported value is residual ambiguity.
The name promises committed error and delivers unresolved uncertainty, and at the principal
budget it orders the arms OPPOSITELY to joint recovery.

This script computes the current metric and six candidates in one pass so they can be
compared on identical episodes, per arm.

    hard_current      the existing metric, reproduced exactly (the control).
    soft_current      the existing graded companion, 1 - 1/|pooled|.
    resolution        share of covered pairs COMMITTED to a single mark (higher better).
    error_rate        false committed marks / committed pairs. The correctness half, which
                      the current metric buries: this is the version space's soundness
                      guarantee expressed as a number.
    ambiguity         share of covered pairs still holding >1 mark (lower better).
    excess_marks      mean(|pooled| - 1) over agreeing pairs. Graded ambiguity: distinguishes
                      "two marks survive" from "three survive", which the indicator cannot.
    severity          (1.0*false + 0.5*ambiguous) / covered. A composite that keeps one
                      number but stops charging an absent claim as much as a false one.
    disagreement      share of pairs where sites hold different TRUE marks. Structural, arm
                      independent, and reported separately rather than folded in.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from itertools import combinations

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ma.baselines import make_baselines                                  # noqa: E402
from ma.evaluate import CLAIM_BACKENDS, _pair_masses, _true_index        # noqa: E402
from ma.evaluate import evaluate_episode                                 # noqa: E402
from ma.policy import IndependentPPO                                     # noqa: E402
from scripts.rescore_from_config import env_from_config                  # noqa: E402


def episode_metrics(env, acc):
    seen = {}
    for agent, window in env.windows.items():
        belief = getattr(window.belief, "last", None)
        if belief is None or not hasattr(belief, "adjacency"):
            continue
        mag = np.asarray(env._true_mag(agent))
        for u, v in combinations(range(window.k), 2):
            key = tuple(sorted((window.nodes[u], window.nodes[v])))
            masses = _pair_masses(belief, u, v)
            t = _true_index(mag, u, v)
            e = seen.setdefault(key, {"marks": [], "truth": [], "soft": []})
            e["marks"].append(frozenset(np.flatnonzero(masses > 1e-9).tolist()))
            e["truth"].append(t)
            e["soft"].append(1.0 - float(masses[t]))
    for e in seen.values():
        acc["covered"] += 1
        truths = set(e["truth"])
        pooled = frozenset.intersection(*e["marks"])
        disagree = len(truths) > 1
        if disagree or not pooled or e["truth"][0] not in pooled:
            acc["disagree"] += 1
            acc["hard_current"] += int(float(np.mean(e["soft"])) > 0.5)
            acc["soft_current"] += float(np.mean(e["soft"]))
            continue
        acc["excess_marks"] += len(pooled) - 1
        acc["soft_current"] += 1.0 - 1.0 / len(pooled)
        if len(pooled) > 1:
            acc["ambiguous"] += 1
            acc["hard_current"] += 1
        else:
            acc["committed"] += 1
            if next(iter(pooled)) != e["truth"][0]:
                acc["false"] += 1
                acc["hard_current"] += 1


def run(result: str, arm: str, episodes: int):
    report = json.loads(pathlib.Path(result).read_text())
    config, seed = report["config"], report.get("seed", 0)
    if config["belief_backend"] not in CLAIM_BACKENDS:
        raise SystemExit("pooling is defined only for the claim backends")
    env = env_from_config(config, seed=seed)
    if arm == "learned":
        import torch
        ck = pathlib.Path(result).with_name(pathlib.Path(result).stem + "_best.pt")
        torch.manual_seed(seed)
        pol = IndependentPPO.load(str(ck), env).policies(deterministic=False)
    else:
        b = {a: make_baselines(env, a, seed) for a in env.topology.agents}
        pol = {a: b[a][arm] for a in env.topology.agents}
    acc = dict(covered=0, committed=0, false=0, ambiguous=0, disagree=0,
               hard_current=0, soft_current=0.0, excess_marks=0)
    succ = []
    for ep in range(episodes):
        r = env.reset(seed=seed * 100_000 + ep)
        while not r.done:
            r = env.step({a: pol[a](env, r) for a in env.topology.agents})
        episode_metrics(env, acc)
        succ.append(float(evaluate_episode(env)["success"]))
    n, agree = max(acc["covered"], 1), max(acc["covered"] - acc["disagree"], 1)
    return {
        "joint_recovery": float(np.mean(succ)),
        "hard_current": acc["hard_current"] / n,
        "soft_current": acc["soft_current"] / n,
        "resolution": acc["committed"] / n,
        "error_rate": acc["false"] / max(acc["committed"], 1),
        "ambiguity": acc["ambiguous"] / n,
        "excess_marks": acc["excess_marks"] / agree,
        "severity": (1.0 * acc["false"] + 0.5 * acc["ambiguous"]) / n,
        "disagreement": acc["disagree"] / n,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="b050,b150,b500")
    ap.add_argument("--arms", default="learned,greedy_uncertainty,random_vary")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", default="results/metric_bakeoff.json")
    args = ap.parse_args(argv)

    out = {}
    for tag in args.cells.split(","):
        src = (ROOT / f"results/budget_tight/k12s50n04{tag}_s0.json")
        if not src.exists():
            src = ROOT / f"results/sweep12k/k12s50n04{tag}_s0.json"
        for arm in args.arms.split(","):
            key = f"{tag}/{arm}"
            out[key] = run(str(src), arm, args.episodes)
            print(f"{key:34s} " + "  ".join(f"{k}={v:.5f}" for k, v in out[key].items()),
                  flush=True)
    (ROOT / args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
