"""Joint success against BUDGET / REQUIRED COVER, one curve per window size.

THE OBJECTION THIS ANSWERS. The scaling figure holds interventions-per-node fixed across
window sizes. "You held moves-per-node fixed, but the cover a window requires is sublinear in
k, so you handed the big windows a more generous budget and the decline you show is a budget
effect." Plotting success against the budget a run actually had, divided by the cover that run
actually required, removes the normalisation from the argument: if the rungs land on one
curve, window size is not what the decline is about.

HOW THE X-AXIS IS OBTAINED WITHOUT RE-TRAINING OR RE-BUDGETING. Every rung is played at ITS
OWN trained budget. The ratio varies anyway, because `required_system` varies episode to
episode -- at k=8 it runs from 6 to 14 over 50 draws, a ratio span of 0.86 to 2.00. So the
curve is built from natural variation in the DENOMINATOR rather than by re-running policies
at budgets they were not trained for, which would put every point off-distribution:
`budget_left` is in the observation, so a policy evaluated at a budget it never saw is
answering a different question. Binning episodes by their own required cover keeps every
point in distribution.

WHAT IS ON EACH AXIS.
  x  budget / required_system -- the system-level cover, the union over agents of the forced
     positions mapped to global node ids, because `budget` is a shared pool of ROUNDS and the
     windows overlap on the shared set. See `scripts/required_cover.py`.
  y  joint success, `ma.evaluate.evaluate_episode`'s `success`: the same event the ladder
     figure plots, so the two are directly comparable.

ORACLE EVIDENCE ONLY, inherited from `required_cover.py`: under sampled evidence no set of
interventions is sufficient with certainty and the denominator is not defined.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np

from ma.baselines import UncertaintyGreedyAgent
from ma.evaluate import evaluate_episode
from ma.policy import IndependentPPO
from scripts.required_cover import closed_form
from scripts.rescore_from_config import env_from_config


def _required_system(env) -> int:
    """The union, over agents, of each window's forced positions as GLOBAL node ids.

    Read after `env.reset`, from the true MAGs, so it is a property of the episode and not of
    what any policy did in it.
    """
    union = set()
    for agent in env.topology.agents:
        window = env.windows[agent]
        for position in closed_form(np.asarray(env._true_mag(agent)), window.k):
            union.add(int(window.nodes[position]))
    return len(union)


def play(env, policies, episodes: int, seed: int) -> List[Dict[str, float]]:
    """One row per episode: the required cover, the budget, and whether the joint succeeded.

    The cover is computed from the reset state BEFORE the episode is played, so the x-axis
    cannot be contaminated by the policy being measured on the y-axis.
    """
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)
    rows = []
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        required = _required_system(env)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
        rows.append({"episode": episode,
                     "required_system": required,
                     "budget": int(env.config.budget),
                     "ratio": env.config.budget / max(required, 1),
                     "success": float(evaluate_episode(env)["success"])})
    return rows


def binned(rows: List[Dict[str, float]], edges: List[float]) -> List[dict]:
    """Success within each ratio bin, with a binomial standard error and the count.

    Bins with fewer than five episodes are reported with their n rather than dropped: a
    curve that thins out at one end is information about the design, and silently removing
    those points is how a figure comes to look tidier than the evidence.
    """
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        sel = [r["success"] for r in rows if low <= r["ratio"] < high]
        if not sel:
            continue
        mean = float(np.mean(sel))
        se = float(np.sqrt(max(mean * (1 - mean), 0.0) / len(sel)))
        out.append({"low": low, "high": high, "mid": (low + high) / 2,
                    "n": len(sel), "success": mean, "se": se})
    return out


def rung(result_path: pathlib.Path, episodes: int, seed: Optional[int],
         arms: List[str]) -> dict:
    report = json.loads(result_path.read_text())
    config = report["config"]
    if config.get("vs_evidence", "oracle") != "oracle":
        raise SystemExit(f"{result_path.name}: sampled evidence -- required cover undefined")
    use_seed = seed if seed is not None else report.get("seed", 0)
    env = env_from_config(config, seed=use_seed)
    k = len(config["topology"]["private"][0]) + len(config["topology"]["exposed"])

    built: Dict[str, dict] = {}
    checkpoint = result_path.with_suffix(".pt")
    if "learned" in arms and checkpoint.exists():
        built["learned"] = IndependentPPO.load(str(checkpoint), env).policies(
            deterministic=False)
    if "greedy" in arms:
        # bar=1.0, the bar the task is GRADED at. The 0.7 default is the handicap that
        # inverted an attribution headline on 2026-08-27; it must never be the reference.
        built["greedy"] = {a: UncertaintyGreedyAgent(a, use_seed, bar=1.0)
                           for a in env.topology.agents}

    out = {"rung": result_path.stem, "k": k, "budget": config["budget"],
           "n_agents": len(config["topology"]["private"]), "seed": use_seed, "arms": {}}
    for label, policies in built.items():
        rows = play(env, policies, episodes, use_seed)
        out["arms"][label] = {"rows": rows,
                              "success": float(np.mean([r["success"] for r in rows])),
                              "mean_ratio": float(np.mean([r["ratio"] for r in rows]))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="rung result .json files")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arms", default="learned,greedy")
    ap.add_argument("--bins", default="0.5,0.8,1.0,1.2,1.5,2.0,3.0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    edges = [float(x) for x in args.bins.split(",")]
    arms = args.arms.split(",")
    payload = {"bins": edges, "episodes": args.episodes, "rungs": []}
    for path in args.results:
        entry = rung(pathlib.Path(path), args.episodes, args.seed, arms)
        for label, arm in entry["arms"].items():
            arm["binned"] = binned(arm["rows"], edges)
        payload["rungs"].append(entry)
        print(f"\n=== {entry['rung']}  k={entry['k']}  agents={entry['n_agents']}  "
              f"budget={entry['budget']} ===")
        for label, arm in entry["arms"].items():
            print(f"  {label:8s} success {arm['success']:.3f}  "
                  f"mean budget/required {arm['mean_ratio']:.2f}")
            for row in arm["binned"]:
                print(f"      ratio {row['low']:.1f}-{row['high']:.1f}  "
                      f"n {row['n']:3d}  success {row['success']:.3f} +/- {row['se']:.3f}")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
