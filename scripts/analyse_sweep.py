"""Collect the sweep's result files into one tidy table.

Every downstream number -- the summary, the charts, the write-up -- comes from the CSV this
produces, so there is exactly one path from raw result to reported claim. The previous
round of this project reported figures assembled by hand from several places and had to
retract them; a single derivation is the cheapest guard against repeating that.

One row per (configuration, seed). Aggregation across seeds happens in the analysis, never
here, because a mean over seeds is what hides a lucky run.

Usage:
    python -m scripts.analyse_sweep --results ~/sa_runs/sweep --out results/sweep
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import numpy as np

# The columns worth carrying forward, in the order they are written.
FIELDS = [
    "tag", "arm", "d", "observation", "seed",
    "gap_closed", "gap_closed_sampled",
    "solve_rate", "greedy_solve_rate", "mean_cost", "cost_ci_low", "cost_ci_high",
    "under_acting_rate", "optimal_rate", "informative_fraction", "mean_regret",
    "final_entropy", "passed", "train_seconds",
    # The reference policies, carried on every row so a row is self-contained: gap-closed
    # is only interpretable next to the anchors it was computed from.
    "ref_random_cost", "ref_greedy_cost", "ref_edge_greedy_cost",
    "ref_no_intervention_solve",
    # Environment and agent levers, so the table can be pivoted on any of them.
    "train_episodes", "budget", "n_obs", "n_int", "identify_threshold",
    "prior", "prior_p", "intervene_scale",
    "entropy_coef", "lr", "step_cost", "hidden", "gamma", "episodes_per_update",
    "n_dags", "n_mecs", "singleton_fraction", "git_commit", "torch",
]


def _get(metrics: Dict, key: str, default=float("nan")):
    value = metrics.get(key, default)
    return default if value is None else value


def load_rows(results_dir: str) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            payload = json.load(f)

        args = payload["args"]
        refs = payload.get("references", {})
        space = payload.get("space", {})
        provenance = payload.get("provenance", {})

        for verdict in payload["per_seed"]:
            det = verdict.get("deterministic", {})
            sampled = verdict.get("sampled", {})
            ci = _get(det, "cost_ci", (float("nan"), float("nan")))
            rows.append({
                "tag": payload.get("tag") or os.path.basename(path)[:-5],
                "arm": _arm_of(payload.get("tag", "")),
                "d": args["d"],
                "observation": args["observation"],
                "seed": verdict.get("seed"),
                "gap_closed": _get(det, "gap_closed"),
                "gap_closed_sampled": _get(sampled, "gap_closed"),
                "solve_rate": _get(det, "solve_rate"),
                "greedy_solve_rate": _get(det, "greedy_solve_rate"),
                "mean_cost": _get(det, "mean_cost"),
                "cost_ci_low": ci[0], "cost_ci_high": ci[1],
                "under_acting_rate": _get(det, "under_acting_rate"),
                "optimal_rate": _get(det, "optimal_rate"),
                "informative_fraction": _get(det, "informative_fraction"),
                "mean_regret": _get(det, "mean_regret"),
                "final_entropy": verdict.get("final_entropy"),
                "passed": verdict.get("passed"),
                "train_seconds": verdict.get("train_seconds"),
                "ref_random_cost": refs.get("random", {}).get("mean_cost"),
                "ref_greedy_cost": refs.get("greedy_oracle", {}).get("mean_cost"),
                "ref_edge_greedy_cost": refs.get("edge_marginal_greedy", {}).get("mean_cost"),
                "ref_no_intervention_solve": refs.get("no_intervention", {}).get("solve_rate"),
                "train_episodes": args["train_episodes"],
                "budget": args.get("budget"), "n_obs": args.get("n_obs"),
                "n_int": args.get("n_int"),
                "identify_threshold": args.get("identify_threshold"),
                "prior": args.get("prior"), "prior_p": args.get("prior_p"),
                "intervene_scale": args.get("intervene_scale"),
                "entropy_coef": args.get("entropy_coef"), "lr": args.get("lr"),
                "step_cost": args.get("step_cost"), "hidden": args.get("hidden"),
                "gamma": args.get("gamma"),
                "episodes_per_update": args.get("episodes_per_update"),
                "n_dags": space.get("n_dags"), "n_mecs": space.get("n_mecs"),
                "singleton_fraction": space.get("singleton_fraction"),
                "git_commit": provenance.get("git_commit", "")[:8],
                "torch": provenance.get("torch"),
            })
    return rows


def _arm_of(tag: str) -> str:
    """The lever a configuration varies, recovered from its tag."""
    if tag.startswith("core"):
        return "core"
    if tag.startswith("d6"):
        return "d6"
    # Tags are "<lever>_<value>"; the value is the final underscore-separated piece.
    return tag.rsplit("_", 1)[0] if "_" in tag else tag


def summarise(rows: List[Dict]) -> List[Dict]:
    """Per configuration: the seed spread, reported as min/median/max, never as a mean.

    The minimum is the headline number. A configuration is only as good as its worst seed,
    and averaging across seeds is exactly how a single lucky run gets mistaken for a
    working method.
    """
    by_tag: Dict[str, List[Dict]] = {}
    for row in rows:
        by_tag.setdefault(row["tag"], []).append(row)

    out = []
    for tag, group in sorted(by_tag.items()):
        gaps = np.array([r["gap_closed"] for r in group], dtype=float)
        finite = gaps[np.isfinite(gaps)]
        out.append({
            "tag": tag, "arm": group[0]["arm"], "n_seeds": len(group),
            "n_passed": sum(1 for r in group if r["passed"]),
            "min_gap": float(finite.min()) if len(finite) else float("nan"),
            "median_gap": float(np.median(finite)) if len(finite) else float("nan"),
            "max_gap": float(finite.max()) if len(finite) else float("nan"),
            "mean_solve": float(np.mean([r["solve_rate"] for r in group])),
            "mean_under_acting": float(np.mean([r["under_acting_rate"] for r in group])),
            "mean_entropy": float(np.mean([r["final_entropy"] for r in group])),
            "mean_optimal": float(np.nanmean([r["optimal_rate"] for r in group])),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True,
                        help="directory of result.json files (may be given twice)",
                        action="append", dest="results_dirs")
    parser.add_argument("--out", type=str, default="results/sweep")
    args = parser.parse_args()

    rows: List[Dict] = []
    for directory in args.results_dirs:
        found = load_rows(os.path.expanduser(directory))
        print(f"{directory}: {len(found)} rows")
        rows.extend(found)

    if not rows:
        raise SystemExit("no result files found")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)
    with open(f"{args.out}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{len(rows)} runs across {len(summary)} configurations\n")
    print(f"{'configuration':<28} {'seeds':>5} {'pass':>5} "
          f"{'min gap':>9} {'median':>8} {'max':>8} {'solve':>6} {'entropy':>8}")
    arm = None
    for s in sorted(summary, key=lambda r: (r["arm"] != "core", r["arm"], r["tag"])):
        if s["arm"] != arm:
            arm = s["arm"]
            print(f"\n[{arm}]")
        print(f"  {s['tag']:<26} {s['n_seeds']:>5} {s['n_passed']:>5} "
              f"{s['min_gap']:>9.3f} {s['median_gap']:>8.3f} {s['max_gap']:>8.3f} "
              f"{s['mean_solve']:>6.2f} {s['mean_entropy']:>8.3f}")

    print(f"\nwritten: {args.out}_runs.csv  {args.out}_summary.json")


if __name__ == "__main__":
    main()
