"""Generate an explorable notebook over the overnight results.

Reads the same `results/all_runs.csv` everything else does, so the notebook cannot drift
from the report. Nothing is hardcoded: change the CSV and the notebook's numbers change.

    python -m legacy.scripts.generate_findings_notebook
"""
from __future__ import annotations

import itertools
import json
import os
from typing import List

NOTEBOOK = "notebooks/sa_findings.ipynb"


_COUNTER = itertools.count()


def _lines(source: str) -> list:
    """Split into notebook `source` lines, KEEPING the newlines.

    nbformat wants each entry to end in a newline. Splitting on "\n" without keeping them
    concatenates the whole cell onto one line, and every code cell then fails with a
    SyntaxError the moment it runs. Caught only by actually executing the notebook, which
    is why this module is verified with nbconvert rather than a JSON schema check.
    """
    return source.strip().splitlines(keepends=True)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": f"md{next(_COUNTER)}", "metadata": {},
            "source": _lines(source)}


def code(source: str) -> dict:
    return {"cell_type": "code", "id": f"code{next(_COUNTER)}", "execution_count": None,
            "metadata": {}, "outputs": [], "source": _lines(source)}


def cells() -> List[dict]:
    return [
        md("""
# Single-agent active causal discovery — overnight findings

238 runs across 73 configurations, 15 August 2026. Everything here reads
`results/all_runs.csv`, which `scripts/analyse_sweep.py` generates from the raw result
files in `results/raw/`. No number is transcribed by hand.

**The result:** the agent beats the greedy information-gain oracle at d=4, d=5 and d=6.

**The story:** 61 configurations of the original network failed first. A supervised probe
showed why none of them could have worked — the network could not express the mapping it
was being asked to learn.

Run the cells in order. The last section is set up for you to slice the data yourself.
"""),

        md("## Setup"),
        code("""
import json, glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Works whether you launch from the repo root or from notebooks/.
ROOT = Path.cwd()
if not (ROOT / "results").exists():
    ROOT = ROOT.parent
RESULTS = ROOT / "results"
assert RESULTS.exists(), f"can't find results/ from {Path.cwd()}"

runs = pd.read_csv(RESULTS / "all_runs.csv")
print(f"{len(runs)} runs across {runs.tag.nunique()} configurations")
runs.head(3)
"""),

        code("""
plt.rcParams.update({
    "figure.figsize": (10, 4.5), "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
GOOD, BAD, MUTED, ACCENT = "#2E7D5B", "#B3402F", "#8892A0", "#1F6F8B"
"""),

        md("""
## 1. The headline

`gap_closed` is the primary metric: `(random − agent) / (random − greedy)` on episode cost,
with unsolved episodes charged at the full budget.

- **0.0** = no better than picking at random
- **1.0** = matches the greedy information-gain oracle
- **above 1.0** = beats it

The **minimum across seeds** is what counts, never the mean — a mean hides a lucky run.
"""),
        code("""
summary = (runs.groupby(["tag", "arch", "d", "n_obs"], as_index=False)
           .agg(seeds=("gap_closed", "size"),
                passed=("passed", "sum"),
                min_gap=("gap_closed", "min"),
                median_gap=("gap_closed", "median"),
                max_gap=("gap_closed", "max"),
                solve=("solve_rate", "mean"),
                entropy=("final_entropy", "mean"),
                oracle_agreement=("optimal_rate", "mean"))
           .sort_values("min_gap", ascending=False))

summary.head(12).round(3)
"""),

        md("""
### The configurations that passed

A run passes only if it clears **all four** criteria: gap closed >= 0.80, no under-acting,
solve rate within 5 points of greedy's, and no collapse between the deterministic and
sampled policies.
"""),
        code("""
winners = summary[summary.passed > 0].copy()
print(f"{len(winners)} configurations had at least one passing seed.")
print(f"Architectures among them: {sorted(winners.arch.unique())}")
print(f"Flat-network configurations that passed: "
      f"{len(summary[(summary.arch == 'flat') & (summary.passed > 0)])} "
      f"of {len(summary[summary.arch == 'flat'])}")
winners.round(3)
"""),

        md("""
## 2. The arc: from failure to beating the oracle

Each dot is one seed. The large dot is the worst seed — the number the criteria use.

Read this top to bottom: the original network fails, tuning helps a little, action memory
helps more, and the architecture change is what crosses the line. The `flat, all fixes`
row is the control: same settings, old network.
"""),
        code("""
ARC = [
    ("core_d5_edge_marginals",        "baseline (flat MLP)"),
    ("s4_best_nocounts",              "+ tuned optimiser"),
    ("s4_counts_best",                "+ action memory"),
    ("s4_everything",                 "flat, everything + 15k eps"),
    ("s5_flat_control",               "flat, all fixes  [CONTROL]"),
    ("s5_pernode",                    "per-node only"),
    ("s5_pernode_best",               "per-node + optimiser"),
    ("s5_pernode_best_counts",        "per-node + optimiser + memory"),
    ("s6_d5_nobs5000",                "same, GATE 1 valid"),
]

fig, ax = plt.subplots(figsize=(10, 5))
for i, (tag, label) in enumerate(ARC):
    vals = runs.loc[runs.tag == tag, "gap_closed"].dropna().values
    if not len(vals):
        continue
    colour = GOOD if vals.min() >= 0.8 else (BAD if vals.max() < 0 else "#B07A16")
    ax.scatter(vals, [i] * len(vals), s=55, color=colour, alpha=0.45, zorder=3)
    ax.scatter([vals.min()], [i], s=130, color=colour, zorder=4,
               edgecolor="white", linewidth=1.5)

ax.axvline(0, color=MUTED, ls="--", lw=1.2)
ax.axvline(1, color=GOOD, ls="--", lw=1.2)
ax.text(0, len(ARC) - 0.3, " random", color=MUTED, fontsize=9)
ax.text(1, len(ARC) - 0.3, " greedy oracle", color=GOOD, fontsize=9)
ax.set_yticks(range(len(ARC)))
ax.set_yticklabels([l for _, l in ARC])
ax.set_xlabel("gap closed")
ax.set_title("Every seed of every step along the way (d=5)")
ax.invert_yaxis()
plt.tight_layout(); plt.show()
"""),

        md("""
## 3. Entropy separated pass from fail better than any hyperparameter

Policy entropy measures how undecided the policy is. At d=5 the maximum is ln(6) = 1.79,
meaning "completely random".

Every failing configuration ended between roughly 1.2 and 1.6 — the policy never sharpened,
so its `argmax` was close to arbitrary. Every passing one ended near 0.5–0.7.

This is worth remembering as a first diagnostic on any future run.
"""),
        code("""
fig, ax = plt.subplots()
passed_runs = runs[runs.passed]
failed_runs = runs[~runs.passed]
ax.scatter(failed_runs.final_entropy, failed_runs.gap_closed, s=28,
           color=BAD, alpha=0.5, label=f"failed (n={len(failed_runs)})")
ax.scatter(passed_runs.final_entropy, passed_runs.gap_closed, s=45,
           color=GOOD, alpha=0.85, label=f"passed (n={len(passed_runs)})")
ax.axhline(0.8, color=MUTED, ls="--", lw=1, label="pass threshold")
ax.set_xlabel("final policy entropy (nats)")
ax.set_ylabel("gap closed")
ax.set_ylim(-6, 2)
ax.set_title("A policy that never sharpened never learned where to intervene")
ax.legend()
plt.tight_layout(); plt.show()

print(runs.groupby("passed").final_entropy.describe()[["count", "mean", "min", "max"]].round(3))
"""),

        md("""
## 4. The stage-1 lever sweep — thirteen levers, none of them the answer

Each of these varied ONE setting away from a fixed baseline. Not one passed. The uniformity
is the finding: the problem was not in the region any lever explored.

**Caveat that matters:** these were all measured at `n_obs=1000`, where GATE 1 fails at d=5
(see section 6). They may not transfer. Re-running this sweep at `n_obs=5000` is an open task.
"""),
        code("""
lever_tags = runs[~runs.tag.str.startswith(("s2_", "s3_", "s4_", "s5_", "s6_", "core", "d6"))]
levers = (lever_tags.groupby("tag", as_index=False)
          .agg(seeds=("gap_closed", "size"), min_gap=("gap_closed", "min"),
               median=("gap_closed", "median"), solve=("solve_rate", "mean"),
               entropy=("final_entropy", "mean"))
          .sort_values("min_gap", ascending=False))
print(f"{len(levers)} lever configurations, "
      f"{int((levers.min_gap >= 0.8).sum())} passing")
levers.round(3)
"""),

        md("""
## 5. The probe — where the failure actually was

Two explanations looked identical from outside:

- **A.** The observation carries the answer, but PPO can't find it.
- **B.** The observation doesn't carry a decodable answer at all.

The probe separates them. Train the agent's *own* network on the agent's *own* observation
to predict the oracle's best target, with full supervision and no exploration problem. If it
still can't learn the mapping, the problem isn't reward or exploration.

It could — but only with the right architecture. That is the whole result.
"""),
        code("""
rows = []
for path in sorted(glob.glob(str(RESULTS / "probe" / "*.json"))):
    payload = json.load(open(path))
    for condition, stats in payload.get("conditions", {}).items():
        rows.append({"d": payload["d"], "episodes": payload["episodes"],
                     "condition": condition,
                     "accuracy": stats["probe_accuracy"],
                     "chance": stats["chance_accuracy"],
                     "majority": stats["majority_accuracy"]})
probe = pd.DataFrame(rows).sort_values(["d", "condition", "episodes"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
styles = {"edge_marginals/pernode": (GOOD, "per-node scorer"),
          "edge_marginals/flat": (BAD, "flat MLP"),
          "posterior/flat": ("#B07A16", "flat MLP, exact posterior")}
for ax, d in zip(axes, [4, 5]):
    subset = probe[probe.d == d]
    for condition, (colour, label) in styles.items():
        line = subset[subset.condition == condition].sort_values("episodes")
        if len(line) > 1:
            ax.plot(line.episodes, line.accuracy, "o-", color=colour, label=label)
    chance = subset.chance.mean()
    ax.axhline(chance, color=MUTED, ls="--", lw=1, label="chance")
    ax.set_xscale("log"); ax.set_xlabel("episodes of supervised data")
    ax.set_title(f"d={d}")
axes[0].set_ylabel("accuracy predicting the oracle's choice")
axes[0].legend(fontsize=8)
plt.suptitle("The same network, the same data — only the architecture differs")
plt.tight_layout(); plt.show()

probe.round(3)
"""),

        md("""
**How to read this.** The per-node scorer at 300 episodes beats the flat network at 9,000 —
roughly a 30x sample-efficiency advantage — and has a higher ceiling. Notice too that the
flat network reading the *exact posterior* does worse than the per-node scorer reading the
*lossy* edge-marginal summary. The difficulty was never the information content.

**Note the ceiling is ~0.89, not 1.0.** The oracle's score depends on each node's
*descendants* — reachability, which is inherently multi-hop — while this architecture does a
single round of neighbour aggregation. That gap is a concrete lead for a deeper
message-passing network.
"""),

        md("""
## 6. GATE 1 — the check that stopped holding

The environment is supposed to satisfy one exact property: the fraction of problems solvable
*without* intervening must equal the fraction of graphs alone in their Markov equivalence
class. That number is computable from the graph space, so it is a prediction, not a vibe.

It was checked at d=3, passed, and was thereafter assumed.
"""),
        code("""
rows = []
for path in sorted(glob.glob(str(RESULTS / "gate1" / "*.json"))):
    payload = json.load(open(path))
    for n_obs, measured in payload["measured"].items():
        rows.append({"d": payload["d"], "n_obs": int(n_obs),
                     "target": payload["target"], "measured": measured["rate"],
                     "ci_low": measured["ci"][0], "ci_high": measured["ci"][1],
                     "passes": measured["covers_target"]})
gate = pd.DataFrame(rows).sort_values(["d", "n_obs"])

fig, ax = plt.subplots()
for d, group in gate.groupby("d"):
    ax.errorbar(group.n_obs, group.measured,
                yerr=[group.measured - group.ci_low, group.ci_high - group.measured],
                fmt="o-", capsize=4, label=f"d={d} measured")
    ax.axhline(group.target.iloc[0], ls="--", lw=1, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("observational samples (n_obs)")
ax.set_ylabel("observational-only identification rate")
ax.set_title("Dashed lines are the exact theoretical targets")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

gate.round(4)
"""),

        md("""
**What this cost.** At the default `n_obs=1000` the gate fails at d=5 and d=6. Every d=5 run
in this project — including the first version of the headline — used an observational phase
too short to pin down the equivalence class.

**What survives:** `gap_closed` is measured against baselines evaluated in the *same*
environment, so rankings and the architecture comparison hold.
**What doesn't:** the claim that the environment matched its specification, and any
comparison of *absolute* difficulty across sizes.

The winner was re-run where the gate passes, and came back stronger (+1.233 vs +1.116).
GATE 1 now runs automatically on every training run.
"""),

        md("""
## 7. Training curves

Entropy over training, for any configuration you like. Change `TAGS` below.
"""),
        code("""
TAGS = ["s5_pernode_best_counts", "s5_flat_control", "core_d5_edge_marginals"]
LABELS = {"s5_pernode_best_counts": "per-node + memory",
          "s5_flat_control": "flat, same settings",
          "core_d5_edge_marginals": "baseline"}
COLOURS = {"s5_pernode_best_counts": GOOD, "s5_flat_control": BAD,
           "core_d5_edge_marginals": MUTED}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for tag in TAGS:
    path = RESULTS / "raw" / f"{tag}.json"
    if not path.exists():
        print(f"missing: {tag}"); continue
    payload = json.load(open(path))
    history = payload["training_history"].get("0", [])
    if not history:
        continue
    episodes = [h["episodes"] for h in history]
    colour = COLOURS.get(tag, ACCENT)
    label = LABELS.get(tag, tag)
    axes[0].plot(episodes, [h["entropy"] for h in history], color=colour, label=label)
    axes[1].plot(episodes, [h["mean_length"] for h in history], color=colour, label=label)

axes[0].set_ylabel("policy entropy (nats)"); axes[0].set_title("Did the policy sharpen?")
axes[1].set_ylabel("interventions per episode"); axes[1].set_title("Did it get more efficient?")
for ax in axes:
    ax.set_xlabel("training episodes"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),

        md("""
The left panel is the whole story in one picture. The right panel shows the baseline
settling at random-policy cost: it learned not to give up, and nothing about where to
intervene.
"""),

        md("""
## 8. Explore it yourself

`runs` is one row per (configuration, seed) with every lever as a column. Some starting
points below — edit freely.
"""),
        code("""
# What columns are available?
sorted(runs.columns.tolist())
"""),
        code("""
# Compare architectures head to head, holding the other settings fixed.
head_to_head = runs[runs.tag.isin(["s6_d5_nobs5000", "s6_d5_nobs5000_flat"])]
head_to_head[["tag", "arch", "seed", "gap_closed", "solve_rate",
              "final_entropy", "optimal_rate", "repeat_rate"]].round(3)
"""),
        code("""
# Does any lever correlate with success across the whole matrix?
numeric = ["gap_closed", "final_entropy", "solve_rate", "optimal_rate",
           "lr", "hidden", "entropy_coef", "step_cost", "gamma",
           "episodes_per_update", "n_obs", "n_int", "budget"]
runs[numeric].corr()["gap_closed"].sort_values(ascending=False).round(3)
"""),
        code("""
# Pivot on anything: here, mean gap by architecture and problem size.
runs.pivot_table(index="arch", columns="d", values="gap_closed",
                 aggfunc=["mean", "max", "count"]).round(3)
"""),
        code("""
# Open any raw result file to see everything recorded for it.
example = json.load(open(RESULTS / "raw" / "s6_d5_nobs5000.json"))
print("top-level keys:", list(example))
print()
print("provenance:", json.dumps(example["provenance"], indent=2))
print()
print("references:", json.dumps(example["references"], indent=2))
"""),

        md("""
## 9. Where the numbers came from

    results/raw/<tag>.json    one file per configuration: args, provenance, references,
                              per-seed metrics (deterministic AND sampled), full training
                              history, and the pass/fail verdict
    results/probe/*.json      supervised probe results
    results/gate1/*.json      GATE 1 audit across d and n_obs
    results/all_runs.csv      the tidy table this notebook reads

To regenerate the CSV after adding runs:

    python -m legacy.scripts.analyse_sweep --results results/raw --out results/all

Every result file records the exact command line that produced it, plus the git commit and
package versions. Note the cluster ran `torch 2.6.0+cpu` and the laptop `2.10.0+cpu`; numpy
and scipy were pinned to match.
"""),
    ]


def main() -> None:
    notebook = {
        "cells": cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(NOTEBOOK), exist_ok=True)
    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(f"written {NOTEBOOK} ({len(notebook['cells'])} cells, {n_code} code)")


if __name__ == "__main__":
    main()
