"""Every figure in the results chapter, regenerated from the JSON on disk.

WHY A SCRIPT AND NOT A NOTEBOOK. A figure is a claim, and a claim in this project has to be
traceable to the run that produced it. Each function below names its input files, so a figure
can be re-derived after a re-run instead of being trusted because it exists. `notebooks/`
carries a thin wrapper for interactive work; this file is what the thesis builds against.

SHOW THE SEEDS. Where a cell is three seeds, the individual seeds are drawn alongside the
summary. At this sample size the spread IS the result, and a mean over three runs that hides a
collapsed seed is the failure mode this project has hit repeatedly.

Usage:  .venv/bin/python scripts/figures.py [--out thesis/figures]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Serif to sit beside LaTeX body text without looking pasted in. Colours are the Okabe-Ito
# colourblind-safe set; the greys carry the un-emphasised arms.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
})
LEARNED, MYOPIC, RANDOM, THIRD = "#0072B2", "#D55E00", "#999999", "#009E73"
KS = [4, 8, 12, 20, 30]


def _ckpt(k: int, which: str):
    """results/ckpt/k04_best.json etc -- one entry per seed."""
    return json.loads((ROOT / f"results/ckpt/k{k:02d}_{which}.json").read_text())


def _sweep_success(k: int):
    """Joint recovery rate per seed, from the sweep run's own evaluation pass."""
    out = []
    for path in sorted(glob.glob(str(ROOT / f"results/sweep/oracle/k{k:02d}s50n04b150_s*.json"))):
        arms = json.loads(pathlib.Path(path).read_text())["arms"]
        out.append((arms["learned"]["success"], arms["greedy_uncertainty"]["success"]))
    return out


# ---------------------------------------------------------------------------------------
def fig_crossover(out: pathlib.Path):
    """THE headline figure: both criteria change sign between k=8 and k=12."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.0, 5.2), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.15]})

    gaps_mean, gaps_seed = [], []
    for k in KS:
        pairs = _sweep_success(k)
        gaps = [l - g for l, g in pairs]
        gaps_mean.append(np.mean(gaps))
        gaps_seed.append(gaps)
    top.axhline(0, color="black", lw=0.8)
    for x, gaps in zip(KS, gaps_seed):
        top.scatter([x] * len(gaps), gaps, s=14, color=LEARNED, alpha=0.45, zorder=3)
    top.plot(KS, gaps_mean, "o-", color=LEARNED, lw=1.6, ms=5, zorder=4)
    top.set_ylabel("joint recovery rate\nlearned $-$ myopic")
    top.set_title("Both criteria change sign between $k_v=8$ and $k_v=12$")
    top.annotate("myopic sufficient", xy=(5.2, -0.055), fontsize=8, color="#555555")
    top.annotate("learning pays", xy=(21, 0.06), fontsize=8, color="#555555")

    for label, arm, colour in (("learned (selected)", "learned", LEARNED),
                               ("myopic", "greedy", MYOPIC),
                               ("random", "random_vary", RANDOM)):
        means, seeds = [], []
        for k in KS:
            vals = [r["means"][arm]["hard"] for r in _ckpt(k, "best")]
            means.append(np.mean(vals))
            seeds.append(vals)
        # 0 cannot be drawn on a log axis; floor it below the smallest non-zero value and
        # mark it, rather than dropping the point that matters most.
        floor = 1e-5
        for x, vals in zip(KS, seeds):
            bottom.scatter([x] * len(vals), [max(v, floor) for v in vals],
                           s=12, color=colour, alpha=0.4, zorder=3)
        bottom.plot(KS, [max(m, floor) for m in means], "o-", color=colour,
                    lw=1.6, ms=5, label=label, zorder=4)
    bottom.set_yscale("log")
    bottom.set_ylim(7e-6, 2e-1)
    bottom.set_xlabel("window size $k_v$")
    bottom.set_ylabel("SHD on committed marks\n(pooled global graph)")
    bottom.annotate("0 errors in 600 episodes", xy=(20, 1e-5), xytext=(11.5, 3.3e-5),
                    fontsize=7.5, color=LEARNED,
                    arrowprops=dict(arrowstyle="->", color=LEARNED, lw=0.7))
    bottom.legend(loc="upper right", frameon=False)
    for ax in (top, bottom):
        ax.set_xticks(KS)
        ax.axvspan(8, 12, color="black", alpha=0.045, zorder=0)
    fig.tight_layout()
    fig.savefig(out / "crossover.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_checkpoint(out: pathlib.Path):
    """The selected and final policies agree below the crossover and diverge above it."""
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    floor = 1e-5
    for which, label, colour, style in (("best", "selected (early-stopped)", LEARNED, "-"),
                                        ("final", "final update", THIRD, "--")):
        means = [np.mean([r["means"]["learned"]["hard"] for r in _ckpt(k, which)]) for k in KS]
        for k in KS:
            vals = [r["means"]["learned"]["hard"] for r in _ckpt(k, which)]
            ax.scatter([k] * len(vals), [max(v, floor) for v in vals],
                       s=12, color=colour, alpha=0.4, zorder=3)
        ax.plot(KS, [max(m, floor) for m in means], "o" + style, color=colour,
                lw=1.6, ms=5, label=label, zorder=4)
    myopic = [np.mean([r["means"]["greedy"]["hard"] for r in _ckpt(k, "best")]) for k in KS]
    ax.plot(KS, myopic, "-", color=MYOPIC, lw=1.2, label="myopic", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(7e-6, 3e-2)
    ax.set_xticks(KS)
    ax.axvspan(8, 12, color="black", alpha=0.045, zorder=0)
    ax.set_xlabel("window size $k_v$")
    ax.set_ylabel("SHD on committed marks")
    ax.set_title("Checkpoint choice is inert below the crossover and decisive above it")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "checkpoint.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_attribution_law(out: pathlib.Path):
    """Predicted against measured attribution. Source: scripts/attr_model.py, run live."""
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/attr_model.py")],
                          capture_output=True, text=True, cwd=ROOT)
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 7 and parts[0].startswith("k") and parts[1].isdigit():
            # parts[6] is the residual as printed; deriving it from the rounded
            # predicted/measured columns loses a digit and understates the maximum.
            rows.append((parts[0], int(parts[1]), float(parts[4]), float(parts[5]),
                         abs(float(parts[6]))))
    if not rows:
        print("!! attr_model.py produced no parseable rows; skipping the law figure")
        return

    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    lim = 0.95
    ax.plot([0, lim], [0, lim], color="black", lw=0.8, zorder=1)
    applies = [r for r in rows if r[1] > 1]
    excluded = [r for r in rows if r[1] <= 1]
    ax.scatter([r[2] for r in applies], [r[3] for r in applies], s=42, color=LEARNED,
               zorder=4, label="two or more peers")
    ax.scatter([r[2] for r in excluded], [r[3] for r in excluded], s=42, facecolors="none",
               edgecolors=MYOPIC, zorder=4, label="one peer (model excludes)")
    for cell, peers, pred, meas, _r in rows:
        ax.annotate(cell.replace("k", "$k$="), (pred, meas), fontsize=6.5,
                    xytext=(4, -3), textcoords="offset points", color="#555555")
    resid = max(r for _, peers, _p, _m, r in rows if peers > 1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"predicted: $0.76 \times$ share of single-pair latents")
    ax.set_ylabel("measured share attributed")
    ax.set_title(f"Closed form predicts attribution\nlargest residual {resid:.3f} where the "
                 f"model applies", fontsize=9)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "attribution_law.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_federation(out: pathlib.Path):
    """RQ3: coordination strategies, and the cost of partitioning the learner."""
    def arm(pattern, key):
        vals = []
        for path in sorted(glob.glob(str(ROOT / pattern))):
            arms = json.loads(pathlib.Path(path).read_text())["arms"]
            if arms.get(key):
                vals.append(arms[key]["success"])
        return vals

    cells = [("$k_v=12$", "results/central/v2_k12_{a}_s*.json"),
             ("$k_v=20$", "results/central/v2_k20_{a}_s*.json")]
    series = [("random", "random_vary", RANDOM),
              ("myopic, fixed partition", "greedy_partitioned", THIRD),
              ("myopic, uncoordinated", "greedy_uncertainty", MYOPIC),
              ("learned (federated)", "learned", LEARNED)]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, (title, pattern) in zip(axes, cells):
        for index, (label, key, colour) in enumerate(series):
            vals = arm(pattern.format(a="A"), key)
            if not vals:
                continue
            ax.bar(index, np.mean(vals), 0.62, color=colour, alpha=0.85, zorder=2)
            ax.scatter([index] * len(vals), vals, s=13, color="black", alpha=0.55, zorder=4)
        central = arm(pattern.format(a="E"), "learned")
        if central:
            ax.bar(len(series), np.mean(central), 0.62, color=LEARNED, alpha=0.4,
                   hatch="//", edgecolor=LEARNED, zorder=2)
            ax.scatter([len(series)] * len(central), central, s=13, color="black",
                       alpha=0.55, zorder=4)
        ax.set_xticks(range(len(series) + 1))
        ax.set_xticklabels([s[0] for s in series] + ["learned (centralised)"],
                           rotation=32, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("joint recovery rate")
    fig.suptitle("Learned coordination beats convention; centralising the learner adds nothing",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "federation.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------
CELL_RE = re.compile(r"k(\d+)s(\d+)n(\d+)b(\d+)")
WINDOW_FLOOR, WINDOW_TAIL = 0.70, 10


def _sweep_rows():
    """One row per (cell, seed), with the competence-floor statistic attached."""
    rows = []
    for path in sorted((ROOT / "results/sweep/oracle").glob("*.json")):
        m = CELL_RE.match(path.stem)
        if not m:
            continue
        d = json.loads(path.read_text())
        if "arms" not in d:
            continue
        tail = [h.get("window_rate", 0.0) for h in (d.get("history") or [])[-WINDOW_TAIL:]]
        rows.append(dict(k=int(m[1]), sigma=int(m[2]) / 100, n=int(m[3]), beta=int(m[4]) / 100,
                         seed=d.get("seed"),
                         wr=(sum(tail) / len(tail) if tail else 0.0), arms=d["arms"]))
    return rows


def fig_sweep_grid(out: pathlib.Path):
    """THE backbone figure: every swept axis against both baselines, on both metrics.

    Positions are evenly spaced rather than linear. Beta runs 1.0 to 5.0 and a linear axis
    crushes the low end, which is where the cells that separate the arms actually sit.
    """
    rows = _sweep_rows()
    axes_spec = [
        ("window size $k_v$", "k",
         lambda r: r["sigma"] == .5 and r["n"] == 4 and r["beta"] == 1.5),
        ("agents $K$", "n",
         lambda r: r["k"] == 12 and r["sigma"] == .5 and r["beta"] == 1.5),
        ("contended fraction $\\sigma$", "sigma",
         lambda r: r["k"] == 12 and r["n"] == 4 and r["beta"] == 1.5),
        ("budget multiplier $\\beta$", "beta",
         lambda r: r["k"] == 12 and r["sigma"] == .5 and r["n"] == 4),
    ]
    series = [("learned", "learned", LEARNED, "o", 1.9),
              ("myopic (greedy)", "greedy_uncertainty", MYOPIC, "s", 1.5),
              ("random", "random_vary", RANDOM, "^", 1.3)]
    metrics = [("global_hard_shd", "SHD on committed marks\n(lower is better)", True),
               ("success", "joint recovery rate\n(higher is better)", False)]
    floor = 1e-5

    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.8))
    for col, (xlabel, key, keep) in enumerate(axes_spec):
        sel = [r for r in rows if keep(r)]
        xs = sorted({r[key] for r in sel})
        pos = list(range(len(xs)))
        gone = [r for r in sel if r["wr"] < WINDOW_FLOOR]
        for row, (mkey, ylabel, logy) in enumerate(metrics):
            ax = axes[row][col]
            ceiling = [np.mean([r["arms"]["oracle_cover"][mkey] for r in sel
                                if r[key] == x and "oracle_cover" in r["arms"]] or [np.nan])
                       for x in xs]
            ax.plot(pos, [max(v, floor) if logy else v for v in ceiling], ls=":", color=THIRD,
                    lw=1.2, zorder=2,
                    label="full-coverage reference" if (row == 0 and col == 0) else None)
            for label, arm, colour, marker, lw in series:
                means, seeds = [], []
                for x in xs:
                    vals = [r["arms"][arm][mkey] for r in sel
                            if r[key] == x and r["wr"] >= WINDOW_FLOOR and arm in r["arms"]]
                    means.append(np.mean(vals) if vals else np.nan)
                    seeds.append(vals)
                for p_, vals in zip(pos, seeds):
                    ax.scatter([p_] * len(vals),
                               [max(v, floor) if logy else v for v in vals],
                               s=11, color=colour, alpha=.35, zorder=3)
                ax.plot(pos, [max(m, floor) if logy else m for m in means], marker=marker,
                        ls="-", color=colour, lw=lw, ms=4.5, zorder=4,
                        label=label if (row == 0 and col == 0) else None)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(floor * .7, 3e-1)
                ax.axhspan(floor * .7, floor * 1.6, color="black", alpha=.05, zorder=0)
            else:
                ax.set_ylim(-.03, 1.05)
            ax.set_xticks(pos)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.set_xlim(-.4, len(xs) - .6)
            if row == 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
        if gone:
            axes[1][col].text(.98, .06,
                              f"{len(gone)} seed{'s' if len(gone) > 1 else ''} excluded",
                              transform=axes[1][col].transAxes, ha="right", fontsize=7,
                              color="#B00020")
    axes[0][0].legend(frameon=False, fontsize=7.6, loc="lower left", handlelength=1.6)
    fig.suptitle("Learned experiment selection against myopic and random baselines, "
                 "on every swept axis\n"
                 "Final-policy evaluation, 3 seeds per cell; runs below the competence floor "
                 "excluded", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "sweep_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for name, fn in (("sweep_grid", fig_sweep_grid),
                     ("crossover", fig_crossover), ("checkpoint", fig_checkpoint),
                     ("attribution_law", fig_attribution_law), ("federation", fig_federation)):
        try:
            fn(out)
            print(f"  wrote {name}.pdf")
        except Exception as exc:                    # a missing cell must not kill the rest
            print(f"  !! {name}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
