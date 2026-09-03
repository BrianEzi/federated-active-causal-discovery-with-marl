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
# FIGURE_GUIDELINES.md, applied 3 Sep. Author at PRINT size: \textwidth is 5.40 in, so every
# figure is authored at FULL, TWOTHIRD or HALF and included at 1.0, 0.667 or 0.5 \textwidth
# with no scaling. Fonts below are therefore rendered sizes: 9 pt labels, 8 pt ticks, nothing
# under 8 pt on the page. Grids wider than FULL do not exist; they are split into subfigure
# panels (sweep_grid_[abcd].pdf, federation_[ab].pdf).
FULL, TWOTHIRD, HALF = 5.40, 3.60, 2.70

# In-figure titles are off: the caption carries the message, per standard practice and rule 7
# of the guidelines. One switch rather than deleted calls, so Brian can reverse it in one line.
SHOW_TITLES = False


def _title(ax, text, **kw):
    if SHOW_TITLES:
        ax.set_title(text, **kw)


def _suptitle(fig, text, **kw):
    if SHOW_TITLES:
        fig.suptitle(text, **kw)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})
LEARNED, MYOPIC, RANDOM, THIRD = "#0072B2", "#D55E00", "#999999", "#009E73"
KS = [4, 8, 12, 20, 30]


# Where a 12,000-episode measurement of the window axis lives. k=4, 8 and 12 were retrained
# into results/sweep12k/; k=20 and k=30 were always at 12,000 and are measured in
# results/rerows/. Both were produced by scripts/global_shd_paired.py under a seeded
# evaluation. results/ckpt/ is the superseded pre-fix set and is not read by anything here.
CKPT12 = {4:  "results/sweep12k/shd{fin}/k04s50n04b150.json",
          8:  "results/sweep12k/shd{fin}/k08s50n04b150.json",
          12: "results/sweep12k/shd{fin}/k12s50n04b150.json",
          20: "results/rerows/k20_{which}.json",
          30: "results/rerows/k30_{which}.json"}


def _ckpt(k: int, which: str):
    """One entry per seed, at 12,000 training episodes, for checkpoint convention `which`."""
    tmpl = CKPT12[k]
    path = ROOT / tmpl.format(which=which, fin="" if which == "best" else "_final")
    if not path.exists():
        raise FileNotFoundError(f"no 12,000-episode {which} measurement for k={k}: {path}")
    return json.loads(path.read_text())


def _sweep_success(k: int, budget: int = 4000):
    """Joint recovery rate per seed, from the run's own evaluation pass.

    WHICH BUDGET, AND WHY IT HAS TO BE A PARAMETER. `results/sweep/oracle/` is NOT one design:
    k=4, 8 and 12 trained for 4,000 episodes and k=20 and 30 for 12,000. Drawing all five from
    that directory puts a budget change inside the x-axis, so a line through it shows window
    size and training budget together. `results/sweep12k/` holds k=4, 8 and 12 at 12,000; the
    two largest windows are the same files either way, because they were always at 12,000.
    """
    d = "results/sweep12k" if budget == 12000 else "results/sweep/oracle"
    paths = sorted(glob.glob(str(ROOT / d / f"k{k:02d}s50n04b150_s*.json")))
    if not paths and budget == 12000:          # k=20 and k=30 live only in the sweep directory
        paths = sorted(glob.glob(str(ROOT / f"results/sweep/oracle/k{k:02d}s50n04b150_s*.json")))
    out = []
    for path in paths:
        arms = json.loads(pathlib.Path(path).read_text())["arms"]
        out.append((arms["learned"]["success"], arms["greedy_uncertainty"]["success"]))
    return out


# ---------------------------------------------------------------------------------------
def _recovery(k: int, budget: int):
    """Joint recovery per seed: (learned, greedy) pairs on the run_arm convention.

    Recovery comes from the run's own 200-episode eval pass at the FINAL update
    (ma/evaluate.py::run_arm), because that is the only place it was ever recorded; the
    paired SHD tool records SHD fields only. Guarded by the recorded train_episodes because
    results/sweep/oracle mixes designs: k<=12 trained for 4,000 episodes, and the k=20/30
    run JSONs were overwritten by the 12,000-episode copies on 1 Sep. The 4,000-episode
    policies at k=20 and 30 therefore have no run file at all; their recovery is measured
    from the surviving u0249 checkpoints (update 249 = episode 4,000 exactly) by
    scripts/recovery_paired.py, which calls the same run_arm on the same episode seeds.
    """
    if budget == 4000 and k in (20, 30):
        path = ROOT / f"results/rerows/k{k}_u0249_recovery.json"
        if not path.exists():
            return []
        return [(e["arms"]["learned"]["success"],
                 e["arms"]["greedy_uncertainty"]["success"])
                for e in json.loads(path.read_text())]
    d = "results/sweep12k" if budget == 12000 else "results/sweep/oracle"
    paths = sorted(glob.glob(str(ROOT / d / f"k{k:02d}s50n04b150_s*.json")))
    if not paths and budget == 12000:      # k=20 and 30 live only in the sweep directory
        paths = sorted(glob.glob(str(ROOT / f"results/sweep/oracle/k{k:02d}s50n04b150_s*.json")))
    out = []
    for path in paths:
        r = json.loads(pathlib.Path(path).read_text())
        if r.get("config", {}).get("train_episodes") != budget:
            continue
        out.append((r["arms"]["learned"]["success"],
                    r["arms"]["greedy_uncertainty"]["success"]))
    return out


def _shd4(k: int):
    """Hard SHD per seed for the policy AFTER 4,000 episodes, one convention along the line:
    the final update of a 4,000-episode run, which for k=20 and 30 is the u0249 checkpoint."""
    path = (ROOT / f"results/rerows/k{k}_u0249.json" if k in (20, 30)
            else ROOT / f"results/rerows/k{k:02d}_final.json")
    if not path.exists():
        return []
    return [e["means"]["learned"]["hard"] for e in json.loads(path.read_text())]


def fig_window_budget(out: pathlib.Path):
    """The window axis at both training budgets: one figure, both reported metrics.

    Replaces the crossover, checkpoint and crossover_budget figures (3 Sep, Brian's call):
    the recovery-gap panel repeated what this figure's recovery panel shows, the old SHD
    panel repeated the sweep grid's window panel without the budget comparison that
    justifies it, and selected-against-final lives in tab:checkpoint. The story is one
    sentence -- at 4,000 episodes the learned arm trails the myopic rule almost everywhere,
    at 12,000 it trails nowhere -- and it needs both budgets on the same axes, not three
    figures.

    CONVENTIONS, stated here because the two panels cannot share one. Recovery is the run's
    own eval pass at the final update (the only checkpoint it was recorded at). SHD is the
    seeded paired evaluation: selected checkpoint at 12,000 (the chapter's reporting
    convention), final update at 4,000 (the policy after exactly 4,000 episodes). The
    myopic rule does not train, so it is one line per panel, not one per budget.
    """
    fig, (top, bot) = plt.subplots(1, 2, figsize=(FULL, 3.0))
    L4 = RANDOM   # grey: the undertrained policy, matching the retired budget figure

    # (a) joint recovery rate
    for budget, colour, style, label in ((4000, L4, "o--", "learned, 4,000 episodes"),
                                         (12000, LEARNED, "o-", "learned, 12,000 episodes")):
        xs, means = [], []
        for k in KS:
            pairs = _recovery(k, budget)
            if not pairs:
                continue
            vals = [l for l, _ in pairs]
            top.scatter([k] * len(vals), vals, s=12, color=colour, alpha=0.4, zorder=3)
            xs.append(k)
            means.append(np.mean(vals))
        top.plot(xs, means, style, color=colour, lw=1.6, ms=5, label=label, zorder=4)
    my_rec = []
    for k in KS:
        vals = [g for _, g in _recovery(k, 12000)]
        top.scatter([k] * len(vals), vals, s=12, color=MYOPIC, alpha=0.4, zorder=2)
        my_rec.append(np.mean(vals))
    top.plot(KS, my_rec, "-", color=MYOPIC, lw=1.4, label="myopic", zorder=2)
    top.set_ylabel(r"joint recovery rate ($\uparrow$)")
    top.set_ylim(0, 1.02)
    top.legend(loc="lower left", frameon=False)
    _title(top, "The sign change belongs to the budget, not the window")

    # (b) SHD on committed marks
    floor = 1e-5
    for getter, colour, style, label in (
            (_shd4, L4, "o--", "learned, 4,000 episodes"),
            (lambda k: [r["means"]["learned"]["hard"] for r in _ckpt(k, "best")],
             LEARNED, "o-", "learned, 12,000 episodes (selected)")):
        xs, means = [], []
        for k in KS:
            vals = getter(k)
            if not vals:
                continue
            bot.scatter([k] * len(vals), [max(v, floor) for v in vals],
                        s=12, color=colour, alpha=0.4, zorder=3)
            xs.append(k)
            means.append(np.mean(vals))
        bot.plot(xs, [max(m, floor) for m in means], style, color=colour,
                 lw=1.6, ms=5, label=label, zorder=4)
    my_shd = [np.mean([r["means"]["greedy"]["hard"] for r in _ckpt(k, "best")]) for k in KS]
    bot.plot(KS, my_shd, "-", color=MYOPIC, lw=1.4, label="myopic", zorder=2)
    bot.set_yscale("log")
    bot.set_ylim(7e-6, 2e-1)
    for ax in (top, bot):
        ax.set_xlabel("window size $k_v$")
    bot.set_ylabel("SHD on committed marks ($\\downarrow$)\n(pooled global graph)")
    bot.annotate("0 errors in 600 episodes", xy=(20, 1e-5), xytext=(11.5, 3.3e-5),
                 fontsize=7.5, color=LEARNED,
                 arrowprops=dict(arrowstyle="->", color=LEARNED, lw=0.7))
    for ax in (top, bot):
        ax.set_xticks(KS)
    fig.tight_layout()
    fig.savefig(out / "window_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_nint(out: pathlib.Path):
    """Transfer to sampled evidence as the interventional sample size grows.

    The k=8 policies trained at 12,000 episodes under oracle evidence, evaluated at the
    selected checkpoint under SAMPLED evidence with n_int swept from 10 to 10,000 --
    Brian's request, 3 Sep. Every point is scripts/global_shd_paired.py with
    --override_n_int, 200 paired episodes, sampled action selection, three seeds. The
    baselines re-run at every n_int because their beliefs read the same test statistics.
    """
    grid = [10, 30, 100, 200, 1000, 3000, 10000]
    files = {}
    for n in grid:
        got = sorted((ROOT / "results/nint_curve").glob(f"nint{n:05d}_s*.json"))
        if got:
            files[n] = got
    if not files:
        raise FileNotFoundError("no results/nint_curve measurements yet")
    incomplete = {n: len(v) for n, v in files.items() if len(v) < 3}
    if incomplete or len(files) < len(grid):
        # Partial grids have said the opposite of complete ones four times this week; draw
        # nothing rather than a curve that will be quoted.
        raise RuntimeError(f"nint grid incomplete: {sorted(files)} of {grid}, "
                           f"short cells {incomplete} -- not drawing a partial curve")
    fig, ax = plt.subplots(figsize=(FULL, 3.2))
    floor = 1e-5
    for arm, colour, label in (("learned", LEARNED, "learned (oracle-trained, transferred)"),
                               ("greedy", MYOPIC, "myopic"),
                               ("random_vary", RANDOM, "random")):
        means = []
        for n in grid:
            vals = [json.loads(f.read_text())[0]["means"][arm]["hard"] for f in files[n]]
            ax.scatter([n] * len(vals), [max(v, floor) for v in vals],
                       s=12, color=colour, alpha=0.4, zorder=3)
            means.append(np.mean(vals))
        ax.plot(grid, [max(m, floor) for m in means], "o-", color=colour, lw=1.6, ms=5,
                label=label, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(grid)
    ax.set_xticklabels([str(n) for n in grid], fontsize=7.5)
    ax.set_xlabel(r"interventional samples per intervention $n_{\mathrm{int}}$")
    ax.set_ylabel(r"SHD on committed marks ($\downarrow$)")
    _title(ax, "Evidence quality is a sample-size dial at evaluation time")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "nint.pdf", bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(TWOTHIRD, TWOTHIRD))
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
    _title(ax, f"Closed form predicts attribution\nlargest residual {resid:.3f} where the "
                 f"model applies", fontsize=9)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "attribution_law.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_federation(out: pathlib.Path):
    """RQ3, split per FIGURE_GUIDELINES.md rule 4 into two print-true subfigure panels.

    (a) coordination strategies at both cells, authored FULL; (b) the six-seed paired
    comparison on the primary metric, authored HALF. The old 9.6-inch three-panel version
    printed its text at 5.0 pt.
    """
    def arm(pattern, key):
        vals = []
        for path in sorted(glob.glob(str(ROOT / pattern))):
            arms = json.loads(pathlib.Path(path).read_text())["arms"]
            if arms.get(key):
                vals.append(arms[key]["success"])
        return vals

    # BUDGETS DIFFER BETWEEN THESE ROWS AND THE FIGURE SAYS SO. The k=12 ladder is the
    # 12,000-episode retrain set once all twelve exist; the k=20 ladder was always at 12,000.
    twelve = sorted((ROOT / "results/central12k").glob("v2_k12_?_s?.json"))
    k12 = ("results/central12k/v2_k12_{a}_s?.json" if len(twelve) >= 12
           else "results/central/v2_k12_{a}_s*.json")
    cells = [("$k_v=12$", k12), ("$k_v=20$", "results/central/v2_k20_{a}_s*.json")]
    series = [("random", "random_vary", RANDOM),
              ("myopic, fixed partition", "greedy_partitioned", THIRD),
              ("myopic, uncoordinated", "greedy_uncertainty", MYOPIC),
              ("learned (federated)", "learned", LEARNED)]

    # (a) coordination strategies.
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.2), sharey=True)
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
        ax.set_xticklabels([s_[0] for s_ in series] + ["learned (centralised)"],
                           rotation=32, ha="right", fontsize=8)
        _title(ax, title)
        ax.annotate(title, xy=(0.04, 0.92), xycoords="axes fraction", fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel(r"joint recovery rate ($\uparrow$)")
    fig.tight_layout()
    fig.savefig(out / "federation_a.pdf", bbox_inches="tight")
    plt.close(fig)

    # (b) the six-seed paired panel, from the measured 12,000-episode ladder.
    lad = {}
    for k in ("A_best", "E_best"):
        q = ROOT / f"results/rerows/ladder12k_{k}.json"
        if q.exists():
            lad[k] = {r["seed"]: r for r in json.loads(q.read_text())}
    if len(lad) != 2:
        print("!! ladder measurements absent; federation_b skipped")
        return
    fig, ax3 = plt.subplots(figsize=(HALF, 2.9))
    seeds = sorted(lad["A_best"])
    ds, ses = [], []
    for sd in seeds:
        x = np.array(lad["A_best"][sd]["rows"]["learned"]["hard"])
        y = np.array(lad["E_best"][sd]["rows"]["learned"]["hard"])
        d = x - y
        ds.append(d.mean())
        ses.append(d.std(ddof=1) / np.sqrt(len(d)))
    ax3.axhline(0, color="black", lw=0.8, zorder=1)
    ax3.errorbar(seeds, ds, yerr=[2 * e for e in ses], fmt="o", color=LEARNED,
                 ms=4.5, lw=1.1, capsize=2.5, zorder=3)
    lim = max(abs(d) + 2 * e for d, e in zip(ds, ses)) * 1.3
    ax3.set_ylim(-lim, lim)
    ax3.set_xticks(seeds)
    ax3.set_xlabel("seed")
    ax3.set_ylabel("paired difference in SHD ($\\downarrow$)\nfederated $-$ centralised", fontsize=8)
    ax3.annotate("above 0: centralising wins", xy=(0.04, 0.92), xycoords="axes fraction",
                 fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(out / "federation_b.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------
CELL_RE = re.compile(r"k(\d+)s(\d+)n(\d+)b(\d+)")
WINDOW_FLOOR, WINDOW_TAIL = 0.70, 10


def _sweep_rows():
    """One row per (cell, seed), with the competence-floor statistic attached."""
    # 12,000 episodes everywhere. results/sweep12k/ holds the 18 retrained cells; k=20 and
    # k=30 exist only in results/sweep/oracle/ and were always trained at 12,000, so they are
    # added from there. Nothing at 4,000 episodes reaches this figure.
    rows = []
    seen = set()
    sources = list(sorted((ROOT / "results/sweep12k").glob("k*s*n*b*_s?.json")))
    sources += [q for q in sorted((ROOT / "results/sweep/oracle").glob("k*s*n*b*_s?.json"))
                if q.stem.startswith(("k20", "k30"))]
    for path in sources:
        m = CELL_RE.match(path.stem)
        if not m or path.stem in seen:
            continue
        seen.add(path.stem)
        d = json.loads(path.read_text())
        if "arms" not in d or d.get("config", {}).get("train_episodes") != 12000:
            continue
        tail = [h.get("window_rate", 0.0) for h in (d.get("history") or [])[-WINDOW_TAIL:]]
        cell = re.match(r"(k\d+s\d+n\d+b\d+)_s(\d)", path.stem).group(1)
        rows.append(dict(k=int(m[1]), sigma=int(m[2]) / 100, n=int(m[3]), beta=int(m[4]) / 100,
                         seed=d.get("seed"), cell=cell,
                         wr=(sum(tail) / len(tail) if tail else 0.0), arms=d["arms"],
                         shd=_measured_shd(cell, d.get("seed"))))
    return rows


# Cells whose selected-checkpoint measurement was taken during the undertraining work and
# lives outside results/sweep12k/shd/. Same script, same conventions.
_PRE12 = {"k12s50n05b150": "results/longcheck/shd_n05_12k.json",
          "k12s75n04b150": "results/longcheck/shd_s75_12k.json",
          "k12s50n08b150": "results/longcheck/shd_n08_12k.json",
          "k12s50n10b150": "results/longcheck/shd_n10_12k.json"}
_SHD_CACHE: dict = {}


def _measured_shd(cell: str, seed):
    """Per-arm structural distance from a PAIRED MEASUREMENT, not from the run's own field.

    WHY THIS EXISTS. Each training run records `global_hard_shd` from its own evaluation pass,
    which scores the policy at its last update. That is not what the thesis reports, and on a
    long run the two differ by up to a factor of 300 on the same seed -- the defect that put a
    ratio of 20.79 in a table where the measured value is 0.06. Every SHD panel in this file
    now reads scripts/global_shd_paired.py output at the selected checkpoint instead.
    """
    if cell not in _SHD_CACHE:
        if cell in _PRE12:
            q = ROOT / _PRE12[cell]
        elif cell.startswith(("k20", "k30")):
            q = ROOT / f"results/rerows/{cell[:3]}_best.json"
        else:
            q = ROOT / f"results/sweep12k/shd/{cell}.json"
        _SHD_CACHE[cell] = ({e["seed"]: e["means"] for e in json.loads(q.read_text())}
                            if q.exists() else {})
    means = _SHD_CACHE[cell].get(seed)
    if not means:
        return {}
    return {"learned": means["learned"]["hard"], "greedy_uncertainty": means["greedy"]["hard"],
            "random_vary": means["random_vary"]["hard"]}


def fig_sweep_grid(out: pathlib.Path):
    """The backbone result, split into four half-width subfigure panels, one per swept axis.

    FIGURE_GUIDELINES.md rule 4: the old single 12-inch grid printed its 9 pt text at 4.0 pt.
    Each panel is authored at HALF (2.70 in), the width it prints in a 2x2 subcaption block,
    so its fonts are print-true. Panel (a) carries the legend for all four.

    Positions are evenly spaced rather than linear. Beta runs 1.0 to 5.0 and a linear axis
    crushes the low end, which is where the cells that separate the arms actually sit.
    """
    rows = _sweep_rows()
    axes_spec = [
        ("a", "window size $k_v$", "k",
         lambda r: r["sigma"] == .5 and r["n"] == 4 and r["beta"] == 1.5),
        ("b", "agents $K$", "n",
         lambda r: r["k"] == 12 and r["sigma"] == .5 and r["beta"] == 1.5),
        ("c", "contended fraction $\\sigma$", "sigma",
         lambda r: r["k"] == 12 and r["n"] == 4 and r["beta"] == 1.5),
        ("d", "budget multiplier $\\beta$", "beta",
         lambda r: r["k"] == 12 and r["sigma"] == .5 and r["n"] == 4),
    ]
    series = [("learned", "learned", LEARNED, "o", 1.9),
              ("myopic (greedy)", "greedy_uncertainty", MYOPIC, "s", 1.5),
              ("random", "random_vary", RANDOM, "^", 1.3)]
    metrics = [("global_hard_shd", "SHD on committed\nmarks ($\\downarrow$)", True),
               ("success", "joint recovery\nrate ($\\uparrow$)", False)]
    floor = 1e-5

    for tag, xlabel, key, keep in axes_spec:
        sel = [r for r in rows if keep(r)]
        xs = sorted({r[key] for r in sel})
        pos = list(range(len(xs)))
        gone = [r for r in sel if r["wr"] < WINDOW_FLOOR]
        fig, panel = plt.subplots(2, 1, figsize=(HALF, 3.45), sharex=True,
                                  gridspec_kw={"hspace": 0.14})
        for row, (mkey, ylabel, logy) in enumerate(metrics):
            ax = panel[row]
            # The full-coverage reference exists only as a run-recorded field, so it is drawn
            # on the recovery row and omitted from the SHD row rather than mixing a recorded
            # number into a panel of measured ones.
            if mkey != "global_hard_shd":
                ceiling = [np.mean([r["arms"]["oracle_cover"][mkey] for r in sel
                                    if r[key] == x and "oracle_cover" in r["arms"]] or [np.nan])
                           for x in xs]
                ax.plot(pos, ceiling, ls=":", color=THIRD, lw=1.2, zorder=2,
                        label="full coverage" if (row == 1 and tag == "a") else None)
            for label, arm, colour, marker, lw in series:
                means, seeds = [], []
                for x in xs:
                    if mkey == "global_hard_shd":
                        vals = [r["shd"][arm] for r in sel
                                if r[key] == x and r["wr"] >= WINDOW_FLOOR and arm in r["shd"]]
                    else:
                        vals = [r["arms"][arm][mkey] for r in sel
                                if r[key] == x and r["wr"] >= WINDOW_FLOOR and arm in r["arms"]]
                    means.append(np.mean(vals) if vals else np.nan)
                    seeds.append(vals)
                for p_, vals in zip(pos, seeds):
                    ax.scatter([p_] * len(vals),
                               [max(v, floor) if logy else v for v in vals],
                               s=10, color=colour, alpha=.35, zorder=3)
                ax.plot(pos, [max(m, floor) if logy else m for m in means], marker=marker,
                        ls="-", color=colour, lw=lw, ms=4,
                        label=label if (row == 1 and tag == "a") else None, zorder=4)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(floor * .7, 3e-1)
                ax.axhspan(floor * .7, floor * 1.6, color="black", alpha=.05, zorder=0)
            else:
                ax.set_ylim(-.03, 1.05)
            ax.set_xticks(pos)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.set_xlim(-.4, len(xs) - .6)
            # Every panel prints alone, so every panel is self-describing: no shared-edge
            # label suppression across subfigures.
            ax.set_ylabel(ylabel, fontsize=8)
        panel[1].set_xlabel(xlabel)
        if gone:
            panel[1].text(.97, .07,
                          f"{len(gone)} seed{'s' if len(gone) > 1 else ''} excluded",
                          transform=panel[1].transAxes, ha="right", fontsize=8,
                          color="#B00020")
        if tag == "a":
            # In the recovery panel's empty middle band: random sits near zero and the other
            # arms above 0.8, so the centre of that panel is the one region nothing crosses.
            panel[1].legend(frameon=False, fontsize=8, loc="center right", handlelength=1.5)
        fig.savefig(out / f"sweep_grid_{tag}.pdf", bbox_inches="tight")
        plt.close(fig)


def fig_pair_class(out: pathlib.Path):
    """Errors by pair class at both training budgets, which is what the table said in numbers.

    Counts rather than rates. The unscored class has 27,000 observations against 673,200, so a
    rate compresses eleven errors and zero errors into two numbers a reader cannot tell apart.
    The myopic and random arms do not train, so they are drawn as horizontal references: the
    figure is one arm moving against two that cannot.
    """
    src = {4000: ROOT / "results/shd_by_class_naxis_det.json",
           12000: ROOT / "results/shd_by_class_naxis_12k.json"}
    if not all(q.exists() for q in src.values()):
        print("!! pair-class data incomplete; skipping")
        return
    tot = {}
    for budget, q in src.items():
        d = json.loads(q.read_text())
        for arm in ("learned", "greedy", "random"):
            tot[(budget, arm)] = (
                sum(e["arms"][arm]["private_incident"] * e["arms"][arm]["n_private"] for e in d),
                sum(e["arms"][arm]["shared_shared"] * e["arms"][arm]["n_shared"] for e in d))
    n_priv = sum(e["arms"]["learned"]["n_private"]
                 for e in json.loads(src[12000].read_text()))
    n_shar = sum(e["arms"]["learned"]["n_shared"]
                 for e in json.loads(src[12000].read_text()))

    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL, 2.9))
    budgets = [4000, 12000]
    xs = [0, 1]
    for ax, idx, title, denom in ((left, 0, f"scored pairs ({n_priv:,} observations)", n_priv),
                                  (right, 1, f"unscored pairs ({n_shar:,} observations)", n_shar)):
        # Offsets differ per arm: learned and myopic converge on the scored panel and their
        # labels would sit on top of each other at a shared offset.
        # Which arms crowd each other differs between the panels: on the scored panel the
        # learned and myopic lines converge, on the unscored panel the learned and random
        # lines do. Offsets are set per panel rather than per arm for that reason.
        offs = {"learned": 9, "greedy": -14, "random": 9} if idx == 0 else \
               {"learned": 9, "greedy": 9, "random": -14}
        for arm, label, colour in (("learned", "learned", LEARNED),
                                   ("greedy", "myopic", MYOPIC),
                                   ("random", "random", RANDOM)):
            dy = offs[arm]
            ys = [tot[(b, arm)][idx] for b in budgets]
            style = "o-" if arm == "learned" else "o--"
            ax.plot(xs, ys, style, color=colour, lw=1.8 if arm == "learned" else 1.1,
                    ms=5, label=label, zorder=4 if arm == "learned" else 2)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}", (x, y), fontsize=7.5, color=colour,
                            xytext=(0, dy), textcoords="offset points", ha="center",
                            zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(["4,000", "12,000"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_xlabel("training episodes")
        _title(ax, title, fontsize=9)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-0.5, 1.2e5)
    left.set_ylabel(r"errors committed ($\downarrow$)")
    left.legend(loc="lower left", frameon=False)
    _suptitle(fig, "Training moves the class the policy is scored on, and not the other",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "pair_class.pdf", bbox_inches="tight")
    plt.close(fig)



def fig_answer_rate(out: pathlib.Path):
    """The dose-response curve: transfer against the answer rate the policy trained under.

    The title says "improves transfer" and not "and saturates". The flattening at the left is
    visible, but the gradient per unit rho is not smooth -- it nearly stops between 0.90 and
    0.85 and resumes -- and with three seeds per rate the plateau is not established as real.

    Source is results/power/rho/deterministic/, which is authoritative; the sibling directory
    is the pre-fix grid and is superseded. The myopic arm is one number at every rate by
    construction -- the same per-episode vectors are reused across rates -- so it is drawn as a
    single reference line, and its being identical is itself the check that the comparison is
    paired over the same episodes.
    """
    src = sorted(glob.glob(str(ROOT / "results/power/rho/deterministic/xfer_rho*_s?.json")))
    if not src:
        print("!! answer-rate grid absent; skipping")
        return
    per = {}
    for f in src:
        rho = float(re.search(r"rho([\d.]+)_", pathlib.Path(f).stem).group(1))
        for e in json.loads(pathlib.Path(f).read_text()):
            per.setdefault(rho, []).append((e["means"]["learned"]["hard"],
                                            e["means"]["greedy"]["hard"],
                                            e["paired"]["learned-greedy"]["delta"],
                                            e["paired"]["learned-greedy"]["significant"]))
    # LINEAR in rho, not evenly spaced. Even spacing reads better -- the rates are 0.50, 0.70,
    # 0.80, 0.85, 0.90, 0.95, 1.00 and the upper four crowd -- but it misrepresents the shape.
    # The 0.70->0.50 step spans four times the rho interval of 0.90->0.85, so even spacing
    # makes a curve whose gradient per unit rho is NOT smooth look like smooth saturation.
    # Agent B found the same distortion in the numeric version of this claim on 3 Sep.
    # Legibility is not worth a false shape on a dose-response figure.
    rhos = sorted(per)
    pos = rhos
    at = {r: r for r in rhos}
    fig, (top, bottom) = plt.subplots(1, 2, figsize=(FULL, 2.9))
    myopic = np.mean([v[1] for r in rhos for v in per[r]])
    top.axhline(myopic, color=MYOPIC, lw=1.3, ls="--", zorder=2,
                label=f"myopic ({myopic:.4f}, all rates)")
    means = [np.mean([v[0] for v in per[r]]) for r in rhos]
    for r in rhos:
        top.scatter([at[r]] * len(per[r]), [v[0] for v in per[r]], s=14, color=LEARNED,
                    alpha=0.4, zorder=3)
    top.plot(pos, means, "o-", color=LEARNED, lw=1.7, ms=5, zorder=4, label="learned")
    top.set_ylabel(r"SHD on committed marks ($\downarrow$)")
    top.legend(loc="upper right", frameon=False, fontsize=7)
    _title(top, "Degrading the training evidence improves transfer")

    bottom.axhline(0, color="black", lw=0.8, zorder=2)
    for r in rhos:
        for _l, _g, d, sig in per[r]:
            bottom.scatter([at[r]], [d], s=22, zorder=3,
                           color=LEARNED if sig else "none",
                           edgecolors=LEARNED, linewidths=0.9)
    bottom.plot(pos, [np.mean([v[2] for v in per[r]]) for r in rhos], "-",
                color=LEARNED, lw=1.5, zorder=4, label="sampled (trained policy)")
    # The argmax derivative of the same policies, drawn because the convention is part of the
    # claim: the ordering of rates survives it, the threshold does not, and 87% of the shift
    # is pairs left undetermined once the policy cannot sample its way out of a state.
    am = {}
    for f in sorted(glob.glob(str(ROOT / "results/power/rho/argmax_det/argmax_rho*_s?.json"))):
        rho = float(re.search(r"rho([\d.]+)_", pathlib.Path(f).stem).group(1))
        for e in json.loads(pathlib.Path(f).read_text()):
            am.setdefault(rho, []).append(e["paired"]["learned-greedy"]["delta"])
    if am:
        bottom.plot([at[r] for r in rhos if r in am],
                    [np.mean(am[r]) for r in rhos if r in am], "^--",
                    color=MYOPIC, lw=1.2, ms=4.5, zorder=3, label="argmax derivative")
        bottom.legend(loc="upper right", frameon=False, fontsize=7)
    top.set_xlabel(r"answer rate $\rho$ trained under")
    bottom.set_xlabel(r"answer rate $\rho$ trained under")
    bottom.set_ylabel("paired difference in SHD ($\\downarrow$)\nlearned $-$ myopic")
    bottom.annotate("filled: ahead beyond 2 SE", xy=(0.04, 0.05),
                    xycoords="axes fraction", fontsize=7, color="#555555")
    for ax in (top, bottom):
        ax.set_xticks(rhos)
        # labels on alternate ticks: linear in rho without 0.80-1.00 overprinting at half width
        ax.set_xticklabels([f"{r:g}" if r in (0.5, 0.7, 0.8, 0.9, 1.0) else ""
                            for r in rhos], fontsize=7.5)
        ax.set_xlim(1.04, 0.46)   # oracle on the left, noise increasing rightward
    fig.tight_layout()
    fig.savefig(out / "answer_rate.pdf", bbox_inches="tight")
    plt.close(fig)



def fig_credit(out: pathlib.Path):
    """Turn-aware credit, measured, at both window sizes.

    k=8: both optimisers degrade about an order of magnitude without the fix -- the
    recorded-field version's federation-specific interaction does not exist. k=12: the pooled
    cell sits at the floor in both states and is uninformative; the federated arm degrades,
    one seed carrying it, as one seed carries every credit-off degradation measured. All runs
    4,000 episodes; per-seed dots are the caveat drawn.
    """
    panels = [("$k_v=8$", "k08s50n04b150"), ("$k_v=12$", "k12s50n04b150")]
    data = {}
    for _, cell in panels:
        for arm in ("pooled", "E4"):
            for state in ("credit", "nocredit"):
                q = ROOT / f"results/credit/shd/{cell}_{arm}_{state}.json"
                if not q.exists():
                    print("!! credit measurement incomplete; skipping figure")
                    return
                data[(cell, arm, state)] = [e["means"]["learned"]["hard"]
                                            for e in json.loads(q.read_text())]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.0), sharey=True)
    floor = 3e-5
    for ax, (title, cell) in zip(axes, panels):
        for arm, label, colour, dx in (("pooled", "pooled", THIRD, -0.045),
                                       ("E4", "federated", LEARNED, 0.045)):
            means = [np.mean(data[(cell, arm, st)]) for st in ("credit", "nocredit")]
            ax.plot([0 + dx, 1 + dx], [max(m, floor) for m in means], "o-", color=colour,
                    lw=1.7, ms=5.5, label=label if cell.startswith("k08") else None, zorder=4)
            for k_, st in enumerate(("credit", "nocredit")):
                vals = data[(cell, arm, st)]
                ax.scatter([k_ + dx] * len(vals), [max(v, floor) for v in vals],
                           s=14, color=colour, alpha=0.4, zorder=3)
            # Annotate the ratio only where BOTH states are off the floor: a saturated cell
            # has no headroom, and a ratio of two floor values reads as a finding it is not.
            if min(means) > 10 * floor:
                ax.annotate(f"{means[1]/means[0]:.0f}$\\times$", xy=(1 + dx, means[1]),
                            xytext=(7, -2), textcoords="offset points", fontsize=8,
                            color=colour)
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["credit on", "credit off"])
        ax.set_xlim(-0.35, 1.45)
        ax.annotate(title, xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)
    axes[0].set_ylabel(r"SHD on committed marks ($\downarrow$)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "credit.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_fixedpolicy(out: pathlib.Path):
    """The fixed-policy decomposition: one policy per training rate, evaluated at every rate.

    Separates an environment property from a learning property. If withheld answers were
    simply a harder evidence regime, both fixed policies would degrade together; instead the
    oracle-trained policy degrades 295x across the range and the rho=0.5-trained policy 27x,
    and the curves are indistinguishable at full evidence. Axis linear in rho, same rule as
    fig_answer_rate.
    """
    per = {}
    for f in sorted(glob.glob(str(ROOT / "results/power/rho/evalsweep_det/fixed_rho*_s?_evalp*.json"))):
        m = re.search(r"fixed_rho([\d.]+)_s(\d)_evalp([\d.]+)", pathlib.Path(f).stem)
        e = json.loads(pathlib.Path(f).read_text())[0]
        per.setdefault((float(m.group(1)), float(m.group(3))), []).append(
            e["means"]["learned"]["hard"])
    if not per:
        print("!! evalsweep_det absent; skipping")
        return
    evals = sorted({k[1] for k in per})
    fig, ax = plt.subplots(figsize=(TWOTHIRD, 3.1))
    floor = 1e-4
    for trained, colour, label in ((1.0, MYOPIC, r"trained at $\rho=1.0$ (oracle)"),
                                   (0.5, LEARNED, r"trained at $\rho=0.5$")):
        means = [np.mean(per[(trained, ev)]) for ev in evals]
        for ev in evals:
            vals = per[(trained, ev)]
            ax.scatter([ev] * len(vals), [max(v, floor) for v in vals], s=13,
                       color=colour, alpha=0.4, zorder=3)
        ax.plot(evals, [max(v, floor) for v in means], "o-", color=colour, lw=1.7,
                ms=5, label=label, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel(r"evaluation answer rate $\rho$")
    ax.set_ylabel(r"SHD on committed marks ($\downarrow$)")
    ax.set_xticks(evals)
    ax.invert_xaxis()          # reading left to right = answers progressively withheld
    _title(ax, "Held fixed, the oracle-trained policy degrades $295\\times$\n"
                 "and the adapted one $27\\times$", fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fixedpolicy.pdf", bbox_inches="tight")
    plt.close(fig)



def fig_inregime(out: pathlib.Path):
    """The answer-rate grid read a second way: the same policies in their own regimes.

    Left: the in-regime paired delta by rate -- the cost or gain of the dial where the policy
    actually lives. Right: measured in-regime against measured transfer, one point per cell.
    Both axes are global_shd_paired.py output (results/power/rho/inregime_det/ and
    deterministic/); the recorded-field version of this comparison is superseded.
    """
    inreg, xfer = {}, {}
    for f in sorted(glob.glob(str(ROOT / "results/power/rho/inregime_det/rho*_s?.json"))):
        m = re.search(r"rho([\d.]+)_s(\d)", pathlib.Path(f).stem)
        e = json.loads(pathlib.Path(f).read_text())[0]
        inreg[(float(m.group(1)), int(m.group(2)))] = e["paired"]["learned-greedy"]["delta"]
    for f in sorted(glob.glob(str(ROOT / "results/power/rho/deterministic/xfer_rho*_s?.json"))):
        m = re.search(r"rho([\d.]+)_s(\d)", pathlib.Path(f).stem)
        e = json.loads(pathlib.Path(f).read_text())[0]
        xfer[(float(m.group(1)), int(m.group(2)))] = e["paired"]["learned-greedy"]["delta"]
    if not inreg or not xfer:
        print("!! in-regime or transfer measurements absent; skipping")
        return
    rhos = sorted({k[0] for k in inreg})

    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL, 2.9),
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    left.axhline(0, color="black", lw=0.8, zorder=1)
    means = [np.mean([inreg[(r, s_)] for s_ in (0, 1, 2)]) for r in rhos]
    for r in rhos:
        left.scatter([r] * 3, [inreg[(r, s_)] for s_ in (0, 1, 2)], s=13, color=LEARNED,
                     alpha=0.4, zorder=3)
    left.plot(rhos, means, "o-", color=LEARNED, lw=1.7, ms=4.5, zorder=4)
    # Ticks at every rate, labels on alternate ones: the axis stays linear in rho (the
    # truthful shape) without the 0.80-1.00 labels overprinting at panel width.
    left.set_xticks(rhos)
    left.set_xticklabels([f"{r:g}" if r in (0.5, 0.7, 0.8, 0.9, 1.0) else "" for r in rhos])
    left.set_xlabel(r"answer rate $\rho$")
    left.invert_xaxis()        # oracle on the left, noise increasing rightward
    left.set_ylabel("paired difference in SHD ($\\downarrow$)\nlearned $-$ myopic, in-regime", fontsize=8)
    left.annotate(r"$\rho=0.95$: worse in-regime" + "\non 3 of 3 seeds",
                  xy=(0.95, 0.0041), xytext=(0.66, 0.0038), fontsize=8, color="#666666",
                  ha="center", arrowprops=dict(arrowstyle="->", color="#666666", lw=0.7))

    right.axhline(0, color="black", lw=0.7, zorder=1)
    right.axvline(0, color="black", lw=0.7, zorder=1)
    for r in rhos:
        for s_ in (0, 1, 2):
            if (r, s_) in xfer:
                right.scatter(inreg[(r, s_)], xfer[(r, s_)], s=16, color=LEARNED,
                              alpha=0.55, zorder=3)
    right.set_xlabel(r"paired difference in SHD, in-regime ($\downarrow$)", fontsize=8)
    right.set_ylabel(r"paired difference in SHD, transfer ($\downarrow$)", fontsize=8)
    right.annotate("Spearman $+0.795$\n(21 cells)", xy=(0.05, 0.95),
                   xycoords="axes fraction", va="top", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(out / "inregime.pdf", bbox_inches="tight")
    plt.close(fig)



def fig_generator(out: pathlib.Path):
    """The generator control: SF against ER at the principal cell, per arm.

    The learned arm holds near zero on both families; the myopic rule degrades fifty-fold on
    Erdos-Renyi. Densities are near-matched (50.0 against 53.6 true edges, same prior_p), so
    the family is the operative difference. Same visual shape as fig_credit on purpose: two
    conditions, the interesting fact being which line moves.
    """
    sf = ROOT / "results/sweep12k/shd/k12s50n04b150.json"
    er = ROOT / "results/generator12k/shd_er_best.json"
    if not (sf.exists() and er.exists()):
        print("!! generator measurements absent; skipping")
        return
    data = {"scale-free": json.loads(sf.read_text()), "Erd\u0151s--R\u00e9nyi": json.loads(er.read_text())}
    fig, ax = plt.subplots(figsize=(TWOTHIRD, 3.1))
    floor = 5e-5
    xs = {k: i for i, k in enumerate(data)}
    arm_map = [("learned", "learned", LEARNED, 0.045),
               ("myopic", "greedy", MYOPIC, -0.045),
               ("random", "random_vary", RANDOM, 0.0)]
    for label, key, colour, dx in arm_map:
        means = []
        for fam, d in data.items():
            vals = [max(e["means"][key]["hard"], floor) for e in d]
            ax.scatter([xs[fam] + dx] * len(vals), vals, s=14, color=colour, alpha=0.4,
                       zorder=3)
            means.append(np.mean([e["means"][key]["hard"] for e in d]))
        ax.plot([0 + dx, 1 + dx], [max(m, floor) for m in means], "o-", color=colour,
                lw=1.7 if key == "learned" else 1.2, ms=5, label=label, zorder=4)
        if key == "greedy":
            ax.annotate(f"$\\times${means[1]/means[0]:.0f}", xy=(1 + dx, means[1]),
                        xytext=(7, -1), textcoords="offset points", fontsize=8, color=colour)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["scale-free\n(all reported results)", "Erd\u0151s--R\u00e9nyi\n(control)"])
    ax.set_xlim(-0.35, 1.4)
    ax.set_ylabel(r"SHD on committed marks ($\downarrow$)")
    ax.legend(loc="center left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "generator.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_training_signal(out: pathlib.Path):
    """What each evidence regime is like to TRAIN under, at the k=8 cell.

    Motivates the partial oracle as a measurement rather than a convenience: genuine
    finite-sample evidence never reaches the competence floor at the sweep's budget, while the
    partial oracle trains and (Figure answer_rate) transfers. Window rate over the last ten
    checkpoints, one dot per seed, the exclusion floor drawn.
    """
    def wr(path):
        d = json.loads(pathlib.Path(path).read_text())
        h = d.get("history") or []
        return float(np.mean([u["window_rate"] for u in h[-10:]]))

    # Budgets differ between the groups and are stated in the caption, not squeezed into
    # tick labels that collide at TWOTHIRD width.
    groups = [
        ("oracle", "results/sweep12k/k08s50n04b150_s?.json", LEARNED),
        ("partial oracle\n$\\rho=0.85$", "results/power/rho/rho0.85_s?.json", THIRD),
        ("sampled\n$n_{\\mathrm{int}}=200$", "results/sampled_ref/k08s50n04b150i0200_s?.json", MYOPIC),
    ]
    fig, ax = plt.subplots(figsize=(TWOTHIRD, 2.9))
    for i, (label, pat, colour) in enumerate(groups):
        vals = [wr(f) for f in sorted(glob.glob(str(ROOT / pat)))]
        if not vals:
            print(f"!! no runs for {pat}"); continue
        ax.scatter([i] * len(vals), vals, s=26, color=colour, zorder=3)
        ax.plot([i - 0.14, i + 0.14], [np.mean(vals)] * 2, color=colour, lw=1.6, zorder=4)
    ax.axhline(WINDOW_FLOOR, color="#B00020", lw=1.0, ls="--", zorder=2)
    ax.annotate("competence floor", xy=(1.62, WINDOW_FLOOR), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color="#B00020")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("per-window solve rate ($\\uparrow$)\n(last ten checkpoints)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_signal.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="thesis/figures")
    args = ap.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for name, fn in (("sweep_grid_[abcd]", fig_sweep_grid),
                     ("window_budget", fig_window_budget), ("nint", fig_nint),
                     ("attribution_law", fig_attribution_law),
                     ("federation_[ab]", fig_federation),
                     ("pair_class", fig_pair_class),
                     ("answer_rate", fig_answer_rate),
                     ("credit", fig_credit),
                     ("fixedpolicy", fig_fixedpolicy),
                     ("inregime", fig_inregime),
                     ("generator", fig_generator),
                     ("training_signal", fig_training_signal)):
        try:
            fn(out)
            print(f"  wrote {name}.pdf")
        except Exception as exc:                    # a missing cell must not kill the rest
            print(f"  !! {name}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
