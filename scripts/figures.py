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
def fig_crossover(out: pathlib.Path):
    """The window axis at both training budgets.

    The 4,000-episode line crosses zero between k=8 and k=12 and the 12,000-episode line does
    not cross at all. Reporting only the first, which is what this figure used to do, reports
    the budget as though it were the window. The lower panel is SHD at the selected checkpoint
    from results/rerows, which is the 4,000-episode design for k<=12 and 12,000 for k=20 and 30;
    its budget is stated in the caption rather than mixed silently.
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.0, 5.2), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.15]})

    top.axhline(0, color="black", lw=0.8)
    # Two lines, because the sign change is a property of the 4,000-episode budget and not of
    # the window. At 12,000 the gap is positive at every window and there is no crossing.
    for budget, colour, style, label in ((4000, MYOPIC, "o--", "4,000 episodes"),
                                         (12000, LEARNED, "o-", "12,000 episodes")):
        ks, means = [], []
        for k in KS:
            pairs = _sweep_success(k, budget)
            if not pairs:
                continue
            gaps = [l - g for l, g in pairs]
            ks.append(k)
            means.append(np.mean(gaps))
            top.scatter([k] * len(gaps), gaps, s=13, color=colour, alpha=0.4, zorder=3)
        top.plot(ks, means, style, color=colour, lw=1.6, ms=5, zorder=4, label=label)
    top.set_ylabel("joint recovery rate\nlearned $-$ myopic")
    top.set_title("The sign change belongs to the budget, not the window")
    top.legend(loc="lower right", frameon=False, fontsize=8)
    top.annotate("no 4,000-episode runs\nexist at $k_v=20$ or $30$", xy=(19, -0.05),
                 fontsize=7, color="#666666")

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
    fig.tight_layout()
    fig.savefig(out / "crossover.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_checkpoint(out: pathlib.Path):
    """Selected against final checkpoint, at 12,000 episodes on every window.

    The title used to read "inert below the crossover and decisive above it". At a uniform
    12,000 episodes there is no crossover to be below, and the earlier reading was confounded:
    the windows where the conventions agreed were exactly the cells trained for 4,000 episodes
    and the ones where they diverged were exactly the two trained for 12,000. Selection runs
    once per checkpoint, so three times the updates give it three times the chances to retain
    an exploratory policy. That is a budget effect, not a window effect.
    """
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
    ax.set_xlabel("window size $k_v$")
    ax.set_ylabel("SHD on committed marks")
    ax.set_title("Neither checkpoint convention is safe alone at 12,000 episodes")
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

    # BUDGETS DIFFER BETWEEN THESE ROWS AND THE FIGURE SAYS SO. The k=12 ladder trained for
    # 4,000 episodes and the k=20 ladder for 12,000. Retrains of the k=12 arms at 12,000 are
    # in results/central12k/ and this function switches to them automatically once all twelve
    # exist, so the figure cannot silently keep reporting the shorter budget.
    twelve = sorted((ROOT / "results/central12k").glob("v2_k12_?_s?.json"))
    k12 = ("results/central12k/v2_k12_{a}_s?.json" if len(twelve) >= 12
           else "results/central/v2_k12_{a}_s*.json")
    cells = [("$k_v=12$", k12), ("$k_v=20$", "results/central/v2_k20_{a}_s*.json")]
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
            # The full-coverage reference exists only as a run-recorded field, so it is drawn
            # on the recovery row and omitted from the SHD row rather than mixing a recorded
            # number into a panel of measured ones.
            if mkey != "global_hard_shd":
                ceiling = [np.mean([r["arms"]["oracle_cover"][mkey] for r in sel
                                    if r[key] == x and "oracle_cover" in r["arms"]] or [np.nan])
                           for x in xs]
                ax.plot(pos, [max(v, floor) if logy else v for v in ceiling], ls=":",
                        color=THIRD, lw=1.2, zorder=2,
                        label="full-coverage reference" if col == 0 else None)
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
                 "12,000 training episodes throughout, 3 seeds per cell. SHD from 200 paired "
                 "episodes at the selected checkpoint; recovery from the final policy.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "sweep_grid.pdf", bbox_inches="tight")
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

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 3.1))
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
        ax.set_title(title, fontsize=9)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-0.5, 1.2e5)
    left.set_ylabel("errors committed")
    left.legend(loc="lower left", frameon=False)
    fig.suptitle("Training moves the class the policy is scored on, and not the other",
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
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.2, 5.0), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1]})
    myopic = np.mean([v[1] for r in rhos for v in per[r]])
    top.axhline(myopic, color=MYOPIC, lw=1.3, ls="--", zorder=2,
                label=f"myopic ({myopic:.4f}, identical at every rate)")
    means = [np.mean([v[0] for v in per[r]]) for r in rhos]
    for r in rhos:
        top.scatter([at[r]] * len(per[r]), [v[0] for v in per[r]], s=14, color=LEARNED,
                    alpha=0.4, zorder=3)
    top.plot(pos, means, "o-", color=LEARNED, lw=1.7, ms=5, zorder=4, label="learned")
    top.set_ylabel("SHD under sampled evidence")
    top.legend(loc="upper left", frameon=False, fontsize=7.5)
    top.set_title("Degrading the training evidence improves transfer")

    bottom.axhline(0, color="black", lw=0.8, zorder=2)
    for r in rhos:
        for _l, _g, d, sig in per[r]:
            bottom.scatter([at[r]], [d], s=22, zorder=3,
                           color=LEARNED if sig else "none",
                           edgecolors=LEARNED, linewidths=0.9)
    bottom.plot(pos, [np.mean([v[2] for v in per[r]]) for r in rhos], "-",
                color=LEARNED, lw=1.5, zorder=4)
    bottom.set_xlabel(r"answer rate $\rho$ the policy trained under")
    bottom.set_ylabel("paired difference\nlearned $-$ myopic")
    bottom.annotate("filled: ahead beyond 2 SE", xy=(0.52, 0.012), fontsize=7,
                    color="#555555")
    for ax in (top, bottom):
        ax.set_xticks(rhos)
        ax.set_xticklabels([f"{r:g}" for r in rhos], fontsize=7.5)
        ax.set_xlim(0.46, 1.04)
    fig.tight_layout()
    fig.savefig(out / "answer_rate.pdf", bbox_inches="tight")
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
                     ("attribution_law", fig_attribution_law), ("federation", fig_federation),
                     ("pair_class", fig_pair_class),
                     ("answer_rate", fig_answer_rate)):
        try:
            fn(out)
            print(f"  wrote {name}.pdf")
        except Exception as exc:                    # a missing cell must not kill the rest
            print(f"  !! {name}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
