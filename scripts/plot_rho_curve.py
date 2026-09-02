"""Three annotated views of the answer-rate sweep.

Deliberately three panels rather than one, because the result is a RELATIONSHIP between two
measurements and a single panel hides whichever one it does not plot:

  1. TRANSFER -- learned minus greedy hard SHD against rho, with per-seed points shown, not
     just the mean. The zero line is the whole claim: below it the learned policy beats the
     myopic rule under evidence it never trained on.
  2. IN-REGIME -- the same policies scored in their own training regime. Plotted on the same
     rho axis so the inversion is visible by eye rather than asserted in prose.
  3. THE INVERSION DIRECTLY -- in-regime quality against transfer quality, one point per rate.
     If in-regime predicted transfer these would fall on a rising line; the claim is that they
     do the opposite.

Per-seed points are drawn on panel 1 because the across-seed spread is the term that decides
whether two rates differ, and a mean with an error bar invites the reader to forget there are
only three of them.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_transfer(directory):
    cells = {}
    for path in sorted(glob.glob(os.path.join(directory, "xfer_rho*.json"))):
        m = re.search(r"xfer_rho([0-9.]+)_s(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        payload = json.loads(open(path).read())
        e = payload[0] if isinstance(payload, list) else payload
        d = e["paired"].get("learned-greedy")
        if d:
            cells.setdefault(float(m.group(1)), []).append(d["delta"])
    return cells


def load_inregime(directory):
    rows = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "rho*_s*.json"))):
        if "xfer" in os.path.basename(path):
            continue
        m = re.search(r"rho([0-9.]+)_s(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        d = json.loads(open(path).read())
        a = d["arms"]
        rows[float(m.group(1))].append((a["learned"]["success"],
                                        a["learned"]["global_hard_shd"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/power/rho")
    ap.add_argument("--out", default="results/power/rho/rho_curve.png")
    args = ap.parse_args(argv)

    xfer = load_transfer(args.dir)
    inreg = load_inregime(args.dir)
    if not xfer:
        print("no transfer data yet")
        return

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 15.0))
    fig.suptitle("Partial-oracle answer rate: in-regime quality inverts transfer quality\n"
                 "k=8, 4 agents, budget 70, 8000 train episodes, 200 paired eval episodes",
                 fontsize=12, y=0.995)

    # -- panel 1: transfer -------------------------------------------------------------------
    ax = axes[0]
    rates = sorted(xfer)
    means = [float(np.mean(xfer[r])) for r in rates]
    ses = [float(np.std(xfer[r], ddof=1) / np.sqrt(len(xfer[r]))) if len(xfer[r]) > 1
           else np.nan for r in rates]
    for r in rates:                                   # per-seed points behind the mean
        ax.plot([r] * len(xfer[r]), xfer[r], "o", color="0.7", ms=5, zorder=1)
    ax.errorbar(rates, means, yerr=ses, fmt="o-", color="C0", lw=2, ms=8, capsize=4, zorder=3,
                label="mean over seeds (+/- 1 SE)")
    ax.axhline(0, color="crimson", lw=1.6, ls="--", zorder=2)
    ax.text(0.5, 0.0, "  greedy baseline", color="crimson", va="bottom", fontsize=9)
    ax.axhspan(min(means + [0]) * 1.35 if means else -0.02, 0, color="C2", alpha=0.07)
    ax.text(0.995, min(means) * 0.55 if means else -0.01, "learned WINS\n(lower SHD)",
            color="C2", fontsize=9, ha="right", va="center", weight="bold")
    ax.text(0.995, max(means) * 0.7 if max(means) > 0 else 0.005,
            "learned LOSES", color="crimson", fontsize=9, ha="right", va="center", weight="bold")
    ax.set_xlabel(r"answer rate $\rho$  (1.00 = full oracle, lower = more withheld)")
    ax.set_ylabel("hard SHD: learned - greedy")
    ax.set_title("1. Transfer to genuine sampled evidence", fontsize=10, weight="bold")
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # -- panel 2: in-regime ------------------------------------------------------------------
    ax = axes[1]
    ir_rates = sorted(inreg)
    succ = [float(np.mean([v[0] for v in inreg[r]])) for r in ir_rates]
    # ERROR BARS ADDED 2 Sep after this panel misled me. Plotted as bare means, the dip at
    # rho=0.95 looks like a cliff; with the seed spread drawn it is visibly one noisy point
    # among several, and rho=0.80's own three seeds span more than the dip being pointed at.
    # A mean without its spread invites exactly the story I spent two hours chasing.
    ir_se = [float(np.std([v[0] for v in inreg[r]], ddof=1) / np.sqrt(len(inreg[r])))
             if len(inreg[r]) > 1 else np.nan for r in ir_rates]
    ax.errorbar(ir_rates, succ, yerr=ir_se, fmt="s-", color="C1", lw=2, ms=8, capsize=4,
                zorder=3, label="mean over seeds (+/- 1 SE)")
    for r in ir_rates:
        ax.plot([r] * len(inreg[r]), [v[0] for v in inreg[r]], "o", color="0.75", ms=4, zorder=1)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel(r"answer rate $\rho$")
    ax.set_ylabel("in-regime episode success")
    ax.set_title("2. Same policies, scored in their OWN regime (note the seed spread)",
                 fontsize=10, weight="bold")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    if 1.0 in inreg:
        ax.annotate("full oracle:\nnear-perfect here,\nWORST at transfer",
                    xy=(1.0, succ[ir_rates.index(1.0)]), xytext=(0.93, 0.80),
                    fontsize=8.5, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))

    # -- panel 3: the inversion --------------------------------------------------------------
    ax = axes[2]
    common = [r for r in rates if r in inreg]
    x = [float(np.mean([v[0] for v in inreg[r]])) for r in common]
    y = [float(np.mean(xfer[r])) for r in common]
    ax.axhline(0, color="crimson", lw=1.4, ls="--")
    ax.scatter(x, y, s=90, c=[("crimson" if v > 0 else "C2") for v in y], zorder=3)
    for r, xi, yi in zip(common, x, y):
        ax.annotate(f"{r:.2f}", (xi, yi), fontsize=9, va="center",
                    xytext=(9, 6), textcoords="offset points")
    # NO CORRELATION STATISTIC HERE, deliberately. With five points -- two of them (rho=0.90
    # and 0.85) nearly coincident on both axes -- a rank correlation is dominated by
    # tie-breaking noise. It read Spearman +0.10 on this data, which a reader would
    # reasonably take as "no relationship", when the visible structure is that the full
    # oracle sits alone in the top-right and every winning rate is bottom-left. The
    # separation is the finding; a coefficient over five points is not evidence either way,
    # and putting one in the title lends it authority it has not earned.
    ax.set_title("3. In-regime quality vs transfer quality", fontsize=10, weight="bold")
    ax.set_xlabel("in-regime episode success  (higher = better in training regime)")
    ax.set_ylabel("hard SHD: learned - greedy at transfer\n(lower = better)")
    ax.grid(alpha=0.25)
    ax.text(0.03, 0.03, "if in-regime PREDICTED transfer, points would fall\n"
                        "bottom-right. The full oracle (1.00) is top-right:\n"
                        "best in its own regime, worst at transfer.",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))

    n_cells = sum(len(v) for v in xfer.values())
    fig.text(0.01, 0.01, f"{n_cells}/21 transfer cells evaluated  |  grey dots are individual "
                         f"seeds  |  generated {__import__('time').strftime('%H:%M %d %b')}",
             fontsize=7.5, color="0.4")
    fig.tight_layout(rect=[0, 0.02, 1, 0.965])
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  ({n_cells} transfer cells, {len(rates)} rates)")


if __name__ == "__main__":
    main()
