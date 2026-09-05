"""The no-skeleton results, plotted. Exploratory (results/noskel/): NOT wired into any
chapter -- the figure ships only if Brian promotes it. Story: with the supplied skeleton
removed, the assumption-trained policy inverts below random (transfer, k=8), while policies
retrained under the estimated skeleton recover a small significant edge at both cells."""
import json, pathlib, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})
LEARNED, MYOPIC, RANDOM, BAD = "#0072B2", "#D55E00", "#999999", "#CC79A7"

def arm_vals(files, arm):
    out = []
    for f in files:
        e = json.loads((ROOT / f).read_text())[0]
        out.append(e["means"][arm]["hard"])
    return out

k8 = [f"results/noskel/shd_s{s}_best.json" for s in (0, 1, 2)]
k12 = [f"results/noskel/shd_k12_s{s}_best.json" for s in (0, 1, 2)]
xfer = ["results/noskel/k08_estskel_transfer.json"]
# transfer file holds 3 entries (one per seed)
xe = json.loads((ROOT / xfer[0]).read_text())
xfer_learned = [e["means"]["learned"]["hard"] for e in xe]

fig = plt.figure(figsize=(5.40, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 0.9])
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
axes[1].sharey(axes[0])
ax3 = fig.add_subplot(gs[2])
panels = [("$k_v=8$", k8, xfer_learned), ("$k_v=12$", k12, None)]
for ax, (title, files, xf) in zip(axes, panels):
    series = [("learned", arm_vals(files, "learned"), LEARNED),
              ("myopic", arm_vals(files, "greedy"), MYOPIC),
              ("random", arm_vals(files, "random_vary"), RANDOM)]
    if xf is not None:
        series.append(("learned, trained\nWITH skeleton", xf, BAD))
    for i, (label, vals, colour) in enumerate(series):
        ax.bar(i, np.mean(vals), 0.62, color=colour, alpha=0.8, zorder=2)
        ax.scatter([i] * len(vals), vals, s=16, color="black", alpha=0.55, zorder=4)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([s[0] for s in series], fontsize=6.5, rotation=20, ha="right")
    ax.annotate(title, xy=(0.04, 0.93), xycoords="axes fraction", fontsize=9)
axes[0].set_ylabel(r"SHD on committed marks ($\downarrow$)")
axes[0].annotate("estimated skeleton throughout;\njoint recovery is 0 for every arm",
                 xy=(0.04, 0.80), xycoords="axes fraction", fontsize=7, color="#666666")
axes[1].tick_params(labelleft=False)

# The finding lives at a scale the absolute axis cannot show: paired learned - myopic per
# seed with 2 SE bars, the same convention the federation figure uses.
x = 0
for cell, files, colour in (("$k_v=8$", k8, LEARNED), ("$k_v=12$", k12, LEARNED)):
    for f in files:
        e = json.loads((ROOT / f).read_text())[0]
        d = e["paired"]["learned-greedy"]
        ax3.errorbar(x, d["delta"], yerr=2 * d["se"], fmt="o", color=colour, ms=4,
                     lw=1.1, capsize=2.5, zorder=3)
        x += 1
    x += 1
ax3.axhline(0, color="black", lw=0.8, zorder=1)
ax3.set_xticks([1, 5])
ax3.set_xticklabels(["$k_v=8$", "$k_v=12$"], fontsize=8)
ax3.set_ylabel("paired difference in SHD ($\\downarrow$)\nlearned $-$ myopic, per seed",
               fontsize=7)
ax3.annotate("below 0:\nlearned ahead", xy=(0.95, 0.10), xycoords="axes fraction",
             ha="right", fontsize=7, color="#666666")
fig.tight_layout()
fig.savefig(ROOT / "results/noskel/noskel.pdf", bbox_inches="tight")
print("wrote results/noskel/noskel.pdf")
