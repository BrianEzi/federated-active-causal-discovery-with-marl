"""The crossover figure at both training budgets, side by side.

The published version of this figure is built on the 4,000-episode sweep and shows a sign
change between k=8 and k=12. That sign change is where 4,000 episodes stopped being enough.
Drawing both budgets on one pair of axes is the honest replacement: the reader sees the
crossover AND sees it disappear, rather than being told about it.
"""
import json, pathlib, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/Users/brianezinwoke/Workspace/federated-active-causal-discovery-with-marl")
KS = [4, 8, 12, 20, 30]
PRE = {"k12s50n04b150": "results/sweep12k/shd/k12s50n04b150.json"}
LEG = {20: "results/ckpt/k20_best.json", 30: "results/ckpt/k30_best.json"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
L4, L12, MY = "#999999", "#0072B2", "#D55E00"
FLOOR = 1e-5


def load(p):
    q = ROOT / p
    return json.loads(q.read_text()) if q.exists() else None


def series12(k):
    if k in LEG:
        return load(LEG[k])
    cell = {4: "k04s50n04b150", 8: "k08s50n04b150", 12: "k12s50n04b150"}[k]
    return load(f"results/sweep12k/shd/{cell}.json")


fig, ax = plt.subplots(figsize=(5.2, 3.4))
for label, getter, colour, ls in (
        # k=20 and k=30 were trained at 12,000 episodes in the ORIGINAL sweep, so they have no
        # 4,000-episode counterpart. Loading them here would put two 12,000-episode points on a
        # line labelled 4,000 -- the mislabelling this whole figure exists to correct.
        ("learned, 4{,}000 episodes",
         lambda k: load(f"results/ckpt/k{k:02d}_best.json") if k <= 12 else None, L4, "--"),
        ("learned, 12{,}000 episodes", series12, L12, "-")):
    # A missing cell BREAKS the line rather than being interpolated across. Joining k=8 to
    # k=20 through an unmeasured k=12 draws a point that does not exist, and on a log axis it
    # lands close enough to the real value to be believed.
    xs, ys, seeds = [], [], []
    for k in KS:
        d = getter(k)
        if not d:
            xs.append(k); ys.append(np.nan); seeds.append([])
            continue
        v = [e["means"]["learned"]["hard"] for e in d]
        xs.append(k); ys.append(np.mean(v)); seeds.append(v)
    for x, v in zip(xs, seeds):
        ax.scatter([x] * len(v), [max(q, FLOOR) for q in v], s=12, color=colour,
                   alpha=.35, zorder=3)
    ax.plot(xs, [np.nan if np.isnan(m) else max(m, FLOOR) for m in ys], "o" + ls,
            color=colour, lw=1.7, ms=5, label=label.replace("{,}", ","), zorder=4)
    real = [i for i, m in enumerate(ys) if not np.isnan(m)]
    span = range(min(real), max(real) + 1) if real else []
    for i, (x, m) in enumerate(zip(xs, ys)):
        if np.isnan(m) and i in span:
            ax.annotate("not yet\nmeasured", (x, 3e-4), fontsize=6.5, ha="center",
                        color=colour, alpha=.8)

my = [np.mean([e["means"]["greedy"]["hard"] for e in load(f"results/ckpt/k{k:02d}_best.json")])
      for k in KS]
ax.plot(KS, my, "-", color=MY, lw=1.5, label="myopic", zorder=2)
ax.set_yscale("log"); ax.set_ylim(FLOOR * .7, 3e-2); ax.set_xticks(KS)
ax.axvspan(8, 12, color="black", alpha=.05, zorder=0)
ax.set_xlabel("window size $k_v$")
ax.set_ylabel("SHD on committed marks")
ax.set_title("The crossover is a training budget, not a complexity threshold", fontsize=9.5)
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.grid(alpha=.25, lw=.5)
fig.tight_layout()
out = ROOT / "thesis/figures/crossover_budget.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
print("wrote", out.name)
