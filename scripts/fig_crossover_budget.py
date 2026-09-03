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
# Every path below is a seeded-evaluation measurement. results/ckpt/ (the pre-fix set this
# script read until 3 Sep) is superseded and marked so on disk; the provenance check guards
# the registries but not figure scripts, which is how it survived here.
LEG = {20: "results/rerows/k20_best.json", 30: "results/rerows/k30_best.json"}
# The 4,000-episode policies at k=20 and k=30. The run JSONs were overwritten by the
# 12,000-episode seed-coverage copy on 1 Sep, but the u0249 checkpoints -- update 249 is
# episode 4,000 exactly -- survived from the ORIGINAL 31 Aug runs, all three seeds. Training
# is horizon-independent (episode seeds derive from the update index; nothing anneals against
# the total), so these are the genuine 4,000-episode policies. Brian spotted this.
U0249 = {20: "results/rerows/k20_u0249.json", 30: "results/rerows/k30_u0249.json"}
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


fig, ax = plt.subplots(figsize=(5.40, 3.4))   # FIGURE_GUIDELINES: author at print size
def series4(k):
    # One convention along the whole line: the policy AFTER 4,000 episodes (the final update
    # of a 4,000-episode run; u0249 is that same policy for the two long cells).
    if k in U0249:
        return load(U0249[k])
    return load(f"results/rerows/k{k:02d}_final.json")


for label, getter, colour, ls in (
        ("learned, policy after 4{,}000 episodes", series4, L4, "--"),
        ("learned, 12{,}000 episodes (selected)", series12, L12, "-")):
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

my = [np.mean([e["means"]["greedy"]["hard"] for e in load(f"results/rerows/k{k:02d}_best.json")])
      for k in KS]
ax.plot(KS, my, "-", color=MY, lw=1.5, label="myopic", zorder=2)
ax.set_yscale("log"); ax.set_ylim(FLOOR * .7, 3e-2); ax.set_xticks(KS)
ax.set_xlabel("window size $k_v$")
ax.set_ylabel("SHD on committed marks")
# Title dropped per FIGURE_GUIDELINES rule 7; the caption carries the message.
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.grid(alpha=.25, lw=.5)
fig.tight_layout()
out = ROOT / "thesis/figures/crossover_budget.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=130)
print("wrote", out.name)
