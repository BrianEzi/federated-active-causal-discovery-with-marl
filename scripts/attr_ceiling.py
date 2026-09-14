"""Is the two-child ceiling a LAW, or a fact about one configuration?

WHAT WAS MEASURED, and why it needs generalising. At k=12 with 4 agents, a latent group is
recovered if and only if it has exactly TWO children: 80.6% of two-child groups settled
against 0.0% of every larger group, across 63 of them, with zero errors anywhere. The share
of two-child groups in the graph distribution is 37.7% and the measured attribution rate of
every competent policy is 36.8%.

THE ARGUMENT, from `cb/attribution.py`. A two-child group explains ONE pair, so there is no
finer hypothesis to separate it from and ownership is the whole question -- one partner
message settles it. A group with three or more children explains a CLIQUE, and separating it
from several smaller latents requires a PARTIAL response: some of its pairs moving while
others do not. That requires the owner to probe its private variables ONE AT A TIME. No
policy in this project does, so responses are always total and atomicity never fires.

WHAT THIS SCRIPT TESTS. The argument makes a prediction for every configuration, not just the
one it was found in: measured attribution should track the two-child share, whatever that
share happens to be. This computes both per cell and prints them side by side, plus the
by-size breakdown that shows whether the cliff is still a cliff.

THE SHARPEST CELL IS n=2. With a single partner there is no ownership question at all -- the
owner is forced -- so EVERY group should resolve regardless of size and the cliff should
VANISH. If it does not, the explanation above is wrong and the two-child correlation is a
coincidence. That cell is worth more than all the others together.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cb.attribution import observable_groups, score_groups                # noqa: E402
from cb.component_attribution import ComponentAttributedBackend           # noqa: E402
from scripts.attr_scale import build_env, drive                           # noqa: E402


def predicted_ceiling(env, agents, episodes: int, seed_offset: int = 10_000):
    """Share of TRUE groups with exactly two children, over fresh episodes.

    Drawn from episodes the measurement does not use, so the prediction cannot be fitted to
    the same draws it is later compared against.
    """
    sizes: collections.Counter = collections.Counter()
    for episode in range(episodes):
        env.reset(seed=seed_offset + episode)
        for agent in agents:
            for group in observable_groups(env.true_adjacency, env.topology, agent):
                sizes[len(group.children)] += 1
    total = sum(sizes.values())
    return (sizes.get(2, 0) / total if total else float("nan")), sizes, total


def run_cell(k, sigma, n_agents, budget, episodes, n_obs, n_int, cap):
    env = build_env(k, sigma, n_agents, budget, n_obs, n_int)
    agents = list(env.topology.agents)
    predicted, size_hist, n_groups = predicted_ceiling(env, agents, max(episodes // 2, 10))

    by_size = collections.defaultdict(collections.Counter)
    right = wrong = total = 0
    start = time.time()
    for episode in range(episodes):
        backends = {a: ComponentAttributedBackend(env.windows[a].k, agent=a,
                                                  n_agents=len(agents), evidence="oracle",
                                                  max_component_pairs=cap)
                    for a in agents}
        drive(env, backends, episode)
        for agent in agents:
            score = score_groups(backends[agent].last, backends[agent].true_groups, bar=1.0)
            right += score["right"]; wrong += score["wrong"]; total += score["total"]
            for group, outcome, _ in score["detail"]:
                by_size[len(group.children)][outcome] += 1
    seconds = (time.time() - start) / max(episodes, 1)
    measured = right / total if total else float("nan")
    return {"k": k, "sigma": sigma, "n_agents": n_agents, "budget": budget,
            "episodes": episodes, "predicted_ceiling": predicted, "measured": measured,
            "right": right, "wrong": wrong, "total": total,
            "seconds_per_episode": seconds,
            "true_group_sizes": {str(s): c for s, c in sorted(size_hist.items())},
            "by_size": {str(s): dict(c) for s, c in sorted(by_size.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="12:0.5:4:60,20:0.5:4:80,30:0.5:4:100,"
                                       "12:0.5:2:60,12:0.5:3:60,12:0.5:8:60,"
                                       "12:0.25:4:60,12:0.75:4:60")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n_obs", type=int, default=50)
    ap.add_argument("--n_int", type=int, default=10)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--out", default="results/attr_ceiling.json")
    args = ap.parse_args()

    rows = []
    header = (f"{'cell':>16s} {'predicted':>10s} {'measured':>9s} {'diff':>7s} "
              f"{'right':>6s} {'wrong':>6s} {'total':>6s} {'s/ep':>7s}")
    print(header); print("-" * len(header))
    for spec in args.cells.split(","):
        k, sigma, n_agents, budget = spec.split(":")
        row = run_cell(int(k), float(sigma), int(n_agents), int(budget),
                       args.episodes, args.n_obs, args.n_int, args.cap)
        name = f"k{int(k)}s{int(float(sigma)*100):02d}n{int(n_agents):02d}"
        row["cell"] = name; rows.append(row)
        print(f"{name:>16s} {row['predicted_ceiling']:10.3f} {row['measured']:9.3f} "
              f"{row['measured'] - row['predicted_ceiling']:+7.3f} {row['right']:6d} "
              f"{row['wrong']:6d} {row['total']:6d} {row['seconds_per_episode']:7.2f}",
              flush=True)
        # The cliff itself: recovery rate by group size.
        parts = []
        for size in sorted(int(s) for s in row["by_size"]):
            c = row["by_size"][str(size)]
            tot = sum(c.values())
            parts.append(f"{size}ch {100.0 * c.get('right', 0) / max(tot, 1):.0f}% (n={tot})")
        print(f"{'':>16s} cliff: {'  '.join(parts)}", flush=True)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
