"""Why does the myopic rule collapse on Erdos-Renyi? Measure its decision statistic.

THE CLAIM IN sec:res_generator, previously untested: the rule ranks variables by the
undetermined marks they touch, and that ranking is "sharp on a hub-concentrated family,
where one intervention settles many pairs, yet nearly flat on a uniform-edge family, where
the counts run to ties and the choice degenerates towards arbitrary".

That is a statement about `UncertaintyGreedyAgent._unsure_touching`, so it can be measured
directly rather than argued from the generator's definition. At each agent's first decision
of an episode -- the belief seeded, nothing intervened on yet -- this records the score
vector the rule actually ranks on, and reports how decisive it is:

    TIES AT THE TOP   how many candidates share the maximum. A rule choosing among six tied
                      candidates is picking at random five times in six.
    MARGIN            best minus runner-up, in undetermined marks. Zero means no signal.
    DEGREE SPREAD     the generator-level quantity the ranking is meant to reflect.

Same principal cell, same episode seeds, both families: nothing differs but the generator.

    .venv/bin/python scripts/generator_decision_probe.py --episodes 60
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ma.baselines import UncertaintyGreedyAgent                      # noqa: E402
from scripts.rescore_from_config import env_from_config              # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="results/sweep12k/k12s50n04b150_s0.json")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", default="results/generator_decision_probe.json")
    args = ap.parse_args(argv)

    config = json.loads((ROOT / args.source).read_text())["config"]
    out = {}
    for family in ("sf", "er"):
        env = env_from_config(dict(config, graph_model=family), seed=0)
        ties, margins, degrees, tops = [], [], [], []
        for ep in range(args.episodes):
            env.reset(seed=90_000 + ep)          # SAME episode seeds in both families
            deg = np.asarray(env.true_adjacency)
            deg = (deg + deg.T).sum(axis=1)
            degrees.append(float(deg.std() / max(deg.mean(), 1e-9)))
            for agent in env.topology.agents:
                g = UncertaintyGreedyAgent(agent, 0, bar=1.0)
                window = env.windows[agent]
                belief = window.belief.last
                if belief is None:
                    continue
                # EXACTLY the rule's own decision: scores over the AUTHORITY nodes only,
                # which is the set it may actually intervene on, and ties broken uniformly
                # at random (ma/baselines.py: self.rng.choice(candidates)).
                counts = g._unsure_touching(belief, window.k)
                scores = np.array([counts[window.pos[n]] for n in window.authority])
                order = np.sort(scores)[::-1]
                top = order[0]
                if top <= 0:
                    continue
                ties.append(int((scores == top).sum()))
                margins.append(float(top - order[1]) if len(order) > 1 else float(top))
                tops.append(float(top))
        out[family] = {
            "episodes": args.episodes,
            "decisions": len(ties),
            "mean_ties_at_top": float(np.mean(ties)),
            "share_with_a_unique_best": float(np.mean([t == 1 for t in ties])),
            "mean_margin_marks": float(np.mean(margins)),
            "share_zero_margin": float(np.mean([m == 0 for m in margins])),
            "mean_top_score": float(np.mean(tops)),
            "degree_coefficient_of_variation": float(np.mean(degrees)),
        }
    (ROOT / args.out).write_text(json.dumps(out, indent=1))
    print(f"{'':22s} {'scale-free':>12s} {'Erdos-Renyi':>12s}")
    for key in ("decisions", "mean_ties_at_top", "share_with_a_unique_best",
                "mean_margin_marks", "share_zero_margin", "mean_top_score",
                "degree_coefficient_of_variation"):
        a, b = out["sf"][key], out["er"][key]
        fmt = (lambda v: f"{v:12.3f}") if isinstance(a, float) else (lambda v: f"{v:12d}")
        print(f"{key:22s} {fmt(a)} {fmt(b)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
