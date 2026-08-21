"""Why did GATE 4 fail? Is the rescue mechanism false, or is my protocol blocking it?

GATE 4 found that B intervening on its own private node does NOT help a confounded A --
identification 0/29 in every arm, mean posterior mass on the truth ~0. MA_DESIGN section 4
predicted the opposite, and predicted it would work "with no disclosure at all".

Two candidate explanations, and they have very different consequences:

  (1) THE MECHANISM IS FALSE. Breaking the confounder does not actually make A's induced
      DAG identifiable. The coordination story collapses and the design needs rethinking.

  (2) THE POOLING IS THE PROBLEM. The mechanism works, but A cannot exploit it because A
      pools every row it has ever seen into one dataset. Under my no-disclosure decision
      (MA_BUILD_LOG) A is never told that B intervened, so A cannot tell the confounded
      rows from the clean ones. A mixture of two regimes is not fit by any single DAG
      either, so breaking the confounding in HALF the rows buys nothing.

These are separable. This script gives A a dataset drawn ENTIRELY from the regime where
B holds its private node under intervention -- no pooling, no mixture, nothing to
disentangle. If A identifies there, explanation (2) is right: the mechanism is real and my
protocol is what blocks it. If A still fails, explanation (1) is right.

Reported alongside: the same episodes with a purely observational dataset of the same size,
so the comparison is regime-vs-regime and not sample-size-vs-sample-size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legacy.ma_v1.env import AgentView
from ma.projection import bidirected_pairs
from ma.topology import Topology
from sa.scm import sample_multi, sample_scm_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--out", default="results/ma/rescue_diagnostic.json")
    args = ap.parse_args()

    topology = Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    view = AgentView("A", topology)
    z_b = topology.b_private[0]
    rng = np.random.default_rng(args.seed)

    rows = []
    for ep in range(args.episodes):
        truth = topology.sample_dag(rng, p=0.5)
        params = sample_scm_params(truth, rng)
        confounded = len(bidirected_pairs(truth, view.nodes)) > 0
        if not confounded:
            continue

        true_index = view.true_index(truth)
        no_mask = np.zeros((args.n, view.k))

        # Arm 1: observational only. The status quo -- A alone, confounded.
        obs, _ = sample_multi(params, args.n, rng)
        post_obs = view.posterior(obs[:, view.nodes], no_mask)

        # Arm 2: EVERY row drawn while B holds its private node under intervention. The
        # confounding path is cut for the whole dataset, so there is no mixture.
        cut, _ = sample_multi(params, args.n, rng, intervene_nodes={z_b})
        post_cut = view.posterior(cut[:, view.nodes], no_mask)

        # Arm 3: half and half -- what my protocol actually gives A, at best.
        half = args.n // 2
        obs_h, _ = sample_multi(params, half, rng)
        cut_h, _ = sample_multi(params, half, rng, intervene_nodes={z_b})
        post_mix = view.posterior(
            np.vstack([obs_h, cut_h])[:, view.nodes], np.zeros((2 * half, view.k)))

        rows.append({
            "episode": ep,
            "mass_observational": float(post_obs[true_index]),
            "mass_cut": float(post_cut[true_index]),
            "mass_mixed": float(post_mix[true_index]),
        })

    def report(key):
        mass = np.array([r[key] for r in rows])
        ident = mass >= args.threshold
        n = len(mass)
        z = 1.96
        p = ident.mean()
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half_w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return {
            "n": int(n),
            "identified_rate": float(p),
            "ci": [float(max(0.0, centre - half_w)), float(min(1.0, centre + half_w))],
            "mean_mass": float(mass.mean()),
            "median_mass": float(np.median(mass)),
        }

    out_report = {arm: report(f"mass_{arm}")
                  for arm in ("observational", "cut", "mixed")}

    print(f"confounded episodes: {len(rows)}\n")
    for arm in ("observational", "cut", "mixed"):
        r = out_report[arm]
        print(f"  {arm:>14}: identified {r['identified_rate']:.3f} "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
              f"mean mass {r['mean_mass']:.4f}  median {r['median_mass']:.4f}")

    verdict = ("pooling -- the mechanism is real, the protocol blocks it"
               if out_report["cut"]["ci"][0] > out_report["observational"]["ci"][1]
               else "mechanism -- breaking the confounder does not restore identifiability")
    out_report["verdict"] = verdict
    print(f"\nverdict: {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "report": out_report,
                              "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
