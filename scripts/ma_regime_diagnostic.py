"""GATE 4 failed twice. This isolates the remaining blocker.

Established so far, both measured:

  1. A randomised `do()` on the confounder does NOT cut confounding. It replaces one latent
     common cause with another that A still cannot see. Clamping does cut it: with the
     whole dataset drawn under a clamp, A's mean posterior mass on its true induced DAG
     goes from 0.0000 to 0.39.

  2. Clamping inside the actual environment STILL rescues nothing -- 0/38 confounded
     episodes. The difference between the two settings is POOLING. In the environment A
     holds 2000 confounded observational rows plus at most 1600 clean clamped rows, all in
     one undifferentiated dataset, and no single DAG fits a mixture of two regimes. A
     cannot separate them because, under the no-disclosure decision, A is never told the
     regime changed.

So the question this script answers: if A could tell the clean rows from the confounded
ones, would the rescue work? Arms, on confounded episodes only:

    pooled        2000 observational + clamped rows, all mixed.  (what the env does now)
    regime        the clamped rows ONLY.                          (one disclosure bit)
    regime+own    clamped rows in which A also varies its own targets, round-robin.
                  (B clamps while A experiments -- the actual coordinated behaviour)

PRE-REGISTERED PREDICTION, before the numbers exist:
    pooled ~ 0, regime ~ 0.18 (the ceiling measured for observational-only clean data,
    limited by Markov equivalence ties), regime+own well above both, because A's own
    interventions orient what the clean regime has made identifiable.

    If regime+own is high, the conclusion is sharp and reportable: the minimum viable
    disclosure for coordination is NOT the |X|^2 ancestral-order bits of MA_DESIGN
    section 5 -- which today measured ~0.005 bits of value per bit disclosed -- but a
    single REGIME bit per round. That would be a genuinely different protocol from the
    one in the design document, arrived at by measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legacy.ma_v1.baselines import GreedyAgentPolicy
from legacy.ma_v1.env import CLAMP, VARY, AgentView, MAConfig, TwoAgentEnv
from ma.projection import bidirected_pairs
from ma.topology import Topology
from sa.scm import sample_multi, sample_scm_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--n_obs", type=int, default=2000)
    ap.add_argument("--n_int", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ma/regime_diagnostic.json")
    args = ap.parse_args()

    topology = Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    view = AgentView("A", topology)
    z_b = topology.b_private[0]
    rng = np.random.default_rng(args.seed)

    # Greedy needs an env instance only to read the window; it never steps it here.
    scratch_env = TwoAgentEnv(MAConfig(topology=topology), seed=args.seed)
    greedy = GreedyAgentPolicy("A", scratch_env, seed=args.seed)

    rows = []
    for ep in range(args.episodes):
        truth = topology.sample_dag(rng, p=0.5)
        params = sample_scm_params(truth, rng)
        if not bidirected_pairs(truth, view.nodes):
            continue
        true_index = view.true_index(truth)

        obs, _ = sample_multi(params, args.n_obs, rng)
        no_mask_obs = np.zeros((args.n_obs, view.k))

        # B clamps its private node for every round. A does nothing of its own yet.
        clamped_blocks = []
        for _ in range(args.rounds):
            block, _ = sample_multi(params, args.n_int, rng,
                                    intervene_nodes={z_b: 0.0})
            clamped_blocks.append(block)
        clamped = np.vstack(clamped_blocks)
        clamp_mask = np.zeros((len(clamped), view.k))

        # Arm: pooled.
        pooled = np.vstack([obs, clamped])
        post_pooled = view.posterior(pooled[:, view.nodes],
                                     np.vstack([no_mask_obs, clamp_mask]))

        # Arm: regime -- the clean rows only.
        post_regime = view.posterior(clamped[:, view.nodes], clamp_mask)

        # Arm: regime + A's own interventions, chosen greedily against A's belief as it
        # develops, all inside the clean regime.
        belief = post_regime.copy()
        joint_rows, joint_mask = [clamped], [clamp_mask]
        for _ in range(args.rounds):
            scores = greedy.scores(belief)
            best = int(np.argmax(scores))
            target = view.authority[best]
            block, _ = sample_multi(params, args.n_int, rng,
                                    intervene_nodes={z_b: 0.0,
                                                     target: scratch_env.config.intervene_scale})
            mask = np.zeros((args.n_int, view.k))
            mask[:, view.pos[target]] = 1.0
            joint_rows.append(block)
            joint_mask.append(mask)
            belief = view.posterior(np.vstack(joint_rows)[:, view.nodes],
                                    np.vstack(joint_mask))
        post_joint = belief

        rows.append({
            "episode": ep,
            "mass_pooled": float(post_pooled[true_index]),
            "mass_regime": float(post_regime[true_index]),
            "mass_regime_own": float(post_joint[true_index]),
        })

    def report(key):
        mass = np.array([r[key] for r in rows])
        ident = mass >= args.threshold
        n = len(mass)
        z = 1.96
        p = ident.mean()
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return {"n": int(n), "identified_rate": float(p),
                "ci": [float(max(0.0, centre - half)), float(min(1.0, centre + half))],
                "mean_mass": float(mass.mean()),
                "median_mass": float(np.median(mass))}

    out_report = {arm: report(f"mass_{arm}")
                  for arm in ("pooled", "regime", "regime_own")}

    print(f"confounded episodes: {len(rows)}\n")
    for arm in ("pooled", "regime", "regime_own"):
        r = out_report[arm]
        print(f"  {arm:>11}: identified {r['identified_rate']:.3f} "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
              f"mean mass {r['mean_mass']:.4f}  median {r['median_mass']:.4f}")

    passed = out_report["regime_own"]["ci"][0] > out_report["pooled"]["ci"][1]
    out_report["rescue_works_with_regime_bit"] = bool(passed)
    print(f"\nrescue with a regime bit: {'WORKS' if passed else 'DOES NOT WORK'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "report": out_report,
                               "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
