"""Which regime-scoring rule removes the valley, and does any of them fix confounding?

Last night's diagnosis: the SUBSET rule creates a valley in the payoff. Sweeping the
probability that the partner clamps, on unconfounded episodes, identification went
0.815 -> 0.721 -> 0.919 -> 0.991 for p = 0, 0.25, 0.5, 1.0. A learner starting near zero
sees a negative gradient and never crosses.

This measures all four rules over the same episodes and the same clamp probabilities, so
the comparison is like-for-like.

TWO SEPARATE QUESTIONS, and they should not be conflated:

  Q1 (gradient) Is identification on UNCONFOUNDED episodes monotone non-decreasing in the
                clamp probability? If yes, the valley is gone and a learner can climb.
  Q2 (target)   Does identification on CONFOUNDED episodes rise with the clamp probability?
                This is what coordination is actually for.

PRE-REGISTERED PREDICTIONS, before the numbers exist:

  POOLED      flat and near zero on confounded, flat on unconfounded. No valley, no payoff.
  SUBSET      reproduces last night: valley on unconfounded, strong payoff on confounded.
  JOINT       valley gone on unconfounded, since clean rows are added rather than
              substituted. On confounded I expect it to UNDERPERFORM subset, because the
              dirty regime still prefers a structure that mimics the confounding and it
              carries most of the rows.
  JOINT_CONF  no valley AND the confounded payoff, because the confounding is modelled
              explicitly as a per-pair flag on the dirty regime rather than being forced
              into the DAG.

  If JOINT_CONF does not beat JOINT on confounded episodes, my confinement-based
  representation is not buying anything and the simpler rule should be preferred.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from legacy.ma_v1.env import AgentView
from ma.projection import bidirected_pairs
from ma.score_regimes import RULES, RegimeScorer
from ma.topology import Topology
from sa.scm import sample_multi, sample_scm_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--n_obs", type=int, default=2000)
    ap.add_argument("--n_int", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--clamp_probs", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ma/regime_scoring.json")
    args = ap.parse_args()

    topology = Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    view = AgentView("A", topology)
    z_b = topology.b_private[0]
    scorer = RegimeScorer(view, [view.pos[s] for s in topology.exposed])

    results = {rule: {p: {"conf": [], "unconf": []} for p in args.clamp_probs}
               for rule in RULES}

    t0 = time.perf_counter()
    for ep in range(args.episodes):
        # One RNG stream per episode so the true graph and SCM are IDENTICAL across every
        # rule and every clamp probability. The arms differ only in what is being tested.
        rng = np.random.default_rng(args.seed * 1_000_003 + ep)
        truth = topology.sample_dag(rng, p=0.5)
        params = sample_scm_params(truth, rng)
        confounded = len(bidirected_pairs(truth, view.nodes)) > 0
        true_index = view.true_index(truth)

        obs, _ = sample_multi(params, args.n_obs, rng)

        for prob in args.clamp_probs:
            blocks = [obs]
            clean_flags = [np.zeros(args.n_obs, dtype=bool)]
            masks = [np.zeros((args.n_obs, view.k))]

            for _ in range(args.rounds):
                clamping = rng.random() < prob
                targets = {z_b: 0.0} if clamping else {}
                # A always experiments on one of its own nodes, so the comparison is not
                # confounded by A doing nothing. Chosen at random rather than greedily:
                # a fixed, policy-free protocol keeps this a measurement of the SCORING
                # rule, not of a policy.
                own = int(rng.choice(view.authority))
                targets[own] = 2.0
                block, _ = sample_multi(params, args.n_int, rng, intervene_nodes=targets)
                mask = np.zeros((args.n_int, view.k))
                mask[:, view.pos[own]] = 1.0
                blocks.append(block)
                masks.append(mask)
                clean_flags.append(np.full(args.n_int, clamping, dtype=bool))

            samples = np.vstack(blocks)[:, view.nodes]
            known = np.vstack(masks)
            clean = np.concatenate(clean_flags)

            for rule in RULES:
                post = scorer.log_posterior(samples, known, clean, rule)
                bucket = "conf" if confounded else "unconf"
                results[rule][prob][bucket].append(float(post[true_index]))

        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{args.episodes} episodes "
                  f"[{time.perf_counter() - t0:.0f}s]", flush=True)

    report = {}
    print(f"\n{'rule':>11} {'p(clamp)':>9} {'conf':>16} {'unconf':>16}")
    for rule in RULES:
        report[rule] = {}
        for prob in args.clamp_probs:
            row = {}
            for bucket in ("conf", "unconf"):
                mass = np.array(results[rule][prob][bucket])
                row[bucket] = {
                    "n": int(len(mass)),
                    "identified": float(np.mean(mass >= args.threshold)) if len(mass) else None,
                    "mean_mass": float(mass.mean()) if len(mass) else None,
                }
            report[rule][str(prob)] = row
            def fmt(bucket):
                value = row[bucket]["identified"]
                text = "    n/a" if value is None else f"{value:>7.3f}"
                return f"{text} (n={row[bucket]['n']:>3})"
            print(f"{rule:>11} {prob:>9.2f} {fmt('conf')} {fmt('unconf')}")

    # Q1: is the unconfounded curve monotone non-decreasing?
    print("\nQ1 -- valley check (unconfounded must be monotone non-decreasing):")
    for rule in RULES:
        curve = [report[rule][str(p)]["unconf"]["identified"] for p in args.clamp_probs]
        if any(c is None for c in curve):
            print(f"  {rule:>11}: insufficient episodes")
            report[rule]["valley_free"] = None
            continue
        drops = [round(curve[i + 1] - curve[i], 4) for i in range(len(curve) - 1)]
        worst = min(drops) if drops else 0.0
        verdict = "no valley" if worst >= -0.02 else f"VALLEY (worst step {worst:+.3f})"
        report[rule]["valley_free"] = bool(worst >= -0.02)
        print(f"  {rule:>11}: {[round(c, 3) for c in curve]}  -> {verdict}")

    print("\nQ2 -- confounded payoff (identification at p=1.0 minus at p=0.0):")
    for rule in RULES:
        lo = report[rule][str(args.clamp_probs[0])]["conf"]["identified"]
        hi = report[rule][str(args.clamp_probs[-1])]["conf"]["identified"]
        if lo is None or hi is None:
            print(f"  {rule:>11}: insufficient confounded episodes")
            report[rule]["confounded_payoff"] = None
            continue
        report[rule]["confounded_payoff"] = float(hi - lo)
        print(f"  {rule:>11}: {lo:.3f} -> {hi:.3f}  ({hi - lo:+.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "report": report}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
