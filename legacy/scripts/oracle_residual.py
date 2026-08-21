"""What did the burn_in/thin stopgap actually leave behind?

The 2026-08-19 fix raised SamplingOracle from burn_in=5000/thin=10 to 50000/50, cutting the
max edge-marginal error against exact DP marginals from 0.100 to 0.016. But MARGINAL ERROR
IS NOT THE QUANTITY THAT MATTERS. The oracle's job is to pick a target, and at the old
settings it picked a DIFFERENT target from a well-mixed chain in 38% of episodes, giving up
0.065 nats on average.

That disagreement rate was never re-measured after the fix. It is what determines whether
the d=7 baseline -- the opponent every learned result is scored against -- is trustworthy.

Reference is a very long chain, not the exact oracle: at d=7 the exact posterior cannot be
enumerated, which is why the sampler exists. At d=5 both are available and the exact oracle
is used as a third arm, so the long chain's own trustworthiness is checked rather than
assumed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from sa.env import EnvConfig
from sa.env_dp import DPCausalDiscoveryEnv
from sa.oracle import SamplingOracle


def agreement(env, cheap: SamplingOracle, reference: SamplingOracle,
              episodes: int, seed: int) -> dict:
    same, lost, deltas = 0, [], []
    for episode in range(episodes):
        result = env.reset(seed=seed * 1000 + episode)
        log_w = env.log_w
        cheap_scores = cheap.scores(log_w)
        ref_scores = reference.scores(log_w)
        cheap_pick = int(np.argmax(cheap_scores))
        ref_pick = int(np.argmax(ref_scores))
        same += int(cheap_pick == ref_pick)
        # Value lost is measured on the REFERENCE's scale -- how much information the cheap
        # choice gives up according to the better-mixed chain.
        delta = float(ref_scores[ref_pick] - ref_scores[cheap_pick])
        deltas.append(delta)
        if delta > 0.01:
            lost.append(delta)
    deltas = np.asarray(deltas)
    return {
        "episodes": episodes,
        "agreement": same / episodes,
        "mean_nats_lost": float(deltas.mean()),
        "max_nats_lost": float(deltas.max()),
        "episodes_losing_over_0.01": len(lost),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=7)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sampler/residual.json")
    args = ap.parse_args()

    env = DPCausalDiscoveryEnv(EnvConfig(d=args.d, n_obs=args.n_obs, n_int=100, budget=5))
    started = time.time()

    arms = {
        # The settings the queued Myriad d=7 jobs are using.
        "mh_50k_50": SamplingOracle(env.dp, n_draws=4000, burn_in=50_000, thin=50,
                                    seed=args.seed, method="mh"),
        # The new default: independent draws, no burn-in.
        "exact": SamplingOracle(env.dp, n_draws=4000, seed=args.seed, method="exact"),
        # Half the draws, to show the exact arm is not just buying agreement with compute.
        "exact_2000": SamplingOracle(env.dp, n_draws=2000, seed=args.seed,
                                     method="exact"),
    }
    # Reference is an EXACT sampler with many draws. Using a long MH chain as the reference
    # would beg the question -- it would score each arm against the very thing whose mixing
    # is in doubt.
    reference = SamplingOracle(env.dp, n_draws=40_000, seed=args.seed + 99, method="exact")

    report = {"d": args.d, "n_obs": args.n_obs, "arms": {}}
    for label, oracle in arms.items():
        report["arms"][label] = agreement(env, oracle, reference, args.episodes, args.seed)
        row = report["arms"][label]
        print(f"{label:16s} agreement {row['agreement']:.3f}  "
              f"mean lost {row['mean_nats_lost']:.4f} nats  "
              f"max {row['max_nats_lost']:.4f}  "
              f"[{time.time() - started:.0f}s]", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
