"""What does the skeleton assumption actually cost?

THE ASSUMPTION IS NOT WHAT IT LOOKS LIKE. `FactoredBackend` seeds each pair's adjacency from
the true MAG, which reads as oracle knowledge. It is not: a MAG's adjacencies are exactly the
pairs no OBSERVED conditioning set can separate, so the skeleton is recoverable from
observational data alone. Measured 31 Aug 2026, `observational_skeleton` agrees with the true
MAG skeleton on 1,350/1,350 pairs at k=6 and 3,360/3,360 at k=8 -- 100%, zero spurious, zero
missed. What the backend supplies is what FCI's adjacency phase recovers.

SO THE REAL QUESTION IS FINITE SAMPLES. The default hands over the INFINITE-DATA answer. Real
adjacency estimation from `n_obs` rows makes errors, and the two kinds are not symmetric:

    MISSED adjacency     the pair is closed to NONE. The agent will never claim it, and the
                         claim is scored against a truth that says it is adjacent -- a
                         permanent, silent loss.
    SPURIOUS adjacency   the pair is opened to {FWD, BACK, BI} but no intervention can ever
                         settle it, because there is nothing there to find. It stays unsure
                         forever, so it CAPS identification outright.

This script estimates the skeleton by conditional-independence testing at a given `n_obs`,
reports both error kinds separately, and then measures identification seeded from the
estimated skeleton against the same episodes seeded from the true one.

    .venv/bin/python scripts/skeleton_ablation.py --episodes 30 --n_obs 60,200,1000

Reports the cost as a function of sample size, so the assumption becomes a measured quantity
rather than a caveat.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cb.citest import FisherZ                                          # noqa: E402
from cb.claims import score_window                                    # noqa: E402
from cb.factored import FactoredBackend                               # noqa: E402
from ma.env import MAConfig, TwoAgentEnv                              # noqa: E402
from ma.topology import federated_topology                            # noqa: E402


def estimate_skeleton(data: np.ndarray, k: int, alpha: float, max_cond: int = 2):
    """PC-style adjacency search on `data`: keep a pair unless some conditioning set of
    size <= `max_cond` renders it independent.

    `max_cond` is capped deliberately. The full search is exponential in the conditioning
    set size and that is precisely the cost the assumption avoids -- so a bounded search is
    both the honest approximation and the one a practitioner would actually run.
    """
    adjacency = np.ones((k, k), dtype=bool)
    np.fill_diagonal(adjacency, False)
    tester = FisherZ(data, alpha=alpha)
    for u, v in itertools.combinations(range(k), 2):
        others = [w for w in range(k) if w not in (u, v)]
        removed = False
        for size in range(0, min(max_cond, len(others)) + 1):
            for cond in itertools.combinations(others, size):
                try:
                    if tester.independent(u, v, list(cond)):
                        adjacency[u, v] = adjacency[v, u] = False
                        removed = True
                        break
                except Exception:
                    continue
            if removed:
                break
    return adjacency


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private", type=int, default=3)
    ap.add_argument("--shared", type=int, default=3)
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n_obs", default="60,200,1000")
    ap.add_argument("--alpha", default="0.01",
                    help="comma-separated. The errors are overwhelmingly MISSED edges, which "
                         "is an underpowered test rather than a fundamental limit, and alpha "
                         "is the dial: a liberal alpha keeps more edges, trading spurious "
                         "adjacencies (which cap identification) against missed ones (which "
                         "lose a claim outright).")
    ap.add_argument("--max_cond", type=int, default=2)
    ap.add_argument("--out", default="results/skeleton_ablation.json")
    args = ap.parse_args(argv)

    sizes = [int(x) for x in args.n_obs.split(",")]
    alphas = [float(x) for x in str(args.alpha).split(",")]
    rows = []
    print(f"{'n_obs':>7s} {'alpha':>7s} {'skel acc':>9s} {'spur':>5s} {'missed':>7s} "
          f"{'claims right':>13s} {'vs true skel':>13s} {'identified':>11s}")
    for n_obs, alpha in ((n, a) for n in sizes for a in alphas):
        cfg = MAConfig(topology=federated_topology(args.agents, args.private, args.shared),
                       n_obs=n_obs, n_int=40, budget=30, turn_order="round_robin",
                       belief_backend="factored", action_modes=("vary",), claim_bar=1.0,
                       reward_criterion="claims", policy_arch="gnn_portable",
                       graph_model="sf", sf_m=2, episode_mix="confounded",
                       vs_evidence="oracle")
        env = TwoAgentEnv(cfg)
        k = args.private + args.shared
        pairs_seen = correct = spurious = missed = 0
        id_est = id_true = windows = 0
        claim_est = claim_true = claim_n = 0.0
        for episode in range(args.episodes):
            env.reset(seed=episode)
            for agent in env.topology.agents:
                window = env.windows[agent]
                mag = env._true_mag(agent)
                data = env.samples[:, window.nodes]
                estimated = estimate_skeleton(data, k, alpha, args.max_cond)
                for u, v in itertools.combinations(range(k), 2):
                    truly = bool(mag[u, v] or mag[v, u])
                    got = bool(estimated[u, v])
                    pairs_seen += 1
                    correct += (truly == got)
                    spurious += (got and not truly)
                    missed += (truly and not got)
                # identification under each skeleton, with the SAME full intervention set,
                # so the only thing that differs is what the belief started from
                private = [window.pos[n] for n in window.private]
                for label, skel in (("est", estimated), ("true", None)):
                    backend = FactoredBackend(k, evidence="oracle")
                    backend.reset(mag, skeleton=skel)
                    from cb.versionspace import reveal
                    for x in range(k):
                        backend._apply_ancestry(x, reveal(backend.truth, k, x))
                    from cb.factored import FactoredBelief
                    belief = FactoredBelief(backend._possible, k)
                    score = score_window(belief, mag, private, bar=1.0)
                    # Claim-level accuracy alongside identification: identification is
                    # zero-tolerance, so ONE missed adjacency destroys a whole window and the
                    # metric reads as a cliff. The claim fraction shows the gradient
                    # underneath it, which is what says how badly a wrong skeleton hurts
                    # rather than merely that it does.
                    fraction = (score.n_right / score.n_claims) if score.n_claims else 0.0
                    if label == "est":
                        id_est += int(score.identified); claim_est += fraction
                    else:
                        id_true += int(score.identified); claim_true += fraction
                claim_n += 1
                windows += 1
        row = {"n_obs": n_obs, "alpha": alpha, "pairs": pairs_seen,
               "accuracy": correct / pairs_seen, "spurious": spurious, "missed": missed,
               "windows": windows, "identified_estimated": id_est / windows,
               "identified_true": id_true / windows,
               "claims_estimated": claim_est / max(claim_n, 1),
               "claims_true": claim_true / max(claim_n, 1)}
        rows.append(row)
        print(f"{n_obs:7d} {alpha:7.3g} {row['accuracy']:8.1%} {spurious:5d} {missed:7d} "
              f"{row['claims_estimated']:12.1%} {row['claims_true']:12.1%} "
              f"{row['identified_estimated']:10.1%}")

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=1))
    print(f"\nwrote {path}")
    print("`identified (true)` is the ceiling this setup assumes; the gap to `(est)` is what")
    print("the skeleton assumption is worth at that sample size, with EVERY node intervened")
    print("on -- so it isolates the skeleton and nothing else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
