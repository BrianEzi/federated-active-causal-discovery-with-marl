"""Is lowering `n_obs` lengthening the horizon, or handing the agent skeleton work?

The d=7 `n_obs` sweep (job 152604) found the agent's advantage over the myopic oracle
GROWS as `n_obs` falls -- median gap closed 1.001 at 20000, 1.043 at 10000, 1.130 at
5000. But GATE 1 starts failing over the same range. Two incompatible readings:

  (a) HORIZON. Less observational data leaves more to discover, the optimal plan is
      longer, and planning finally pays. The window is open and the result is real.

  (b) SKELETON. Less observational data leaves the skeleton unsettled, so the agent is
      being scored on a different and easier-to-beat task than the one the design
      intends ("orient within the equivalence class"). GATE 1 failing says exactly this:
      the environment is under-powered by our own pre-registered standard.

They are distinguishable at step 0, before any agent acts. Under (a) the skeleton is
settled at every `n_obs` and only orientation is open. Under (b) skeleton error rises
sharply as `n_obs` falls.

PRE-REGISTERED PREDICTION, recorded before the numbers exist:
    I expect (b) to carry a substantial part of it -- GATE 1 failing at 5000 is hard to
    read any other way -- but not all of it, because greedy's own cost rises over the
    same range (1.94 -> 2.35), which is a horizon effect and cannot be a scoring
    artefact. The informative outcome is the SPLIT, not which label wins.

Decision rule fixed now:
    skeleton error at n_obs=5000 within 25% of its value at 20000  -> (a), result stands
    more than 2x its value at 20000                                -> (b), the low-n_obs
                                                                      arms are not
                                                                      measuring planning
    in between                                                     -> report both, claim
                                                                      neither
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sa.backend import Backend
from sa.env import EnvConfig


def diagnose(n_obs: int, d: int, episodes: int, seed: int) -> dict:
    """Step-0 belief quality: how much of the remaining uncertainty is skeleton, and how
    much is orientation?"""
    cfg = EnvConfig(d=d, n_obs=n_obs, budget=20, observation_kinds=None) \
        if False else EnvConfig(d=d, n_obs=n_obs, budget=20)
    backend = Backend(cfg, force_dp=True, seed=seed)
    env = backend.make_env()

    skeleton_err, orient_err, true_mass, identified, n_edges = [], [], [], [], []
    triu = np.triu_indices(d, k=1)

    for ep in range(episodes):
        result = env.reset(seed=seed * 100_000 + ep)
        truth = np.asarray(env.true_adjacency) > 0.5
        marg = np.clip(env.dp.edge_marginals_onepass(env.log_w), 0.0, 1.0)

        # Skeleton belief: probability the pair is adjacent in EITHER direction. The two
        # orientations are mutually exclusive in a DAG, so the sum is a probability.
        adjacent = np.clip(marg + marg.T, 0.0, 1.0)
        true_adjacent = truth | truth.T
        # Expected number of skeleton mistakes: sum over unordered pairs of the
        # probability mass sitting on the wrong adjacency answer.
        skeleton_err.append(float(np.abs(adjacent - true_adjacent)[triu].sum()))

        # Orientation belief, CONDITIONAL on the pair being adjacent -- this is the part
        # the design intends the agent to resolve by intervening. Averaged over the true
        # edges only, so it is not diluted by the (many) non-adjacent pairs.
        wrong = 0.0
        for i, j in zip(*np.nonzero(truth)):
            total = adjacent[i, j]
            wrong += (marg[j, i] / total) if total > 1e-12 else 0.5
        n = int(truth.sum())
        orient_err.append(wrong / n if n else 0.0)
        n_edges.append(n)

        true_mass.append(float(result.info["true_mass"]))
        identified.append(bool(result.identified))

    def stat(values):
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                "se": float(arr.std(ddof=1) / np.sqrt(len(arr)))}

    return {
        "n_obs": n_obs,
        "episodes": episodes,
        # Expected skeleton mistakes at step 0. THE decisive number.
        "skeleton_error": stat(skeleton_err),
        # Expected fraction of true edges pointing the wrong way, given adjacency.
        "orientation_error": stat(orient_err),
        "true_mass": stat(true_mass),
        "gate1_rate": float(np.mean(identified)),
        "mean_true_edges": float(np.mean(n_edges)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=7)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_obs", type=int, nargs="+",
                    default=[2000, 5000, 10000, 20000, 40000])
    ap.add_argument("--out", default="results/step0/step0_d7.json")
    args = ap.parse_args()

    rows = []
    for n_obs in args.n_obs:
        t0 = time.perf_counter()
        row = diagnose(n_obs, args.d, args.episodes, args.seed)
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        print(f"n_obs={n_obs:>6}  skeleton_err={row['skeleton_error']['mean']:.3f}"
              f" +/- {row['skeleton_error']['se']:.3f}"
              f"  orient_err={row['orientation_error']['mean']:.3f}"
              f"  true_mass={row['true_mass']['mean']:.3f}"
              f"  gate1={row['gate1_rate']:.4f}"
              f"  [{row['seconds']:.0f}s]", flush=True)

    baseline = next((r for r in rows if r["n_obs"] == 20000), None)
    verdict = None
    if baseline is not None:
        low = next((r for r in rows if r["n_obs"] == 5000), None)
        if low is not None:
            ref = baseline["skeleton_error"]["mean"]
            ratio = low["skeleton_error"]["mean"] / ref if ref > 1e-9 else float("inf")
            verdict = {
                "skeleton_error_ratio_5000_over_20000": ratio,
                "reading": ("horizon" if ratio <= 1.25 else
                            "skeleton" if ratio >= 2.0 else "ambiguous"),
            }
            print(f"\nratio = {ratio:.2f} -> {verdict['reading']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows,
                               "verdict": verdict}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
