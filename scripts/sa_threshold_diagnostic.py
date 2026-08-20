"""Does the 0.7 identification threshold survive growing `d`?

GATE 1 failed at `d=6` on the LOW side: the observational-only identification rate was
0.025 [0.005, 0.050] against a singleton-MEC fraction of 0.081. Failing low means graphs
that ARE identifiable from observation alone are not being identified -- the opposite of
the leak this project was built to fix.

The hypothesis under test: **the criterion is at fault, not the environment.** Identification
demands >= 0.7 of the posterior on the true DAG. At `d=6` that mass is spread over 3.78M
DAGs, so a singleton-MEC graph can be the clear winner and still hold far less than 0.7.

The measurement, on singleton-MEC graphs with observational data only:

  mass       posterior mass on the true DAG
  local_max  does the true DAG beat every single-edge perturbation of itself (add, delete
             or reverse one edge, acyclic results only)? A cheap stand-in for "is it the
             argmax", usable at `d` where enumeration is not.
  rank       the true rank, by enumeration -- only at d <= 5, and reported alongside
             `local_max` so the proxy can be checked against the thing it stands in for.

If mass falls steeply with `d` while `local_max` stays high, the graph is still being found
and it is the THRESHOLD that has stopped being reachable.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from sa.dp import DPPosterior
from sa.graphs import build_graph_space, is_singleton_mec
from sa.score import get_score
from sa.scm import sample_multi, sample_scm_params


def sample_singleton_dag(d: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Draw from the Erdos-Renyi DAG prior until the graph is alone in its MEC."""
    while True:
        order = rng.permutation(d)
        adjacency = np.zeros((d, d), dtype=bool)
        for i in range(d):
            for j in range(i + 1, d):
                if rng.random() < p:
                    adjacency[order[i], order[j]] = True
        if is_singleton_mec(adjacency):
            return adjacency


def perturbations(adjacency: np.ndarray):
    """Every graph one edge away: add, delete, or reverse. Cyclic results are skipped."""
    from sa.graphs import is_acyclic
    d = adjacency.shape[0]
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            candidate = adjacency.copy()
            if adjacency[i, j]:
                candidate[i, j] = False
                yield candidate                      # delete
                reversed_ = candidate.copy()
                reversed_[j, i] = True
                if is_acyclic(reversed_):
                    yield reversed_                  # reverse
            else:
                candidate[i, j] = True
                if is_acyclic(candidate):
                    yield candidate                  # add


def log_score_of(dp: DPPosterior, log_w: np.ndarray, adjacency: np.ndarray) -> float:
    """Unnormalised log score -- the partition function cancels in every comparison here."""
    d = adjacency.shape[0]
    bits = 1 << np.arange(d)
    total = 0.0
    for node in range(d):
        mask = int(np.dot(np.asarray(adjacency)[:, node] > 0.5, bits))
        total += log_w[node, dp._mask_to_index[node, mask]]
    return float(total)


def run_d(d: int, episodes: int, n_obs: int, prior_p: float, seed: int,
          enumerate_rank: bool) -> dict:
    rng = np.random.default_rng(seed)
    score = get_score("bge", d)
    dp = DPPosterior.for_prior(d, score, kind="erdos_renyi", p=prior_p)

    space = build_graph_space(d) if enumerate_rank else None
    engine = prior_array = None
    if space is not None:
        from sa.posterior import PosteriorEngine
        from sa.priors import erdos_renyi_prior
        engine = PosteriorEngine(space, score)
        # The SAME prior family the DP uses, so any disagreement between the two paths is
        # a real disagreement rather than a misspecification we introduced here.
        prior_array = erdos_renyi_prior(space, p=prior_p)

    rows = []
    for episode in range(episodes):
        adjacency = sample_singleton_dag(d, prior_p, rng)
        params = sample_scm_params(adjacency, rng)
        samples, _ = sample_multi(params, n_obs, rng)
        intervened = np.zeros_like(samples, dtype=bool)

        log_w = dp.log_weights(samples, intervened)
        mass = float(np.exp(dp.log_prob_dag(log_w, adjacency)))

        truth_score = log_score_of(dp, log_w, adjacency)
        local_max = all(log_score_of(dp, log_w, other) <= truth_score
                        for other in perturbations(adjacency))

        row = {"mass": mass, "local_max": bool(local_max)}
        if engine is not None:
            posterior = engine.posterior(samples, intervened, prior=prior_array)
            key = np.all(space.dags == adjacency, axis=(1, 2))
            index = int(np.flatnonzero(key)[0])
            row["rank"] = int((posterior > posterior[index]).sum() + 1)
            row["enumerated_mass"] = float(posterior[index])
        rows.append(row)

    masses = np.array([r["mass"] for r in rows])
    out = {
        "d": d, "episodes": episodes, "n_obs": n_obs, "prior_p": prior_p,
        "mass_median": float(np.median(masses)),
        "mass_mean": float(masses.mean()),
        "mass_p10": float(np.percentile(masses, 10)),
        "mass_p90": float(np.percentile(masses, 90)),
        "frac_mass_over_0.7": float((masses >= 0.7).mean()),
        "frac_local_max": float(np.mean([r["local_max"] for r in rows])),
        "rows": rows,
    }
    if engine is not None:
        ranks = np.array([r["rank"] for r in rows])
        out["frac_rank_1"] = float((ranks == 1).mean())
        out["rank_median"] = float(np.median(ranks))
        # The proxy is only useful if it agrees with the thing it stands in for.
        agree = np.mean([(r["rank"] == 1) == r["local_max"] for r in rows])
        out["local_max_agrees_with_rank1"] = float(agree)
        enum = np.array([r["enumerated_mass"] for r in rows])
        out["dp_vs_enumerated_max_abs_diff"] = float(np.abs(enum - masses).max())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", default="3,4,5,6")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--prior_p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enumerate_upto", type=int, default=5)
    ap.add_argument("--out", default="results/threshold_diagnostic.json")
    args = ap.parse_args(argv)

    report = {"args": vars(args), "by_d": {}}
    for d in [int(x) for x in args.dims.split(",")]:
        started = time.time()
        block = run_d(d, args.episodes, args.n_obs, args.prior_p, args.seed,
                      enumerate_rank=(d <= args.enumerate_upto))
        block["seconds"] = time.time() - started
        report["by_d"][str(d)] = block
        line = (f"d={d}  mass median {block['mass_median']:.3f} "
                f"(p10 {block['mass_p10']:.3f}, p90 {block['mass_p90']:.3f})  "
                f">=0.7 in {block['frac_mass_over_0.7']:.1%}  "
                f"local-max {block['frac_local_max']:.1%}")
        if "frac_rank_1" in block:
            line += (f"  rank-1 {block['frac_rank_1']:.1%}  "
                     f"proxy agrees {block['local_max_agrees_with_rank1']:.1%}  "
                     f"dp-vs-enum {block['dp_vs_enumerated_max_abs_diff']:.2e}")
        print(line + f"  [{block['seconds']:.0f}s]", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
