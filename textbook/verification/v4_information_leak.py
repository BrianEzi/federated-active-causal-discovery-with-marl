"""V4 -- The information leak: how a scorer can solve the task without intervening.

Part II, Chapter 6. Four separate failures are reproduced and measured:

  [1] UNPENALISED PROFILE LIKELIHOOD inverts the ranking: denser DAGs always fit
      better, so the complete graphs take the top of the posterior and the true
      sparse graph is pushed down.
  [2] KnownVarianceScore (the previous codebase's estimator) is not a difference
      of a set function, so it VIOLATES SCORE EQUIVALENCE and separates Markov
      equivalent DAGs from observational data alone.
  [3] Consequently it "identifies" graphs no intervention was needed for: the
      observational-only identification rate blows past the exact singleton
      fraction. This is the GATE 1 failure, quantified.
  [4] With EQUAL error variances the leak is available to any scorer that can
      exploit it -- but BGe still cannot, because score equivalence is structural.
      This separates "the data are identifiable" from "the estimator identifies".

Run:  python textbook/verification/v4_information_leak.py
"""
from __future__ import annotations

import itertools

import numpy as np

from sa.graphs import build_graph_space
from sa.posterior import PosteriorEngine, is_identified
from sa.score import BGeScore, BICScore, KnownVarianceScore
from sa.scm import SCMParams, sample, sample_scm_params

RNG = np.random.default_rng(11)


class ProfileLikelihood:
    """Maximised Gaussian log-likelihood with NO complexity penalty.

    This is the estimator the two-agent codebase used. It was valid there only
    because all eight candidate graphs had exactly three edges, so no penalty was
    needed to compare them. The moment all DAGs are admitted the omission is fatal.
    """

    def __init__(self, d, ridge=1e-9):
        self.d, self.ridge = d, ridge

    def local_score(self, node, parents, samples):
        parents = sorted(int(p) for p in parents)
        n = samples.shape[0]
        if n == 0:
            return 0.0
        y = samples[:, int(node)] - samples[:, int(node)].mean()
        if parents:
            X = samples[:, parents] - samples[:, parents].mean(axis=0)
            beta = np.linalg.solve(X.T @ X + self.ridge * np.eye(len(parents)), X.T @ y)
            resid = y - X @ beta
        else:
            resid = y
        var = max(float(np.mean(resid ** 2)), 1e-12)
        return -0.5 * n * (np.log(2 * np.pi * var) + 1.0)


def dag_score(score, adjacency, data):
    d = adjacency.shape[0]
    return sum(score.local_score(j, tuple(np.flatnonzero(adjacency[:, j] > 0.5)), data)
               for j in range(d))


def all_scores(score, space, data):
    return np.array([dag_score(score, space.dags[i], data) for i in range(space.n_dags)])


def posterior_from(scores):
    p = np.exp(scores - scores.max())
    return p / p.sum()


def within_class_spread(scores, space):
    return max(scores[np.flatnonzero(space.mec_id == c)].ptp()
               for c in range(space.n_mecs) if (space.mec_id == c).sum() > 1)


def main() -> None:
    print("=" * 76)
    print("V4  The information leak")
    print("=" * 76)

    d = 3
    space = build_graph_space(d)
    truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8)   # X1->X2->X3
    truth_idx = next(i for i in range(space.n_dags)
                     if np.array_equal(space.dags[i], truth))
    params = sample_scm_params(truth, RNG)
    data, _ = sample(params, 1000, RNG)
    n_edges = space.dags.reshape(space.n_dags, -1).sum(axis=1)

    # ---------------------------------------------------------------- [1] ----
    print("\n[1] Unpenalised profile likelihood INVERTS the ranking")
    print("    (true graph is the 2-edge chain X1 -> X2 -> X3, N = 1000)")
    prof = all_scores(ProfileLikelihood(d), space, data)
    p_prof = posterior_from(prof)
    dense = n_edges == 3
    order = np.argsort(-p_prof)
    print(f"    {'rank':<6}{'edges':<7}{'log lik':>13}{'posterior':>12}")
    for r in range(6):
        i = order[r]
        print(f"    {r+1:<6}{n_edges[i]:<7}{prof[i]:>13.4f}{p_prof[i]:>12.4f}")
    rank_true = int(np.flatnonzero(order == truth_idx)[0]) + 1
    print(f"    -> the {int(dense.sum())} densest (3-edge) DAGs hold "
          f"{p_prof[dense].sum():.1%} of the mass")
    print(f"    -> the TRUE 2-edge graph ranks {rank_true} of {space.n_dags} "
          f"with posterior {p_prof[truth_idx]:.4f}")
    print(f"    -> maximised log-likelihood is monotone in edge count: "
          f"min over 3-edge DAGs {prof[dense].min():.3f} "
          f">= max over 2-edge DAGs {prof[n_edges == 2].max():.3f}: "
          f"{prof[dense].min() >= prof[n_edges == 2].max() - 1e-9}")

    print("\n    By contrast, BGe integrates the parameters out:")
    bge = BGeScore(d)
    stats = bge.sufficient_stats(data)
    bsc = np.array([sum(bge.local_score_from_stats(
        j, tuple(np.flatnonzero(space.dags[i][:, j] > 0.5)), stats) for j in range(d))
        for i in range(space.n_dags)])
    p_bge = posterior_from(bsc)
    border = np.argsort(-p_bge)
    print(f"    -> 3-edge DAGs hold {p_bge[dense].sum():.1%} of the mass")
    print(f"    -> the true graph ranks "
          f"{int(np.flatnonzero(border == truth_idx)[0]) + 1} "
          f"with posterior {p_bge[truth_idx]:.4f}")
    print(f"    -> its equivalence class holds "
          f"{p_bge[space.mec_id == space.mec_id[truth_idx]].sum():.4f}, split "
          f"{space.mec_sizes[space.mec_id[truth_idx]]} ways -- the tie is INTACT")

    # ---------------------------------------------------------------- [2] ----
    print("\n[2] Score equivalence, scorer by scorer (within-class spread, nats)")
    print(f"    {'scorer':<24}{'within-class spread':>22}{'verdict':>16}")
    for name, sc in [("BGe", BGeScore(d)),
                     ("BIC", BICScore(d)),
                     ("profile likelihood", ProfileLikelihood(d)),
                     ("KnownVarianceScore", KnownVarianceScore(d, noise_scale=1.0))]:
        s = all_scores(sc, space, data)
        spread = within_class_spread(s, space)
        verdict = "EQUIVALENT" if spread < 1e-7 else "*** LEAKS ***"
        print(f"    {name:<24}{spread:>22.6e}{verdict:>16}")

    print("\n    Why KnownVarianceScore leaks, algebraically. Its local term is")
    print("        s(j | Pa) = -RSS(j | Pa) / (2 sigma^2) - (n/2) log(2 pi sigma^2)")
    print("    with sigma FIXED. Summing over a DAG gives")
    print("        S(G) = -(1/2 sigma^2) sum_j RSS(j | Pa_j)  -  (nd/2) log(2 pi sigma^2),")
    print("    whose second term is constant across DAGs, so the ranking is decided")
    print("    entirely by sum_j RSS(j | Pa_j). That is NOT of the form")
    print("    sum_j [m(Pa_j u {j}) - m(Pa_j)] for any set function m, so the")
    print("    cancellation in the covered-edge-reversal argument (v2, section 6)")
    print("    does not occur. The free variance in BGe/BIC is precisely what")
    print("    restores it: a fitted sigma^2_j makes each term a log-determinant")
    print("    ratio, which IS a set-function difference.")
    kv = KnownVarianceScore(d, noise_scale=1.0)
    s_kv = all_scores(kv, space, data)
    cls = space.mec_id[truth_idx]
    members = np.flatnonzero(space.mec_id == cls)
    print(f"\n    Demonstration on the true graph's class (size {len(members)}):")
    for m in members:
        arcs = " ".join(f"{i}->{j}" for i in range(d) for j in range(d)
                        if space.dags[m][i, j])
        print(f"      {arcs:<18} BGe {bsc[m]:>12.4f}   KnownVariance {s_kv[m]:>12.4f}")
    print(f"      BGe spread {bsc[members].ptp():.3e}   "
          f"KnownVariance spread {s_kv[members].ptp():.3f} nats")

    # ---------------------------------------------------------------- [3] ----
    print("\n[3] The consequence: observational identification beyond what theory allows")
    target = space.singleton_fraction
    print(f"    Exact target (fraction of DAGs alone in their class) = {target:.1%}")
    n_ep = 300
    for name, make in [("BGe", lambda: BGeScore(d)),
                       ("KnownVarianceScore", lambda: KnownVarianceScore(d, 1.0))]:
        for equal_noise in (False, True):
            sc = make()
            engine = PosteriorEngine(space, sc)
            rng = np.random.default_rng(4)
            hits = 0
            for _ in range(n_ep):
                gi = rng.integers(space.n_dags)
                a = space.dags[gi]
                pr = sample_scm_params(a, rng)
                if equal_noise:
                    pr = SCMParams(pr.adjacency, pr.weights, np.ones(d))
                dat, iv = sample(pr, 1000, rng)
                post = engine.posterior(dat, iv)
                hits += int(is_identified(post, int(gi), 0.7))
            rate = hits / n_ep
            tag = "equal noise" if equal_noise else "per-node noise"
            flag = ("  <-- LEAK" if rate > target + 0.05 else
                    "  ok" if rate >= target - 0.05 else "  under-powered")
            print(f"    {name:<20} {tag:<16} observational solve rate "
                  f"{rate:>6.1%}{flag}")

    # ---------------------------------------------------------------- [4] ----
    print("\n[4] Separating 'the data are identifiable' from 'the estimator identifies'")
    print("    Peters & Buhlmann (2014): with EQUAL error variances the DAG is")
    print("    identifiable from the observational distribution. Yet BGe's")
    print("    observational solve rate above is unchanged by equal noise, because")
    print("    score equivalence is a property of the SCORE'S FORM (v2 section 6),")
    print("    not of the data. A score-equivalent estimator provably CANNOT exploit")
    print("    the equal-variance signal. Per-node noise in sa/scm.py is therefore")
    print("    defence in depth; the load-bearing fix was replacing the estimator.")

    print("\n    Direct check -- equal-variance data, BGe, within-class spread:")
    a = space.dags[truth_idx]
    pr = sample_scm_params(a, RNG)
    pr_eq = SCMParams(pr.adjacency, pr.weights, np.ones(d))
    dat_eq, _ = sample(pr_eq, 4000, RNG)
    s_eq = all_scores(BGeScore(d), space, dat_eq)
    print(f"      BGe on equal-variance data:  spread {within_class_spread(s_eq, space):.3e}"
          f"  -> still exactly tied")
    s_eq_kv = all_scores(KnownVarianceScore(d, 1.0), space, dat_eq)
    print(f"      KnownVariance on the same data: spread "
          f"{within_class_spread(s_eq_kv, space):.3f} nats  -> separated")

    print("\nALL V4 CHECKS COMPLETE")


if __name__ == "__main__":
    main()
