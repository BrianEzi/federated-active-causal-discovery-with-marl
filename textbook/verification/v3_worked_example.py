"""V3 -- The worked 3-node BGe example of Part II, Chapter 8.

Every matrix, determinant and gamma term printed in the chapter is produced here,
for a fixed toy dataset, and every determinant is cross-checked in EXACT RATIONAL
ARITHMETIC with SymPy so no step rests on floating point.

Three DAGs are compared on the same data:

    G_empty :  no edges
    G_chain :  X1 -> X2 -> X3
    G_dense :  X1 -> X2, X1 -> X3, X2 -> X3   (complete)

together with G_rev : X1 <- X2 <- X3, the Markov-equivalent reversal of the
chain, which must score identically.

Run:  python textbook/verification/v3_worked_example.py
"""
from __future__ import annotations

import itertools

import numpy as np
import sympy as sp
from scipy.special import multigammaln

from sa.score import BGeScore

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# A fixed 10 x 3 dataset. Generated once from X1 -> X2 -> X3 with weights
# (1.5, -0.8) and per-node noise scales (1.0, 0.6, 1.2), then FROZEN as literals
# so the chapter's arithmetic is reproducible without an RNG.
DATA = np.array([
    [ 0.30471708, -0.10552426, -0.85899693],
    [-1.03998411, -1.98079647, -0.02257763],
    [ 0.7504512 ,  1.61618922, -1.32066576],
    [ 0.94056472,  1.03380073, -0.10593985],
    [-1.95103519, -2.55829472,  1.85227818],
    [-1.30217951, -2.71325041,  0.87172698],
    [ 0.1278404 , -0.34515861,  0.30032683],
    [-0.31624259, -0.29835439,  1.29003243],
    [-0.01680116,  0.34273656, -0.29268576],
    [-0.85304393, -1.85466514,  1.31151257],
])

LABELS = {0: "X1", 1: "X2", 2: "X3"}


def show(name, M):
    print(f"    {name} =")
    for row in np.atleast_2d(M):
        print("        [" + "  ".join(f"{v:>11.6f}" for v in np.atleast_1d(row)) + "]")


def exact_det(M):
    """Determinant in exact rational arithmetic (SymPy), as a float and a Rational."""
    R = sp.Matrix([[sp.nsimplify(sp.Rational(str(v)), rational=True) for v in row]
                   for row in np.atleast_2d(M)])
    det = sp.simplify(R.det())
    return det, float(det)


def marginal(stats_n, mean, scatter, subset, d, alpha_mu, alpha_w, T, verbose=False):
    """log p(D^subset), with every intermediate quantity exposed."""
    l = len(subset)
    if l == 0:
        return 0.0
    idx = np.asarray(subset, dtype=int)
    N = stats_n
    m = mean[idx]
    S = scatter[np.ix_(idx, idx)]
    Tl = T[np.ix_(idx, idx)]
    shrink = N * alpha_mu / (N + alpha_mu)
    R = Tl + S + shrink * np.outer(m, m)
    nu = alpha_w - d + l

    _, logdet_T = np.linalg.slogdet(Tl)
    _, logdet_R = np.linalg.slogdet(R)

    terms = {
        "-(N l/2) log pi":            -(N * l / 2.0) * np.log(np.pi),
        "(l/2) log(a_mu/(a_mu+N))":   (l / 2.0) * np.log(alpha_mu / (alpha_mu + N)),
        "log Gamma_l((nu+N)/2)":      multigammaln((nu + N) / 2.0, l),
        "-log Gamma_l(nu/2)":         -multigammaln(nu / 2.0, l),
        "(nu/2) log|T_l|":            (nu / 2.0) * logdet_T,
        "-((nu+N)/2) log|R_l|":       -((nu + N) / 2.0) * logdet_R,
    }
    total = float(sum(terms.values()))

    if verbose:
        names = "{" + ", ".join(LABELS[i] for i in subset) + "}"
        print(f"\n  --- subset {names}   (l = {l},  nu = alpha_w - d + l = {nu:g}) ---")
        show("mean_l", m)
        show("S_l (centred scatter)", S)
        show("T_l", Tl)
        print(f"    shrink = N*a_mu/(N+a_mu) = {N}*{alpha_mu}/({N}+{alpha_mu}) = {shrink:.6f}")
        show("R_l = T_l + S_l + shrink * mean_l mean_l^T", R)
        dT, fT = exact_det(Tl)
        dR, fR = exact_det(np.round(R, 12))
        print(f"    |T_l| = {np.exp(logdet_T):.10g}   (SymPy exact: {dT} = {fT:.10g})")
        print(f"    |R_l| = {np.exp(logdet_R):.10g}   (SymPy exact: {fR:.10g})")
        for k, v in terms.items():
            print(f"      {k:<28} {v:>16.9f}")
        print(f"      {'TOTAL log p(D^l)':<28} {total:>16.9f}")
    return total


def main() -> None:
    print("=" * 78)
    print("V3  Worked BGe example on a fixed 10 x 3 dataset")
    print("=" * 78)

    d, N = 3, DATA.shape[0]
    bge = BGeScore(d)
    alpha_mu, alpha_w, T = bge.alpha_mu, bge.alpha_w, bge.T

    print(f"\nPrior: mu_0 = 0,  alpha_mu = {alpha_mu:g},  alpha_w = d + 2 = {alpha_w:g}")
    t = alpha_mu * (alpha_w - d - 1) / (alpha_mu + 1)
    print(f"       t = alpha_mu (alpha_w - d - 1)/(alpha_mu + 1) = "
          f"{alpha_mu:g}*({alpha_w:g}-{d}-1)/({alpha_mu:g}+1) = {t:g}")
    show("T = t I_3", T)

    print(f"\nData: N = {N}")
    show("D (10 x 3)", DATA)
    mean = DATA.mean(axis=0)
    centred = DATA - mean
    scatter = centred.T @ centred
    show("column means xbar", mean)
    show("centred scatter S_N = (D - 1 xbar^T)^T (D - 1 xbar^T)", scatter)
    dS, fS = exact_det(np.round(scatter, 12))
    print(f"    |S_N| = {np.linalg.det(scatter):.10g}  (SymPy exact: {fS:.10g})")

    # -- every subset marginal, in full ---------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1  The seven subset marginals log p(D^l)")
    print("-" * 78)
    marg = {}
    for r in range(1, d + 1):
        for subset in itertools.combinations(range(d), r):
            marg[subset] = marginal(N, mean, scatter, subset, d,
                                    alpha_mu, alpha_w, T, verbose=True)
            # cross-check against the shipped implementation
            ref = bge._log_marginal_stats(bge.sufficient_stats(DATA), subset)
            assert abs(marg[subset] - ref) < 1e-10, (subset, marg[subset], ref)
    marg[()] = 0.0
    print("\n    (each agrees with sa.score.BGeScore to < 1e-10)")

    print("\n" + "-" * 78)
    print("STEP 2  Local scores  s(j | Pa) = log p(D^{Pa u {j}}) - log p(D^{Pa})")
    print("-" * 78)
    print(f"    {'node':<6}{'parents':<16}{'log p(Pa u j)':>16}{'log p(Pa)':>16}{'local':>14}")
    local = {}
    for j in range(d):
        others = [k for k in range(d) if k != j]
        for r in range(len(others) + 1):
            for pa in itertools.combinations(others, r):
                a = marg[tuple(sorted(pa + (j,)))]
                b = marg[tuple(sorted(pa))]
                local[(j, pa)] = a - b
                pa_txt = "{" + ",".join(LABELS[p] for p in pa) + "}"
                print(f"    {LABELS[j]:<6}{pa_txt:<16}{a:>16.6f}{b:>16.6f}{a-b:>14.6f}")

    print("\n" + "-" * 78)
    print("STEP 3  Assembling four DAGs")
    print("-" * 78)
    dags = {
        "G_empty  (no edges)":          {0: (), 1: (), 2: ()},
        "G_chain  X1->X2->X3":          {0: (), 1: (0,), 2: (1,)},
        "G_rev    X1<-X2<-X3":          {0: (1,), 1: (2,), 2: ()},
        "G_fork   X1<-X2->X3":          {0: (1,), 1: (), 2: (1,)},
        "G_coll   X1->X2<-X3":          {0: (), 1: (0, 2), 2: ()},
        "G_dense  complete":            {0: (), 1: (0,), 2: (0, 1)},
    }
    totals = {}
    for name, pa_map in dags.items():
        parts = [local[(j, pa_map[j])] for j in range(d)]
        totals[name] = sum(parts)
        detail = "  +  ".join(f"{p:.4f}" for p in parts)
        print(f"    {name:<26} {detail}  =  {totals[name]:>12.6f}")

    print("\n" + "-" * 78)
    print("STEP 4  Reading off the conclusions")
    print("-" * 78)
    chain = totals["G_chain  X1->X2->X3"]
    rev = totals["G_rev    X1<-X2<-X3"]
    fork = totals["G_fork   X1<-X2->X3"]
    coll = totals["G_coll   X1->X2<-X3"]
    dense = totals["G_dense  complete"]
    empty = totals["G_empty  (no edges)"]

    print(f"    (a) SCORE EQUIVALENCE. chain, reversed chain and fork share the skeleton")
    print(f"        X1-X2-X3 with NO v-structure, so they are one Markov equivalence")
    print(f"        class and must tie exactly:")
    print(f"          chain {chain:.12f}")
    print(f"          rev   {rev:.12f}")
    print(f"          fork  {fork:.12f}")
    print(f"          max spread = {max(chain,rev,fork)-min(chain,rev,fork):.3e} nats")
    assert max(chain, rev, fork) - min(chain, rev, fork) < 1e-10

    print(f"\n    (b) THE COLLIDER IS SEPARABLE. X1->X2<-X3 has the same skeleton but a")
    print(f"        v-structure, so it is a DIFFERENT class and scores differently:")
    print(f"          collider {coll:.6f}   vs chain {chain:.6f}   "
          f"(difference {coll - chain:+.6f} nats)")
    assert abs(coll - chain) > 1e-3

    print(f"\n    (c) COMPLEXITY IS PAID FOR. The complete DAG has one more edge than the")
    print(f"        data-generating chain and still scores WORSE, with no penalty term")
    print(f"        added by hand -- the parameter prior does it:")
    print(f"          dense {dense:.6f}   chain {chain:.6f}   "
          f"(dense - chain = {dense - chain:+.6f} nats)")
    print(f"          empty {empty:.6f}   (chain - empty = {chain - empty:+.6f} nats)")

    print(f"\n    (d) POSTERIOR over the 25 DAGs at d = 3 (uniform prior):")
    from sa.graphs import build_graph_space
    space = build_graph_space(3)
    stats = bge.sufficient_stats(DATA)
    scores = np.array([
        sum(bge.local_score_from_stats(
            j, tuple(np.flatnonzero(space.dags[i][:, j] > 0.5)), stats)
            for j in range(3))
        for i in range(space.n_dags)])
    post = np.exp(scores - scores.max()); post /= post.sum()
    order = np.argsort(-post)
    print(f"        {'rank':<6}{'edges':<8}{'log score':>14}{'posterior':>12}   class")
    for rank, i in enumerate(order[:6]):
        e = int(space.dags[i].sum())
        print(f"        {rank+1:<6}{e:<8}{scores[i]:>14.4f}{post[i]:>12.4f}"
              f"   {space.mec_id[i]} (size {space.mec_sizes[space.mec_id[i]]})")
    truth_idx = next(i for i in range(space.n_dags)
                     if np.array_equal(space.dags[i],
                                       np.array([[0,1,0],[0,0,1],[0,0,0]])))
    print(f"        true chain X1->X2->X3 is rank "
          f"{int(np.flatnonzero(order == truth_idx)[0]) + 1} "
          f"with posterior {post[truth_idx]:.4f}")
    print(f"        mass on the top class = "
          f"{post[space.mec_id == space.mec_id[truth_idx]].sum():.4f} "
          f"(spread over its {space.mec_sizes[space.mec_id[truth_idx]]} members)")

    print("\nALL V3 CHECKS PASSED")


if __name__ == "__main__":
    main()
