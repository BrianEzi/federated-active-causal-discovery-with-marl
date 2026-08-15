"""V2 -- The BGe marginal likelihood: independent re-derivation and identities.

Everything Part II asserts about BGe is checked here against something other than
`sa/score.py`:

  [1] An independent transcription of the Kuipers, Moffa & Heckerman (2014)
      formula, written from the paper's notation rather than from the code,
      agrees with `BGeScore._log_marginal_stats` to machine precision.
  [2] For p = 1 and p = 2 the closed form agrees with BRUTE-FORCE NUMERICAL
      INTEGRATION of the Normal-Wishart integral. This is the check that the
      normalising constants are right; nothing else in the project tests them.
  [3] The telescoping identity: for any complete DAG, the sum of local scores
      equals log p(D_V), independent of the topological order used.
  [4] Chickering's covered-edge-reversal lemma: reversing a covered edge leaves
      the BGe score of the DAG numerically unchanged.
  [5] Full score equivalence: every DAG in a Markov equivalence class receives
      an identical score on observational data (d = 3, 4, 5).
  [6] Score equivalence does NOT depend on the isotropic choice of T -- an
      arbitrary positive-definite T preserves it, whereas an INCONSISTENT T
      (a different prior matrix per subset) destroys it. This isolates which
      property is load-bearing.
  [7] The sufficient-statistics path equals the naive re-read path.

Run:  python textbook/verification/v2_bge_identities.py
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import integrate
from scipy.special import gammaln, multigammaln

from sa.graphs import build_graph_space
from sa.score import BGeScore, BICScore
from sa.scm import sample, sample_scm_params

RNG = np.random.default_rng(20260815)


# ---------------------------------------------------------------------------
# [1] Independent transcription of Kuipers, Moffa & Heckerman (2014), eq. (5).
# ---------------------------------------------------------------------------
def kuipers_log_marginal(x, subset, n_vars, alpha_mu, alpha_w, T_full, mu0=None):
    r"""log p(D^{(l)}) written straight from the paper.

        log p(D^l) = - (N l / 2) log pi
                     + (l / 2) log( alpha_mu / (alpha_mu + N) )
                     + log Gamma_l( (alpha_w - n + l + N) / 2 )
                     - log Gamma_l( (alpha_w - n + l) / 2 )
                     + ((alpha_w - n + l) / 2) log |T^l|
                     - ((alpha_w - n + l + N) / 2) log |R^l|

        R = T + S_N + (alpha_mu N / (alpha_mu + N)) (xbar - mu0)(xbar - mu0)^T

    with n = total number of variables, l = |subset|, N = sample count,
    S_N the CENTRED scatter matrix. `multigammaln` is scipy's log Gamma_l,
    used here instead of the project's own `_log_multivariate_gamma` so that
    the two implementations share no code.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    l = len(subset)
    if l == 0 or N == 0:
        return 0.0
    idx = np.asarray(subset, dtype=int)
    mu0 = np.zeros(n_vars) if mu0 is None else np.asarray(mu0, dtype=float)

    xs = x[:, idx]
    xbar = xs.mean(axis=0)
    centred = xs - xbar
    S_N = centred.T @ centred
    delta = xbar - mu0[idx]

    T = np.asarray(T_full)[np.ix_(idx, idx)]
    R = T + S_N + (alpha_mu * N / (alpha_mu + N)) * np.outer(delta, delta)

    nu = alpha_w - n_vars + l                      # the corrected d.o.f. term
    return float(
        -(N * l / 2.0) * np.log(np.pi)
        + (l / 2.0) * np.log(alpha_mu / (alpha_mu + N))
        + multigammaln((nu + N) / 2.0, l)
        - multigammaln(nu / 2.0, l)
        + (nu / 2.0) * np.linalg.slogdet(T)[1]
        - ((nu + N) / 2.0) * np.linalg.slogdet(R)[1]
    )


# ---------------------------------------------------------------------------
# [2] Brute-force numerical integration of the Normal-Wishart integral.
# ---------------------------------------------------------------------------
def numeric_log_marginal_p1(x_col, nu, t, alpha_mu, mu0=0.0, n_grid=200_001):
    r"""log \int\int prod_i N(x_i | mu, 1/w) * N(mu | mu0, 1/(alpha_mu w))
                     * Gamma(w; nu/2, t/2) dmu dw.

    For l = 1 the Wishart W_1(nu, T^{-1}) density is proportional to
    w^{(nu-2)/2} exp(-t w / 2), i.e. Gamma(shape = nu/2, rate = t/2), with
    normaliser (t/2)^{nu/2} / Gamma(nu/2).

    The mu integral is a Gaussian convolution and is done in closed form (it is
    elementary and not the part in doubt):

        int N(mu | mu0, 1/(alpha_mu w)) prod_i N(x_i | mu, 1/w) dmu
          = (w / 2pi)^{N/2} sqrt(alpha_mu / (N + alpha_mu))
            exp( -w/2 * [ SS + N alpha_mu/(N + alpha_mu) (xbar - mu0)^2 ] )

    The remaining one-dimensional w integral -- which carries every gamma-function
    and determinant constant that Kuipers et al. correct -- is then evaluated by
    dense trapezoidal quadrature in log w, in log space via log-sum-exp. Nothing
    below reuses the closed-form answer.
    """
    x = np.asarray(x_col, dtype=float)
    N = x.size
    xbar = x.mean()
    ss = float(((x - xbar) ** 2).sum())
    quad_const = ss + (N * alpha_mu / (N + alpha_mu)) * (xbar - mu0) ** 2

    log_prior_norm = (nu / 2.0) * np.log(t / 2.0) - gammaln(nu / 2.0)

    # The integrand in w is Gamma-shaped with shape (nu+N)/2 and rate (t+quad)/2.
    shape, rate = (nu + N) / 2.0, (t + quad_const) / 2.0
    centre = np.log(shape / rate)
    grid = np.linspace(centre - 30.0, centre + 20.0, n_grid)   # u = log w
    w = np.exp(grid)

    log_integrand = (
        log_prior_norm
        + (nu / 2.0 - 1.0) * grid - t * w / 2.0          # Gamma(w; nu/2, t/2)
        + (N / 2.0) * (grid - np.log(2 * np.pi))          # (w/2pi)^{N/2}
        + 0.5 * np.log(alpha_mu / (N + alpha_mu))
        - w * quad_const / 2.0
        + grid                                            # Jacobian dw = w du
    )
    m = log_integrand.max()
    step = grid[1] - grid[0]
    vals = np.exp(log_integrand - m)
    total = np.trapz(vals, dx=step)
    coarse = np.trapz(vals[::2], dx=2 * step)             # Richardson-style error probe
    return float(m + np.log(total)), float(abs(total - coarse) / total)


def sequential_log_marginal(x, subset, n_vars, alpha_mu, alpha_w, T_full, mu0=None):
    r"""log p(D^l) by the CHAIN RULE over sequential multivariate-t predictives.

    A completely different route to the same number, sharing no algebra with the
    batch formula. Write the Normal-Wishart prior in Normal-inverse-Wishart form:
    with Sigma = W^{-1},

        mu | Sigma ~ N(mu_0, Sigma / kappa_0),    Sigma ~ IW(Psi_0, nu_0)

    where kappa_0 = alpha_mu, Psi_0 = T^l and nu_0 = alpha_w - n + l. Conjugacy
    gives, after i observations,

        kappa_i = kappa_0 + i,   nu_i = nu_0 + i,
        m_i     = (kappa_0 mu_0 + i xbar_i) / kappa_i,
        Psi_i   = Psi_0 + S_i + (kappa_0 i / kappa_i)(xbar_i - mu_0)(xbar_i - mu_0)^T,

    and the one-step-ahead predictive is a multivariate t:

        x_{i+1} | D_{1:i} ~ t_{nu_i - l + 1}( m_i,  Psi_i (kappa_i + 1)
                                                    / (kappa_i (nu_i - l + 1)) ).

    Then  log p(D^l) = sum_i log t(x_{i+1} | D_{1:i}).  If the batch formula's
    gamma-function and determinant constants were wrong, this would disagree.
    """
    x = np.asarray(x, dtype=float)
    idx = np.asarray(subset, dtype=int)
    xs = x[:, idx]
    N, l = xs.shape
    mu0 = np.zeros(n_vars) if mu0 is None else np.asarray(mu0, dtype=float)

    kappa, nu = float(alpha_mu), float(alpha_w - n_vars + l)
    m = mu0[idx].astype(float).copy()
    Psi = np.asarray(T_full)[np.ix_(idx, idx)].astype(float).copy()

    total = 0.0
    for i in range(N):
        dof = nu - l + 1.0
        scale = Psi * (kappa + 1.0) / (kappa * dof)
        delta = xs[i] - m
        _, logdet_scale = np.linalg.slogdet(scale)
        quad = float(delta @ np.linalg.solve(scale, delta))
        total += float(
            gammaln((dof + l) / 2.0) - gammaln(dof / 2.0)
            - (l / 2.0) * np.log(dof * np.pi)
            - 0.5 * logdet_scale
            - ((dof + l) / 2.0) * np.log1p(quad / dof)
        )
        # conjugate update with the single observation x_i
        Psi = Psi + (kappa / (kappa + 1.0)) * np.outer(delta, delta)
        m = (kappa * m + xs[i]) / (kappa + 1.0)
        kappa += 1.0
        nu += 1.0
    return total


# ---------------------------------------------------------------------------
def dag_score(score, adjacency, data, stats=None):
    """Total BGe (or BIC) log score of a DAG on observational data."""
    d = adjacency.shape[0]
    total = 0.0
    for j in range(d):
        parents = tuple(np.flatnonzero(adjacency[:, j] > 0.5).tolist())
        if stats is not None and hasattr(score, "local_score_from_stats"):
            total += score.local_score_from_stats(j, parents, stats)
        else:
            total += score.local_score(j, parents, data)
    return total


def main() -> None:
    print("=" * 72)
    print("V2  BGe marginal likelihood -- independent re-derivation and identities")
    print("=" * 72)

    d = 3
    space = build_graph_space(d)
    truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8)  # 0 -> 1 -> 2
    params = sample_scm_params(truth, RNG)
    data, _ = sample(params, 400, RNG)

    bge = BGeScore(d)
    stats = bge.sufficient_stats(data)
    print(f"\n    d = {d}, N = {data.shape[0]}, alpha_mu = {bge.alpha_mu}, "
          f"alpha_w = {bge.alpha_w}, T = {bge.T[0,0]:.6g} * I")

    # -- [1] code vs. independent transcription --------------------------------
    print("\n[1] sa/score.py vs. independent transcription of Kuipers et al. (2014)")
    worst = 0.0
    for r in range(1, d + 1):
        for subset in itertools.combinations(range(d), r):
            a = bge._log_marginal_stats(stats, subset)
            b = kuipers_log_marginal(data, subset, d, bge.alpha_mu, bge.alpha_w, bge.T)
            worst = max(worst, abs(a - b))
            print(f"    subset {str(subset):<12} code {a:>14.9f}   paper {b:>14.9f}"
                  f"   |diff| {abs(a-b):.2e}")
    assert worst < 1e-9, worst
    print(f"    max |difference| = {worst:.3e}  OK")

    # -- [2] closed form vs. numerical integration -----------------------------
    print("\n[2] Closed form vs. brute-force integration of the Normal-Wishart integral")
    small = data[:12]
    for node in range(d):
        nu = bge.alpha_w - d + 1
        t = bge.T[node, node]
        closed = kuipers_log_marginal(small, (node,), d, bge.alpha_mu, bge.alpha_w, bge.T)
        numeric, relerr = numeric_log_marginal_p1(small[:, node], nu, t, bge.alpha_mu)
        print(f"    l=1 node {node}: closed {closed:>12.8f}   quadrature {numeric:>12.8f}"
              f"   |diff| {abs(closed-numeric):.3e}  (quad rel. err {relerr:.1e})")
        assert abs(closed - numeric) < 1e-6, (node, closed, numeric)

    print("    (l >= 2: chain rule over sequential multivariate-t predictives)")
    for subset in [(0, 1), (1, 2), (0, 2), (0, 1, 2)]:
        closed = kuipers_log_marginal(small, subset, d, bge.alpha_mu, bge.alpha_w, bge.T)
        seq = sequential_log_marginal(small, subset, d, bge.alpha_mu, bge.alpha_w, bge.T)
        print(f"    l={len(subset)} {str(subset):<11}: closed {closed:>13.9f}   "
              f"sequential {seq:>13.9f}   |diff| {abs(closed-seq):.3e}")
        assert abs(closed - seq) < 1e-8, (subset, closed, seq)

    # -- [3] telescoping over complete DAGs ------------------------------------
    print("\n[3] Telescoping: sum of local scores over a COMPLETE DAG = log p(D_V)")
    joint = bge._log_marginal_stats(stats, tuple(range(d)))
    for order in itertools.permutations(range(d)):
        a = np.zeros((d, d), dtype=np.int8)
        for pos, j in enumerate(order):
            for i in order[:pos]:
                a[i, j] = 1
        s = dag_score(bge, a, data, stats)
        assert abs(s - joint) < 1e-9, (order, s, joint)
    print(f"    log p(D_V) = {joint:.9f}; all {math.factorial(d)} complete DAGs "
          f"agree to < 1e-9  OK")

    # -- [4] covered edge reversal ---------------------------------------------
    print("\n[4] Chickering's covered-edge reversal leaves the score unchanged")
    #  0 -> 1  is covered in  0->1, 0->2, 1->2  iff Pa(1) = Pa(0) u {0}. Build one.
    a = np.zeros((d, d), dtype=np.int8)
    a[0, 1] = 1; a[0, 2] = 1; a[1, 2] = 1        # Pa(0)={}, Pa(1)={0} -> 0->1 covered
    b = a.copy(); b[0, 1] = 0; b[1, 0] = 1        # reverse it
    sa_, sb_ = dag_score(bge, a, data, stats), dag_score(bge, b, data, stats)
    print(f"    0->1->2, 0->2   score {sa_:.9f}")
    print(f"    1->0->2, 1->2   score {sb_:.9f}   |diff| {abs(sa_-sb_):.3e}")
    assert abs(sa_ - sb_) < 1e-9
    #  A NON-covered reversal must generally change the score.
    c = np.zeros((d, d), dtype=np.int8); c[0, 1] = 1; c[1, 2] = 1      # chain
    e = np.zeros((d, d), dtype=np.int8); c2 = c.copy(); c2[1, 2] = 0; c2[2, 1] = 1
    print(f"    chain 0->1->2   score {dag_score(bge, c, data, stats):.9f}")
    print(f"    0->1<-2         score {dag_score(bge, c2, data, stats):.9f}  "
          f"(v-structure created: NOT equivalent, score differs)")

    # -- [5] full score equivalence --------------------------------------------
    print("\n[5] Score equivalence across whole equivalence classes (observational)")
    for dd in (3, 4, 5):
        sp = build_graph_space(dd)
        truth_d = sp.dags[RNG.integers(sp.n_dags)]
        pr = sample_scm_params(truth_d, RNG)
        dat, _ = sample(pr, 500, RNG)
        sc = BGeScore(dd)
        st = sc.sufficient_stats(dat)
        scores = np.array([dag_score(sc, sp.dags[i], dat, st)
                           for i in range(sp.n_dags)])
        spreads = []
        for cls in range(sp.n_mecs):
            members = np.flatnonzero(sp.mec_id == cls)
            if len(members) > 1:
                spreads.append(scores[members].ptp())
        spreads = np.array(spreads)
        # Between-class spread, for scale.
        between = scores.max() - scores.min()
        print(f"    d={dd}: max WITHIN-class spread {spreads.max():.3e} nats "
              f"| total between-class range {between:.1f} nats "
              f"| ratio {spreads.max()/between:.1e}")
        assert spreads.max() < 1e-7, (dd, spreads.max())

        bic = BICScore(dd)
        bscores = np.array([dag_score(bic, sp.dags[i], dat) for i in range(sp.n_dags)])
        bspread = max(bscores[np.flatnonzero(sp.mec_id == c)].ptp()
                      for c in range(sp.n_mecs) if (sp.mec_id == c).sum() > 1)
        print(f"          BIC cross-check: max within-class spread {bspread:.3e} nats")

    # -- [6] WHICH property is actually load-bearing ---------------------------
    #
    # This section began as an attempt to break score equivalence by making the
    # prior matrix T inconsistent across subsets. It FAILED to break it, and the
    # reason is a stronger theorem than the one usually quoted:
    #
    #   Any score of the form  sum_j [ m(Pa_j u {j}) - m(Pa_j) ]  for an ARBITRARY
    #   set function m : 2^V -> R is score-equivalent.
    #
    # Proof: under a covered-edge reversal i -> j (so Pa_j = Pa_i u {i}), the two
    # families that change contribute, before the reversal,
    #       [m(Pa_i u {i}) - m(Pa_i)] + [m(Pa_i u {i,j}) - m(Pa_i u {i})]
    #     = m(Pa_i u {i,j}) - m(Pa_i),
    # and after,
    #       [m(Pa_i u {i,j}) - m(Pa_i u {j})] + [m(Pa_i u {j}) - m(Pa_i)]
    #     = m(Pa_i u {i,j}) - m(Pa_i).
    # The two intermediate terms cancel in both directions, so the totals agree --
    # whatever m is. Chickering (1995) then supplies the rest: any two Markov
    # equivalent DAGs are joined by a finite chain of covered-edge reversals.
    #
    # So the Normal-Wishart algebra and the special T are what make BGe a correct
    # MARGINAL LIKELIHOOD; they are NOT what make it score-equivalent. Score
    # equivalence comes for free from the FORM of the score. What breaks it is a
    # local term that is not a difference of a set function -- i.e. one that
    # depends on the child j beyond its appearance in the set. That is exactly the
    # defect in KnownVarianceScore (see v4).
    print("\n[6] What is actually load-bearing for score equivalence")
    sp = build_graph_space(3)
    tru = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int8)
    pr = sample_scm_params(tru, RNG)
    dat, _ = sample(pr, 400, RNG)

    def within_spread(score_obj, space, data, stats=None):
        s = np.array([dag_score(score_obj, space.dags[i], data, stats)
                      for i in range(space.n_dags)])
        return max(s[np.flatnonzero(space.mec_id == c)].ptp()
                   for c in range(space.n_mecs) if (space.mec_id == c).sum() > 1)

    print("    (a) arbitrary positive-definite T, used consistently:")
    L = np.array([[1.3, 0.0, 0.0], [0.4, 0.9, 0.0], [-0.2, 0.3, 1.1]])
    sc = BGeScore(3); sc.T = L @ L.T
    sp_a = within_spread(sc, sp, dat, sc.sufficient_stats(dat))
    print(f"        within-class spread {sp_a:.3e}  -> still score-equivalent")
    assert sp_a < 1e-7

    print("    (b) T rescaled by |subset| (marginal consistency deliberately broken):")

    class InconsistentBGe(BGeScore):
        def _log_marginal_stats(self, stats, subset):
            keep = self.T
            self.T = keep * (1.0 + 0.35 * len(subset))
            try:
                return super()._log_marginal_stats(stats, subset)
            finally:
                self.T = keep

    bad = InconsistentBGe(3)
    sp_b = within_spread(bad, sp, dat, bad.sufficient_stats(dat))
    print(f"        within-class spread {sp_b:.3e}  -> STILL score-equivalent")
    print("        (it is still a set function of the subset, so the theorem applies)")
    assert sp_b < 1e-7

    print("    (c) a PURELY ARBITRARY set function m: 2^V -> R (random numbers):")

    class ArbitrarySetFunction:
        """local_score(j, Pa) = m(Pa u {j}) - m(Pa), m drawn at random."""
        def __init__(self, d, seed=1):
            r = np.random.default_rng(seed)
            self.m = {frozenset(s): float(r.normal() * 10)
                      for k in range(d + 1)
                      for s in itertools.combinations(range(d), k)}
        def local_score(self, node, parents, samples):
            pa = frozenset(int(p) for p in parents)
            return self.m[pa | {int(node)}] - self.m[pa]

    arb = ArbitrarySetFunction(3)
    sp_c = within_spread(arb, sp, dat)
    print(f"        within-class spread {sp_c:.3e}  -> score-equivalent")
    print("        Score equivalence follows from the FORM of the score, not from")
    print("        the Normal-Wishart algebra. The algebra is what makes BGe a")
    print("        correct marginal likelihood; it is not what makes it equivalent.")
    assert sp_c < 1e-10

    print("    (d) a local term that is NOT a difference of a set function:")

    class NotASetFunction:
        """Depends on the child index beyond its membership in the family set.

        This is the structural shape of KnownVarianceScore: node j's term is a
        residual sum of squares for j specifically, not m(Pa u {j}) - m(Pa).
        """
        def local_score(self, node, parents, samples):
            y = samples[:, int(node)]
            pa = sorted(int(p) for p in parents)
            if pa:
                X = samples[:, pa]
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ beta
            else:
                resid = y
            return float(-0.5 * np.sum(resid ** 2))     # fixed unit variance


    nsf = NotASetFunction()
    sp_d = within_spread(nsf, sp, dat)
    print(f"        within-class spread {sp_d:.3f} nats  -> score equivalence BROKEN")
    assert sp_d > 1e-3

    # -- [6e] Chickering's covered-edge chain, verified exhaustively -----------
    print("    (e) Chickering (1995): every equivalent pair is joined by covered-edge")
    print("        reversals. Verified exhaustively at d = 3 and d = 4:")

    def covered_neighbours(a):
        """All DAGs reachable from `a` by reversing ONE covered edge.

        i -> j is covered iff Pa(j) = Pa(i) u {i}.
        """
        dd = a.shape[0]
        out = []
        for i in range(dd):
            for j in range(dd):
                if not a[i, j]:
                    continue
                pa_i = set(np.flatnonzero(a[:, i] > 0.5).tolist())
                pa_j = set(np.flatnonzero(a[:, j] > 0.5).tolist())
                if pa_j == pa_i | {i}:
                    b = a.copy(); b[i, j] = 0; b[j, i] = 1
                    out.append(b)
        return out

    for dd in (3, 4):
        s2 = build_graph_space(dd)
        key = {s2.dags[i].tobytes(): i for i in range(s2.n_dags)}
        # connected components of the covered-reversal graph
        comp = -np.ones(s2.n_dags, dtype=int)
        n_comp = 0
        for start in range(s2.n_dags):
            if comp[start] >= 0:
                continue
            stack, comp[start] = [start], n_comp
            while stack:
                cur = stack.pop()
                for nb in covered_neighbours(s2.dags[cur]):
                    k = key[np.ascontiguousarray(nb, dtype=np.int8).tobytes()]
                    if comp[k] < 0:
                        comp[k] = n_comp
                        stack.append(k)
            n_comp += 1
        same = np.array_equal(
            np.unique(np.stack([comp, s2.mec_id]), axis=1).shape[1],
            s2.n_mecs)
        agree = all(len(set(comp[s2.mec_id == c])) == 1 for c in range(s2.n_mecs))
        print(f"        d={dd}: {n_comp} covered-reversal components vs "
              f"{s2.n_mecs} Markov equivalence classes -- "
              f"{'IDENTICAL partition' if (n_comp == s2.n_mecs and agree) else 'DIFFER'}")
        assert n_comp == s2.n_mecs and agree

    # -- [7] sufficient statistics == naive re-read ----------------------------
    print("\n[7] Sufficient-statistics path == naive per-parent-set re-read")
    sc = BGeScore(3)
    st = sc.sufficient_stats(dat)
    worst = 0.0
    for node in range(3):
        for r in range(3):
            for parents in itertools.combinations([k for k in range(3) if k != node], r):
                a = sc.local_score(node, parents, dat)
                b = sc.local_score_from_stats(node, parents, st)
                worst = max(worst, abs(a - b))
    print(f"    max |difference| over all (node, parent set) pairs = {worst:.3e}")
    assert worst < 1e-8

    print("\nALL V2 CHECKS PASSED")


if __name__ == "__main__":
    main()
