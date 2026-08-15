"""Structure scores for linear Gaussian DAGs.

Why this module exists at all, and why it is not the previous codebase's scorer:

The two-agent estimator used a *profile* likelihood -- fit each node's regression by OLS,
plug the fit back in, evaluate. That was valid there only because all 8 candidate graphs
had exactly 3 edges, so they were equally complex and no penalty was needed. That
assumption dies the moment all DAGs are allowed: a graph with more parents always fits
better because it has more free parameters. Measured at d=3 over all 25 DAGs, the six
densest tie at the top holding 67% of the posterior mass while the true 2-edge graph
ranks 9th. An unpenalised score does not merely blur the answer, it inverts it.

Two scores are provided, both **score-equivalent** on observational data -- meaning every
DAG in a Markov equivalence class receives an identical score. That property is not a
nicety: it is the formal statement that observational data cannot distinguish within a
class (Chickering 2002), and it is what makes interventions necessary. A scorer that
violates it is leaking information, which is exactly the bug that sank the previous
round (see docs/THEORY_NOTES.md #1/#2).

  bge -- the marginal likelihood under a Normal-Wishart prior (Geiger & Heckerman 2002;
         corrected formulation in Kuipers, Moffa & Heckerman 2014). Fully Bayesian:
         parameters are integrated out rather than fitted, so complexity is handled by
         the maths rather than by a penalty term bolted on. This is the default.

  bic  -- maximised log-likelihood minus (k/2)·log n (Schwarz 1978). An asymptotic
         approximation to the above. Kept because it is simple enough to verify by
         inspection, which makes it a useful independent check on `bge`.

Both are decomposable: the score of a DAG is the sum over nodes of a local term
depending only on that node and its parents. That is what makes exact enumeration cheap
-- see `sa/posterior.py`.

Interventional data (hard interventions): a sample in which node j was itself set by
intervention says nothing about j's own parents, because the intervention replaced j's
structural equation. Those samples are dropped from j's local term but retained as parent
values for every other node's terms (Cooper & Yoo 1999). Note this is precisely why score
equivalence holds only on *observational* data -- once different nodes are scored on
different sample subsets, equivalence classes separate, which is the entire point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.special import gammaln


@dataclass(frozen=True)
class GaussianStats:
    """Sufficient statistics for the BGe marginal likelihood.

    BGe depends on the data ONLY through the sample count, the column means and the
    centred scatter matrix -- everything else in `_log_marginal` is a function of those
    three. Crucially the statistics for a *subset* of columns are plain submatrices of
    the full-column versions (a column's mean does not depend on which other columns are
    selected, and scatter[a, b] = sum_i c_ia c_ib regardless of the subset). So the whole
    table of local scores can be built from one O(n d^2) pass instead of re-reading all n
    rows once per (node, parent-set) pair -- of which there are d * 2^(d-1), i.e. 160
    re-reads at d=5.

    This is an exact restatement, not an approximation: the only difference from the
    naive path is BLAS summation order, which shows up at ~1e-12 relative.
    """

    n: int
    mean: np.ndarray     # [d]
    scatter: np.ndarray  # [d, d], centred


def _log_multivariate_gamma(a: float, p: int) -> float:
    """log of the multivariate gamma function Gamma_p(a)."""
    if p == 0:
        return 0.0
    i = np.arange(1, p + 1)
    return (p * (p - 1) / 4.0) * np.log(np.pi) + np.sum(gammaln(a + (1.0 - i) / 2.0))


class BGeScore:
    """Marginal likelihood of a linear Gaussian DAG under a Normal-Wishart prior.

    Parameters follow the standard defaults: prior mean zero, `alpha_mu = 1`, and
    `alpha_w = d + 2` (the smallest value giving a proper prior). The prior scale matrix
    is `t * I` with `t = alpha_mu * (alpha_w - d - 1) / (alpha_mu + 1)`, the choice that
    makes the prior marginally consistent across subsets -- which is what guarantees
    score equivalence.
    """

    def __init__(self, d: int, alpha_mu: float = 1.0, alpha_w: float = None):
        self.d = d
        self.alpha_mu = float(alpha_mu)
        # Must exceed d + 1 for the Wishart prior to be proper.
        self.alpha_w = float(d + 2) if alpha_w is None else float(alpha_w)
        if self.alpha_w <= d + 1:
            raise ValueError(f"alpha_w must be > d + 1 = {d + 1}, got {self.alpha_w}")
        t = self.alpha_mu * (self.alpha_w - d - 1) / (self.alpha_mu + 1)
        self.T = t * np.eye(d)

    def sufficient_stats(self, samples: np.ndarray) -> GaussianStats:
        """One O(n d^2) pass producing everything `local_score_from_stats` needs."""
        x = np.asarray(samples, dtype=float)
        n = x.shape[0]
        if n == 0:
            return GaussianStats(0, np.zeros(self.d), np.zeros((self.d, self.d)))
        mean = x.mean(axis=0)
        centered = x - mean
        return GaussianStats(n, mean, centered.T @ centered)

    def _log_marginal(self, samples: np.ndarray, subset: Sequence[int]) -> float:
        """log p(data restricted to `subset`) under the joint Normal-Wishart model.

        The local score for a node is the difference of two of these (with and without
        the node added to its parent set), which is the standard way BGe is assembled
        and is what makes it decompose.
        """
        return self._log_marginal_stats(self.sufficient_stats(samples), subset)

    def _log_marginal_stats(self, stats: GaussianStats, subset: Sequence[int]) -> float:
        p = len(subset)
        if p == 0:
            return 0.0
        n = stats.n
        if n == 0:
            return 0.0

        idx = np.asarray(subset, dtype=int)
        mean = stats.mean[idx]
        scatter = stats.scatter[np.ix_(idx, idx)]

        # Prior mean is zero, so the mean-shift correction uses `mean` directly.
        shrink = n * self.alpha_mu / (n + self.alpha_mu)
        R = self.T[np.ix_(idx, idx)] + scatter + shrink * np.outer(mean, mean)

        awp = self.alpha_w - self.d + p
        sign_T, logdet_T = np.linalg.slogdet(self.T[np.ix_(idx, idx)])
        sign_R, logdet_R = np.linalg.slogdet(R)
        if sign_T <= 0 or sign_R <= 0:
            raise np.linalg.LinAlgError("non-positive-definite matrix in BGe score")

        return (
            -n * p / 2.0 * np.log(np.pi)
            + p / 2.0 * np.log(self.alpha_mu / (n + self.alpha_mu))
            + _log_multivariate_gamma((n + awp) / 2.0, p)
            - _log_multivariate_gamma(awp / 2.0, p)
            + awp / 2.0 * logdet_T
            - (n + awp) / 2.0 * logdet_R
        )

    def local_score(self, node: int, parents: Sequence[int], samples: np.ndarray) -> float:
        """log p(node's data | parents' data). `samples` must already exclude rows where
        `node` was itself intervened on."""
        return self.local_score_from_stats(node, parents, self.sufficient_stats(samples))

    def local_score_from_stats(self, node: int, parents: Sequence[int],
                               stats: GaussianStats) -> float:
        """As `local_score`, but reusing statistics computed once for the whole table.

        `stats` must have been built from the same already-filtered sample subset that
        `local_score` would have received -- i.e. rows where `node` was intervened on are
        the caller's responsibility to remove, exactly as before.
        """
        parents = sorted(int(p) for p in parents)
        with_node = sorted(parents + [int(node)])
        return (self._log_marginal_stats(stats, with_node)
                - self._log_marginal_stats(stats, parents))


class BICScore:
    """Maximised Gaussian log-likelihood minus (k/2)·log n.

    Score-equivalent for linear Gaussian models: equivalent DAGs share a skeleton, hence
    an identical parameter count, and attain the same maximised likelihood. Provided as
    an independently-verifiable cross-check on `BGeScore` rather than as the default --
    it is an asymptotic approximation and is less trustworthy at the small sample sizes
    an episode begins with.
    """

    def __init__(self, d: int, ridge: float = 1e-8):
        self.d = d
        self.ridge = ridge

    def local_score(self, node: int, parents: Sequence[int], samples: np.ndarray) -> float:
        parents = sorted(int(p) for p in parents)
        n = samples.shape[0]
        if n == 0:
            return 0.0
        # Centre both sides so the fit implicitly includes an intercept. Without this the
        # parents branch fits through the origin while the no-parents branch subtracts a
        # mean, and the inconsistency breaks score equivalence -- measured at 7e-2 spread
        # within a class, versus ~1e-14 once centred.
        y = samples[:, int(node)]
        y = y - y.mean()
        if parents:
            X = samples[:, parents]
            X = X - X.mean(axis=0)
            gram = X.T @ X + self.ridge * np.eye(len(parents))
            beta = np.linalg.solve(gram, X.T @ y)
            residual = y - X @ beta
        else:
            residual = y
        variance = max(float(np.mean(residual ** 2)), 1e-12)
        log_lik = -0.5 * n * (np.log(2 * np.pi * variance) + 1.0)
        # Parameters for this node: one coefficient per parent, an intercept, a variance.
        k = len(parents) + 2
        return log_lik - 0.5 * k * np.log(n)


class KnownVarianceScore:
    """DEFECTIVE BY DESIGN -- reproduces the previous codebase's scorer. Do not use for
    experiments; it exists so the gates can be tested against a known-bad estimator.

    This is the profile likelihood with the environment's *true, shared* error variance
    plugged in rather than fitted. That single choice is what invalidated the previous
    round of results. Supplying a known variance breaks score equivalence, and when all
    nodes share one noise scale the DAG becomes fully identifiable from observational
    data (Peters & Buehlmann 2014) -- so the estimator recovers the graph before any
    intervention is taken and the active-discovery task collapses. Measured on the old
    environment: 98% observational recovery, and roughly half of all episodes solved with
    the agent doing nothing.

    Worth stating plainly, because it is easy to get backwards: the leak lives in the
    *estimator*, not in the data. `BGeScore` is score-equivalent by construction, so it
    cannot separate Markov-equivalent DAGs however the noise is distributed -- feeding it
    equal-variance data changes nothing. Per-node noise scales in `sa/scm.py` are
    therefore defence in depth rather than the load-bearing fix, and they matter most if
    a non-score-equivalent estimator (a learned one, say) is ever swapped in.
    """

    def __init__(self, d: int, noise_scale: float = 1.0, ridge: float = 1e-6):
        self.d = d
        self.noise_scale = noise_scale
        self.ridge = ridge

    def local_score(self, node: int, parents: Sequence[int], samples: np.ndarray) -> float:
        parents = sorted(int(p) for p in parents)
        n = samples.shape[0]
        if n == 0:
            return 0.0
        y = samples[:, int(node)]
        if parents:
            X = samples[:, parents]
            gram = X.T @ X + self.ridge * np.eye(len(parents))
            residual = y - X @ np.linalg.solve(gram, X.T @ y)
        else:
            residual = y
        var = max(self.noise_scale ** 2, 1e-8)
        return float(-0.5 * np.sum(residual ** 2) / var - 0.5 * n * np.log(2 * np.pi * var))


def get_score(name, d: int):
    """Factory. Accepts a name, or any object already exposing `local_score`, so tests
    and diagnostics can inject a custom scorer without going through the registry."""
    if hasattr(name, "local_score"):
        return name
    if name == "bge":
        return BGeScore(d)
    if name == "bic":
        return BICScore(d)
    if name == "known_variance":
        return KnownVarianceScore(d)
    raise ValueError(
        f"unknown score {name!r}; expected 'bge', 'bic', 'known_variance', or a scorer object"
    )
