"""The exact sampler must agree with the DP it is meant to replace sampling for.

Two independent things are checked, and they matter for different reasons:

  log Z    the layered source-decomposition and the DP's signed sink recurrence share no
           code path, so agreement is real evidence that both are right. This is the
           strongest check available -- it compares a closed-form total against a closed-form
           total, with no Monte Carlo noise in between.

  marginals draws must reproduce the DP's exact edge marginals to Monte Carlo tolerance, and
           the error must FALL with more draws. A biased sampler agrees on log Z and still
           produces the wrong marginals, so the second check is not implied by the first.
"""
from __future__ import annotations

import numpy as np
import pytest

from sa.dag_samplers import LayeredExactSampler
from sa.dp import DPPosterior
from sa.score import BGeScore


def problem(d: int, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    for j in range(1, d):
        x[:, j] += 0.9 * x[:, j - 1]
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    return dp, dp.log_weights(x, np.zeros((n, d), dtype=bool))


@pytest.mark.parametrize("d", [3, 4, 5, 6])
def test_layered_partition_function_matches_the_dp(d):
    """No shared code path, so this is a genuine cross-check rather than a tautology."""
    dp, log_w = problem(d)
    sampler = LayeredExactSampler(dp, log_w)
    assert abs(sampler.log_partition() - float(dp.log_partition(log_w))) < 1e-9


@pytest.mark.parametrize("d", [3, 4])
def test_draws_reproduce_exact_edge_marginals(d):
    dp, log_w = problem(d)
    exact = dp.edge_marginals_onepass(log_w)
    draws = LayeredExactSampler(dp, log_w).sample(4000, rng=np.random.default_rng(0))
    assert np.abs(draws.mean(axis=0) - exact).max() < 0.03


def test_error_falls_with_more_draws():
    """Independent draws, so error must shrink. A sampler stuck in a mode would flatten --
    which is exactly how the partition-MCMC bug was caught."""
    dp, log_w = problem(5)
    exact = dp.edge_marginals_onepass(log_w)
    sampler = LayeredExactSampler(dp, log_w)
    errors = []
    for n in (250, 8000):
        draws = sampler.sample(n, rng=np.random.default_rng(1))
        errors.append(float(np.abs(draws.mean(axis=0) - exact).max()))
    assert errors[-1] < errors[0]


@pytest.mark.parametrize("d", [3, 4, 5])
def test_every_draw_is_acyclic(d):
    """Acyclicity is structural here -- the layer construction cannot emit a cycle -- so a
    failure would mean the decomposition itself is wrong, not that a check was missed."""
    dp, log_w = problem(d)
    draws = LayeredExactSampler(dp, log_w).sample(400, rng=np.random.default_rng(2))
    for adjacency in draws:
        reach = adjacency.astype(bool).copy()
        for m in range(d):
            reach |= reach[:, [m]] & reach[[m], :]
        assert not np.any(np.diag(reach))


def test_caching_does_not_change_the_draws():
    """The per-state distribution caches are a pure speedup (26s -> 1.8s at d=7). Same seed
    must give the same draws whether or not the cache is warm."""
    dp, log_w = problem(4)
    sampler = LayeredExactSampler(dp, log_w)
    first = sampler.sample(200, rng=np.random.default_rng(3))
    second = sampler.sample(200, rng=np.random.default_rng(3))
    assert np.array_equal(first, second)
