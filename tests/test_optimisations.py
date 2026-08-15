"""The optimised hot paths must be exact restatements, not approximations.

Every test here pins a fast path against the slow one it replaced. The point is that a
future change to either side breaks a test rather than silently shifting every number in
the results directory -- these paths sit underneath every reported metric, so a quiet
divergence would be indistinguishable from a scientific finding.
"""
import itertools

import numpy as np
import pytest

from sa.graphs import build_graph_space
from sa.posterior import PosteriorEngine, edge_marginals
from sa.score import BGeScore, BICScore


def _data(n, d, seed=0, frac_intervened=0.2):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n, d))
    intervened = np.zeros((n, d))
    k = int(n * frac_intervened)
    for node in range(d):
        rows = rng.choice(n, size=k, replace=False)
        intervened[rows, node] = 1.0
    return samples, intervened


# --------------------------------------------------------------------------------------
# BGe sufficient statistics
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("n", [10, 300, 5000])
def test_stats_path_matches_naive_local_score(d, n):
    """local_score_from_stats == the original per-call recomputation, for every parent set."""
    score = BGeScore(d)
    samples, _ = _data(n, d, seed=d * 100 + n)
    stats = score.sufficient_stats(samples)

    for node in range(d):
        others = [k for k in range(d) if k != node]
        for r in range(len(others) + 1):
            for parents in itertools.combinations(others, r):
                fast = score.local_score_from_stats(node, parents, stats)
                # Reconstruct the pre-optimisation computation explicitly, so this test
                # does not merely compare the new code against itself.
                slow = _naive_local_score(score, node, parents, samples)
                assert fast == pytest.approx(slow, rel=1e-9, abs=1e-9)


def _naive_local_score(score, node, parents, samples):
    """The original implementation, inlined so the reference cannot drift."""
    def marginal(subset):
        p = len(subset)
        if p == 0 or samples.shape[0] == 0:
            return 0.0
        n = samples.shape[0]
        idx = np.asarray(subset, dtype=int)
        x = samples[:, idx]
        mean = x.mean(axis=0)
        centered = x - mean
        scatter = centered.T @ centered
        shrink = n * score.alpha_mu / (n + score.alpha_mu)
        R = score.T[np.ix_(idx, idx)] + scatter + shrink * np.outer(mean, mean)
        awp = score.alpha_w - score.d + p
        from sa.score import _log_multivariate_gamma
        _, logdet_T = np.linalg.slogdet(score.T[np.ix_(idx, idx)])
        _, logdet_R = np.linalg.slogdet(R)
        return (
            -n * p / 2.0 * np.log(np.pi)
            + p / 2.0 * np.log(score.alpha_mu / (n + score.alpha_mu))
            + _log_multivariate_gamma((n + awp) / 2.0, p)
            - _log_multivariate_gamma(awp / 2.0, p)
            + awp / 2.0 * logdet_T
            - (n + awp) / 2.0 * logdet_R
        )

    parents = sorted(int(p) for p in parents)
    return marginal(sorted(parents + [int(node)])) - marginal(parents)


def test_stats_handle_empty_sample_set():
    """A node intervened on in every row contributes nothing -- must not divide by zero."""
    score = BGeScore(4)
    stats = score.sufficient_stats(np.zeros((0, 4)))
    assert stats.n == 0
    assert score.local_score_from_stats(0, (1, 2), stats) == 0.0


# --------------------------------------------------------------------------------------
# Score table, gather and edge marginals
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [3, 4, 5])
def test_score_table_respects_intervention_masking(d):
    """The stats path must still drop rows where the node itself was intervened on.

    This is the one place the optimisation could plausibly go wrong: statistics are now
    built per node, so a bug would silently score every node on the full sample set --
    which is exactly the Cooper & Yoo rule being violated, and would look like unusually
    good performance rather than like a crash.
    """
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    samples, intervened = _data(400, d, seed=7, frac_intervened=0.5)

    table = engine.local_score_table(samples, intervened)
    for node in range(d):
        usable = intervened[:, node] < 0.5
        subset = samples[usable]
        for i, parents in enumerate(engine.parent_sets[node]):
            expected = _naive_local_score(engine.score, node, parents, subset)
            assert table[node, i] == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("d", [3, 4, 5])
def test_flat_gather_matches_broadcast_gather(d):
    """Bit-identical: the flat index is a pure reindexing, no arithmetic changes."""
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    rng = np.random.default_rng(1)
    table = rng.normal(size=(d, engine.n_parent_sets))

    rows = np.arange(d)[None, :]
    expected = table[rows, engine.parent_set_ids].sum(axis=1)
    actual = table.ravel()[engine._flat_ids].sum(axis=1)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("d", [3, 4, 5])
def test_engine_edge_marginals_match_module_function(d):
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    rng = np.random.default_rng(2)
    posterior = rng.random(space.n_dags)
    posterior /= posterior.sum()

    expected = edge_marginals(space, posterior)
    actual = engine.edge_marginals(posterior)
    assert np.allclose(actual, expected, atol=1e-12)
    # Diagonal must stay exactly zero: no DAG has a self-loop, so a non-zero here would
    # mean the parent-set membership table is misindexed.
    assert np.all(np.diag(actual) == 0.0)


@pytest.mark.parametrize("d", [3, 4])
def test_posterior_unchanged_end_to_end(d):
    """The number every downstream metric reads must be unchanged."""
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    samples, intervened = _data(500, d, seed=3)

    table = engine.local_score_table(samples, intervened)
    rows = np.arange(d)[None, :]
    reference = table[rows, engine.parent_set_ids].sum(axis=1)
    assert np.allclose(engine.log_scores(samples, intervened), reference, atol=1e-12)


def test_log_prior_cache_does_not_leak_between_priors():
    """The cache is keyed on array identity -- a different prior must not reuse it.

    Guarding the failure mode that would matter: two priors in one process (the
    reference runs and the agent runs share an engine in some scripts), where a stale
    cache would silently score every graph under the wrong prior.
    """
    d = 4
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    samples, intervened = _data(300, d, seed=11)

    uniform = np.full(space.n_dags, 1.0 / space.n_dags)
    skewed = np.linspace(1.0, 5.0, space.n_dags)
    skewed /= skewed.sum()

    first = engine.posterior(samples, intervened, uniform)
    second = engine.posterior(samples, intervened, skewed)
    third = engine.posterior(samples, intervened, uniform)

    assert not np.allclose(first, second)      # cache did not stick
    assert np.allclose(first, third)           # and returning is still consistent

    # And each matches an engine that has never seen another prior.
    fresh = PosteriorEngine(space, BGeScore(d))
    assert np.allclose(second, fresh.posterior(samples, intervened, skewed))


def test_non_stats_scorer_still_works():
    """BIC has no sufficient_stats; it must take the original path, not crash."""
    d = 4
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BICScore(d))
    samples, intervened = _data(200, d, seed=5)
    table = engine.local_score_table(samples, intervened)
    assert np.all(np.isfinite(table[:, : len(engine.parent_sets[0])]))
