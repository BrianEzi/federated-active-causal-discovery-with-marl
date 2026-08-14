"""Tests for the exact posterior.

Two things matter most here. The caching must not change any answer -- it is a pure
speed optimisation, so a cached score must equal the naive one exactly. And the
observational posterior must be flat within a Markov equivalence class, because that
tie is what interventions exist to break; if it is ever broken without intervening,
information is leaking.
"""
import numpy as np
import pytest

from sa.graphs import build_graph_space
from sa.posterior import (
    PosteriorEngine,
    edge_marginals,
    is_identified,
    mec_posterior,
)
from sa.score import BGeScore, get_score


def chain_data(n: int, d: int = 3, seed: int = 0):
    """0 -> 1 -> 2 with per-node noise scales. Returns (samples, intervened)."""
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.2, 1.0, d)
    x = np.zeros((n, d))
    x[:, 0] = rng.normal(0, scales[0], n)
    x[:, 1] = 1.5 * x[:, 0] + rng.normal(0, scales[1], n)
    x[:, 2] = -1.2 * x[:, 1] + rng.normal(0, scales[2], n)
    return x, np.zeros((n, d))


@pytest.fixture(scope="module")
def engine3():
    space = build_graph_space(3)
    return space, PosteriorEngine(space, BGeScore(3))


# --- correctness of the caching --------------------------------------------------

def test_cached_scores_match_the_naive_computation(engine3):
    space, engine = engine3
    samples, intervened = chain_data(200)
    fast = engine.log_scores(samples, intervened)
    naive = np.array([
        sum(engine.score.local_score(j, np.flatnonzero(dag[:, j] > 0.5), samples)
            for j in range(space.d))
        for dag in space.dags
    ])
    np.testing.assert_allclose(fast, naive, rtol=0, atol=1e-9)


def test_local_score_table_has_the_expected_size(engine3):
    """d * 2^(d-1) distinct terms -- 12 at d=3 -- not one per DAG."""
    space, engine = engine3
    assert sum(len(s) for s in engine.parent_sets) == space.d * 2 ** (space.d - 1)


# --- the property interventions exist to break -----------------------------------

def test_observational_posterior_is_flat_within_each_class(engine3):
    space, engine = engine3
    samples, intervened = chain_data(300)
    post = engine.posterior(samples, intervened)
    for mec in range(space.n_mecs):
        members = np.flatnonzero(space.mec_id == mec)
        if len(members) < 2:
            continue
        spread = float(post[members].max() - post[members].min())
        assert spread < 1e-9, (
            f"class {mec} is not flat observationally (spread {spread:.2e}) -- "
            f"orientation information is leaking without any intervention."
        )


def test_intervening_breaks_the_tie_within_a_class(engine3):
    """The whole point of the project: an intervention must separate DAGs that
    observational data cannot."""
    space, engine = engine3
    rng = np.random.default_rng(3)
    n = 400

    # Truth is 0 -> 1 -> 2. Intervene on node 1 in half the samples: under do(X1), node 1
    # no longer depends on node 0, but node 2 still follows node 1.
    scales = rng.uniform(0.2, 1.0, 3)
    x = np.zeros((n, 3))
    intervened = np.zeros((n, 3))
    intervened[n // 2:, 1] = 1.0
    x[:, 0] = rng.normal(0, scales[0], n)
    x[:, 1] = np.where(intervened[:, 1] > 0.5,
                       rng.normal(0, 1.0, n),
                       1.5 * x[:, 0] + rng.normal(0, scales[1], n))
    x[:, 2] = -1.2 * x[:, 1] + rng.normal(0, scales[2], n)

    post = engine.posterior(x, intervened)
    truth = np.zeros((3, 3), dtype=np.int8); truth[0, 1] = 1; truth[1, 2] = 1
    t_idx = next(i for i in range(space.n_dags) if np.array_equal(space.dags[i], truth))
    members = space.mec_members(t_idx)
    assert len(members) > 1, "test needs a non-singleton class to be meaningful"

    spread = float(post[members].max() - post[members].min())
    assert spread > 1e-6, "intervening did not separate Markov-equivalent DAGs"
    assert post[t_idx] == pytest.approx(post[members].max()), (
        "the true DAG is not the best-supported member of its own class after intervening"
    )


# --- basic sanity ----------------------------------------------------------------

def test_posterior_is_a_distribution(engine3):
    space, engine = engine3
    samples, intervened = chain_data(150)
    post = engine.posterior(samples, intervened)
    assert post.shape == (space.n_dags,)
    assert post.min() >= 0.0
    assert float(post.sum()) == pytest.approx(1.0)


def test_posterior_is_uniform_with_no_data(engine3):
    space, engine = engine3
    post = engine.posterior(np.zeros((0, 3)), np.zeros((0, 3)))
    np.testing.assert_allclose(post, 1.0 / space.n_dags)


# --- derived representations -----------------------------------------------------

def test_edge_marginals_shape_and_range(engine3):
    space, engine = engine3
    samples, intervened = chain_data(200)
    marg = edge_marginals(space, engine.posterior(samples, intervened))
    assert marg.shape == (space.d, space.d)
    assert marg.min() >= -1e-12 and marg.max() <= 1 + 1e-12
    assert np.allclose(np.diag(marg), 0.0), "self-edges must be impossible"


def test_edge_marginals_of_a_point_mass_recover_that_graph(engine3):
    space, _ = engine3
    point = np.zeros(space.n_dags); point[7] = 1.0
    np.testing.assert_allclose(edge_marginals(space, point), space.dags[7])


def test_mec_posterior_sums_to_one_and_groups_correctly(engine3):
    space, engine = engine3
    samples, intervened = chain_data(200)
    post = engine.posterior(samples, intervened)
    mecs = mec_posterior(space, post)
    assert mecs.shape == (space.n_mecs,)
    assert float(mecs.sum()) == pytest.approx(1.0)
    assert float(mecs[space.mec_id[3]]) >= float(post[3]) - 1e-12


# --- the identification criterion ------------------------------------------------

def test_identification_requires_breaking_the_tie(engine3):
    """A DAG in a class of size k caps at 1/k until an intervention separates it, so a
    0.9 threshold is unreachable observationally for non-singletons. This is the
    property that makes the criterion meaningful rather than an arbitrary cutoff."""
    space, engine = engine3
    samples, intervened = chain_data(2000)
    post = engine.posterior(samples, intervened)
    for i in range(space.n_dags):
        if len(space.mec_members(i)) > 1:
            assert not is_identified(post, i), (
                f"DAG {i} declared identified from observational data despite sharing a "
                f"class with {len(space.mec_members(i)) - 1} others."
            )


def test_identification_threshold_is_respected():
    post = np.array([0.95, 0.05])
    assert is_identified(post, 0)
    assert not is_identified(post, 1)
    assert not is_identified(np.array([0.89, 0.11]), 0)


def test_engine_works_with_either_score():
    space = build_graph_space(3)
    samples, intervened = chain_data(200)
    for name in ("bge", "bic"):
        engine = PosteriorEngine(space, get_score(name, 3))
        post = engine.posterior(samples, intervened)
        assert float(post.sum()) == pytest.approx(1.0)
