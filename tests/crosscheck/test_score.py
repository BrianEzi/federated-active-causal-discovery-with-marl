"""Tests for the structure scores.

The headline test is `test_score_is_equivalent_within_markov_equivalence_classes`. Score
equivalence is the formal statement that observational data cannot distinguish DAGs
inside a class -- the property that makes interventions necessary and therefore makes
this project's task exist. A scorer that violates it leaks orientation information, which
is exactly the failure that invalidated the previous round of results.

It has already earned its keep: it caught a real bug in `BICScore`, which centred `y` in
the no-parents branch but fitted without an intercept in the parents branch (within-class
spread 7e-2, versus ~1e-13 once fixed).
"""
import numpy as np
import pytest

from ma.graphs import build_graph_space
from crosscheck.score import BGeScore, BICScore, get_score


def total_score(score, dag: np.ndarray, samples: np.ndarray) -> float:
    """Sum of local scores over nodes -- the score of the whole DAG."""
    return sum(
        score.local_score(j, np.flatnonzero(dag[:, j] > 0.5), samples)
        for j in range(dag.shape[0])
    )


def posterior_over(score, space, samples: np.ndarray) -> np.ndarray:
    s = np.array([total_score(score, a, samples) for a in space.dags])
    p = np.exp(s - s.max())
    return p / p.sum()


def chain_data(n: int, d: int = 3, seed: int = 0) -> np.ndarray:
    """0 -> 1 -> 2, with a DIFFERENT noise scale per node.

    Per-node noise is deliberate: equal error variances would make the DAG identifiable
    from observational data alone (Peters & Buehlmann 2014), which is the leak this
    rebuild exists to remove.
    """
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.2, 1.0, d)
    x = np.zeros((n, d))
    x[:, 0] = rng.normal(0, scales[0], n)
    x[:, 1] = 1.5 * x[:, 0] + rng.normal(0, scales[1], n)
    x[:, 2] = -1.2 * x[:, 1] + rng.normal(0, scales[2], n)
    return x


def chain_index(space) -> int:
    truth = np.zeros((3, 3), dtype=np.int8)
    truth[0, 1] = 1
    truth[1, 2] = 1
    return next(i for i in range(space.n_dags) if np.array_equal(space.dags[i], truth))


@pytest.fixture(params=["bge", "bic"])
def score_name(request):
    return request.param


# --- the property everything depends on ------------------------------------------

def test_score_is_equivalent_within_markov_equivalence_classes(score_name):
    """Every DAG in a class must score identically on observational data."""
    d = 3
    space = build_graph_space(d)
    score = get_score(score_name, d)
    samples = chain_data(300, d)
    scores = np.array([total_score(score, a, samples) for a in space.dags])

    for mec in range(space.n_mecs):
        members = np.flatnonzero(space.mec_id == mec)
        if len(members) < 2:
            continue
        spread = float(scores[members].max() - scores[members].min())
        assert spread < 1e-8, (
            f"{score_name}: class {mec} spans {spread:.2e} in score, so observational "
            f"data is separating Markov-equivalent DAGs -- information is leaking."
        )


# --- the failure that motivated this module --------------------------------------

def test_dense_graphs_do_not_dominate_the_posterior(score_name):
    """Regression test on the exact measured failure of the previous scorer.

    With an unpenalised profile likelihood, the six densest DAGs at d=3 tied at the top
    holding 67% of posterior mass while the true 2-edge graph ranked 9th of 25. Both
    scores here must put the true class ahead of the fully-connected class.
    """
    d = 3
    space = build_graph_space(d)
    samples = chain_data(1000, d)
    post = posterior_over(get_score(score_name, d), space, samples)

    edge_counts = space.dags.reshape(space.n_dags, -1).sum(1)
    densest_mass = float(post[edge_counts == edge_counts.max()].sum())
    true_class_mass = float(post[space.mec_members(chain_index(space))].sum())

    assert true_class_mass > densest_mass, (
        f"{score_name}: densest class holds {densest_mass:.3f} versus {true_class_mass:.3f} "
        f"for the true class -- complexity is not being penalised."
    )


def test_empty_graph_wins_when_variables_are_independent(score_name):
    """A score that invents structure from noise is unusable. Independent data must
    favour the empty graph."""
    d = 3
    space = build_graph_space(d)
    rng = np.random.default_rng(1)
    samples = rng.normal(0, 1, (1000, d))
    post = posterior_over(get_score(score_name, d), space, samples)

    edge_counts = space.dags.reshape(space.n_dags, -1).sum(1)
    empty = int(np.flatnonzero(edge_counts == 0)[0])
    assert post[empty] == post.max(), f"{score_name}: empty graph is not the MAP"
    assert post[empty] > 0.5, f"{score_name}: only {post[empty]:.3f} on the empty graph"


def test_posterior_concentrates_on_the_true_class_as_data_grows(score_name):
    """Consistency: more data must mean more mass on the truth, monotonically enough to
    distinguish a working score from one that has plateaued."""
    d = 3
    space = build_graph_space(d)
    score = get_score(score_name, d)
    idx = space.mec_members(chain_index(space))

    small = float(posterior_over(score, space, chain_data(100, d))[idx].sum())
    large = float(posterior_over(score, space, chain_data(10000, d))[idx].sum())

    assert large > small, f"{score_name}: mass fell from {small:.3f} to {large:.3f}"
    assert large > 0.85, f"{score_name}: only {large:.3f} on the true class at n=10000"


# --- mechanics -------------------------------------------------------------------

def test_local_score_is_order_invariant_in_its_parents():
    d = 3
    samples = chain_data(200, d)
    for score in (BGeScore(d), BICScore(d)):
        a = score.local_score(2, [0, 1], samples)
        b = score.local_score(2, [1, 0], samples)
        assert a == pytest.approx(b)


def test_empty_sample_set_contributes_nothing():
    """A node intervened on in every sample has no usable rows and must contribute zero,
    not a NaN or an exception."""
    d = 3
    empty = np.zeros((0, d))
    for score in (BGeScore(d), BICScore(d)):
        assert score.local_score(0, [], empty) == 0.0
        assert score.local_score(0, [1], empty) == 0.0


def test_bge_rejects_an_improper_prior():
    with pytest.raises(ValueError):
        BGeScore(3, alpha_w=3.0)  # must exceed d + 1


def test_get_score_rejects_unknown_names():
    with pytest.raises(ValueError):
        get_score("nonsense", 3)
