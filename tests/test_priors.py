"""Tests for the graph priors.

The load-bearing test is `test_uniform_is_erdos_renyi_at_p_half`. That equivalence is why
the original uniform sampling was never actually off the research standard, and it is the
reason switching to ER is a re-parameterisation rather than a change of graph family.
"""
import numpy as np
import pytest

from ma.graphs import build_graph_space
from ma.priors import (
    build_prior,
    erdos_renyi_prior,
    expected_degree_p,
    expected_edges_p,
    prior_mean_mec_size,
    prior_singleton_fraction,
    scale_free_prior,
    uniform_prior,
)


@pytest.fixture(scope="module")
def space3():
    return build_graph_space(3)


@pytest.fixture(scope="module")
def space4():
    return build_graph_space(4)


# --- the equivalence ---------------------------------------------------------------

def test_uniform_is_erdos_renyi_at_p_half(space4):
    """P(G) ~ p^|E| (1-p)^(pairs-|E|) is constant at p=0.5, so ER(0.5) IS uniform."""
    np.testing.assert_allclose(erdos_renyi_prior(space4, 0.5), uniform_prior(space4))


def test_lower_p_shifts_mass_toward_sparser_graphs(space4):
    edges = space4.dags.reshape(space4.n_dags, -1).sum(1)
    sparse = float((erdos_renyi_prior(space4, 0.2) * edges).sum())
    dense = float((erdos_renyi_prior(space4, 0.8) * edges).sum())
    uniform = float((uniform_prior(space4) * edges).sum())
    assert sparse < uniform < dense


# --- basic well-formedness ---------------------------------------------------------

@pytest.mark.parametrize("kind", ["uniform", "erdos_renyi", "scale_free"])
def test_priors_are_distributions(space3, kind):
    prior = build_prior(space3, kind=kind, p=0.4)
    assert prior.shape == (space3.n_dags,)
    assert prior.min() >= 0.0
    assert float(prior.sum()) == pytest.approx(1.0)


def test_scale_free_reduces_to_erdos_renyi_at_gamma_zero(space4):
    np.testing.assert_allclose(
        scale_free_prior(space4, 0.4, gamma=0.0), erdos_renyi_prior(space4, 0.4)
    )


def test_scale_free_favours_concentrated_degree_distributions(space4):
    """A star (one hub) should gain mass relative to a chain (even degrees) at gamma > 0."""
    er = erdos_renyi_prior(space4, 0.4)
    sf = scale_free_prior(space4, 0.4, gamma=3.0)

    degree = space4.dags.sum(axis=1) + space4.dags.sum(axis=2)
    spread = degree.max(axis=1) - degree.min(axis=1)
    three_edge = space4.dags.reshape(space4.n_dags, -1).sum(1) == 3
    most_even = np.flatnonzero(three_edge & (spread == spread[three_edge].min()))[0]
    most_concentrated = np.flatnonzero(three_edge & (spread == spread[three_edge].max()))[0]

    assert sf[most_concentrated] / er[most_concentrated] > sf[most_even] / er[most_even]


def test_invalid_arguments_are_rejected(space3):
    with pytest.raises(ValueError):
        erdos_renyi_prior(space3, 0.0)
    with pytest.raises(ValueError):
        erdos_renyi_prior(space3, 1.0)
    with pytest.raises(ValueError):
        build_prior(space3, kind="nonsense")


# --- parameterisation helpers -------------------------------------------------------

def test_expected_degree_p_saturates_at_small_d():
    """Records why `expected_degree` is a poor knob below d ~ 8: degree 2 on 3 nodes means
    every node touches every other, i.e. the complete graph."""
    assert expected_degree_p(3, 2.0) == pytest.approx(1.0, abs=1e-5)
    assert expected_degree_p(9, 2.0) == pytest.approx(0.25)


def test_expected_edges_p_matches_the_pair_count():
    assert expected_edges_p(4, 3.0) == pytest.approx(0.5)   # 3 of 6 pairs
    assert expected_edges_p(5, 5.0) == pytest.approx(0.5)   # 5 of 10 pairs


# --- the summaries the gates depend on ------------------------------------------------

def test_singleton_fraction_under_uniform_matches_the_space(space4):
    """GATE 1's target under a uniform prior must equal the plain singleton fraction."""
    assert prior_singleton_fraction(space4, uniform_prior(space4)) == pytest.approx(
        space4.singleton_fraction
    )


def test_sparser_priors_have_smaller_equivalence_classes(space4):
    """The difficulty/realism trade-off, pinned as a test so it cannot be forgotten:
    sparse graphs sit in smaller classes, and class size drives interventions needed."""
    sparse = prior_mean_mec_size(space4, erdos_renyi_prior(space4, 0.2))
    dense = prior_mean_mec_size(space4, erdos_renyi_prior(space4, 0.8))
    assert sparse < dense


def test_singleton_fraction_moves_with_sparsity(space4):
    sparse = prior_singleton_fraction(space4, erdos_renyi_prior(space4, 0.2))
    dense = prior_singleton_fraction(space4, erdos_renyi_prior(space4, 0.8))
    assert sparse > dense, "sparse graphs should more often be identifiable observationally"
