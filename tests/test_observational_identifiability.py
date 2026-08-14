"""Regression tests for the observational-shortcut bug (found 2026-08-14).

The bug: `bayes_optimal_estimator` substituted the environment's true, shared `noise_scale`
for every node's error variance. Because our SCM gives every node the SAME noise scale, and a
linear Gaussian SEM with equal error variances is fully identifiable from observational data
(Peters & Bühlmann 2014), this let the estimator recover the true DAG *before any intervention
was taken* -- collapsing the active-discovery task the whole project is about, and silently
making the oracle-agreement metric vacuous (every legal target tied at zero discriminating
value, so every choice scored as "optimal").

These tests lock in the corrected behaviour: with the error variance fitted by MLE, the
observational posterior must be MEC-limited, i.e. it must NOT reliably identify the true DAG
from observational data alone. See docs/THEORY_NOTES.md entries #1 and #2.
"""
import numpy as np
import jax
import pytest

from src.generators import get_all_4node_topologies, generate_scm_params
from src.marl.bayes_optimal_estimator import compute_hypothesis_posterior
from src.marl.oracle_policy import expected_discrimination


def _observational_samples(hypothesis_idx: int, n: int, noise_scale: float, seed: int):
    """Draws purely observational samples from topology `hypothesis_idx`."""
    matrices, orders = get_all_4node_topologies()
    params = generate_scm_params(jax.random.PRNGKey(seed), matrices[hypothesis_idx], 0)
    weights = np.array(params.W)  # weights[i, j] is the weight of edge j -> i
    rng = np.random.default_rng(seed)

    samples = np.zeros((n, 4))
    for node in np.array(orders[hypothesis_idx]):
        node = int(node)
        samples[:, node] = samples @ weights[node, :] + rng.normal(0, noise_scale, n)
    return samples, np.zeros((n, 4))  # no interventions


def _observational_accuracy(use_known_variance: bool, trials: int = 80, n: int = 100):
    """Fraction of trials where the observational posterior's MAP is the true topology."""
    all_adj = np.array(get_all_4node_topologies()[0])
    hits = 0
    for t in range(trials):
        true_idx = t % 8
        samples, interv = _observational_samples(true_idx, n, noise_scale=0.1, seed=t)
        posterior = compute_hypothesis_posterior(
            samples, interv, all_adj, noise_scale=0.1, use_known_variance=use_known_variance
        )
        hits += int(np.argmax(posterior) == true_idx)
    return hits / trials


def test_known_variance_reproduces_the_observational_shortcut():
    """Documents the bug rather than hiding it: with the true shared variance supplied,
    the estimator identifies the DAG observationally almost perfectly. This is the
    behaviour that made the task degenerate, kept only for reproducing old results."""
    accuracy = _observational_accuracy(use_known_variance=True)
    assert accuracy > 0.9, (
        f"Expected the known-variance path to still exhibit the equal-variance shortcut "
        f"(~0.98 observed when characterised); got {accuracy:.3f}."
    )


def test_fitted_variance_cannot_identify_the_dag_observationally():
    """The core property this project depends on: observational data alone must NOT
    determine the DAG, otherwise interventions are unnecessary and there is no task."""
    accuracy = _observational_accuracy(use_known_variance=False)
    assert accuracy < 0.6, (
        f"Observational-only MAP accuracy is {accuracy:.3f}, which is too high: the estimator can "
        f"identify the true DAG without intervening, so the active-discovery task is degenerate. "
        f"Expected at most the reciprocal MEC size (~0.25-0.5) for these spanning-tree topologies, "
        f"and in practice arbitrary tie-breaking within the class -- see "
        f"test_fitted_variance_posterior_is_uniform_over_the_equivalence_class."
    )


def test_fitted_variance_posterior_is_uniform_over_the_equivalence_class():
    """The corrected estimator should degrade to *exactly* 'right equivalence class,
    orientation undetermined' -- not to uselessness, and not to a spurious preference.

    Note deliberately why MAP accuracy is NOT tested here: with the variance profiled out
    the score is score-equivalent (Chickering 2002), so members of the same Markov
    equivalence class receive *bit-for-bit tied* posterior mass. `argmax` over exact ties
    is decided by floating-point ordering, which makes MAP accuracy a measurement of tie
    -breaking rather than of the estimator. The meaningful assertions are that the true
    hypothesis always sits in the tied-best set, and that its posterior mass equals the
    reciprocal of that set's size.
    """
    all_adj = np.array(get_all_4node_topologies()[0])
    for t in range(24):
        true_idx = t % 8
        samples, interv = _observational_samples(true_idx, 100, noise_scale=0.1, seed=t)
        posterior = compute_hypothesis_posterior(samples, interv, all_adj, noise_scale=0.1)

        # Tolerance rather than exact equality: the 1e-6 ridge in the OLS solve perturbs
        # score equivalence very slightly (observed spread ~2e-3), so members of one class
        # tie to about three decimal places rather than bit-for-bit.
        tied = np.where(posterior >= posterior.max() - 0.02)[0]
        assert true_idx in tied, (
            f"seed={t}: true topology {true_idx} is not among the tied-best hypotheses "
            f"{tied.tolist()}; the likelihood is actively preferring a wrong structure."
        )
        assert posterior[true_idx] == pytest.approx(1.0 / len(tied), abs=0.02), (
            f"seed={t}: expected posterior mass to be uniform over the {len(tied)}-member "
            f"equivalence class ({1.0 / len(tied):.3f}); got {posterior[true_idx]:.3f}."
        )
        assert len(tied) > 1, (
            f"seed={t}: the equivalence class collapsed to a single hypothesis, i.e. the DAG "
            f"is observationally identified -- the shortcut has reappeared."
        )


def test_oracle_has_real_discriminating_work_to_do_observationally():
    """The metric-level consequence. Under the bug, the posterior concentrated so hard that
    `expected_discrimination` returned all-zeros and `oracle_best_targets` marked EVERY legal
    node optimal -- which is why measured oracle agreement was 99-100% while being vacuous.
    After the fix, some node must carry non-zero discriminating value."""
    all_adj = np.array(get_all_4node_topologies()[0])
    vacuous = 0
    trials = 40
    for t in range(trials):
        samples, interv = _observational_samples(t % 8, 100, noise_scale=0.1, seed=t)
        posterior = compute_hypothesis_posterior(samples, interv, all_adj, noise_scale=0.1)
        if expected_discrimination(posterior, all_adj).max() <= 1e-12:
            vacuous += 1
    vacuous_rate = vacuous / trials
    assert vacuous_rate < 0.1, (
        f"{vacuous_rate:.1%} of observational states leave the oracle with no discriminating "
        f"choice, so oracle-agreement would be vacuously 100% there. Expected ~0%."
    )
