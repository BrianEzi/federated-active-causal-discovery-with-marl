"""Block 1 acceptance test: the subset DP must reproduce enumeration exactly.

Checked DIRECTLY against the enumerated posterior, never through a downstream consumer.
Measuring a new implementation through something that reads it -- a sampler through the
oracle, a posterior through an environment's solve rate -- conflates correctness with the
consumer's own behaviour, and cost three debugging rounds on 2026-08-15.

The comparison is meaningful only because both paths share `LocalScorer`, so any
disagreement is the DP recurrence and not a difference in what is being scored.

d=6 builds 3,781,503 graphs and is the slowest test in the suite by a wide margin. It is
here rather than in a script because d=6 is the largest place ground truth exists, and the
whole point of the DP is to go past it -- an untested boundary is where a scaling bug would
live.
"""
import numpy as np
import pytest

from sa.dp import DPPosterior
from sa.graphs import build_graph_space
from sa.posterior import PosteriorEngine
from sa.priors import build_prior, erdos_renyi_prior, scale_free_prior, uniform_prior
from sa.score import BGeScore


def _data(n, d, seed=0, intervene=None):
    """`intervene` is a list of nodes intervened on in the first fifth of the rows."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n, d))
    intervened = np.zeros((n, d))
    for node in intervene or []:
        intervened[: n // 5, node] = 1.0
    return samples, intervened


def _reference(d, samples, intervened, prior):
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    log_scores = engine.log_scores(samples, intervened)
    log_unnorm = log_scores + np.log(np.maximum(prior(space), 1e-300))
    m = log_unnorm.max()
    log_z = m + np.log(np.exp(log_unnorm - m).sum())
    posterior = np.exp(log_unnorm - log_z)
    return space, engine, posterior, log_z


# --------------------------------------------------------------------------------------
# The acceptance test
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [3, 4, 5, 6])
def test_log_partition_matches_enumeration(d):
    samples, intervened = _data(500, d, seed=d, intervene=[1])
    space, _, _, log_z_enum = _reference(d, samples, intervened, uniform_prior)

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    log_w = dp.log_weights(samples, intervened)
    log_z_dp, growth = dp.log_partition_diagnostic(log_w)

    # The enumerated constant carries the uniform prior's 1/N; the DP drops constant
    # factors, so the two differ by exactly log N and that offset must be removed rather
    # than absorbed into a loose tolerance.
    assert log_z_dp - np.log(space.n_dags) == pytest.approx(log_z_enum, abs=1e-8)
    assert growth < 1e6, f"cancellation growth {growth:.2e} at d={d}"


@pytest.mark.parametrize("d", [3, 4, 5, 6])
def test_edge_marginals_match_enumeration(d):
    samples, intervened = _data(500, d, seed=d + 10, intervene=[0, 2] if d > 2 else [0])
    space, engine, posterior, _ = _reference(d, samples, intervened, uniform_prior)
    exact = engine.edge_marginals(posterior)

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    approx = dp.edge_marginals(dp.log_weights(samples, intervened))

    assert np.abs(exact - approx).max() < 1e-9
    assert np.all(np.diag(approx) == 0.0)


@pytest.mark.parametrize("d", [3, 4, 5])
def test_true_dag_probability_matches_enumeration(d):
    """The quantity `is_identified` thresholds -- the one number the env actually reads."""
    samples, intervened = _data(400, d, seed=d + 20, intervene=[1])
    space, _, posterior, _ = _reference(d, samples, intervened, uniform_prior)

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    log_w = dp.log_weights(samples, intervened)
    log_z = dp.log_partition(log_w)

    rng = np.random.default_rng(0)
    for index in rng.choice(space.n_dags, size=min(25, space.n_dags), replace=False):
        got = np.exp(dp.log_prob_dag(log_w, space.dags[index], log_z=log_z))
        assert got == pytest.approx(float(posterior[index]), rel=1e-7, abs=1e-12)


# --------------------------------------------------------------------------------------
# Block 2 acceptance test: one-pass edge marginals
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [3, 4, 5, 6])
def test_onepass_marginals_match_enumeration(d):
    """Pinned against ENUMERATION, not against `edge_marginals`.

    Comparing the two DP routes to each other would pass if they shared a mistake in the
    weights, the prior term or the parent-set indexing -- everything except the recurrence
    itself. Enumeration is the only reference that does not share code with either.
    """
    samples, intervened = _data(500, d, seed=d + 40, intervene=[0, 2] if d > 2 else [0])
    _, engine, posterior, _ = _reference(d, samples, intervened, uniform_prior)
    exact = engine.edge_marginals(posterior)

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    onepass = dp.edge_marginals_onepass(dp.log_weights(samples, intervened), check=True)

    assert np.abs(exact - onepass).max() < 1e-9
    assert np.all(np.diag(onepass) == 0.0)


def test_onepass_agrees_with_constrained_runs_under_a_non_uniform_prior():
    """The prior enters the backward pass through the same weights as the forward one; a
    prior applied on only one side would still look right at p=0.5, where it is zero."""
    d = 5
    samples, intervened = _data(400, d, seed=44, intervene=[3])
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="erdos_renyi", p=0.25)
    log_w = dp.log_weights(samples, intervened)
    assert np.abs(dp.edge_marginals(log_w)
                  - dp.edge_marginals_onepass(log_w)).max() < 1e-12


def test_euler_identity_holds_for_every_node():
    """`sum_P w_v(P) dZ/dw_v(P) == Z`, because Z has degree exactly one in each node's
    weights -- every DAG picks exactly one parent set per node.

    This is the internal check that a misindexed backward pass cannot survive, and it
    needs no ground truth, so it remains available at d where enumeration does not.
    """
    for d in (4, 6, 8):
        rng = np.random.default_rng(d)
        samples = rng.normal(size=(300, d))
        dp = DPPosterior.for_prior(d, BGeScore(d), kind="erdos_renyi", p=0.35)
        log_w = dp.log_weights(samples, np.zeros((300, d)))
        dp.edge_marginals_onepass(log_w, check=True)   # raises AssertionError if violated


def test_onepass_is_faster_than_constrained_runs_at_d6():
    """The pre-registered block 2 acceptance threshold: >= 5x at d=6.

    A timing test is ordinarily a bad test, but the whole justification for adding a second
    implementation of a quantity that already worked is the speedup. If it is not there,
    the extra code is a liability and this should fail rather than pass quietly.
    """
    import time
    d = 6
    samples, intervened = _data(400, d, seed=6, intervene=[1])
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    log_w = dp.log_weights(samples, intervened)

    dp.edge_marginals_onepass(log_w)          # warm any first-call cost
    t0 = time.perf_counter()
    dp.edge_marginals(log_w)
    slow = time.perf_counter() - t0
    t0 = time.perf_counter()
    dp.edge_marginals_onepass(log_w)
    fast = time.perf_counter() - t0
    assert slow / fast >= 5.0, f"only {slow / fast:.1f}x -- measured 6.4x on 2026-08-16"


def test_moebius_transpose_is_the_adjoint_of_zeta():
    """<zeta(w), y> == <w, moebius_transpose(y)> for random vectors.

    Pins the one line the backward pass depends on most and the one that would be easiest
    to write with the addition running the wrong way -- a bug that leaves every marginal
    in [0, 1] and merely wrong.
    """
    from sa.dp import moebius_transpose, zeta
    d = 5
    rng = np.random.default_rng(0)
    w = rng.normal(size=(1, 1 << d))
    y = rng.normal(size=(1, 1 << d))
    assert float((zeta(w.copy(), d) @ y.T).item()) == pytest.approx(
        float((w @ moebius_transpose(y.copy(), d).T).item()), rel=1e-12)


# --------------------------------------------------------------------------------------
# Priors: the DP must be scoring the SAME model, not a similar one
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
def test_erdos_renyi_prior_is_reproduced_exactly(p):
    """ER is modular because |E| = sum_i |Pa_i|. If that identity were mishandled the
    posterior would still look plausible -- just tilted toward the wrong density."""
    d = 4
    samples, intervened = _data(300, d, seed=7, intervene=[2])
    _, _, posterior, _ = _reference(
        d, samples, intervened, lambda s: erdos_renyi_prior(s, p))

    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    exact = engine.edge_marginals(posterior)

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="erdos_renyi", p=p)
    approx = dp.edge_marginals(dp.log_weights(samples, intervened))
    assert np.abs(exact - approx).max() < 1e-9


def test_uniform_equals_erdos_renyi_at_half():
    """Stated in sa/priors.py; pinned here because the DP relies on it."""
    d = 4
    dp_u = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    dp_e = DPPosterior.for_prior(d, BGeScore(d), kind="erdos_renyi", p=0.5)
    assert dp_u.log_edge_odds == pytest.approx(dp_e.log_edge_odds, abs=1e-12)


def test_non_modular_prior_is_refused_not_approximated():
    """The dangerous failure is silence: a scale-free prior quietly ignored would give a
    posterior under a different model that still looks reasonable at small d."""
    with pytest.raises(ValueError, match="not modular"):
        DPPosterior.for_prior(4, BGeScore(4), kind="scale_free")


def test_scale_free_is_actually_non_modular():
    """Guards the claim above: if scale_free ever became ER-equivalent, the refusal would
    be spurious and this test says so rather than leaving the restriction unexamined."""
    space = build_graph_space(4, fast=True)
    assert not np.allclose(scale_free_prior(space, 0.3, gamma=1.0),
                           erdos_renyi_prior(space, 0.3))


# --------------------------------------------------------------------------------------
# Interventions and edges
# --------------------------------------------------------------------------------------

def test_intervention_changes_the_dp_posterior_the_same_way():
    """Interventional data is what breaks Markov equivalence; if the DP dropped the
    intervened rows differently from the enumerated path the two would agree
    observationally and diverge exactly where the experiment lives."""
    d = 4
    samples, _ = _data(600, d, seed=31)
    obs = np.zeros((600, d))
    itv = np.zeros((600, d))
    itv[:200, 2] = 1.0

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    prior = uniform_prior(space)

    for intervened in (obs, itv):
        exact = engine.edge_marginals(engine.posterior(samples, intervened, prior))
        approx = dp.edge_marginals(dp.log_weights(samples, intervened))
        assert np.abs(exact - approx).max() < 1e-9

    assert not np.allclose(
        dp.edge_marginals(dp.log_weights(samples, obs)),
        dp.edge_marginals(dp.log_weights(samples, itv)), atol=1e-6)


def test_empty_sample_set_recovers_the_prior():
    """With no data every DAG's weight is its prior alone, so the DP must return the
    prior's own edge marginals rather than a degenerate answer."""
    d = 4
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    prior = build_prior(space, kind="erdos_renyi", p=0.3)
    expected = engine.edge_marginals(prior)

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="erdos_renyi", p=0.3)
    log_w = dp.log_weights(np.zeros((0, d)), np.zeros((0, d)))
    assert np.abs(dp.edge_marginals(log_w) - expected).max() < 1e-9


def test_log_prob_dag_rejects_a_self_loop():
    d = 4
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    log_w = dp.log_weights(*_data(100, d, seed=1))
    bad = np.zeros((d, d))
    bad[1, 1] = 1.0
    with pytest.raises(ValueError, match="own parent"):
        dp.log_prob_dag(log_w, bad)


def test_scorer_is_shared_with_the_enumerated_engine():
    """Both paths must read the same local scores; otherwise the acceptance tests above
    compare two models rather than two algorithms."""
    d = 4
    samples, intervened = _data(200, d, seed=3, intervene=[0])
    engine = PosteriorEngine(build_graph_space(d, fast=True), BGeScore(d))
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    assert np.array_equal(engine.local_score_table(samples, intervened),
                          dp.scorer.table(samples, intervened))
    assert engine.parent_sets == dp.scorer.parent_sets
