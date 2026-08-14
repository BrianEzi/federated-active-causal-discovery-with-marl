"""Tests for the SCM and the environment, including GATE 1.

The gate test is the important one. It is the check that was missing from the previous
round, where the environment let roughly half of all episodes be solved without acting
and nothing in the test suite noticed.
"""
import numpy as np
import pytest

from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig
from sa.gates import bootstrap_ci, check_gate_1
from sa.graphs import build_graph_space
from sa.scm import sample, sample_scm_params, topological_order


# --- SCM -------------------------------------------------------------------------

def test_topological_order_places_parents_first():
    a = np.zeros((4, 4), dtype=np.int8)
    a[0, 1] = 1; a[1, 2] = 1; a[0, 3] = 1
    order = topological_order(a)
    position = {int(node): i for i, node in enumerate(order)}
    for i in range(4):
        for j in range(4):
            if a[i, j]:
                assert position[i] < position[j]


def test_topological_order_rejects_cycles():
    a = np.zeros((3, 3), dtype=np.int8)
    a[0, 1] = 1; a[1, 2] = 1; a[2, 0] = 1
    with pytest.raises(ValueError):
        topological_order(a)


def test_noise_scales_differ_across_nodes():
    """The whole point of the rebuild. Equal error variances make the DAG identifiable
    observationally (Peters & Buehlmann 2014), which removes the need to intervene."""
    rng = np.random.default_rng(0)
    a = np.zeros((4, 4), dtype=np.int8); a[0, 1] = 1
    params = sample_scm_params(a, rng)
    assert len(np.unique(params.noise_scales)) == 4


def test_weights_are_nonzero_exactly_on_edges():
    rng = np.random.default_rng(0)
    a = np.zeros((3, 3), dtype=np.int8); a[0, 1] = 1; a[1, 2] = 1
    params = sample_scm_params(a, rng)
    assert params.weights[0, 1] != 0 and params.weights[1, 2] != 0
    assert params.weights[1, 0] == 0 and params.weights[2, 0] == 0
    assert np.all(np.abs(params.weights[a > 0.5]) >= 0.5)  # detectable


def test_intervention_marks_only_the_intervened_node():
    rng = np.random.default_rng(0)
    a = np.zeros((3, 3), dtype=np.int8); a[0, 1] = 1; a[1, 2] = 1
    params = sample_scm_params(a, rng)
    samples, intervened = sample(params, 50, rng, intervene_node=1)
    assert intervened[:, 1].all()
    assert not intervened[:, 0].any() and not intervened[:, 2].any()
    assert samples.shape == (50, 3)


def test_intervened_node_becomes_independent_of_its_parents():
    """A hard intervention replaces the structural equation, so the correlation with the
    parent must vanish -- this is what makes the intervention informative."""
    rng = np.random.default_rng(2)
    a = np.zeros((2, 2), dtype=np.int8); a[0, 1] = 1
    params = sample_scm_params(a, rng)

    obs, _ = sample(params, 4000, rng)
    itv, _ = sample(params, 4000, rng, intervene_node=1)
    assert abs(np.corrcoef(obs[:, 0], obs[:, 1])[0, 1]) > 0.3
    assert abs(np.corrcoef(itv[:, 0], itv[:, 1])[0, 1]) < 0.1


def test_intervention_values_vary():
    """A constant intervention value has no variance, leaving descendants' dependence on
    it unidentifiable; the value must vary per sample."""
    rng = np.random.default_rng(0)
    a = np.zeros((2, 2), dtype=np.int8)
    params = sample_scm_params(a, rng)
    samples, _ = sample(params, 200, rng, intervene_node=0)
    assert samples[:, 0].std() > 0.5


# --- environment ------------------------------------------------------------------

@pytest.fixture(scope="module")
def env3():
    return CausalDiscoveryEnv(EnvConfig(d=3, n_obs=200))


def test_reset_produces_a_valid_belief(env3):
    result = env3.reset(seed=0)
    assert result.posterior.shape == (env3.space.n_dags,)
    assert float(result.posterior.sum()) == pytest.approx(1.0)
    assert result.n_interventions == 0


def test_step_consumes_budget_and_accumulates_data(env3):
    env3.reset(seed=1)
    before = env3.samples.shape[0]
    result = env3.step(0)
    assert result.n_interventions == 1
    assert env3.samples.shape[0] == before + env3.config.n_int
    assert env3.intervened[-1, 0] == 1.0


def test_episode_ends_when_budget_is_exhausted():
    env = CausalDiscoveryEnv(EnvConfig(d=3, n_obs=200, budget=3))
    result = env.reset(seed=5)
    steps = 0
    while not result.done and steps < 20:
        result = env.step(steps % 3)
        steps += 1
    assert result.done
    assert result.n_interventions <= 3


def test_passing_ends_the_episode(env3):
    env3.reset(seed=2)
    result = env3.step(PASS_ACTION)
    assert result.done and result.info["passed"]
    assert result.n_interventions == 0


def test_invalid_actions_are_rejected(env3):
    env3.reset(seed=3)
    with pytest.raises(ValueError):
        env3.step(99)
    with pytest.raises(ValueError):
        env3.step(-2)


def test_step_before_reset_is_an_error():
    with pytest.raises(RuntimeError):
        CausalDiscoveryEnv(EnvConfig(d=3)).step(0)


def test_force_index_pins_the_true_graph(env3):
    for idx in (0, 7, 12):
        assert env3.reset(seed=0, force_index=idx).info["true_index"] == idx


def test_same_seed_reproduces_the_episode(env3):
    a = env3.reset(seed=42).posterior.copy()
    b = env3.reset(seed=42).posterior.copy()
    np.testing.assert_allclose(a, b)


# --- observations ------------------------------------------------------------------

def test_observation_shapes_match_declared_dims(env3):
    env3.reset(seed=0)
    for kind, dim in env3.observation_dim.items():
        assert env3.observation(kind).shape == (dim,)


def test_edge_marginal_observation_excludes_self_edges(env3):
    env3.reset(seed=0)
    # d(d-1) off-diagonal entries plus one budget element.
    assert env3.observation("edge_marginals").shape[0] == 3 * 2 + 1


def test_unknown_observation_kind_is_rejected(env3):
    env3.reset(seed=0)
    with pytest.raises(ValueError):
        env3.observation("something_else")


# --- GATE 1 -------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(0.5, 0.1, 500)
    low, high = bootstrap_ci(values)
    assert low < values.mean() < high


def test_bootstrap_ci_handles_empty_input():
    low, high = bootstrap_ci(np.array([]))
    assert np.isnan(low) and np.isnan(high)


@pytest.mark.parametrize("d", [3, 4])
def test_gate_1_passes_at_default_settings(d):
    """The environment must require interventions.

    Observational-only identification must match the fraction of DAGs alone in their
    Markov equivalence class -- 16.0% at d=3, 10.87% at d=4. Excess means orientation
    information is leaking; a large shortfall means the estimator cannot identify even
    the graphs theory says are identifiable.
    """
    result = check_gate_1(EnvConfig(d=d), n_episodes=150, seed=1)
    assert result.passed, str(result)


def test_gate_1_detects_the_historical_leaky_estimator():
    """Guard on the guard: GATE 1 must catch the defect that invalidated the last round.

    The leak lives in the *estimator*, not the data. `KnownVarianceScore` reproduces the
    old scorer -- profile likelihood with the true shared variance plugged in -- which
    breaks score equivalence and, on equal-variance data, identifies the DAG before any
    intervention. Paired with equal noise scales this is exactly the old setup.
    """
    # 600 episodes rather than 150: the leak is ~26% against a 16% target, a ~10pp effect,
    # and at 150 episodes the sampling noise (+/- ~6pp) can drop it under the gate's
    # tolerance and make this test flaky. Measured at 1000 episodes: leaky 26.0%, BGe 13.8%.
    leaky = EnvConfig(d=3, score="known_variance", noise_range=(1.0, 1.0))
    result = check_gate_1(leaky, n_episodes=600, seed=1)
    assert not result.passed and "LEAK" in result.detail, (
        f"GATE 1 failed to detect the known-variance shortcut: {result}"
    )


def test_equal_noise_alone_does_not_leak_through_a_score_equivalent_scorer():
    """Records a finding that is easy to get backwards, and that cost real time to learn.

    BGe is score-equivalent by construction, so it cannot separate Markov-equivalent DAGs
    regardless of how the noise is distributed. Equal variances make the DAG identifiable
    *in principle*, but only an estimator that assumes a known or shared variance can act
    on it. So per-node noise in sa/scm.py is defence in depth, not the load-bearing fix --
    which matters if a non-score-equivalent estimator is ever swapped in.
    """
    equal_noise = EnvConfig(d=3, noise_range=(1.0, 1.0))  # BGe, but equal variances
    result = check_gate_1(equal_noise, n_episodes=150, seed=1)
    assert result.passed, (
        f"BGe is score-equivalent and must be immune to the equal-variance shortcut: {result}"
    )
