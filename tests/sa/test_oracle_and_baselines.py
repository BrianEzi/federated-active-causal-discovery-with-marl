"""Tests for the information-gain oracle, the baseline policies, and GATE 2.

The oracle is both a baseline and a measuring instrument, so its failure modes matter.
The most important test here is `test_score_is_zero_when_no_graph_can_be_discriminated`:
when the oracle has no preference, every choice is trivially "optimal", and aggregating
`is_optimal` over such steps is what produced a retracted 99.4-100% agreement figure that
was 93-98% vacuous.
"""
import numpy as np
import pytest

from sa.baselines import GreedyOraclePolicy, RandomPolicy, make_baselines, no_intervention_policy
from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig
from sa.gates import bootstrap_ci, check_gate_2, run_policy
from sa.graphs import build_graph_space
from sa.oracle import InterventionOracle


@pytest.fixture(scope="module")
def space3():
    return build_graph_space(3)


@pytest.fixture(scope="module")
def oracle3(space3):
    return InterventionOracle(space3)


# --- the scoring rule -------------------------------------------------------------

def test_score_is_zero_when_the_posterior_is_a_point_mass(space3, oracle3):
    """Nothing left to learn means no intervention has value. This is the case that must
    never be counted as the agent choosing well."""
    point = np.zeros(space3.n_dags)
    point[5] = 1.0
    np.testing.assert_allclose(oracle3.scores(point), 0.0, atol=1e-12)


def test_score_is_zero_when_no_graph_can_be_discriminated(space3, oracle3):
    """Two DAGs with identical descendant sets from every node cannot be separated by any
    intervention, so every score must be zero even though the posterior is uncertain."""
    reach_groups = oracle3.signatures
    pair = None
    for i in range(space3.n_dags):
        for j in range(i + 1, space3.n_dags):
            if np.array_equal(reach_groups[i], reach_groups[j]):
                pair = (i, j)
                break
        if pair:
            break
    assert pair is not None, "expected at least one indistinguishable pair at d=3"

    post = np.zeros(space3.n_dags)
    post[list(pair)] = 0.5
    np.testing.assert_allclose(oracle3.scores(post), 0.0, atol=1e-12)


def test_score_is_positive_when_graphs_disagree(space3, oracle3):
    """A uniform posterior over all DAGs must leave some node worth intervening on."""
    uniform = np.full(space3.n_dags, 1.0 / space3.n_dags)
    assert oracle3.scores(uniform).max() > 0.1


def test_score_equals_shannon_entropy_of_the_outcome(space3, oracle3):
    """The criterion IS expected information gain, not a proxy: the outcome is a
    deterministic function of the graph, so I(graph; outcome) = H(outcome)."""
    rng = np.random.default_rng(0)
    post = rng.dirichlet(np.ones(space3.n_dags))
    for node in range(space3.d):
        mass = np.bincount(oracle3.signatures[:, node], weights=post)
        mass = mass[mass > 0]
        expected = float(-np.sum(mass * np.log(mass)))
        assert oracle3.scores(post)[node] == pytest.approx(expected)


def test_scores_are_bounded_by_log_of_the_group_count(space3, oracle3):
    uniform = np.full(space3.n_dags, 1.0 / space3.n_dags)
    scores = oracle3.scores(uniform)
    for node in range(space3.d):
        assert scores[node] <= np.log(oracle3.n_groups[node]) + 1e-9


# --- target selection -------------------------------------------------------------

def test_best_targets_returns_all_ties(space3, oracle3):
    """Ties are genuinely equivalent; marking one arbitrarily correct would make the
    metric measure enumeration order rather than the agent."""
    point = np.zeros(space3.n_dags); point[5] = 1.0
    _, best = oracle3.best_targets(point)
    assert best.all(), "with all scores zero every target ties and must be marked best"


def test_best_action_breaks_ties_randomly(space3, oracle3):
    point = np.zeros(space3.n_dags); point[5] = 1.0
    rng = np.random.default_rng(0)
    picks = {oracle3.best_action(point, rng) for _ in range(50)}
    assert len(picks) > 1, "tie-breaking is deterministic, biasing the oracle by node index"


def test_score_choice_flags_uninformative_steps(space3, oracle3):
    """The field whose absence produced a retracted result."""
    point = np.zeros(space3.n_dags); point[5] = 1.0
    result = oracle3.score_choice(0, point)
    assert result["informative"] == 0.0
    assert result["is_optimal"] == 1.0, "vacuously optimal -- must be excluded by informative"

    uniform = np.full(space3.n_dags, 1.0 / space3.n_dags)
    scores = oracle3.scores(uniform)
    best_choice = oracle3.score_choice(int(np.argmax(scores)), uniform)
    assert best_choice["informative"] == 1.0
    assert best_choice["is_optimal"] == 1.0
    assert best_choice["regret"] == pytest.approx(0.0)


def test_regret_is_positive_for_a_worse_choice(space3, oracle3):
    uniform = np.full(space3.n_dags, 1.0 / space3.n_dags)
    scores = oracle3.scores(uniform)
    worst = int(np.argmin(scores))
    if scores[worst] < scores.max() - 1e-9:
        assert oracle3.score_choice(worst, uniform)["regret"] > 0.0


# --- baseline policies -------------------------------------------------------------

def test_no_intervention_policy_always_passes():
    assert no_intervention_policy(None, None) == PASS_ACTION


def test_random_policy_stays_in_range_and_never_passes():
    env = CausalDiscoveryEnv(EnvConfig(d=4, n_obs=200))
    env.reset(seed=0)
    policy = RandomPolicy(seed=0)
    picks = [policy(env, None) for _ in range(100)]
    assert all(0 <= p < 4 for p in picks)
    assert len(set(picks)) > 1


def test_greedy_oracle_passes_when_nothing_is_informative(space3):
    env = CausalDiscoveryEnv(EnvConfig(d=3, n_obs=200))
    result = env.reset(seed=0)
    policy = GreedyOraclePolicy(space3, seed=0)

    class Solved:
        posterior = np.zeros(space3.n_dags)
    Solved.posterior[5] = 1.0
    assert policy(env, Solved) == PASS_ACTION


def test_no_intervention_baseline_matches_the_gate_1_rate():
    """Cross-check between two independently computed numbers: the no-intervention
    baseline's success rate must equal the observational-only rate GATE 1 measures."""
    cfg = EnvConfig(d=3)
    result = run_policy(cfg, no_intervention_policy, 200, seed=0)
    space = CausalDiscoveryEnv(cfg).space
    low, high = bootstrap_ci(result["identified"], seed=0)
    assert low <= space.singleton_fraction <= high, (
        f"no-intervention rate CI ({low:.3f}-{high:.3f}) excludes the singleton fraction "
        f"{space.singleton_fraction:.3f}"
    )


def test_make_baselines_provides_the_standard_set(space3):
    baselines = make_baselines(space3)
    assert set(baselines) == {"no_intervention", "random", "greedy_oracle"}


# --- GATE 2 --------------------------------------------------------------------------

@pytest.mark.parametrize("d", [3, 4])
def test_gate_2_passes_at_default_settings(d):
    """The oracle must identify the graph in measurably fewer interventions than random.

    Measured on interventions rather than success rate, because with a generous budget
    both succeed almost always and the rate saturates. Reference values at 200 episodes:
    d=3 oracle 1.12 vs random 1.55; d=4 oracle 1.38 vs random 2.53.
    """
    cfg = EnvConfig(d=d)
    space = CausalDiscoveryEnv(cfg).space
    baselines = make_baselines(space, seed=0)
    result = check_gate_2(cfg, baselines["random"], baselines["greedy_oracle"],
                          n_episodes=120, seed=1)
    assert result.passed, str(result)


def test_gate_2_reports_failure_when_the_two_policies_are_identical():
    """Guard on the guard: comparing the oracle against itself must not pass."""
    cfg = EnvConfig(d=3)
    space = CausalDiscoveryEnv(cfg).space
    oracle = make_baselines(space, seed=0)["greedy_oracle"]
    result = check_gate_2(cfg, oracle, oracle, n_episodes=80, seed=1)
    assert not result.passed and "OVERLAP" in result.detail
