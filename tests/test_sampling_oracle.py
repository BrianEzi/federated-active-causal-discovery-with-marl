"""Block 3: the enumeration-free oracle and GATE 1 target.

Three things stop existing at d=7, and all three are needed before a d=7 number means
anything: the belief (block 1), the greedy baseline's oracle, and GATE 1's target. This
file covers the last two.

Everything is checked against the exact object it replaces, at sizes where that object
exists. Nothing is checked through a downstream consumer -- measuring a sampler through
the oracle instead of against the exact posterior is what cost three debugging rounds on
2026-08-15, because a correctness bug and a mixing problem look identical from there.
"""
import numpy as np
import pytest

from sa.dp import DPPosterior
from sa.gates import estimate_singleton_fraction
from sa.graphs import build_graph_space, is_singleton_mec
from sa.oracle import InterventionOracle, SamplingOracle
from sa.priors import erdos_renyi_prior, prior_singleton_fraction, uniform_prior
from sa.sampler import descendant_codes, mh_sample
from sa.score import BGeScore


def _data(n, d, seed=0, intervene=(1,)):
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(n, d))
    intervened = np.zeros((n, d))
    for node in intervene:
        intervened[: n // 5, node] = 1.0
    return samples, intervened


# --------------------------------------------------------------------------------------
# The sampler itself, against the exact posterior
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [4, 5, 6])
def test_sampler_reproduces_exact_edge_marginals(d):
    """Direct check, and deliberately the FIRST one.

    A sampler validated only through the oracle can be systematically wrong while looking
    like it merely mixes slowly. The discarded Gibbs sampler scored 0.068/0.389/0.123 total
    variation here against this sampler's 0.022/0.009/0.004, which settled in one run what
    three rounds of oracle-level measurement had not.
    """
    samples, intervened = _data(500, d, seed=d)
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    log_w = dp.log_weights(samples, intervened)

    exact = dp.edge_marginals_onepass(log_w)
    draws, acceptance = mh_sample(log_w, dp._mask_to_index, d, 4000,
                                  rng=np.random.default_rng(d + 1))
    assert np.abs(exact - draws.mean(axis=0)).max() < 0.03
    # An acceptance rate at either extreme means the chain is not exploring, and would
    # make the assertion above pass for the wrong reason on an unlucky seed.
    assert 0.01 < acceptance < 0.95


def test_sampler_draws_only_acyclic_graphs():
    """The move set guards acyclicity by hand; a cycle would corrupt every descendant set
    downstream and would not show up as an obviously wrong number."""
    from sa.graphs import is_acyclic
    d = 5
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    draws, _ = mh_sample(dp.log_weights(*_data(300, d, seed=2)), dp._mask_to_index, d,
                         500, rng=np.random.default_rng(0))
    assert all(is_acyclic(g) for g in draws)


def test_sampler_targets_the_prior_when_given_prior_only_weights():
    """Prior-only weights must produce the prior, since that is how GATE 1's target is
    estimated. Checked on edge marginals, where the prior's exact answer is known."""
    d = 4
    space = build_graph_space(d, fast=True)
    from sa.posterior import PosteriorEngine
    engine = PosteriorEngine(space, BGeScore(d))
    expected = engine.edge_marginals(erdos_renyi_prior(space, 0.3))

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="erdos_renyi", p=0.3)
    draws, _ = mh_sample(dp.log_prior_term, dp._mask_to_index, d, 20000,
                         rng=np.random.default_rng(0))
    assert np.abs(expected - draws.mean(axis=0)).max() < 0.02


def test_descendant_codes_match_the_enumerated_closure():
    from sa.graphs import descendants
    d = 5
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    draws, _ = mh_sample(dp.log_weights(*_data(300, d, seed=4)), dp._mask_to_index, d,
                         200, rng=np.random.default_rng(1))
    codes = descendant_codes(draws)
    bits = (1 << np.arange(d)).astype(np.int64)
    for k in range(0, len(draws), 17):
        assert np.array_equal(codes[k], descendants(draws[k]).astype(np.int64) @ bits)


# --------------------------------------------------------------------------------------
# GATE 1's target without enumeration
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [3, 4, 5])
def test_covered_edge_test_matches_enumerated_equivalence_classes(d):
    """A DAG is alone in its class iff it has no covered edge (Chickering 1995).

    This is the whole reason GATE 1 survives past d=6: it is a per-graph test needing no
    comparison to any other graph. Verified exhaustively here, and separately confirmed on
    all 3,781,503 graphs at d=6 (2026-08-16), which is too slow for the suite.
    """
    space = build_graph_space(d, fast=True)
    assert np.array_equal(is_singleton_mec(space.dags), space.mec_sizes[space.mec_id] == 1)


def test_covered_edge_test_accepts_a_single_adjacency():
    """The collider A->B<-C is alone in its class; the chain A->B->C is not."""
    collider = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert is_singleton_mec(collider) is True
    assert is_singleton_mec(chain) is False


@pytest.mark.parametrize("d,p", [(4, 0.5), (5, 0.5), (6, 0.5), (5, 0.3)])
def test_sampled_singleton_fraction_is_unbiased(d, p):
    """Block 3 acceptance test, stated as a z-score rather than as CI containment.

    The pre-registered form was "the estimate's CI contains the exact value at d=4,5,6".
    That was a poorly designed test and it is recorded rather than quietly replaced: a 95%
    interval misses 5% of the time by construction, so across the nine configurations
    actually measured there was a ~37% chance of at least one miss even with a perfect
    estimator -- and on 2026-08-16 one config (d=5, p=0.3) duly missed by 0.00025.

    Chasing that miss produced a *wrong* diagnosis first (chains all starting from the
    empty graph, which is itself a singleton). Random initialisation did not help and the
    sign of the deviation flipped with burn-in, which is the signature of noise, not bias.
    The properly powered check across d=4,5,6 x p=0.3,0.5,0.7 gave max |z| = 1.86 and mean
    z = -0.34 -- no detectable bias at a standard error of ~0.0013.

    So the claim being tested is "unbiased", and that is what is asserted.
    """
    space = build_graph_space(d, fast=True)
    exact = prior_singleton_fraction(space, erdos_renyi_prior(space, p))

    result = estimate_singleton_fraction(d, p=p, n_chains=16, n_samples=1000, seed=3)
    per_chain = result["per_chain"]
    se = per_chain.std(ddof=1) / np.sqrt(len(per_chain))
    assert abs(result["estimate"] - exact) / se < 3.5, (
        f"z = {(result['estimate'] - exact) / se:+.2f}; exact {exact:.5f}, "
        f"estimate {result['estimate']:.5f} +- {se:.5f}")


def test_singleton_fraction_falls_as_graphs_get_denser():
    """Sanity direction: denser graphs sit in larger equivalence classes, so fewer of them
    are alone. Cheap, and it would catch a sign error in the prior that the z-test above
    could absorb into its tolerance."""
    values = [estimate_singleton_fraction(4, p=p, n_chains=8, n_samples=1000, seed=4)
              ["estimate"] for p in (0.3, 0.5, 0.7)]
    assert values[0] > values[1] > values[2]


# --------------------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [4, 5])
def test_sampled_oracle_scores_track_the_exact_oracle(d):
    """Scores, not just choices: a systematic distortion could preserve the argmax on easy
    beliefs and fail exactly on the close calls that decide the baseline."""
    from sa.posterior import PosteriorEngine
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")

    samples, intervened = _data(600, d, seed=d + 7, intervene=(0, 2))
    exact = InterventionOracle(space).scores(
        engine.posterior(samples, intervened, uniform_prior(space)))
    sampled = SamplingOracle(dp, n_draws=16000, seed=0).scores(
        dp.log_weights(samples, intervened))

    assert np.abs(exact - sampled).max() < 0.05
    assert sampled.shape == (d,)
    assert np.all(sampled >= -1e-12)


def test_sampled_oracle_shares_the_choice_logic_with_the_exact_one():
    """Ties returned as a set, random tie-breaking, and the `informative` flag are the
    parts whose absence produced a retracted result. Both oracles must inherit the same
    implementation rather than reimplementing it."""
    from sa.oracle import _OracleChoices
    assert issubclass(SamplingOracle, _OracleChoices)
    assert issubclass(InterventionOracle, _OracleChoices)
    for name in ("best_targets", "best_action", "score_choice"):
        assert getattr(SamplingOracle, name) is getattr(InterventionOracle, name)


def test_sampled_oracle_reports_no_preference_on_an_uninformative_belief():
    """With a posterior concentrated on one graph every target has zero entropy, so
    `informative` must be false -- the guard against the vacuous-optimality metric."""
    d = 4
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    log_w = np.full((d, dp.scorer.n_parent_sets), -1e3)
    log_w[:, 0] = 0.0                       # only the empty parent set is plausible
    result = SamplingOracle(dp, n_draws=2000, seed=0).score_choice(0, log_w)
    assert result["informative"] == 0.0


def test_sampled_oracle_is_reproducible_from_its_seed():
    d = 4
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    log_w = dp.log_weights(*_data(300, d, seed=5))
    first = SamplingOracle(dp, n_draws=2000, seed=11).scores(log_w)
    second = SamplingOracle(dp, n_draws=2000, seed=11).scores(log_w)
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------------------
# The edge-marginal greedy opponent on the DP path
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [4, 5])
def test_independent_edge_belief_matches_the_enumerated_one(d):
    """The opponent that makes d=7 comparable with d=4/5/6.

    Checked on the BELIEF rather than on the choices, because the belief is pure algebra
    -- the independent-edge product is modular, so building it as a log-weight table must
    reproduce the enumerated distribution exactly, not approximately. Any tolerance here
    would hide a real error; the Monte Carlo slack belongs to the oracle, which is tested
    separately.
    """
    from sa.baselines import EdgeMarginalGreedyDPPolicy, EdgeMarginalGreedyPolicy
    from sa.env import CausalDiscoveryEnv, EnvConfig
    from sa.posterior import PosteriorEngine

    space = build_graph_space(d, fast=True)
    env = CausalDiscoveryEnv(EnvConfig(d=d, n_obs=1500, budget=5), space=space)
    result = env.reset(seed=1)
    env.step(0)
    result = env.step(1)

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="erdos_renyi", p=0.5)
    log_w = dp.log_weights(env.samples, env.intervened)

    expected = EdgeMarginalGreedyPolicy(space, seed=0).approximate_posterior(result.posterior)
    table = EdgeMarginalGreedyDPPolicy(dp, n_draws=100, seed=0).approximate_belief(log_w)

    engine = PosteriorEngine(space, BGeScore(d))
    rebuilt = table.ravel()[engine._flat_ids].sum(axis=1)
    rebuilt = np.exp(rebuilt - rebuilt.max())
    rebuilt /= rebuilt.sum()
    assert np.abs(expected - rebuilt).max() < 1e-9


def test_independent_edge_weights_are_modular():
    """Every DAG's weight must factorise per node, which is what lets the DP and the
    sampler consume the approximation without a graph list. If it did not factorise the
    table would be silently wrong rather than raising."""
    from sa.dp import independent_edge_log_weights
    d = 4
    dp = DPPosterior.for_prior(d, BGeScore(d), kind="uniform")
    rng = np.random.default_rng(0)
    marginals = rng.uniform(0.05, 0.95, size=(d, d))
    np.fill_diagonal(marginals, 0.0)
    table = independent_edge_log_weights(marginals, dp.scorer, d)

    for node in range(d):
        for i, parents in enumerate(dp.scorer.parent_sets[node]):
            expected = sum(
                np.log(marginals[j, node]) if j in parents else np.log1p(-marginals[j, node])
                for j in range(d) if j != node)
            assert table[node, i] == pytest.approx(expected, abs=1e-12)


def test_both_greedy_baselines_exist_on_both_paths():
    """A missing baseline name fails only AFTER the expensive references are computed, and
    silently changes which opponent the headline is measured against."""
    from sa.backend import Backend
    from sa.env import EnvConfig
    required = {"random", "greedy_oracle", "edge_marginal_greedy", "no_intervention"}
    enumerated = Backend(EnvConfig(d=4), force_dp=False).make_baselines()
    sampled = Backend(EnvConfig(d=4), force_dp=True).make_baselines()
    assert required <= set(enumerated)
    assert required <= set(sampled)
