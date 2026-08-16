"""The DP environment must be the same environment, not a similar one.

`DPCausalDiscoveryEnv` exists to run at d=7, where nothing can be checked against anything.
Its entire claim to trustworthiness is that at d=4 and d=5, where the enumerated
environment also exists, the two produce **the same numbers step for step** on a shared
seed. That equivalence is what these tests pin.
"""
import numpy as np
import pytest

from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig
from sa.env_dp import DPCausalDiscoveryEnv
from sa.graphs import build_graph_space, is_singleton_mec
from sa.priors import erdos_renyi_prior, prior_singleton_fraction


def _config(d, **kwargs):
    base = dict(d=d, n_obs=1000, n_int=100, budget=5, prior="erdos_renyi", prior_p=0.5)
    base.update(kwargs)
    return EnvConfig(**base)


@pytest.mark.parametrize("d", [4, 5])
def test_matches_the_enumerated_environment_step_for_step(d):
    """The acceptance test for the whole DP path.

    Same true graph, same seed, same actions -- so the SCM draws are identical and any
    difference is the belief machinery alone. Compares every quantity that exists in both:
    the true DAG's posterior mass, the edge-marginal observation, and all three episode
    flags.
    """
    cfg = _config(d)
    space = build_graph_space(d, fast=True)
    enumerated = CausalDiscoveryEnv(cfg, space=space)
    dp_env = DPCausalDiscoveryEnv(cfg)

    worst_mass = worst_marginal = 0.0
    for trial in range(6):
        index = int(np.random.default_rng(trial).integers(space.n_dags))
        a = enumerated.reset(seed=trial, force_index=index)
        b = dp_env.reset(seed=trial, force_adjacency=space.dags[index])
        assert np.array_equal(enumerated.samples, dp_env.samples), "SCM draws diverged"

        for step in range(cfg.budget):
            worst_mass = max(worst_mass, abs(a.info["true_mass"] - b.info["true_mass"]))
            worst_marginal = max(worst_marginal, np.abs(
                enumerated.observation("edge_marginals")
                - dp_env.observation("edge_marginals")).max())
            assert a.identified == b.identified
            assert a.done == b.done
            assert a.info["is_singleton"] == b.info["is_singleton"]
            if a.done:
                break
            a = enumerated.step(step % d)
            b = dp_env.step(step % d)

    assert worst_mass < 1e-9
    assert worst_marginal < 1e-9


def test_passing_ends_the_episode_the_same_way():
    cfg = _config(4)
    space = build_graph_space(4, fast=True)
    dp_env = DPCausalDiscoveryEnv(cfg)
    dp_env.reset(seed=0, force_adjacency=space.dags[10])
    result = dp_env.step(PASS_ACTION)
    assert result.info["passed"] and result.done and result.n_interventions == 0


def test_true_graphs_are_drawn_from_the_prior():
    """The graphs come from an MH chain, not from a list, so the distribution they follow
    is a claim that needs checking. The singleton fraction is the right statistic: it is
    exactly what GATE 1 compares against, so a biased graph sampler would corrupt the gate
    rather than merely the flavour of the episodes.
    """
    d = 4
    cfg = _config(d, n_obs=10)          # tiny: only the graph draw matters here
    space = build_graph_space(d, fast=True)
    expected = prior_singleton_fraction(space, erdos_renyi_prior(space, 0.5))

    env = DPCausalDiscoveryEnv(cfg)
    flags = []
    for episode in range(400):
        flags.append(env.reset(seed=episode).info["is_singleton"])
    observed = float(np.mean(flags))
    # 400 draws, p ~ 0.11: standard error ~0.016, so 4 SE is ~0.06.
    assert abs(observed - expected) < 0.06, f"{observed:.3f} vs prior {expected:.3f}"


def test_consecutive_episodes_do_not_reuse_the_same_graph():
    """The chain is advanced between episodes rather than restarted. If it were not
    advanced far enough, consecutive episodes would share a true graph and every result
    would be correlated in a way nothing downstream would reveal."""
    env = DPCausalDiscoveryEnv(_config(5, n_obs=10))
    graphs = [env.reset(seed=e).info["true_adjacency"].copy() for e in range(30)]
    repeats = sum(np.array_equal(graphs[i], graphs[i + 1]) for i in range(len(graphs) - 1))
    assert repeats < 10, f"{repeats} of 29 consecutive pairs identical"


def test_the_enumerated_posterior_observation_is_refused_not_faked():
    """It cannot exist at the sizes this class is for, and returning something plausible
    instead would silently change what the agent is being trained on."""
    env = DPCausalDiscoveryEnv(_config(4, n_obs=10))
    env.reset(seed=0)
    with pytest.raises(ValueError, match="does not exist"):
        env.observation("posterior")


def test_non_modular_prior_is_refused_at_construction():
    """Fails when the environment is built, not on some later step, so a misconfigured run
    cannot get as far as producing numbers."""
    with pytest.raises(ValueError, match="not modular"):
        DPCausalDiscoveryEnv(_config(4, prior="scale_free"))


def test_observation_dimension_matches_what_is_returned():
    for include_counts in (False, True):
        env = DPCausalDiscoveryEnv(_config(5, n_obs=10, include_counts=include_counts))
        env.reset(seed=0)
        assert env.observation("edge_marginals").shape == (
            env.observation_dim["edge_marginals"],)


def test_singleton_flag_agrees_with_the_enumerated_class_size():
    """`is_singleton` switches from a class-size lookup to the covered-edge test; the two
    must agree, or GATE 1 measures a different quantity on the two paths."""
    d = 5
    space = build_graph_space(d, fast=True)
    rng = np.random.default_rng(0)
    for index in rng.choice(space.n_dags, size=200, replace=False):
        assert bool(is_singleton_mec(space.dags[index])) == \
            bool(space.mec_sizes[space.mec_id[index]] == 1)


def test_runs_at_d7_where_nothing_can_be_enumerated():
    """The point of the exercise. 1.14 billion DAGs; no list is built anywhere."""
    env = DPCausalDiscoveryEnv(_config(7, n_obs=500, budget=3))
    result = env.reset(seed=0)
    assert 0.0 <= result.info["true_mass"] <= 1.0
    result = env.step(2)
    assert result.n_interventions == 1
    assert env.observation("edge_marginals").shape == (7 * 6 + 1,)
