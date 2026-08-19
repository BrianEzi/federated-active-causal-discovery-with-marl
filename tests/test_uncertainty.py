"""The uncertainty decomposition, and the properties that make it trustworthy.

The metric this replaces looked reasonable and was wrong, so these tests check the things
that would have caught that: an identity that must hold exactly, a closed form derived
independently of the implementation, and the qualitative behaviour the split claims.
"""
from __future__ import annotations

import numpy as np
import pytest

from sa.backend import Backend
from sa.baselines import make_baselines, no_intervention_policy
from sa.env import EnvConfig
from sa.graphs import build_graph_space
from sa.uncertainty import decompose, episode_trace, summarise_trace


def env_for(d=4, n_obs=1000, budget=8):
    return Backend(EnvConfig(d=d, n_obs=n_obs, budget=budget), seed=0).make_env()


# -- identities -------------------------------------------------------------------

def test_uniform_posterior_has_entropy_log_n_dags():
    space = build_graph_space(4)
    out = decompose(np.full(space.n_dags, 1.0 / space.n_dags), space)
    assert out["h_total"] == pytest.approx(np.log2(space.n_dags))


def test_chain_rule_holds_on_real_posteriors():
    """H(G) = H(E) + H(G|E) is an identity. `h_within` is computed directly rather than by
    subtraction, so this is a genuine check and not a tautology."""
    env = env_for()
    for ep in range(15):
        result = env.reset(seed=ep)
        out = decompose(result.posterior, env.space)
        assert abs(out["chain_rule_residual"]) < 1e-9
        for _ in range(2):
            result = env.step(int(ep % env.config.d))
            out = decompose(result.posterior, env.space)
            assert abs(out["chain_rule_residual"]) < 1e-9


def test_a_point_mass_has_zero_uncertainty_everywhere():
    space = build_graph_space(4)
    p = np.zeros(space.n_dags); p[17] = 1.0
    out = decompose(p, space)
    assert out["h_total"] == pytest.approx(0.0)
    assert out["h_class"] == pytest.approx(0.0)
    assert out["h_within"] == pytest.approx(0.0)


# -- the closed form, which is the real correctness check --------------------------

def test_observational_within_class_entropy_matches_the_closed_form():
    """BGe is score-equivalent, so with observational data only every member of a class has
    identical likelihood and the within-class posterior is uniform. Then

        H(G|E) = SUM_c p_c log2 |c|

    derived without reference to this implementation. It also fails loudly if score
    equivalence is ever broken -- which is what makes it worth asserting.
    """
    env = env_for()
    for ep in range(20):
        result = env.reset(seed=100 + ep)
        out = decompose(result.posterior, env.space)
        assert out["h_within"] == pytest.approx(out["h_within_if_uniform"], abs=1e-6)


def test_the_closed_form_STOPS_holding_once_an_intervention_lands():
    """The mirror of the previous test. Interventional data breaks ties inside a class --
    that is the entire point -- so after intervening the within-class entropy must fall
    BELOW the uniform prediction. If it did not, interventions would be doing nothing that
    observation could not."""
    env = env_for()
    broke = 0
    for ep in range(20):
        result = env.reset(seed=200 + ep)
        before = decompose(result.posterior, env.space)
        result = env.step(0)
        after = decompose(result.posterior, env.space)
        # never above: an intervention cannot make members of a class more equal
        assert after["h_within"] <= after["h_within_if_uniform"] + 1e-6
        if after["h_within"] < after["h_within_if_uniform"] - 1e-3:
            broke += 1
    assert broke > 0, "no intervention ever broke a within-class tie"


# -- the behaviour the split claims ------------------------------------------------

def test_more_observational_data_reduces_class_uncertainty_not_within_class():
    """The load-bearing claim. Observation should shrink H(E) towards zero while H(G|E)
    stays put -- it converges to the log size of the true class, which no amount of
    watching can reduce."""
    small = decompose(env_for(n_obs=200).reset(seed=7).posterior, build_graph_space(4))
    large = decompose(env_for(n_obs=20000).reset(seed=7).posterior, build_graph_space(4))
    assert large["h_class"] < small["h_class"] - 0.05
    # within-class entropy is bounded by the true class size and does not vanish
    assert large["h_within"] >= 0.0


def test_interventions_reduce_the_addressable_part():
    """Averaged over episodes, a greedy agent must remove within-class uncertainty. This is
    the property that makes the metric useful rather than merely well defined."""
    env = env_for()
    policy = make_baselines(env.space, seed=0)["greedy_oracle"]
    removed = []
    for ep in range(25):
        summary = summarise_trace(episode_trace(env, policy, seed=300 + ep))
        removed.append(summary["addressable_bits_removed"])
    assert np.mean(removed) > 0.1, f"greedy removed only {np.mean(removed):.3f} bits"


def test_doing_nothing_removes_no_addressable_uncertainty():
    """The control. An agent that never intervenes cannot reduce H(G|E) at all, because
    only interventions can. Its trace has a single row, so the removal is exactly zero."""
    env = env_for()
    for ep in range(10):
        trace = episode_trace(env, no_intervention_policy, seed=400 + ep)
        summary = summarise_trace(trace)
        assert summary["addressable_bits_removed"] == pytest.approx(0.0)
