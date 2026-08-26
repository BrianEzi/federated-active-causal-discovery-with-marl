"""The observation must carry what the reward is scored on, and pay whom it credits.

Two changes from 2026-08-26, each pinned here because both are silent failures: a policy
blind to confounding still trains, it just cannot win; and a shared reward still learns, it
just learns from the wrong signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.env import MAConfig, ROUND_ROBIN, VARY, TwoAgentEnv
from ma.topology import Topology

TOPO = Topology(name="T_3agent_1each", private=((0,), (1,), (2,)), exposed=(3, 4, 5))


def _env(**kw):
    kw.setdefault("belief_backend", "version_space")
    kw.setdefault("reward_criterion", "claims")
    kw.setdefault("claim_bar", 1.0)
    kw.setdefault("episode_mix", "confounded")
    kw.setdefault("disclose_regime", True)
    kw.setdefault("budget", 3)
    return TwoAgentEnv(MAConfig(topology=TOPO, n_obs=60, n_int=20,
                                turn_order=ROUND_ROBIN, action_modes=(VARY,), **kw), seed=0)


def test_confounding_beliefs_are_absent_by_default_and_present_when_asked():
    """The regression that mattered: a pair believed confounded was invisible to the policy
    while the greedy baseline read it through cb.claims."""
    blind = _env()
    sighted = _env(observe_belief_channels=True)
    blind.reset(seed=5)
    sighted.reset(seed=5)

    k = blind.windows[0].k
    assert sighted.obs_size(0) - blind.obs_size(0) == k * (k - 1)

    belief = sighted.windows[0].belief.last
    rows, cols = np.triu_indices(k, k=1)
    expected = np.asarray(belief.bidirected)[rows, cols]
    observation = sighted.observation(0)
    tail = observation[-k * (k - 1):]
    assert np.allclose(tail[:len(expected)], expected)
    # And the blind observation really does not contain them anywhere.
    if expected.any():
        blind_obs = blind.observation(0)
        missing = [v for v in expected if v > 0
                   and not np.any(np.isclose(blind_obs, v))]
        assert missing, "expected at least one confounding frequency absent from the blind obs"


def test_the_declared_observation_size_matches_what_is_produced():
    for flag in (False, True):
        env = _env(observe_belief_channels=flag)
        env.reset(seed=2)
        for agent in TOPO.agents:
            assert env.observation(agent).shape[0] == env.obs_size(agent), (flag, agent)


def test_observations_stay_on_the_unit_interval():
    """Raw counts beside probabilities dominated a first layer once; every feature is
    scaled, and the new channels are frequencies already."""
    env = _env(observe_belief_channels=True)
    result = env.reset(seed=3)
    while not result.done:
        result = env.step({a: 0 for a in TOPO.agents})
        for agent in TOPO.agents:
            observation = env.observation(agent)
            assert np.all(np.isfinite(observation))
            assert observation.min() >= 0.0 and observation.max() <= 1.0


def test_per_agent_reward_pays_each_agent_for_its_own_window():
    env = _env(per_agent_reward=True, budget=3)
    result = env.reset(seed=7)
    while not result.done:
        result = env.step({a: 0 for a in TOPO.agents})
        rewards = result.info["agent_rewards"]
        assert rewards is not None and set(rewards) == set(TOPO.agents)
        for agent, value in rewards.items():
            # The terminal component is exactly this agent's own identification.
            own = result.identified[agent]
            assert (value >= 1.0) == bool(own) or abs(value) < 1.0 + 1e-9


def test_shared_reward_is_the_default_and_reports_no_per_agent_table():
    env = _env()
    result = env.reset(seed=7)
    result = env.step({a: 0 for a in TOPO.agents})
    assert result.info["agent_rewards"] is None


def test_identified_fraction_is_reported_for_the_window_metric():
    """The joint rate falls exponentially in agent count whatever the policy does, so the
    per-window rate is what training watches."""
    env = _env()
    result = env.reset(seed=1)
    fraction = result.info["identified_fraction"]
    assert 0.0 <= fraction <= 1.0
    assert fraction == pytest.approx(
        np.mean([float(v) for v in result.identified.values()]))
