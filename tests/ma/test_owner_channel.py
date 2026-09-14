"""The policy must be able to see WHO it blames, not merely THAT a pair is confounded.

`AttributedBelief.owner_channel` has existed since the attributed backend landed and its
docstring says it "replaces the single bidirected channel in the observation". Nothing ever
called it, so every attributed run in the repo trained a policy that saw "this pair is
confounded" and never "and I blame agent 2" -- the same blindfold `_belief_channels` records
fixing one level down for confounding itself.
"""
import numpy as np
import pytest

from ma.env import ATTRIBUTED, MAConfig, TwoAgentEnv
from ma.topology import federated_topology


def _env(owner: bool, agents=3, arch="gnn"):
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(agents, 2, 3), n_obs=60, n_int=20, budget=12,
        turn_order="round_robin", belief_backend=ATTRIBUTED, action_modes=("vary",),
        claim_bar=1.0, reward_criterion="claims", policy_arch=arch, graph_model="sf",
        sf_m=2, episode_mix="confounded", vs_evidence="oracle", disclose_regime=True,
        observe_belief_channels=True, observe_owner_channel=owner))


def test_declared_obs_size_matches_the_vector_actually_produced():
    """A mismatch here is the failure that kills a run at load time, hours in."""
    for owner in (False, True):
        env = _env(owner)
        env.reset(seed=0)
        for agent in env.topology.agents:
            assert env.obs_size(agent) == len(env.observation(agent))


def test_the_channel_adds_exactly_one_column_per_pair_per_agent():
    off, on = _env(False), _env(True)
    off.reset(seed=0); on.reset(seed=0)
    k = off.windows[0].k
    expected = (k * (k - 1) // 2) * off.topology.n_agents
    assert on.obs_size(0) - off.obs_size(0) == expected


def test_it_carries_real_ownership_belief_not_zeros():
    """A channel of zeros would pass every width check and teach the policy nothing."""
    env = _env(True)
    env.reset(seed=0)
    k, n = env.windows[0].k, env.topology.n_agents
    tail = env.observation(0)[-(k * (k - 1) // 2) * n:]
    assert np.count_nonzero(tail) > 0


def test_it_is_off_by_default_so_existing_checkpoints_still_load():
    env = TwoAgentEnv(MAConfig(
        topology=federated_topology(3, 2, 3), n_obs=60, n_int=20, budget=12,
        belief_backend=ATTRIBUTED, action_modes=("vary",), claim_bar=1.0,
        reward_criterion="claims", policy_arch="gnn", observe_belief_channels=True))
    assert env.config.observe_owner_channel is False


def test_width_is_stable_before_the_belief_exists():
    """Reset builds windows before any belief is populated; the zero-fill must be the same
    width as the populated channel or the first observation of every episode is malformed."""
    env = _env(True)
    k, n = env.windows[0].k, env.topology.n_agents
    blank = env._belief_channels(0)
    env.reset(seed=0)
    assert len(blank) == len(env._belief_channels(0)) == k * (k - 1) + (k * (k - 1) // 2) * n


def test_a_non_attributed_backend_degrades_to_zeros_rather_than_raising():
    """The factored belief has no `owner_channel`. Turning the flag on there is a
    configuration mistake, but it must not take a run down mid-episode."""
    env = TwoAgentEnv(MAConfig(
        topology=federated_topology(3, 2, 3), n_obs=60, n_int=20, budget=12,
        belief_backend="factored", action_modes=("vary",), claim_bar=1.0,
        reward_criterion="claims", policy_arch="gnn", graph_model="sf", sf_m=2,
        vs_evidence="oracle", observe_belief_channels=True, observe_owner_channel=True))
    env.reset(seed=0)
    assert env.obs_size(0) == len(env.observation(0))
