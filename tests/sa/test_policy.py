"""Tests for the PPO agent.

Mechanics only -- whether it *learns well* is what `scripts/run_experiment.py` measures
against the pinned criteria, and that is not something a unit test should assert.

`test_observation_features_are_on_a_comparable_scale` encodes a real bug: the budget
feature was a raw count sitting at 20.0 while posterior entries averaged 0.04, a ~500x
mismatch that saturated the tanh trunk and drowned out the belief the agent acts on.
"""
import numpy as np
import pytest
import torch

from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig
from sa.graphs import build_graph_space
from sa.policy import ActorCritic, PPOAgent, PPOConfig, action_to_env


@pytest.fixture(scope="module")
def space3():
    return build_graph_space(3)


@pytest.fixture(scope="module")
def agent(space3):
    return PPOAgent(EnvConfig(d=3, n_obs=200, budget=10),
                    PPOConfig(total_episodes=64, episodes_per_update=32, seed=0),
                    space=space3)


# --- action mapping -------------------------------------------------------------------

def test_last_action_index_means_pass():
    assert action_to_env(3, 3) == PASS_ACTION
    assert action_to_env(0, 3) == 0
    assert action_to_env(2, 3) == 2


def test_action_space_has_one_slot_per_node_plus_pass(agent):
    assert agent.n_actions == agent.d + 1


# --- network --------------------------------------------------------------------------

def test_network_starts_near_uniform():
    """Small final-layer weights matter: a sharp initial policy never explores."""
    net = ActorCritic(obs_dim=10, n_actions=4)
    logits, _ = net(torch.zeros(10))
    assert float(logits.detach().abs().max()) < 0.1


def test_forward_shapes():
    net = ActorCritic(obs_dim=7, n_actions=4)
    logits, value = net(torch.zeros((5, 7)))
    assert logits.shape == (5, 4) and value.shape == (5,)


# --- observations -----------------------------------------------------------------------

def test_observation_features_are_on_a_comparable_scale(space3):
    """Regression test on the scale mismatch that stopped learning entirely."""
    env = CausalDiscoveryEnv(EnvConfig(d=3, n_obs=200, budget=20), space=space3)
    env.reset(seed=0)
    for kind in ("posterior", "edge_marginals"):
        obs = env.observation(kind)
        assert obs.min() >= -1e-9 and obs.max() <= 1.0 + 1e-9, (
            f"{kind}: features outside [0, 1] ({obs.min():.3f} to {obs.max():.3f}); the "
            f"budget feature was previously a raw count of 20 alongside ~0.04 probabilities."
        )


def test_agent_observation_dim_matches_the_environment(space3):
    for kind in ("posterior", "edge_marginals"):
        a = PPOAgent(EnvConfig(d=3, n_obs=200), PPOConfig(observation=kind, seed=0),
                     space=space3)
        assert a.obs_dim == a.env.observation_dim[kind]


# --- acting ---------------------------------------------------------------------------

def test_deterministic_action_is_reproducible(agent):
    obs = np.zeros(agent.obs_dim, dtype=np.float32)
    first, _, _ = agent.act(obs, deterministic=True)
    second, _, _ = agent.act(obs, deterministic=True)
    assert first == second


def test_sampled_actions_vary(agent):
    obs = np.zeros(agent.obs_dim, dtype=np.float32)
    torch.manual_seed(0)
    actions = {agent.act(obs)[0] for _ in range(60)}
    assert len(actions) > 1, "a near-uniform initial policy must not be deterministic"


def test_as_policy_returns_valid_environment_actions(agent, space3):
    env = CausalDiscoveryEnv(agent.env_config, space=space3)
    result = env.reset(seed=0)
    policy = agent.as_policy(deterministic=True)
    action = policy(env, result)
    assert action == PASS_ACTION or 0 <= action < agent.d


# --- training loop ----------------------------------------------------------------------

def test_training_runs_and_records_history(agent):
    history = agent.train()
    assert history and {"episodes", "solve_rate", "entropy"} <= set(history[0])
    assert np.isfinite(history[-1]["entropy"])


def test_advantages_are_normalised_and_returns_are_not(agent):
    """Returns must stay on the reward scale: normalising them would corrupt the value
    targets, a bug this project has hit before."""
    rewards = np.array([0.1, 0.2, 1.0, -0.05], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    advantages, returns = agent._advantages(rewards, values, dones)
    assert abs(float(advantages.mean())) < 1e-5
    assert abs(float(advantages.std()) - 1.0) < 1e-3
    assert float(returns.max()) > 1.0  # not squashed to zero mean


def test_advantages_do_not_bootstrap_across_episode_boundaries(agent):
    """A terminal step's advantage must not borrow value from the next episode."""
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    done_then_not = agent._advantages(rewards, values, np.array([1.0, 0.0], dtype=np.float32))[1]
    assert done_then_not[0] == pytest.approx(1.0)
