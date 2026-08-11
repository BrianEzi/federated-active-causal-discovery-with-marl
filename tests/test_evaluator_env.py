import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.types import SCMConfig, MechanismType, NoiseType, ActionCategory
from src.evaluator_env import FederatedCausalEnv

@pytest.fixture
def base_config():
    return SCMConfig(
        d=4,
        K=2,
        mechanism_type=int(MechanismType.LINEAR),
        noise_type=int(NoiseType.GAUSSIAN),
        noise_scale=0.1
    )

def test_env_initialization(base_config):
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0)
    
    assert env.max_steps == 20
    assert len(env.agent_masks) == 2
    
def test_env_reset(base_config):
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0)
    
    key = jax.random.PRNGKey(42)
    obs, info = env.reset(key)
    
    assert "agent_0" in obs
    assert "agent_1" in obs
    assert obs["agent_0"].shape == (env.obs_dim,) # Dynamic obs_dim (65 with obs_feedback=True)
    assert "true_adjacency" in info

def test_env_step(base_config):
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0)
    
    key = jax.random.PRNGKey(42)
    obs, info = env.reset(key)
    
    # Agent 0 does Local Intervention on node 0
    # Agent 1 does NOOP on node 0
    joint_actions = {
        "agent_0": (int(ActionCategory.LOCAL_INTERVENTION), 0),
        "agent_1": (int(ActionCategory.NOOP), 0)
    }
    
    predicted_dags = {
        "agent_0": np.zeros((4, 4)),
        "agent_1": np.zeros((4, 4))
    }
    
    next_obs, rewards, done, next_info = env.step(joint_actions, predicted_dags, key)
    
    assert not done
    # Agent 0 budget decreased by 0.5
    assert next_obs["agent_0"][-1] == 9.5
    # Agent 1 budget unchanged
    assert next_obs["agent_1"][-1] == 10.0
    
    assert "agent_0" in rewards
    assert "agent_1" in rewards
    assert "info_gains" in next_info
    assert "agent_0" in next_info["info_gains"]
    assert "agent_1" in next_info["info_gains"]
    assert next_info["info_gains"]["agent_0"] >= 0.0
