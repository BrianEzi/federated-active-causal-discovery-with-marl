import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legacy.src.types import SCMConfig, MechanismType, NoiseType, ActionCategory
from legacy.src.evaluator_env import FederatedCausalEnv

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
        "agent_0": (int(ActionCategory.INTERVENE), 0),
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

def test_obs_feedback_uses_edge_authority_mask_not_observation_mask(base_config):
    """
    Regression test: the predicted-DAG observation-feedback slice (obs_feedback=True) must
    be masked by each agent's narrower edge-authority mask (own local domain + shared
    boundary), not the wider observation mask. The observation mask alone would pass
    through a Z1<->X2 cell for Agent 0 (both nodes are within its obs_masks[0]), even
    though Z1<->X2 crosses a domain it has no authority over.

    self.last_predicted_dag is forced directly (bypassing predict_graph_hypothesis's own
    structural_mask) so this test isolates _get_obs_dict's own masking specifically --
    it must not pass only because an earlier layer already zeroed the value.
    """
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0, obs_feedback=True)
    key = jax.random.PRNGKey(0)
    env.reset(key)

    d = base_config.d
    forced = np.ones((d, d), dtype=np.float32)  # saturate every cell, including forbidden ones
    np.fill_diagonal(forced, 0.0)
    env.last_predicted_dag = forced

    obs_dict = env._get_obs_dict()

    d2 = d * d
    pred_dag_slice_0 = obs_dict["agent_0"][3 * d2: 4 * d2].reshape(d, d)
    pred_dag_slice_1 = obs_dict["agent_1"][3 * d2: 4 * d2].reshape(d, d)

    # Agent 0: Z1(0)<->X2(2) must not leak, even though X2 is within agent_0's obs_masks.
    assert pred_dag_slice_0[0, 2] == 0.0 and pred_dag_slice_0[2, 0] == 0.0
    # Agent 1: X1(1)<->Z2(3) must not leak, even though X1 is within agent_1's obs_masks.
    assert pred_dag_slice_1[1, 3] == 0.0 and pred_dag_slice_1[3, 1] == 0.0
    # Legitimate edges must still pass through.
    assert pred_dag_slice_0[0, 1] == 1.0  # Z1<->X1, agent 0's own local pair
    assert pred_dag_slice_0[1, 2] == 1.0  # X1<->X2, shared boundary
    assert pred_dag_slice_1[2, 3] == 1.0  # X2<->Z2, agent 1's own local pair
