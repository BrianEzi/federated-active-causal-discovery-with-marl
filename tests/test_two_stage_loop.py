import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.types import SCMConfig, MechanismType, NoiseType, InterventionType
from src.evaluator_env import FederatedCausalEnv, build_intervention_spec_jitted

def test_soft_shift_intervention_spec():
    budgets = jnp.array([10.0, 10.0])
    costs = jnp.array([1.0, 1.0])
    agent_masks = jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    
    mask, types, values, costs_out = build_intervention_spec_jitted(
        jnp.array(0), jnp.array(1), jnp.array(2), jnp.array(0),
        budgets, costs, agent_masks, 4, int(InterventionType.SOFT_SHIFT), 2.0
    )
    assert mask[1] == 1.0
    assert types[1] == int(InterventionType.SOFT_SHIFT)
    assert values[1] == 2.0
    assert costs_out[0] == 1.0

def test_two_stage_env_initialization_and_obs_feedback():
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    action_costs = np.array([1.0, 1.0])
    
    env = FederatedCausalEnv(
        config, action_costs,
        estimator_type="analytic",
        intervention_type="soft_shift",
        obs_feedback=True
    )
    
    # Obs dim: 3*d*d + d + 1 + d*d (obs_cov+run_cov+asym+node_intervention_counts+budget+pred_dag)
    #        = 3*16 + 4 + 1 + 16 = 69
    assert env.obs_dim == 69
    
    obs, info = env.reset(jax.random.PRNGKey(42))
    assert "agent_0" in obs
    assert "agent_1" in obs
    assert obs["agent_0"].shape == (69,)
    assert obs["agent_1"].shape == (69,)
    assert "true_adjacency" in info

def test_two_stage_env_step_execution():
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    action_costs = np.array([1.0, 1.0])
    
    env = FederatedCausalEnv(
        config, action_costs,
        estimator_type="analytic",
        intervention_type="soft_shift",
        obs_feedback=True,
        reward_density="dense"
    )
    
    obs, info = env.reset(jax.random.PRNGKey(42))
    
    joint_actions = {
        "agent_0": (0, 1), # Local intervention on X1
        "agent_1": (2, 0)  # NOOP
    }
    
    next_obs, rewards, done, step_info = env.step(joint_actions, predicted_dags=None, key=jax.random.PRNGKey(43))
    
    assert "agent_0" in rewards
    assert "agent_1" in rewards
    assert isinstance(done, bool)
    assert "predicted_dag" in step_info
    assert step_info["predicted_dag"].shape == (4, 4)
