import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.pag import PAGTracker
from src.types import SCMConfig, MechanismType, NoiseType
from src.generators import generate_er_dag, generate_scm_params
from src.evaluator_env import FederatedCausalEnv

def test_pag_orientation():
    """Test 1: Verify PAGTracker correctly converts Xi o-o Xj to Xi -> Xj."""
    d = 4
    tracker = PAGTracker(d)
    
    assert tracker.count_circle_marks() == d * (d - 1)
    
    # Simulate intervention on node 0, and say node 0 and node 1 are dependent (p_value < 0.05)
    # while node 0 and node 2 are independent (p_value > 0.05).
    intervened_nodes = [0]
    p_values = np.ones((d, d))
    p_values[0, 1] = 0.01
    p_values[1, 0] = 0.01
    
    tracker.update_pag_from_intervention(intervened_nodes, p_values, threshold=0.05)
    
    # 0 -> 1 should be TAIL -> ARROW
    assert tracker.P[0, 1] == PAGTracker.TAIL
    assert tracker.P[1, 0] == PAGTracker.ARROW
    
    # 0 -> 2 should be NULL (edge deleted)
    assert tracker.P[0, 2] == PAGTracker.NULL
    assert tracker.P[2, 0] == PAGTracker.NULL
    
    # Circle marks should have decreased
    assert tracker.count_circle_marks() < d * (d - 1)

def test_reward_strictly_positive():
    """Test 2: Assert that orienting an edge produces a strictly positive reward."""
    from src.rewards import compute_global_reward
    
    prev_circles = 12
    curr_circles = 10
    action_costs = np.array([0.1, 0.1])
    violations = 0
    
    reward = compute_global_reward(prev_circles, curr_circles, None, action_costs, violations)
    
    # delta circles = 2, costs = 0.2, violations = 0 => reward = 1.8
    assert reward > 0
    assert reward == pytest.approx(1.8)

def test_federated_causal_env():
    """Test 3: Run a random episode and assert step output shapes match standards."""
    d = 4
    K = 2
    
    config = SCMConfig(
        d=d,
        K=K,
        mechanism_type=MechanismType.LINEAR,
        noise_type=NoiseType.GAUSSIAN,
        noise_scale=0.1
    )
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    adj = generate_er_dag(k1, d, edge_prob=0.8)
    scm_params = generate_scm_params(k2, adj, MechanismType.LINEAR)
    topological_order = jnp.arange(d)
    
    agent_masks = jnp.array([
        [1., 1., 1., 0.],
        [0., 1., 1., 1.]
    ])
    action_costs = jnp.array([1.0, 1.0])
    
    env = FederatedCausalEnv(config, adj, scm_params, topological_order, agent_masks, action_costs)
    
    obs_dict, info = env.reset(key)
    
    assert "agent_0" in obs_dict
    assert "agent_1" in obs_dict
    assert obs_dict["agent_0"].shape == (2 * d * d + d + 1,)
    assert "pag" in info
    
    for i in range(10):
        # joint action: agent 0 intervenes on node 0, agent 1 intervenes on node 2
        joint_actions = {
            "agent_0": 0,
            "agent_1": 2
        }
        
        step_key = jax.random.fold_in(key, i)
        next_obs, reward, terminated, next_info = env.step(joint_actions, step_key)
        
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "pag" in next_info
        
        if terminated:
            break
