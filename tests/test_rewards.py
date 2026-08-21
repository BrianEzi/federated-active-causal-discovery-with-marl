import numpy as np
import jax
import jax.numpy as jnp
import pytest
from legacy.src.rewards import compute_ippo_rewards, jitted_compute_ippo_rewards

def test_compute_ippo_rewards_perfect():
    stitched_dag = np.zeros((4, 4))
    true_dag = np.zeros((4, 4))
    stitched_dag[0, 1] = 1
    true_dag[0, 1] = 1
    
    rewards = compute_ippo_rewards(stitched_dag, true_dag, has_cycle=False)
    
    assert rewards["agent_0"] == 0.0
    assert rewards["agent_1"] == 0.0

def test_compute_ippo_rewards_local_error_agent1():
    stitched_dag = np.zeros((4, 4))
    true_dag = np.zeros((4, 4))
    # Agent 1 hallucinates an edge from Z1 (0) to X1 (1)
    stitched_dag[0, 1] = 1
    
    rewards = compute_ippo_rewards(stitched_dag, true_dag, has_cycle=False)
    
    assert rewards["agent_0"] == -1.0 # Penalized for node 0
    assert rewards["agent_1"] == 0.0  # Unaffected

def test_compute_ippo_rewards_boundary_error():
    stitched_dag = np.zeros((4, 4))
    true_dag = np.zeros((4, 4))
    # Boundary edge missed (X1 -> X2)
    true_dag[1, 2] = 1
    
    rewards = compute_ippo_rewards(stitched_dag, true_dag, has_cycle=False)
    
    assert rewards["agent_0"] == -1.0 # Both share the boundary penalty
    assert rewards["agent_1"] == -1.0

def test_compute_ippo_rewards_cycle_penalty():
    stitched_dag = np.zeros((4, 4))
    true_dag = np.zeros((4, 4))
    
    rewards = compute_ippo_rewards(stitched_dag, true_dag, has_cycle=True, cycle_penalty=10.0)
    
    assert rewards["agent_0"] == -10.0
    assert rewards["agent_1"] == -10.0

def test_compute_ippo_rewards_normalized():
    stitched_dag = np.zeros((4, 4))
    true_dag = np.zeros((4, 4))
    true_dag[1, 2] = 1
    
    # max_steps = 20.0
    rewards = compute_ippo_rewards(stitched_dag, true_dag, has_cycle=False, max_steps=20.0)
    assert rewards["agent_0"] == -1.0 / 20.0
    assert rewards["agent_1"] == -1.0 / 20.0
    
    # JAX version
    r0, r1 = jitted_compute_ippo_rewards(jnp.array(stitched_dag), jnp.array(true_dag), jnp.array(False), max_steps=20.0)
    assert np.isclose(float(r0), -1.0 / 20.0)
    assert np.isclose(float(r1), -1.0 / 20.0)

def test_compute_ippo_rewards_intrinsic():
    stitched_dag = np.zeros((4, 4))
    true_dag = np.zeros((4, 4))
    true_dag[1, 2] = 1
    
    info_gains_np = {"agent_0": 0.40, "agent_1": 0.10}
    rewards = compute_ippo_rewards(
        stitched_dag, true_dag, has_cycle=False, max_steps=20.0,
        info_gains=info_gains_np, intrinsic_coef=0.05
    )
    assert np.isclose(rewards["agent_0"], -1.0 / 20.0 + 0.05 * 0.40)
    assert np.isclose(rewards["agent_1"], -1.0 / 20.0 + 0.05 * 0.10)
    
    # JAX single [d, d]
    info_gains_jax = jnp.array([0.40, 0.10])
    r0, r1 = jitted_compute_ippo_rewards(
        jnp.array(stitched_dag), jnp.array(true_dag), jnp.array(False),
        max_steps=20.0, info_gains=info_gains_jax, intrinsic_coef=0.05
    )
    assert np.isclose(float(r0), -1.0 / 20.0 + 0.05 * 0.40)
    assert np.isclose(float(r1), -1.0 / 20.0 + 0.05 * 0.10)
    
    # JAX batched [B, d, d]
    batch_stitched = jnp.stack([jnp.array(stitched_dag), jnp.array(stitched_dag)])
    batch_true = jnp.stack([jnp.array(true_dag), jnp.array(true_dag)])
    batch_cycle = jnp.array([False, False])
    batch_ig = jnp.array([[0.40, 0.10], [0.80, 0.20]])
    br0, br1 = jitted_compute_ippo_rewards(
        batch_stitched, batch_true, batch_cycle,
        max_steps=20.0, info_gains=batch_ig, intrinsic_coef=0.05
    )
    assert np.isclose(float(br0[0]), -1.0 / 20.0 + 0.05 * 0.40)
    assert np.isclose(float(br1[0]), -1.0 / 20.0 + 0.05 * 0.10)
    assert np.isclose(float(br0[1]), -1.0 / 20.0 + 0.05 * 0.80)
    assert np.isclose(float(br1[1]), -1.0 / 20.0 + 0.05 * 0.20)
