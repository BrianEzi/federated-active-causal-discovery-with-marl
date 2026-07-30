import numpy as np
import pytest
from src.rewards import compute_ippo_rewards

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
