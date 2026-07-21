import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from src.scm.generator import LinearGaussianSCM
from src.rl.env import CausalDiscoveryEnv

def test_environment_mechanics():
    print("Generating baseline data for RL Env testing...")
    scm = LinearGaussianSCM(num_vars=10, random_seed=42)
    data = scm.generate_data(num_samples=100)
    
    print("\nInitializing CausalDiscoveryEnv...")
    env = CausalDiscoveryEnv(data, max_steps=50, step_cost=0.1, invalid_penalty=10.0)
    obs, info = env.reset()
    
    assert obs.sum() == 0, "Initial graph should be empty."
    
    mask = info['action_mask']
    assert len(mask) == env.action_space.n, "Mask length mismatch."
    
    # Let's find the ADD(0 -> 1) action index
    # action_mapping is ordered by ADD, REMOVE, REVERSE, then i, then j (i != j)
    add_0_1_idx = None
    add_1_0_idx = None
    add_2_1_idx = None
    add_0_2_idx = None
    
    for idx, (op, i, j) in enumerate(env.action_mapping):
        if op == env.ACTION_ADD and i == 0 and j == 1:
            add_0_1_idx = idx
        if op == env.ACTION_ADD and i == 1 and j == 0:
            add_1_0_idx = idx
        if op == env.ACTION_ADD and i == 1 and j == 2:
            add_1_2_idx = idx
        if op == env.ACTION_ADD and i == 2 and j == 0:
            add_2_0_idx = idx
        if op == env.ACTION_REMOVE and i == 0 and j == 1:
            rem_0_1_idx = idx
            
    assert mask[add_0_1_idx] == 1, "ADD action should be valid on empty graph."
    assert mask[rem_0_1_idx] == 0, "REMOVE action should be invalid on empty graph."
    
    print("Testing Step: ADD(0 -> 1)")
    obs, reward, term, trunc, info = env.step(add_0_1_idx)
    
    assert env.adjacency_matrix[0, 1] == 1, "Edge 0 -> 1 should be added."
    assert env.adjacency_matrix.sum() == 1, "Only one edge should exist."
    
    print(f"Reward received: {reward:.4f}")
    
    mask = info['action_mask']
    assert mask[add_0_1_idx] == 0, "ADD(0->1) should now be masked."
    assert mask[rem_0_1_idx] == 1, "REMOVE(0->1) should now be valid."
    
    # Test Acyclicity: Add 1->2, then try to Add 2->0 (which creates 0->1->2->0 cycle)
    print("Testing Step: ADD(1 -> 2)")
    env.step(add_1_2_idx)
    assert env.adjacency_matrix[1, 2] == 1
    
    mask = env.get_action_mask()
    assert mask[add_2_0_idx] == 0, "ADD(2->0) should be masked to prevent cycle 0->1->2->0"
    print("Acyclicity mask correctly blocked 2->0.")
    
    # Test Invalid Penalty Bypass
    print("Testing Step: Invalid Action Bypass")
    obs, reward, term, trunc, info = env.step(add_2_0_idx)
    assert reward == -10.0, "Should receive heavy negative penalty for invalid action."
    assert env.adjacency_matrix[2, 0] == 0, "Invalid action should not mutate state."
    
    print("\nAll MDP Environment Tests Passed!")

if __name__ == "__main__":
    test_environment_mechanics()
