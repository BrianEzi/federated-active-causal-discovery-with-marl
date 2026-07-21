import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from src.rl.metrics import compute_shd
from src.rl.agent import DDQNAgent

def test_shd_calculation():
    # Construct a simple true DAG
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1 # 0 -> 1
    true_dag[1, 2] = 1 # 1 -> 2
    
    # 1. Exact match
    assert compute_shd(true_dag, true_dag) == 0
    
    # 2. Missing edge
    pred_missing = true_dag.copy()
    pred_missing[0, 1] = 0
    assert compute_shd(pred_missing, true_dag) == 1
    
    # 3. Extra edge
    pred_extra = true_dag.copy()
    pred_extra[0, 2] = 1
    assert compute_shd(pred_extra, true_dag) == 1
    
    # 4. Reversed edge
    pred_reversed = true_dag.copy()
    pred_reversed[0, 1] = 0
    pred_reversed[1, 0] = 1
    assert compute_shd(pred_reversed, true_dag) == 1
    
    print("SHD tests passed successfully.")

def test_agent_masking():
    agent = DDQNAgent(obs_size=100, action_size=5, lr=1e-3)
    
    # Dummy state
    state = np.zeros(100)
    
    # Restrict mask to only allow action index 3
    mask = np.array([0, 0, 0, 1, 0])
    
    # Check greedy selection (epsilon=0)
    action = agent.select_action(state, mask, epsilon=0.0)
    assert action == 3, f"Agent selected {action}, but mask only allowed 3."
    
    # Check random selection (epsilon=1)
    action_random = agent.select_action(state, mask, epsilon=1.0)
    assert action_random == 3, f"Random sampler selected {action_random}, but mask only allowed 3."
    
    print("Agent mask enforcement passed successfully.")

if __name__ == "__main__":
    test_shd_calculation()
    test_agent_masking()
