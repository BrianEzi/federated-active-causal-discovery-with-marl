import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from src.scm.generator import LinearGaussianSCM
from src.rl.metrics import compute_overlap_metrics

def test_federated_subgraphs():
    scm = LinearGaussianSCM(num_vars=10, random_seed=42)
    true_dag = scm.adjacency_matrix
    
    # Slice exactly as train_federated does
    agent1_true_dag = true_dag[0:6, 0:6]
    agent2_true_dag = true_dag[4:10, 4:10]
    
    # Ground truth edges from generator.py:
    # (0,1), (1,2), (2,3), (3,4), (5,4), (5,6), (6,7), (7,8), (8,9)
    
    # Agent 1 has vars 0 to 5.
    # Edges within Agent 1: (0,1), (1,2), (2,3), (3,4), (5,4)
    assert np.sum(agent1_true_dag) == 5, f"Expected 5 edges for Agent 1, got {np.sum(agent1_true_dag)}"
    assert agent1_true_dag[3, 4] == 1 # V4 -> V5
    assert agent1_true_dag[5, 4] == 1 # V6 -> V5
    
    # Agent 2 has vars 4 to 9.
    # Indices shift by -4. So variables are: 
    # V5=0, V6=1, V7=2, V8=3, V9=4, V10=5
    # Edges within Agent 2: (5,4) -> (1,0); (5,6) -> (1,2); (6,7) -> (2,3); (7,8) -> (3,4); (8,9) -> (4,5)
    assert np.sum(agent2_true_dag) == 5, f"Expected 5 edges for Agent 2, got {np.sum(agent2_true_dag)}"
    assert agent2_true_dag[1, 0] == 1 # V6 -> V5
    assert agent2_true_dag[1, 2] == 1 # V6 -> V7
    
    print("Federated subgraphs extraction passed successfully.")

def test_overlap_metrics():
    # Mock a true dag for agent 2 (Vars V5 to V10)
    # V5 is 0, V6 is 1
    # True edge is V6 -> V5 (1 -> 0)
    true_dag = np.zeros((6, 6))
    true_dag[1, 0] = 1 
    
    # Scenario 1: Perfect prediction (TPR=1.0, FDR=0.0)
    pred_dag = np.zeros((6, 6))
    pred_dag[1, 0] = 1
    tpr, fdr = compute_overlap_metrics(pred_dag, true_dag, 0, 1)
    assert tpr == 1.0 and fdr == 0.0
    
    # Scenario 2: Missed the edge entirely (TPR=0.0, FDR=0.0)
    pred_dag = np.zeros((6, 6))
    tpr, fdr = compute_overlap_metrics(pred_dag, true_dag, 0, 1)
    assert tpr == 0.0 and fdr == 0.0
    
    # Scenario 3: Hallucinated V5 -> V6 (TPR=0.0, FDR=1.0)
    pred_dag = np.zeros((6, 6))
    pred_dag[0, 1] = 1
    tpr, fdr = compute_overlap_metrics(pred_dag, true_dag, 0, 1)
    assert tpr == 0.0 and fdr == 1.0
    
    print("Overlap metrics tests passed successfully.")

if __name__ == "__main__":
    test_federated_subgraphs()
    test_overlap_metrics()
