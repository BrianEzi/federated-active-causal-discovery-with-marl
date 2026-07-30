import numpy as np
import pytest
from src.stitching import stitch_predicted_dags, detect_cycle

def test_detect_cycle_no_cycle():
    # 0 -> 1 -> 2
    adj = np.zeros((3, 3))
    adj[0, 1] = 1
    adj[1, 2] = 1
    assert not detect_cycle(adj)

def test_detect_cycle_with_cycle():
    # 0 -> 1 -> 2 -> 0
    adj = np.zeros((3, 3))
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[2, 0] = 1
    assert detect_cycle(adj)
    
def test_detect_cycle_two_cycle():
    # 0 -> 1 and 1 -> 0
    adj = np.zeros((3, 3))
    adj[0, 1] = 1
    adj[1, 0] = 1
    assert detect_cycle(adj)

def test_stitch_predicted_dags():
    d = 4
    prob_1 = np.zeros((d, d))
    prob_2 = np.zeros((d, d))
    
    # Agent 1 predicts 0 -> 1 and 1 -> 2
    prob_1[0, 1] = 0.9
    prob_1[1, 2] = 0.8
    
    # Agent 2 predicts 1 -> 2 and 2 -> 3
    prob_2[1, 2] = 0.7
    prob_2[2, 3] = 0.95
    
    predicted_probs = {"agent_0": prob_1, "agent_1": prob_2}
    
    stitched_dag, has_cycle = stitch_predicted_dags(predicted_probs, d)
    
    assert not has_cycle
    assert stitched_dag[0, 1] == 1.0
    # Overlapping prediction: (0.8 + 0.7) / 2 = 0.75 > 0.5 -> 1.0
    assert stitched_dag[1, 2] == 1.0
    assert stitched_dag[2, 3] == 1.0
    assert np.sum(stitched_dag) == 3.0

def test_stitch_predicted_dags_conflict():
    d = 4
    prob_1 = np.zeros((d, d))
    prob_2 = np.zeros((d, d))
    
    # Agent 1 predicts 1 -> 2 strongly
    prob_1[1, 2] = 0.9
    
    # Agent 2 predicts 2 -> 1 strongly
    prob_2[2, 1] = 0.9
    
    predicted_probs = {"agent_0": prob_1, "agent_1": prob_2}
    
    stitched_dag, has_cycle = stitch_predicted_dags(predicted_probs, d)
    
    # 1->2 gets 0.45, 2->1 gets 0.45. Both < 0.5, so neither edge is formed!
    # Wait, the threshold is > 0.5. So stitched_dag should have no edges.
    assert stitched_dag[1, 2] == 0.0
    assert stitched_dag[2, 1] == 0.0
    assert not has_cycle
