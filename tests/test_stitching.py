import numpy as np
import jax
import jax.numpy as jnp
import pytest
from src.stitching import stitch_predicted_dags, detect_cycle, jitted_stitch_dags, jitted_detect_cycle

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
    prob_1 = np.zeros((d, d), dtype=np.float32)
    prob_2 = np.zeros((d, d), dtype=np.float32)
    
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
    assert stitched_dag[1, 2] == 1.0
    assert stitched_dag[2, 3] == 1.0
    assert np.sum(stitched_dag) == 3.0

def test_stitch_predicted_dags_conflict_suppression():
    d = 4
    prob_1 = np.zeros((d, d), dtype=np.float32)
    prob_2 = np.zeros((d, d), dtype=np.float32)
    
    # Agent 1 predicts 1 -> 2 strongly
    prob_1[1, 2] = 0.9
    
    # Agent 2 predicts 2 -> 1 equally strongly
    prob_2[2, 1] = 0.9
    
    predicted_probs = {"agent_0": prob_1, "agent_1": prob_2}
    
    stitched_dag, has_cycle = stitch_predicted_dags(predicted_probs, d, margin=0.10)
    
    # Under differential thresholding, 0.9 - 0.9 = 0.0 <= margin
    # Neither direction wins; both are suppressed to avoid 2-cycle collision.
    assert stitched_dag[1, 2] == 0.0
    assert stitched_dag[2, 1] == 0.0
    assert not has_cycle

def test_stitch_predicted_dags_differential_winner():
    d = 4
    prob_1 = np.zeros((d, d), dtype=np.float32)
    prob_2 = np.zeros((d, d), dtype=np.float32)
    
    # Agent 1 predicts 1 -> 2 with high confidence
    prob_1[1, 2] = 0.85
    
    # Agent 2 predicts 2 -> 1 with lower confidence
    prob_2[2, 1] = 0.60
    
    predicted_probs = {"agent_0": prob_1, "agent_1": prob_2}
    
    stitched_dag, has_cycle = stitch_predicted_dags(predicted_probs, d, margin=0.10)
    
    # 0.85 - 0.60 = 0.25 > 0.10 -> 1 -> 2 wins!
    assert stitched_dag[1, 2] == 1.0
    assert stitched_dag[2, 1] == 0.0
    assert not has_cycle

def test_jitted_stitch_dags_equivalence():
    d = 4
    prob_1 = np.zeros((d, d), dtype=np.float32)
    prob_2 = np.zeros((d, d), dtype=np.float32)
    prob_1[0, 1] = 0.8
    prob_1[1, 2] = 0.75
    prob_2[2, 1] = 0.55
    prob_2[2, 3] = 0.9
    
    cpu_dag, cpu_cycle = stitch_predicted_dags({"agent_0": prob_1, "agent_1": prob_2}, d, margin=0.10)
    jax_dag, jax_cycle = jitted_stitch_dags(jnp.array(prob_1), jnp.array(prob_2), d, margin=0.10)
    
    assert np.array_equal(cpu_dag, np.array(jax_dag))
    assert bool(cpu_cycle) == bool(jax_cycle)
