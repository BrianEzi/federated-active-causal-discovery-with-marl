import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legacy.src.types import SCMConfig, MechanismType, NoiseType
from legacy.src.evaluator_env import FederatedCausalEnv
from legacy.src.generators import generate_4node_topologies
from legacy.src.train import parse_topology_list

def test_parse_topology_list():
    assert parse_topology_list("0,1,2") == (0, 1, 2)
    assert parse_topology_list("0 1 2") == (0, 1, 2)
    assert parse_topology_list("[0, 2, 6]") == (0, 2, 6)
    assert parse_topology_list(None) is None
    assert parse_topology_list((0, 1)) == (0, 1)

def test_generate_4node_topologies_custom_subset():
    key = jax.random.PRNGKey(42)
    allowed = (0, 2, 6)
    
    sampled_indices = []
    for i in range(50):
        k_sub = jax.random.fold_in(key, i)
        adj, order = generate_4node_topologies(k_sub, allowed_indices=allowed)
        
        # Check against known matrices for 0, 2, 6
        m0, _ = generate_4node_topologies(key, force_idx=0)
        m2, _ = generate_4node_topologies(key, force_idx=2)
        m6, _ = generate_4node_topologies(key, force_idx=6)
        
        is_0 = np.array_equal(adj, m0)
        is_2 = np.array_equal(adj, m2)
        is_6 = np.array_equal(adj, m6)
        
        assert is_0 or is_2 or is_6, "Sampled topology must be strictly within allowed subset (0, 2, 6)"

def test_env_reset_with_custom_allowed_topologies():
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    env = FederatedCausalEnv(config, action_costs=np.array([1.0, 1.0]), initial_budget=10.0)
    
    key = jax.random.PRNGKey(42)
    allowed = (1, 3, 5) # Reverse Chain, Chain+Collider, Chain+Fork
    
    for i in range(20):
        k_ep = jax.random.fold_in(key, i)
        obs, info = env.reset(k_ep, allowed_topologies=allowed)
        true_adj = info["true_adjacency"]
        
        m1, _ = generate_4node_topologies(key, force_idx=1)
        m3, _ = generate_4node_topologies(key, force_idx=3)
        m5, _ = generate_4node_topologies(key, force_idx=5)
        
        is_1 = np.array_equal(true_adj, m1)
        is_3 = np.array_equal(true_adj, m3)
        is_5 = np.array_equal(true_adj, m5)
        
        assert is_1 or is_3 or is_5, f"Reset true_adjacency must match one of allowed topologies {allowed}"
