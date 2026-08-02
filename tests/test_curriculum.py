import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.generators import generate_4node_topologies
from src.evaluator_env import FederatedCausalEnv
from src.types import SCMConfig
from src.train import get_curriculum_topologies


def test_generate_4node_topologies_allowed_indices():
    """Verify that generate_4node_topologies only samples from allowed_indices."""
    key = jax.random.PRNGKey(42)
    
    # Test Stage 1: only Graph 0
    for i in range(20):
        key, subkey = jax.random.split(key)
        adj_0, order_0 = generate_4node_topologies(subkey, force_idx=0)
        adj_sampled, order_sampled = generate_4node_topologies(subkey, allowed_indices=(0,))
        assert jnp.array_equal(adj_0, adj_sampled)
        assert jnp.array_equal(order_0, order_sampled)

    # Test Stage 2: only Graphs 0 and 1
    sampled_indices = set()
    for i in range(50):
        key, subkey = jax.random.split(key)
        adj_s, _ = generate_4node_topologies(subkey, allowed_indices=(0, 1))
        adj_0, _ = generate_4node_topologies(subkey, force_idx=0)
        adj_1, _ = generate_4node_topologies(subkey, force_idx=1)
        
        is_0 = jnp.array_equal(adj_s, adj_0)
        is_1 = jnp.array_equal(adj_s, adj_1)
        assert is_0 or is_1
        if is_0:
            sampled_indices.add(0)
        if is_1:
            sampled_indices.add(1)
            
    # Both 0 and 1 should have been sampled across 50 iterations
    assert sampled_indices == {0, 1}


def test_get_curriculum_topologies_schedule():
    """Verify curriculum stage transitions across total episode horizons."""
    total_episodes = 100
    # Stage 1: 0% to 20% -> episodes 1 to 20
    # Stage 2: 20% to 50% -> episodes 21 to 50
    # Stage 3: 50% to 100% -> episodes 51 to 100
    
    for ep in range(1, 21):
        topos, stage = get_curriculum_topologies(ep, total_episodes, 0.20, 0.30)
        assert topos == (0,)
        assert stage == 1
        
    for ep in range(21, 51):
        topos, stage = get_curriculum_topologies(ep, total_episodes, 0.20, 0.30)
        assert topos == (0, 1)
        assert stage == 2
        
    for ep in range(51, 101):
        topos, stage = get_curriculum_topologies(ep, total_episodes, 0.20, 0.30)
        assert topos == tuple(range(8))
        assert stage == 3


def test_env_reset_with_curriculum_topologies():
    """Verify FederatedCausalEnv.reset adheres to allowed_topologies."""
    config = SCMConfig(d=4, K=2, mechanism_type=0, noise_type=0, noise_scale=1.0)
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(config=config, action_costs=action_costs, sample_count=100)
    key = jax.random.PRNGKey(123)
    
    # Reference adjacencies
    ref_adj_0, _ = generate_4node_topologies(key, force_idx=0)
    ref_adj_1, _ = generate_4node_topologies(key, force_idx=1)
    
    # Reset restricted to stage 1
    for _ in range(10):
        key, subkey = jax.random.split(key)
        _, info = env.reset(subkey, allowed_topologies=(0,))
        assert np.array_equal(info["true_adjacency"], np.array(ref_adj_0))
        
    # Reset restricted to stage 2
    seen = set()
    for _ in range(30):
        key, subkey = jax.random.split(key)
        _, info = env.reset(subkey, allowed_topologies=(0, 1))
        is_0 = np.array_equal(info["true_adjacency"], np.array(ref_adj_0))
        is_1 = np.array_equal(info["true_adjacency"], np.array(ref_adj_1))
        assert is_0 or is_1
        if is_0:
            seen.add(0)
        if is_1:
            seen.add(1)
    assert seen == {0, 1}
