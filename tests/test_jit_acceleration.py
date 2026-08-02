import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from src.types import SCMConfig, MechanismType, NoiseType, ActionCategory
from src.evaluator_env import FederatedCausalEnv, build_intervention_spec_jitted
from src.marl.ppo_agent import IPPOActor, sample_actions_jitted
from src.stitching import jitted_stitch_dags, jitted_detect_cycle, stitch_predicted_dags, detect_cycle
from src.rewards import jitted_compute_ippo_rewards, compute_ippo_rewards

def test_jitted_cycle_detection_vs_dfs():
    rng = np.random.RandomState(123)
    for _ in range(500):
        adj = (rng.rand(4, 4) > 0.65).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        
        dfs_has_cycle = detect_cycle(adj)
        jax_has_cycle = bool(jitted_detect_cycle(jnp.array(adj)))
        assert dfs_has_cycle == jax_has_cycle

def test_jitted_stitch_and_rewards_equivalence():
    p1 = np.array([
        [0.0, 0.8, 0.2, 0.0],
        [0.1, 0.0, 0.9, 0.0],
        [0.7, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    p2 = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.1],
        [0.0, 0.3, 0.0, 0.9],
        [0.0, 0.2, 0.4, 0.0]
    ], dtype=np.float32)
    
    true_adj = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=np.float32)
    
    # NumPy Reference
    stitched_np, cycle_np = stitch_predicted_dags({"agent_0": p1, "agent_1": p2}, 4)
    rewards_np = compute_ippo_rewards(stitched_np, true_adj, cycle_np)
    
    # JAX JIT Implementation
    stitched_jax, cycle_jax = jitted_stitch_dags(jnp.array(p1), jnp.array(p2), 4)
    r1_jax, r2_jax = jitted_compute_ippo_rewards(stitched_jax, jnp.array(true_adj), cycle_jax)
    
    np.testing.assert_array_equal(np.array(stitched_jax), stitched_np)
    assert bool(cycle_jax) == cycle_np
    assert np.isclose(float(r1_jax), rewards_np["agent_0"])
    assert np.isclose(float(r2_jax), rewards_np["agent_1"])

def test_sample_actions_jitted_shapes_and_masks():
    def forward(obs): return IPPOActor(d=4)(obs)
    trans = hk.without_apply_rng(hk.transform(forward))
    key = jax.random.PRNGKey(0)
    obs = jnp.zeros((1, 17))
    params = trans.init(key, obs)
    cat_logits, target_logits, graph_logits = trans.apply(params, obs)
    
    local_mask = jnp.array([1.0, 1.0, 0.0, 0.0])
    boundary_mask = jnp.array([0.0, 1.0, 1.0, 0.0])
    edge_mask = jnp.ones((4, 4))
    
    k1, key = jax.random.split(key)
    cat, target, lp, gp = sample_actions_jitted(
        cat_logits[0], target_logits[0], graph_logits[0],
        local_mask, boundary_mask, edge_mask, k1
    )
    
    assert cat.shape == ()
    assert target.shape == ()
    assert lp.shape == ()
    assert gp.shape == (4, 4)
    assert 0 <= int(cat) <= 2
    assert 0 <= int(target) <= 3

def test_env_step_jitted_execution():
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    env = FederatedCausalEnv(config, action_costs=np.array([1.0, 1.0]), initial_budget=10.0, sample_count=50, fixed_graph=True)
    key = jax.random.PRNGKey(42)
    obs_dict, info = env.reset(key)
    
    gp0 = jnp.zeros((4, 4))
    gp1 = jnp.zeros((4, 4))
    c0 = jnp.array(int(ActionCategory.LOCAL_INTERVENTION))
    t0 = jnp.array(0) # Node 0 (local to agent 0)
    c1 = jnp.array(int(ActionCategory.NOOP))
    t1 = jnp.array(0)
    
    k_step, key = jax.random.split(key)
    agent_obs, r0, r1, done, final_dag, info_gains = env.step_jitted(c0, t0, gp0, c1, t1, gp1, k_step)
    
    assert agent_obs.shape == (2, 17)
    assert final_dag.shape == (4, 4)
    assert info_gains.shape == (2,)
    assert float(info_gains[0]) >= 0.0
    assert not done
    assert float(env.jax_state.budgets[0]) == 9.0
    assert float(env.jax_state.budgets[1]) == 10.0

