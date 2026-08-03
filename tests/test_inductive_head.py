import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import pytest

from src.types import SCMConfig, MechanismType, NoiseType
from src.evaluator_env import FederatedCausalEnv, compute_invariance_asymmetry_matrix
from src.marl.ppo_agent import InductiveIPPOActor, InductiveIPPORNNActor

def test_compute_invariance_asymmetry_matrix():
    d = 4
    obs_cov = jnp.eye(d)
    int_cov = jnp.zeros((d, d, d))
    
    # Simulate intervention on node 1 causing variance shift in node 2
    # int_cov slice [1, :, :] has Var(X_2 | do(X_1)) = 4.0
    int_cov = int_cov.at[1, 2, 2].set(4.0)
    int_mask = jnp.array([0.0, 1.0, 0.0, 0.0]) # Node 1 intervened
    
    asymmetry = compute_invariance_asymmetry_matrix(obs_cov, int_cov, int_mask)
    
    # A[1, 2] should be positive (evidence for 1 -> 2)
    # A[2, 1] should be negative (exact anti-symmetry)
    assert asymmetry.shape == (d, d)
    assert float(asymmetry[1, 2]) > 0.0
    assert float(asymmetry[2, 1]) == -float(asymmetry[1, 2])
    # Skew-symmetry property: A == -A^T
    assert np.allclose(asymmetry, -asymmetry.T)

def test_inductive_ppo_actor_anti_symmetry():
    d = 4
    gamma = 2.0
    
    def forward(obs):
        actor = InductiveIPPOActor(d=d, gamma=gamma)
        return actor(obs)
        
    actor_trans = hk.without_apply_rng(hk.transform(forward))
    key = jax.random.PRNGKey(42)
    
    # Create 49-dim observation: 16 (obs) + 16 (run) + 16 (asym) + 1 (budget)
    obs_cov = jnp.eye(d).flatten()
    run_cov = jnp.eye(d).flatten()
    
    # Asymmetry matrix with A[1, 2] = 1.5, A[2, 1] = -1.5
    asym_mat = jnp.zeros((d, d)).at[1, 2].set(1.5).at[2, 1].set(-1.5)
    asym_flat = asym_mat.flatten()
    budget = jnp.array([10.0])
    
    obs = jnp.concatenate([obs_cov, run_cov, asym_flat, budget])[None, :] # [1, 49]
    
    params = actor_trans.init(key, obs)
    cat_logits, target_logits, graph_logits = actor_trans.apply(params, obs)
    
    assert cat_logits.shape == (1, 3)
    assert target_logits.shape == (1, d)
    assert graph_logits.shape == (1, d, d)
    
    gl = graph_logits[0]
    
    # Verify that Logit(1 -> 2) - Logit(2 -> 1) is dominated by the asymmetry term
    diff_12 = float(gl[1, 2] - gl[2, 1])
    assert diff_12 > 0.0, "Logit(1 -> 2) should strictly exceed Logit(2 -> 1) when asymmetry is positive"

def test_inductive_ppo_rnn_actor_execution():
    d = 4
    def forward(obs, state):
        actor = InductiveIPPORNNActor(d=d)
        return actor(obs, state)
        
    actor_trans = hk.without_apply_rng(hk.transform(forward))
    key = jax.random.PRNGKey(42)
    
    obs_dim = 3 * d * d + 1
    obs = jnp.zeros((1, obs_dim))
    init_state = InductiveIPPORNNActor.initial_state(1)
    
    params = actor_trans.init(key, obs, init_state)
    (cat_logits, target_logits, graph_logits), next_state = actor_trans.apply(params, obs, init_state)
    
    assert cat_logits.shape == (1, 3)
    assert target_logits.shape == (1, d)
    assert graph_logits.shape == (1, d, d)
    assert next_state.shape == (1, 64)

def test_env_integration_inductive():
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    action_costs = np.array([1.0, 1.0])
    env = FederatedCausalEnv(config, action_costs, initial_budget=10.0, max_steps=5)
    
    key = jax.random.PRNGKey(123)
    obs_dict, info = env.reset(key)
    
    assert "agent_0" in obs_dict
    assert "agent_1" in obs_dict
    assert obs_dict["agent_0"].shape == (49,)
    assert obs_dict["agent_1"].shape == (49,)
