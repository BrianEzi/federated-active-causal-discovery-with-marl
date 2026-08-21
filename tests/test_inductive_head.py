import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import pytest

from legacy.src.types import SCMConfig, MechanismType, NoiseType
from legacy.src.evaluator_env import FederatedCausalEnv, compute_invariance_asymmetry_matrix
from legacy.src.marl.ppo_agent import InductiveIPPOActor, InductiveIPPORNNActor

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
    (cat_logits, target_logits), next_state = actor_trans.apply(params, obs, init_state)
    
    assert cat_logits.shape == (1, 2)
    assert target_logits.shape == (1, d)
    assert next_state.shape == (1, 64)

def test_env_integration_inductive():
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    action_costs = np.array([1.0, 1.0])
    env = FederatedCausalEnv(config, action_costs, initial_budget=10.0, max_steps=5)
    
    key = jax.random.PRNGKey(123)
    obs_dict, info = env.reset(key)
    
    assert "agent_0" in obs_dict
    assert "agent_1" in obs_dict
    assert obs_dict["agent_0"].shape == (env.obs_dim,)
    assert obs_dict["agent_1"].shape == (env.obs_dim,)
