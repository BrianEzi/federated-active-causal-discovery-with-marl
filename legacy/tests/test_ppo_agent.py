import jax
import jax.numpy as jnp
import haiku as hk
import pytest
from legacy.src.marl.ppo_agent import IPPOActor, IPPOCritic, mask_invalid_targets

def test_ippo_actor_shape():
    def forward(obs):
        return IPPOActor(d=4)(obs)
    
    actor = hk.without_apply_rng(hk.transform(forward))
    dummy_obs = jnp.zeros((2, 17)) # batch=2, obs_dim = 16 (cov) + 1 (budget)
    
    key = jax.random.PRNGKey(42)
    params = actor.init(key, dummy_obs)
    
    cat_logits, target_logits = actor.apply(params, dummy_obs)
    
    assert cat_logits.shape == (2, 2)
    assert target_logits.shape == (2, 4)

def test_ippo_critic_shape():
    def forward(obs):
        return IPPOCritic()(obs)
        
    critic = hk.without_apply_rng(hk.transform(forward))
    dummy_obs = jnp.zeros((2, 17))
    
    key = jax.random.PRNGKey(42)
    params = critic.init(key, dummy_obs)
    
    v = critic.apply(params, dummy_obs)
    assert v.shape == (2,)

def test_mask_invalid_targets():
    # cat_action: 0=INTERVENE, 1=NOOP
    cat_actions = jnp.array([0, 1])
    target_logits = jnp.zeros((2, 4))
    
    # Valid intervention mask (local + boundary)
    valid_mask = jnp.array([1, 1, 1, 0])
    
    masked = mask_invalid_targets(cat_actions, target_logits, valid_mask)
    
    # Row 0: INTERVENE. Nodes 0, 1, 2 are valid. Node 3 should be -1e9.
    assert float(masked[0, 0]) == 0.0
    assert float(masked[0, 1]) == 0.0
    assert float(masked[0, 2]) == 0.0
    assert float(masked[0, 3]) < -1e8
    
    # Row 1: NOOP. All targets should be masked (-1e9)
    assert float(masked[1, 0]) < -1e8
    assert float(masked[1, 1]) < -1e8
    assert float(masked[1, 2]) < -1e8
    assert float(masked[1, 3]) < -1e8
