import jax
import jax.numpy as jnp
import haiku as hk
import pytest
from src.marl.ppo_agent import IPPOActor, IPPOCritic, mask_invalid_targets

def test_ippo_actor_shape():
    def forward(obs):
        return IPPOActor(d=4)(obs)
    
    actor = hk.without_apply_rng(hk.transform(forward))
    dummy_obs = jnp.zeros((2, 17)) # batch=2, obs_dim = 16 (cov) + 1 (budget)
    
    key = jax.random.PRNGKey(42)
    params = actor.init(key, dummy_obs)
    
    cat_logits, target_logits, graph_logits = actor.apply(params, dummy_obs)
    
    assert cat_logits.shape == (2, 3)
    assert target_logits.shape == (2, 4)
    assert graph_logits.shape == (2, 4, 4)

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
    # cat_action: 0=LOCAL, 1=PEER, 2=NOOP
    cat_actions = jnp.array([0, 1, 2])
    target_logits = jnp.zeros((3, 4))
    
    # Agent 1: local ownership (0, 1), peer boundary (1, 2)
    local_mask = jnp.array([1, 1, 0, 0])
    boundary_mask = jnp.array([0, 1, 1, 0])
    
    masked = mask_invalid_targets(cat_actions, target_logits, local_mask, boundary_mask)
    
    # Batch 0: LOCAL. Valid targets: 0, 1
    assert masked[0, 0] == 0.0
    assert masked[0, 1] == 0.0
    assert masked[0, 2] < -1e8
    assert masked[0, 3] < -1e8
    
    # Batch 1: PEER. Valid targets: 1, 2 (boundary nodes)
    assert masked[1, 0] < -1e8
    assert masked[1, 1] == 0.0
    assert masked[1, 2] == 0.0
    assert masked[1, 3] < -1e8
    
    # Batch 2: NOOP. Nothing is valid, mask all.
    assert (masked[2] < -1e8).all()
