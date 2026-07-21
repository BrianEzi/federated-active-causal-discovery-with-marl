import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.marl.agent import MLPAgent, RNNAgent, CausalTransformerAgent
from src.marl.mixer import QMIXMixer
from src.marl.trainer import QMIXTrainer

def test_mlp_agent_forward():
    agent = MLPAgent(num_actions=6, hidden_dim=32)
    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((37,))
    params = agent.init(key, dummy_obs)
    q_vals = agent.apply(params, dummy_obs)
    assert q_vals.shape == (6,)

def test_rnn_agent_forward():
    agent = RNNAgent(num_actions=6, hidden_dim=32)
    key = jax.random.PRNGKey(0)
    dummy_carry = jnp.zeros((32,))
    dummy_obs = jnp.zeros((37,))
    params = agent.init(key, dummy_carry, dummy_obs)
    new_carry, q_vals = agent.apply(params, dummy_carry, dummy_obs)
    assert new_carry.shape == (32,)
    assert q_vals.shape == (6,)

def test_transformer_agent_forward():
    agent = CausalTransformerAgent(num_actions=6, hidden_dim=32, num_heads=2)
    key = jax.random.PRNGKey(0)
    dummy_seq = jnp.zeros((1, 10, 37)) # [batch, T, obs_dim]
    params = agent.init(key, dummy_seq)
    q_seq = agent.apply(params, dummy_seq)
    assert q_seq.shape == (1, 10, 6)

@pytest.mark.parametrize("agent_type", ["mlp", "rnn", "transformer"])
def test_qmix_trainer_architectures(agent_type):
    K = 2
    obs_dim = 37
    state_dim = 52
    num_actions = 6
    max_steps = 10
    
    if agent_type == "mlp":
        agent = MLPAgent(num_actions=num_actions, hidden_dim=32)
    elif agent_type == "rnn":
        agent = RNNAgent(num_actions=num_actions, hidden_dim=32)
    elif agent_type == "transformer":
        agent = CausalTransformerAgent(num_actions=num_actions, hidden_dim=32, num_heads=2)
        
    mixer = QMIXMixer(hidden_dim=16)
    trainer = QMIXTrainer(agent, mixer, lr=1e-3, agent_type=agent_type)
    
    key = jax.random.PRNGKey(42)
    train_state, target_state = trainer.init_state(key, obs_dim, state_dim, K, max_steps=max_steps)
    
    # Fake batch [B=2, T=5, K=2]
    batch = {
        'states': jnp.zeros((2, 5, state_dim)),
        'observations': jnp.zeros((2, 5, K, obs_dim)),
        'actions': jnp.zeros((2, 5, K), dtype=jnp.int32),
        'rewards': jnp.zeros((2, 5, 1)),
        'dones': jnp.zeros((2, 5, 1), dtype=jnp.bool_),
        'avail_actions': jnp.ones((2, 5, K, num_actions))
    }
    
    new_state, loss = trainer.train_step(train_state, target_state, batch)
    assert loss >= 0.0
