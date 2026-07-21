import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

class MLPAgent(nn.Module):
    num_actions: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        # observation shape: [obs_dim] or [..., obs_dim]
        x = nn.Dense(self.hidden_dim)(observation)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        q_values = nn.Dense(self.num_actions)(x)
        return q_values

class RNNAgent(nn.Module):
    num_actions: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, carry: jax.Array, observation: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Recurrent GRU Agent for POMDP active causal discovery.
        carry: [..., hidden_dim]
        observation: [..., obs_dim]
        returns: (new_carry, q_values)
        """
        x = nn.Dense(self.hidden_dim)(observation)
        x = nn.relu(x)
        new_carry, x = nn.GRUCell(features=self.hidden_dim)(carry, x)
        q_values = nn.Dense(self.num_actions)(x)
        return new_carry, q_values

    def initialize_carry(self, batch_shape: Tuple[int, ...]) -> jax.Array:
        return jnp.zeros(batch_shape + (self.hidden_dim,))

class CausalTransformerAgent(nn.Module):
    num_actions: int
    hidden_dim: int = 64
    num_heads: int = 2

    @nn.compact
    def __call__(self, obs_sequence: jax.Array) -> jax.Array:
        """
        Self-Attention Trajectory Transformer Agent.
        obs_sequence: [batch, T, obs_dim] or [T, obs_dim]
        returns: q_values [batch, T, num_actions] or [T, num_actions]
        """
        x = nn.Dense(self.hidden_dim)(obs_sequence)
        x = nn.relu(x)
        
        # Self-Attention across sequence length T
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.hidden_dim
        )(x, x)
        
        x = x + attn_out
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        q_values = nn.Dense(self.num_actions)(x)
        return q_values

def mask_q_values(q_values: jax.Array, avail_actions: jax.Array) -> jax.Array:
    """
    Adds -1e9 to unavailable action indices to ensure they are never greedily selected.
    q_values: [batch, num_actions] or [num_actions]
    avail_actions: boolean array of same shape (True = available, False = masked)
    """
    penalty = (1.0 - avail_actions) * (-1e9)
    return q_values + penalty
