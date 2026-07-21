import jax
import jax.numpy as jnp
import flax.linen as nn

class MLPAgent(nn.Module):
    num_actions: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        # observation shape: [2d^2 + d + 1]
        x = nn.Dense(self.hidden_dim)(observation)
        x = nn.relu(x)
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
    # -1e9 penalty for masked actions
    penalty = (1.0 - avail_actions) * (-1e9)
    return q_values + penalty
