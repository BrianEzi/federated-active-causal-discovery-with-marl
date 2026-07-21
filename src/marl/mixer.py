import jax
import jax.numpy as jnp
import flax.linen as nn

class QMIXMixer(nn.Module):
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, q_chosen: jax.Array, state: jax.Array) -> jax.Array:
        """
        Mixes local chosen Q-values into a global Q_tot.
        q_chosen: [batch_size, K] array of chosen Q-values.
        state: [batch_size, state_dim] global state vector.
        """
        K = q_chosen.shape[-1]
        
        # Ensure q_chosen is at least 2D
        # If input is [K], reshape to [1, K]
        is_single = False
        if q_chosen.ndim == 1:
            is_single = True
            q_chosen = jnp.expand_dims(q_chosen, axis=0)
            state = jnp.expand_dims(state, axis=0)
            
        batch_size = q_chosen.shape[0]
        
        # Hypernetwork 1: computes W1 [batch, K, H] and b1 [batch, 1, H]
        w1 = nn.Dense(K * self.hidden_dim)(state)
        w1 = jnp.abs(w1) # Enforce strict monotonicity
        w1 = w1.reshape(batch_size, K, self.hidden_dim)
        
        b1 = nn.Dense(self.hidden_dim)(state)
        b1 = b1.reshape(batch_size, 1, self.hidden_dim)
        
        # Hypernetwork 2: computes W2 [batch, H, 1] and b2 [batch, 1, 1]
        w2 = nn.Dense(self.hidden_dim)(state)
        w2 = nn.relu(w2)
        w2 = nn.Dense(self.hidden_dim)(w2)
        w2 = jnp.abs(w2) # Enforce strict monotonicity
        w2 = w2.reshape(batch_size, self.hidden_dim, 1)
        
        b2 = nn.Dense(self.hidden_dim)(state)
        b2 = nn.relu(b2)
        b2 = nn.Dense(1)(b2)
        b2 = b2.reshape(batch_size, 1, 1)
        
        # Forward pass through the mixer network
        q_chosen_exp = jnp.expand_dims(q_chosen, axis=1) # [batch, 1, K]
        
        # [batch, 1, K] @ [batch, K, H] -> [batch, 1, H]
        hidden = nn.elu(jnp.matmul(q_chosen_exp, w1) + b1)
        
        # [batch, 1, H] @ [batch, H, 1] -> [batch, 1, 1]
        q_tot = jnp.matmul(hidden, w2) + b2
        
        q_tot = jnp.squeeze(q_tot, axis=1) # [batch, 1]
        
        if is_single:
            return q_tot[0]
            
        return q_tot
