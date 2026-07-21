import jax
import jax.numpy as jnp
from src.types import MechanismType, NoiseType, SCMParams

def generate_noise(key: jax.Array, noise_type: int, shape: tuple, scale: float = 1.0) -> jax.Array:
    """Generates noise based on the specified distribution type."""
    def _gaussian(k): return jax.random.normal(k, shape) * scale
    def _gumbel(k): return jax.random.gumbel(k, shape) * scale
    def _uniform(k): return jax.random.uniform(k, shape, minval=-scale, maxval=scale)
    
    return jax.lax.switch(
        noise_type,
        [_gaussian, _gumbel, _uniform],
        key
    )

def apply_mechanism(parent_values: jax.Array, 
                    node_idx: int,
                    mechanism_type: int, 
                    params: SCMParams) -> jax.Array:
    """
    Applies the structural mechanism for a specific node.
    parent_values: [d] array of current node values
    node_idx: integer index of the current node
    """
    
    # W[node_idx, :] contains structural weights from all nodes to node_idx.
    # Non-parents will have a weight of 0 here.
    structural_weights = params.W[node_idx, :]
    
    def _linear(_):
        return jnp.dot(structural_weights, parent_values)
        
    def _nonlinear_anm(_):
        # Mask non-parents explicitly. If a weight is non-zero, it is a parent.
        masked_parents = parent_values * (jnp.abs(structural_weights) > 0).astype(parent_values.dtype)
        
        w1 = params.mlp_w1[node_idx] # [d, hidden_dim]
        b1 = params.mlp_b1[node_idx] # [hidden_dim]
        w2 = params.mlp_w2[node_idx] # [hidden_dim, 1]
        b2 = params.mlp_b2[node_idx] # [1]
        
        h1 = jax.nn.relu(jnp.dot(masked_parents, w1) + b1)
        out = jnp.dot(h1, w2) + b2
        return out[0]
        
    def _post_nonlinear(_):
        # Post-nonlinear transformation on top of the ANM mechanism
        return jax.nn.tanh(_nonlinear_anm(_))
        
    return jax.lax.switch(
        mechanism_type,
        [_linear, _nonlinear_anm, _post_nonlinear],
        None
    )
