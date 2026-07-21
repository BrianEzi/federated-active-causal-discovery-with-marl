import jax
import jax.numpy as jnp

def stouffer_z_score_fusion(p_values_matrix: jax.Array, sample_sizes: jax.Array) -> jax.Array:
    """
    Accepts an array of client-level p-values for edge independence testing across overlapping variables
    and fuses them into a unified global Z-score using Stouffer's method weighted by sample sizes.
    
    p_values_matrix: [m, ...] array of p-values, where m is the number of clients (agents).
    sample_sizes: [m] array of sample sizes for each client.
    Returns: [...] array of global Z-scores.
    """
    # Convert p-values to Z-scores using the inverse normal CDF (probit function)
    # jax.scipy.special.ndtri is the inverse CDF for the standard normal distribution
    # Note: p-values should be bounded away from 0 and 1 to avoid infinity
    eps = 1e-15
    safe_p_values = jnp.clip(p_values_matrix, eps, 1.0 - eps)
    
    # We want upper-tail Z-scores typically, so Z = ndtri(1 - p)
    z_scores = jax.scipy.special.ndtri(1.0 - safe_p_values)
    
    # Calculate weights: sqrt(N_k)
    weights = jnp.sqrt(sample_sizes)
    
    # Broadcast weights to match z_scores shape if necessary
    # z_scores has shape [m, ...]
    broadcast_shape = [-1] + [1] * (z_scores.ndim - 1)
    weights = weights.reshape(broadcast_shape)
    
    # Z_global = sum(sqrt(N_k) * Z_k) / sqrt(sum(N_k))
    weighted_sum = jnp.sum(weights * z_scores, axis=0)
    norm_factor = jnp.sqrt(jnp.sum(sample_sizes))
    
    return weighted_sum / norm_factor

def stitch_global_covariance(local_covariances: jax.Array, 
                             agent_masks: jax.Array, 
                             sample_counts: jax.Array) -> jax.Array:
    """
    Aggregates local sample covariance matrices into an initial d x d global covariance matrix.
    Averages overlapping entries weighted by sample counts and marks unobserved entries as NaN.
    
    local_covariances: [m, d, d]
    agent_masks: [m, d]
    sample_counts: [m]
    Returns: [d, d] global covariance matrix.
    """
    # Create pairwise masks indicating which agent observed both variable i and j
    # agent_masks[k, i] * agent_masks[k, j] = 1 if agent k observes both
    pairwise_masks = agent_masks[:, :, None] * agent_masks[:, None, :] # [m, d, d]
    
    # Reshape sample counts for broadcasting
    weights = sample_counts.reshape(-1, 1, 1) # [m, 1, 1]
    
    # Total valid samples observing each pair (i, j)
    total_weights = jnp.sum(weights * pairwise_masks, axis=0) # [d, d]
    
    # Weighted sum of covariances
    sum_covariances = jnp.sum(weights * pairwise_masks * local_covariances, axis=0) # [d, d]
    
    # Avoid division by zero by safely defaulting to 1.0 where total_weight is 0
    safe_total_weights = jnp.where(total_weights > 0, total_weights, 1.0)
    
    global_cov = sum_covariances / safe_total_weights
    
    # Mark unobserved entries (where total_weights == 0) as NaN
    global_cov = jnp.where(total_weights > 0, global_cov, jnp.nan)
    
    return global_cov
