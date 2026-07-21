import jax
import jax.numpy as jnp
from src.types import SCMConfig, EnvState, InterventionSpec, InterventionType
from src.functional import apply_mechanism, generate_noise
from functools import partial

def _sample_single_scm(key: jax.Array, 
                       state: EnvState, 
                       config: SCMConfig, 
                       intervention_spec: InterventionSpec) -> jax.Array:
    """
    Samples a single realization of all d variables sequentially, following the topological order.
    intervention_spec: Specifies hard, soft-shift, and soft-scale interventions.
    """
    # Initial values for all nodes.
    vals = jnp.zeros(config.d)
    
    noise_key, _ = jax.random.split(key)
    # Pre-sample noise for all variables to avoid doing it inside the scan
    noise = generate_noise(noise_key, config.noise_type, (config.d,), config.noise_scale)

    def scan_fn(current_vals, i):
        # i goes from 0 to d-1. We look up the actual node index in the topological order.
        node_idx = state.topological_order[i]
        
        # Calculate mechanism based on current (parent) values
        mech_val = apply_mechanism(current_vals, node_idx, config.mechanism_type, state.scm_params)
        
        # Get intervention parameters for this node
        is_intervened = intervention_spec.mask[node_idx]
        int_type = intervention_spec.type[node_idx]
        int_val = intervention_spec.value[node_idx]
        
        # Apply SOFT_SCALE (if active, scale the mechanism output, else 1.0)
        is_scale = is_intervened * (int_type == InterventionType.SOFT_SCALE)
        scale_factor = is_scale * int_val + (1.0 - is_scale) * 1.0
        mech_val = mech_val * scale_factor
        
        # Apply SOFT_SHIFT (if active, shift the mechanism output, else 0.0)
        is_shift = is_intervened * (int_type == InterventionType.SOFT_SHIFT)
        shift_amount = is_shift * int_val + (1.0 - is_shift) * 0.0
        mech_val = mech_val + shift_amount
        
        # Add observational noise
        observational_val = mech_val + noise[node_idx]
        
        # Apply HARD intervention (if active, override entirely)
        is_hard = is_intervened * (int_type == InterventionType.HARD)
        final_val = is_hard * (int_val + noise[node_idx]) + (1.0 - is_hard) * observational_val
        
        # Update values vector with the newly computed node value
        next_vals = current_vals.at[node_idx].set(final_val)
        return next_vals, None

    # We use jax.lax.scan over the static sequence [0, 1, ..., d-1]
    final_vals, _ = jax.lax.scan(scan_fn, vals, jnp.arange(config.d))
    return final_vals

@partial(jax.jit, static_argnums=(2, 3))
def sample_scm(key: jax.Array, 
               state: EnvState, 
               config: SCMConfig, 
               num_samples: int, 
               intervention_spec: InterventionSpec) -> jax.Array:
    """
    Samples multiple observations from the SCM.
    Returns: [num_samples, d] array of samples.
    """
    keys = jax.random.split(key, num_samples)
    
    # Vectorize the sampling over the batch of keys
    sampler = jax.vmap(_sample_single_scm, in_axes=(0, None, None, None))
    return sampler(keys, state, config, intervention_spec)
