import jax
import jax.numpy as jnp
from functools import partial
from src.types import SCMConfig, EnvState, InterventionSpec, InterventionType
from src.functional import apply_mechanism, generate_noise

def _sample_single_scm(key: jax.Array, 
                       state: EnvState, 
                       intervention_spec: InterventionSpec,
                       d: int,
                       mechanism_type: int,
                       noise_type: int,
                       noise_scale: float) -> jax.Array:
    """Samples a single realization of all d variables sequentially."""
    vals = jnp.zeros(d)
    
    noise_key, _ = jax.random.split(key)
    noise = generate_noise(noise_key, noise_type, (d,), noise_scale)

    def scan_fn(current_vals, i):
        node_idx = state.topological_order[i]
        
        mech_val = apply_mechanism(current_vals, node_idx, mechanism_type, state.scm_params)
        
        is_intervened = intervention_spec.mask[node_idx]
        int_type = intervention_spec.type[node_idx]
        int_val = intervention_spec.value[node_idx]
        
        is_scale = is_intervened * (int_type == InterventionType.SOFT_SCALE)
        scale_factor = is_scale * int_val + (1.0 - is_scale) * 1.0
        mech_val = mech_val * scale_factor
        
        is_shift = is_intervened * (int_type == InterventionType.SOFT_SHIFT)
        shift_amount = is_shift * int_val + (1.0 - is_shift) * 0.0
        mech_val = mech_val + shift_amount
        
        observational_val = mech_val + noise[node_idx]
        
        is_hard = is_intervened * (int_type == InterventionType.HARD)
        final_val = is_hard * (int_val + noise[node_idx]) + (1.0 - is_hard) * observational_val
        
        next_vals = current_vals.at[node_idx].set(final_val)
        return next_vals, None

    final_vals, _ = jax.lax.scan(scan_fn, vals, jnp.arange(d))
    return final_vals

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _sample_scm_jitted(key: jax.Array, 
                       state: EnvState, 
                       intervention_spec: InterventionSpec,
                       num_samples: int,
                       d: int,
                       mechanism_type: int,
                       noise_type: int,
                       noise_scale: float) -> jax.Array:
    keys = jax.random.split(key, num_samples)
    sampler = jax.vmap(_sample_single_scm, in_axes=(0, None, None, None, None, None, None))
    return sampler(keys, state, intervention_spec, d, mechanism_type, noise_type, noise_scale)

def sample_scm(key: jax.Array, 
               state: EnvState, 
               config: SCMConfig, 
               num_samples: int, 
               intervention_spec: InterventionSpec) -> jax.Array:
    """
    Samples multiple observations from the SCM.
    Returns: [num_samples, d] array of samples.
    """
    # Unwrap config into static integer values for compilation to bypass chex.dataclass hashing issues
    d_static = int(config.d)
    mech_static = int(config.mechanism_type)
    noise_static = int(config.noise_type)
    
    return _sample_scm_jitted(key, state, intervention_spec, num_samples, 
                              d_static, mech_static, noise_static, config.noise_scale)
