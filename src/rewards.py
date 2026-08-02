import numpy as np

def compute_ippo_rewards(
    stitched_dag: np.ndarray, 
    true_dag: np.ndarray, 
    has_cycle: bool, 
    cycle_penalty: float = 10.0, 
    edge_penalty: float = 1.0,
    max_steps: float = 1.0,
    info_gains: dict = None,
    intrinsic_coef: float = 0.0
) -> dict:
    """
    Computes the mixed cooperative SHD penalty reward for the IPPO agents.
    If max_steps > 1.0, normalizes step penalties by max_steps to decouple return scale from horizon.
    If intrinsic_coef > 0.0, adds intrinsic information gain curiosity bonus.
    Returns: {"agent_0": r1, "agent_1": r2}
    """
    diff = np.abs(stitched_dag - true_dag)
    
    # Local penalty for Agent 1 (edges involving its private node Z1=0)
    a1_local_errors = np.sum(diff[0, :]) + np.sum(diff[:, 0])
    
    # Local penalty for Agent 2 (edges involving its private node Z2=3)
    a2_local_errors = np.sum(diff[3, :]) + np.sum(diff[:, 3])
    
    # Boundary errors (edges between X1=1 and X2=2)
    boundary_errors = diff[1, 2] + diff[2, 1]
    
    # Base reward is the negative SHD penalty
    r1 = -(a1_local_errors + boundary_errors) * edge_penalty
    r2 = -(a2_local_errors + boundary_errors) * edge_penalty
    
    if has_cycle:
        r1 -= cycle_penalty
        r2 -= cycle_penalty
        
    scale = 1.0 / max(1.0, float(max_steps))
    r1 = r1 * scale
    r2 = r2 * scale
    
    if info_gains is not None and intrinsic_coef > 0.0:
        r1 += intrinsic_coef * float(info_gains.get("agent_0", 0.0))
        r2 += intrinsic_coef * float(info_gains.get("agent_1", 0.0))
        
    return {"agent_0": float(r1), "agent_1": float(r2)}


import jax
import jax.numpy as jnp
from typing import Tuple, Optional

@jax.jit
def jitted_compute_ippo_rewards(
    stitched_dag: jax.Array, 
    true_dag: jax.Array, 
    has_cycle: jax.Array, 
    cycle_penalty: float = 10.0, 
    edge_penalty: float = 1.0,
    max_steps: float = 1.0,
    info_gains: Optional[jax.Array] = None,
    intrinsic_coef: float = 0.0
) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the mixed cooperative SHD penalty reward for the IPPO agents in JAX.
    If max_steps > 1.0, normalizes step penalties by max_steps.
    If intrinsic_coef > 0.0, adds intrinsic information gain curiosity bonus.
    Supports single [d, d] or batched [B, d, d].
    Returns: (r1, r2)
    """
    diff = jnp.abs(stitched_dag - true_dag)
    scale = 1.0 / jnp.maximum(1.0, max_steps)
    if diff.ndim == 2:
        a1_local_errors = jnp.sum(diff[0, :]) + jnp.sum(diff[:, 0])
        a2_local_errors = jnp.sum(diff[3, :]) + jnp.sum(diff[:, 3])
        boundary_errors = diff[1, 2] + diff[2, 1]
        
        pen = jnp.where(has_cycle, cycle_penalty, 0.0)
        r1 = (-(a1_local_errors + boundary_errors) * edge_penalty - pen) * scale
        r2 = (-(a2_local_errors + boundary_errors) * edge_penalty - pen) * scale
        
        if info_gains is not None:
            r1 = r1 + intrinsic_coef * info_gains[0]
            r2 = r2 + intrinsic_coef * info_gains[1]
            
        return r1, r2
    else:
        a1_local_errors = jnp.sum(diff[:, 0, :], axis=-1) + jnp.sum(diff[:, :, 0], axis=-1)
        a2_local_errors = jnp.sum(diff[:, 3, :], axis=-1) + jnp.sum(diff[:, :, 3], axis=-1)
        boundary_errors = diff[:, 1, 2] + diff[:, 2, 1]
        
        pen = jnp.where(has_cycle, cycle_penalty, 0.0)
        r1 = (-(a1_local_errors + boundary_errors) * edge_penalty - pen) * scale
        r2 = (-(a2_local_errors + boundary_errors) * edge_penalty - pen) * scale
        
        if info_gains is not None:
            r1 = r1 + intrinsic_coef * info_gains[:, 0]
            r2 = r2 + intrinsic_coef * info_gains[:, 1]
            
        return r1, r2

