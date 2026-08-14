import numpy as np

def compute_ippo_rewards(
    stitched_dag: np.ndarray, 
    true_dag: np.ndarray, 
    has_cycle: bool, 
    cycle_penalty: float = 10.0, 
    edge_penalty: float = 1.0,
    max_steps: float = 1.0,
    info_gains: dict = None,
    intrinsic_coef: float = 0.0,
    impact_scores: dict = None,
    impact_coef: float = 0.0,
    reward_density: str = "dense",
    is_terminal: bool = False,
    prev_shd: float = None,
    num_agents: int = 2,
    holding_bonus: float = 0.0,
    action_friction: float = 0.0,
    actions: dict = None,
    remaining_budgets: dict = None,
    budget_bonus_coef: float = 0.1
) -> dict:
    """
    Computes mixed cooperative SHD penalty & ablation rewards for IPPO agents.
    Supports K=2 (federated) and K=1 (single-agent centralized).
    """
    diff = np.abs(stitched_dag - true_dag)
    
    if num_agents == 1:
        current_e1 = float(np.sum(diff))
        current_e2 = 0.0
        current_shd = current_e1
    else:
        # Personal local penalty for Agent 0 (edges involving Z1=0)
        a1_local_errors = np.sum(diff[0, :]) + np.sum(diff[:, 0])
        
        # Personal local penalty for Agent 1 (edges involving Z2=3)
        a2_local_errors = np.sum(diff[3, :]) + np.sum(diff[:, 3])
        
        # Shared boundary errors (edges between X1=1 and X2=2)
        boundary_errors = diff[1, 2] + diff[2, 1]
        
        current_e1 = a1_local_errors + boundary_errors
        current_e2 = a2_local_errors + boundary_errors
        current_shd = a1_local_errors + a2_local_errors + boundary_errors
    
    if reward_density == "sparse":
        if is_terminal:
            r1 = -current_e1 * edge_penalty
            r2 = -current_e2 * edge_penalty
        else:
            r1 = 0.0
            r2 = 0.0
    else:
        if prev_shd is not None:
            if isinstance(prev_shd, (tuple, list)):
                r1 = (prev_shd[0] - current_e1) * edge_penalty
                r2 = (prev_shd[1] - current_e2) * edge_penalty
            elif isinstance(prev_shd, dict):
                r1 = (prev_shd.get("agent_0", current_e1) - current_e1) * edge_penalty
                r2 = (prev_shd.get("agent_1", current_e2) - current_e2) * edge_penalty
            else:
                shd_improvement = prev_shd - current_shd
                r1 = shd_improvement * edge_penalty
                r2 = shd_improvement * edge_penalty
        else:
            r1 = -current_e1 * edge_penalty
            r2 = -current_e2 * edge_penalty
            
        # Holding bonus: reward staying at SHD=0 when cycle-free
        if not has_cycle and holding_bonus > 0.0:
            if current_e1 == 0:
                r1 += holding_bonus
            if current_e2 == 0:
                r2 += holding_bonus
            
        # Action friction penalty
        if actions is not None and action_friction > 0.0:
            if actions.get("agent_0", 1) == 0:  # INTERVENE
                r1 -= action_friction
            if actions.get("agent_1", 1) == 0:  # INTERVENE
                r2 -= action_friction
    
    if has_cycle and (reward_density == "dense" or is_terminal):
        r1 -= cycle_penalty
        r2 -= cycle_penalty
        
    scale = 1.0 / max(1.0, float(max_steps))
    r1 = r1 * scale
    r2 = r2 * scale
    
    # Terminal budget conservation bonus
    if is_terminal and remaining_budgets is not None and budget_bonus_coef > 0.0:
        if current_e1 == 0:
            r1 += budget_bonus_coef * float(remaining_budgets.get("agent_0", 0.0))
        if current_e2 == 0:
            r2 += budget_bonus_coef * float(remaining_budgets.get("agent_1", 0.0))
    
    if info_gains is not None and intrinsic_coef > 0.0:
        r1 += intrinsic_coef * float(info_gains.get("agent_0", 0.0))
        r2 += intrinsic_coef * float(info_gains.get("agent_1", 0.0))
        
    if impact_scores is not None and impact_coef > 0.0:
        r1 += impact_coef * float(impact_scores.get("agent_0", 0.0))
        r2 += impact_coef * float(impact_scores.get("agent_1", 0.0))
        
    if num_agents == 1:
        return {"agent_0": float(r1)}
    return {
        "agent_0": float(r1),
        "agent_1": float(r2)
    }


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
    intrinsic_coef: float = 0.0,
    impact_scores: Optional[jax.Array] = None,
    impact_coef: float = 0.0
) -> Tuple[jax.Array, jax.Array]:
    """
    Computes mixed cooperative rewards for IPPO agents in JAX.
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
            
        if impact_scores is not None:
            r1 = r1 + impact_coef * impact_scores[0]
            r2 = r2 + impact_coef * impact_scores[1]
            
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
            
        if impact_scores is not None:
            r1 = r1 + impact_coef * impact_scores[:, 0]
            r2 = r2 + impact_coef * impact_scores[:, 1]
            
        return r1, r2

