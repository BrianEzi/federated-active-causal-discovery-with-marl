import numpy as np

def compute_global_reward(prev_circle_count: int, 
                          curr_circle_count: int, 
                          joint_action: np.ndarray, 
                          action_costs: np.ndarray,
                          structural_violations: int,
                          num_variables: int = 5,
                          circle_reward: float = 1.0,
                          noop_penalty: float = 0.5,
                          violation_penalty: float = 20.0) -> float:
    """
    Computes the global scalar reward for the federated system.
    """
    # 1. Delta Circles (Positive reward for orienting ambiguous marks)
    delta_circles = prev_circle_count - curr_circle_count
    r_circles = delta_circles * circle_reward
    
    # 2. Action Cost
    total_action_cost = np.sum(action_costs)
    
    # 3. NO-OP penalty when unresolved circles remain
    all_noop = np.all(joint_action == num_variables) if joint_action is not None else False
    r_noop = -noop_penalty if (all_noop and curr_circle_count > 0) else 0.0
    
    # 4. Structural Violation Penalty
    r_violations = -violation_penalty * structural_violations
    
    # R_t
    reward = r_circles - total_action_cost + r_noop + r_violations
    
    return float(reward)


