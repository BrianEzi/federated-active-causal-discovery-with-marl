import numpy as np

def compute_global_reward(prev_circle_count: int, 
                          curr_circle_count: int, 
                          joint_action: np.ndarray, 
                          action_costs: np.ndarray,
                          structural_violations: int,
                          penalty_weight: float = 1.0) -> float:
    """
    Computes the global scalar reward for the federated system.
    
    prev_circle_count: int, number of circle marks in the previous PAG state.
    curr_circle_count: int, number of circle marks in the current PAG state.
    joint_action: [K] array of actions taken by the agents. Assuming actions < d correspond to intervening on a node.
                  We assume that if action < d, it's an intervention.
    action_costs: [K] array of costs for each agent's action.
    structural_violations: int, number of structural violations detected by the PAG Tracker.
    """
    # Delta Circles (Positive reward for orienting ambiguous marks)
    delta_circles = prev_circle_count - curr_circle_count
    
    # Action Cost
    # Only penalize if the action is an intervention (e.g. action < len(d))
    # We will assume all passed action_costs correspond to valid interventions for simplicity,
    # or the wrapper filters them. Let's just sum the provided action_costs array.
    total_action_cost = np.sum(action_costs)
    
    # Penalty
    penalty = penalty_weight * structural_violations
    
    # R_t
    reward = delta_circles - total_action_cost - penalty
    
    return float(reward)
