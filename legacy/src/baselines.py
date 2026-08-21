import numpy as np
from legacy.src.types import ActionCategory, STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK, compute_edge_authority_mask

def estimate_graph_from_obs(obs: np.ndarray, d: int, agent_id: int, threshold: float = 0.25) -> np.ndarray:
    """
    Statistical heuristic graph estimator for baseline agents:
    1. Extracts cov_obs, cov_run, and invariance asymmetry from observation.
    2. Identifies edges via correlation thresholding.
    3. Directs edges using empirical invariance asymmetry from interventions.
    Edges are restricted to the agent's edge-authority domain (its own local pair, or the
    shared boundary pair): an agent may observe a peer's boundary node for covariance purposes,
    but may never assert an edge directly from its own private node into the peer's domain.
    """
    d2 = d * d
    obs_arr = np.asarray(obs).flatten()
    if obs_arr.shape[0] >= 3 * d2:
        cov_obs = obs_arr[:d2].reshape(d, d)
        asym = obs_arr[2*d2:3*d2].reshape(d, d)
    else:
        cov_obs = obs_arr[:d2].reshape(d, d)
        asym = np.zeros((d, d))
        
    diag = np.diag(cov_obs)
    denom = np.sqrt(np.maximum(1e-5, np.outer(diag, diag)))
    corr = np.abs(cov_obs) / denom
    
    graph_pred = np.zeros((d, d))
    obs_nodes = [0, 1, 2] if agent_id == 0 else [1, 2, 3]
    
    for i in obs_nodes:
        for j in obs_nodes:
            if i >= j:
                continue
            if corr[i, j] > threshold:
                if asym[i, j] > 0.05:
                    graph_pred[i, j] = 1.0
                elif asym[j, i] > 0.05:
                    graph_pred[j, i] = 1.0
                else:
                    # Fallback to upper triangular orientation
                    graph_pred[i, j] = 1.0

    if d == 4:
        edge_mask = np.array(compute_edge_authority_mask(STANDARD_LOCAL_MASKS[agent_id], STANDARD_BOUNDARY_MASK))
        graph_pred = graph_pred * edge_mask

    return graph_pred

class RandomAgent:
    def __init__(self, agent_id: int, d: int = 4, threshold: float = 0.25):
        self.agent_id = agent_id
        self.d = d
        self.threshold = threshold
        self.local_nodes = [0, 1] if agent_id == 0 else [2, 3]
        self.boundary_nodes = [1, 2]
        # Valid intervention targets: own local domain, union the shared boundary pair.
        self.valid_targets = sorted(set(self.local_nodes + self.boundary_nodes))

    def act(self, obs, avail_actions=None):
        cat = np.random.choice([ActionCategory.INTERVENE, ActionCategory.NOOP])

        if cat == ActionCategory.INTERVENE:
            target = np.random.choice(self.valid_targets)
        else:
            target = 0

        graph_pred = estimate_graph_from_obs(obs, self.d, self.agent_id, self.threshold)
        return (int(cat), int(target)), graph_pred

class RoundRobinAgent:
    def __init__(self, agent_id: int, d: int = 4, threshold: float = 0.25):
        self.agent_id = agent_id
        self.d = d
        self.threshold = threshold
        self.local_nodes = [0, 1] if agent_id == 0 else [2, 3]
        self.boundary_nodes = [1, 2]
        # Cycle through every valid intervention target: own local domain, union the
        # shared boundary pair.
        self.targets = sorted(set(self.local_nodes + self.boundary_nodes))
        self.step = 0

    def act(self, obs, avail_actions=None):
        target = self.targets[self.step % len(self.targets)]
        self.step += 1
        cat = ActionCategory.INTERVENE

        graph_pred = estimate_graph_from_obs(obs, self.d, self.agent_id, self.threshold)
        return (int(cat), int(target)), graph_pred

class VanillaAgent:
    """
    Vanilla minimal 4-action discrete baseline agent with statistical graph estimation.
    Action 0: Intervene on private Z node (Z1 for agent 0, Z2 for agent 1)
    Action 1: Intervene on local X node (X1 for agent 0, X2 for agent 1)
    Action 2: Intervene on peer boundary X node (X2 for agent 0, X1 for agent 1)
    Action 3: NO-OP (do nothing)
    """
    def __init__(self, agent_id: int, d: int = 4, threshold: float = 0.25):
        self.agent_id = agent_id
        self.d = d
        self.threshold = threshold
        self.z_node = 0 if agent_id == 0 else 3
        self.x_node = 1 if agent_id == 0 else 2
        self.peer_node = 2 if agent_id == 0 else 1

    def act(self, obs, avail_actions=None):
        act_idx = int(np.random.choice([0, 1, 2, 3]))
        if act_idx == 0:
            cat, target = ActionCategory.INTERVENE, self.z_node
        elif act_idx == 1:
            cat, target = ActionCategory.INTERVENE, self.x_node
        elif act_idx == 2:
            cat, target = ActionCategory.INTERVENE, self.peer_node
        else:
            cat, target = ActionCategory.NOOP, 0

        graph_pred = estimate_graph_from_obs(obs, self.d, self.agent_id, self.threshold)
        return (int(cat), int(target)), graph_pred
