import numpy as np
from src.types import ActionCategory

def estimate_graph_from_obs(obs: np.ndarray, d: int, agent_id: int, threshold: float = 0.25) -> np.ndarray:
    """
    Statistical heuristic graph estimator for baseline agents:
    1. Extracts cov_obs, cov_run, and invariance asymmetry from observation.
    2. Identifies edges via correlation thresholding.
    3. Directs edges using empirical invariance asymmetry from interventions.
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
                    
    return graph_pred

class RandomAgent:
    def __init__(self, agent_id: int, d: int = 4, threshold: float = 0.25):
        self.agent_id = agent_id
        self.d = d
        self.threshold = threshold
        self.local_nodes = [0, 1] if agent_id == 0 else [2, 3]
        self.boundary_nodes = [1, 2]
        
    def act(self, obs, avail_actions=None):
        cat = np.random.choice([
            ActionCategory.LOCAL_INTERVENTION, 
            ActionCategory.PEER_REQUEST, 
            ActionCategory.NOOP
        ])
        
        if cat == ActionCategory.LOCAL_INTERVENTION:
            target = np.random.choice(self.local_nodes)
        elif cat == ActionCategory.PEER_REQUEST:
            target = np.random.choice(self.boundary_nodes)
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
        # Only request peer intervention on the other agent's boundary node
        peer_boundary = [2] if agent_id == 0 else [1]
        
        self.targets = self.local_nodes + peer_boundary
        self.step = 0
        
    def act(self, obs, avail_actions=None):
        target = self.targets[self.step % len(self.targets)]
        self.step += 1
        
        if target in self.local_nodes:
            cat = ActionCategory.LOCAL_INTERVENTION
        else:
            cat = ActionCategory.PEER_REQUEST
            
        graph_pred = estimate_graph_from_obs(obs, self.d, self.agent_id, self.threshold)
        return (int(cat), int(target)), graph_pred
