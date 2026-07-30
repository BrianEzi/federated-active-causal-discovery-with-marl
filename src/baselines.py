import numpy as np
from src.types import ActionCategory

class RandomAgent:
    def __init__(self, agent_id: int, d: int = 4):
        self.agent_id = agent_id
        self.d = d
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
            
        # Predict empty graph to avoid cycle penalties
        graph_pred = np.zeros((self.d, self.d))
        
        return (int(cat), int(target)), graph_pred

class RoundRobinAgent:
    def __init__(self, agent_id: int, d: int = 4):
        self.agent_id = agent_id
        self.d = d
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
            
        graph_pred = np.zeros((self.d, self.d))
        
        return (int(cat), int(target)), graph_pred
