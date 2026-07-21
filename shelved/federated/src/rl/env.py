import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from src.rl.bic import GaussianBIC

class CausalDiscoveryEnv(gym.Env):
    """
    RL Environment for Causal Discovery using local discrete edge edits.
    """
    
    # Action Constants
    ACTION_ADD = 0
    ACTION_REMOVE = 1
    ACTION_REVERSE = 2
    
    def __init__(self, data: np.ndarray, max_steps: int = 100,
                 lambda_sparse: float = 0.0, step_cost: float = 0.1, 
                 invalid_penalty: float = 10.0, max_edges: int = None):
        """
        Args:
            data: Observational dataset to score the DAG against.
            max_steps: Maximum number of steps per episode.
            lambda_sparse: Optional L0 regularization term for sparsity.
            step_cost: Cost applied at every step to encourage termination.
            invalid_penalty: Heavy penalty for bypassing the action mask.
            max_edges: Maximum allowed edges (L0 budget constraint).
        """
        super().__init__()
        
        self.data = data
        self.num_samples, self.num_vars = data.shape
        self.max_steps = max_steps
        self.lambda_sparse = lambda_sparse
        self.step_cost = step_cost
        self.invalid_penalty = invalid_penalty
        
        if max_edges is None:
            self.max_edges = (self.num_vars * (self.num_vars - 1)) // 2
        else:
            self.max_edges = max_edges
            
        self.bic_calculator = GaussianBIC(data)
        
        # State observation: flattened adjacency matrix
        self.observation_space = spaces.MultiBinary(self.num_vars * self.num_vars)
        
        # Action mapping
        # Total possible edges is num_vars * (num_vars - 1)
        # For each possible edge, 3 actions: ADD, REMOVE, REVERSE
        self.num_possible_edges = self.num_vars * (self.num_vars - 1)
        self.action_space = spaces.Discrete(3 * self.num_possible_edges)
        
        # Mapping from action index to (operation, i, j)
        self.action_mapping = []
        for op in [self.ACTION_ADD, self.ACTION_REMOVE, self.ACTION_REVERSE]:
            for i in range(self.num_vars):
                for j in range(self.num_vars):
                    if i != j:
                        self.action_mapping.append((op, i, j))
                        
        self.current_step = 0
        self.adjacency_matrix = np.zeros((self.num_vars, self.num_vars), dtype=int)
        self.current_bic = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        # Start with an empty graph
        self.adjacency_matrix = np.zeros((self.num_vars, self.num_vars), dtype=int)
        
        # Initial score
        self.current_bic = self.bic_calculator.compute_graph_bic(self.adjacency_matrix)
        
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        return self.adjacency_matrix.flatten()
        
    def _get_info(self):
        return {
            'action_mask': self.get_action_mask(),
            'bic': self.current_bic,
            'edges': np.sum(self.adjacency_matrix)
        }
        
    def _simulate_action(self, op: int, i: int, j: int) -> np.ndarray:
        """Applies the action to a copy of the adjacency matrix."""
        new_matrix = self.adjacency_matrix.copy()
        if op == self.ACTION_ADD:
            new_matrix[i, j] = 1
        elif op == self.ACTION_REMOVE:
            new_matrix[i, j] = 0
        elif op == self.ACTION_REVERSE:
            new_matrix[i, j] = 0
            new_matrix[j, i] = 1
        return new_matrix

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a binary array of shape (action_space.n,) indicating valid actions.
        0 = Invalid/Masked out, 1 = Valid.
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        current_edges = np.sum(self.adjacency_matrix)
        
        for action_idx, (op, i, j) in enumerate(self.action_mapping):
            # 1. Basic Validity Checks
            if op == self.ACTION_ADD:
                # Can't add if edge already exists, or if opposite edge exists (would create immediate 2-cycle)
                if self.adjacency_matrix[i, j] == 1 or self.adjacency_matrix[j, i] == 1:
                    continue
                # Enforce L0 budget
                if current_edges >= self.max_edges:
                    continue
            elif op == self.ACTION_REMOVE:
                # Can't remove if edge doesn't exist
                if self.adjacency_matrix[i, j] == 0:
                    continue
            elif op == self.ACTION_REVERSE:
                # Can't reverse if edge doesn't exist
                if self.adjacency_matrix[i, j] == 0:
                    continue
                    
            # 2. Acyclicity Check
            if op in [self.ACTION_ADD, self.ACTION_REVERSE]:
                new_matrix = self._simulate_action(op, i, j)
                # Fast cycle check: if we add j <- i, we just need to check if there is a path from j to i
                # For reverse (j -> i becomes i -> j), we check if there is a path from i to j in the original graph without j->i.
                # Easiest and safest is to just check the new graph
                try:
                    # networkx finds cycles fast enough for small graphs (N=10)
                    G = nx.DiGraph(new_matrix)
                    if not nx.is_directed_acyclic_graph(G):
                        continue
                except nx.NetworkXError:
                    continue
                    
            # If we passed all checks, the action is valid
            mask[action_idx] = 1
            
        return mask
        
    def step(self, action: int):
        self.current_step += 1
        
        op, i, j = self.action_mapping[action]
        mask = self.get_action_mask()
        
        # Check if action is valid
        if mask[action] == 0:
            # Invalid action penalized
            reward = -self.invalid_penalty
            terminated = False
            truncated = (self.current_step >= self.max_steps)
            return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        # Apply action
        self.adjacency_matrix = self._simulate_action(op, i, j)
        
        # Calculate new BIC
        new_bic = self.bic_calculator.compute_graph_bic(self.adjacency_matrix)
        
        # Calculate Reward: Delta BIC / N - lambda * ||A||_0 - c_step
        # Note: Lower BIC is better. So positive delta means improvement.
        delta_bic = (self.current_bic - new_bic) / self.num_vars
        l0_norm = np.sum(self.adjacency_matrix)
        
        reward = delta_bic - (self.lambda_sparse * l0_norm) - self.step_cost
        
        # Update current BIC
        self.current_bic = new_bic
        
        terminated = False
        truncated = (self.current_step >= self.max_steps)
        
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()
