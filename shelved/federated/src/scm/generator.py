import numpy as np
import networkx as nx

class LinearGaussianSCM:
    """
    10-variable Linear Gaussian Structural Causal Model.
    
    Generates synthetic data according to a specific Directed Acyclic Graph (DAG)
    which contains an overlapping v-structure (V4 -> V5 <- V6).
    Note: Variables are 0-indexed in code (V1=0, ..., V10=9).
    So the v-structure is 3 -> 4 <- 5.
    """
    
    def __init__(self, num_vars=10, random_seed=42):
        self.num_vars = num_vars
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize adjacency matrix
        # A[i, j] = 1 means i -> j
        self.adjacency_matrix = np.zeros((self.num_vars, self.num_vars), dtype=int)
        
        # Define the edges for the ground truth DAG
        # We need a v-structure at V5 (index 4) from V4 (index 3) and V6 (index 5)
        edges = [
            (0, 1), # V1 -> V2
            (1, 2), # V2 -> V3
            (2, 3), # V3 -> V4
            (3, 4), # V4 -> V5  (part of v-structure)
            (5, 4), # V6 -> V5  (part of v-structure)
            (5, 6), # V6 -> V7
            (6, 7), # V7 -> V8
            (7, 8), # V8 -> V9
            (8, 9)  # V9 -> V10
        ]
        
        for u, v in edges:
            self.adjacency_matrix[u, v] = 1
            
        # Verify it is a DAG
        G = nx.DiGraph(self.adjacency_matrix)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("The defined graph is not a DAG!")
            
        # Create a weight matrix (random weights between 0.5 and 1.5 or -1.5 and -0.5)
        self.weight_matrix = np.zeros_like(self.adjacency_matrix, dtype=float)
        for u, v in edges:
            sign = self.rng.choice([-1, 1])
            weight = sign * self.rng.uniform(0.5, 1.5)
            self.weight_matrix[u, v] = weight
            
        self.topological_order = list(nx.topological_sort(G))

    def generate_data(self, num_samples, noise_std=1.0):
        """
        Generates observational data from the SCM.
        
        Args:
            num_samples: Number of samples to generate.
            noise_std: Standard deviation of the Gaussian noise.
            
        Returns:
            A numpy array of shape (num_samples, num_vars)
        """
        data = np.zeros((num_samples, self.num_vars))
        
        # Generate data in topological order to satisfy causal dependencies
        for node in self.topological_order:
            # Noise term e_j ~ N(0, sigma^2)
            noise = self.rng.normal(0, noise_std, size=num_samples)
            
            # Parents of the current node
            parents = np.where(self.adjacency_matrix[:, node] == 1)[0]
            
            if len(parents) == 0:
                data[:, node] = noise
            else:
                # V_j = sum(W_{i, j} * V_i) + e_j
                parent_values = data[:, parents]
                weights = self.weight_matrix[parents, node]
                data[:, node] = parent_values @ weights + noise
                
        return data
