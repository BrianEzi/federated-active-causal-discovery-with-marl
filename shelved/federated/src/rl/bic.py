import numpy as np

class GaussianBIC:
    """
    Computes the Bayesian Information Criterion (BIC) score for a given Directed Acyclic Graph (DAG)
    assuming a linear Gaussian structural causal model.
    """
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: A numpy array of shape (num_samples, num_vars)
        """
        self.data = data
        self.num_samples, self.num_vars = data.shape

    def compute_node_bic(self, node: int, parents: list[int]) -> float:
        """
        Computes the BIC score for a single node given its parents.
        
        Args:
            node: Index of the target node.
            parents: List of indices of the parent nodes.
            
        Returns:
            The local BIC score for the node.
        """
        y = self.data[:, node]
        
        if len(parents) == 0:
            # No parents: likelihood based on simple mean and variance
            rss = np.sum((y - np.mean(y)) ** 2)
            k = 1  # 1 parameter (the mean, variance is usually counted but we'll stick to edge counts for sparsity)
            # Actually, standard score-based methods penalize number of edges.
            # We'll use k = |parents|. Since k=0 here, penalty is 0.
            # But let's be rigorous: k = len(parents).
            k = 0
        else:
            X = self.data[:, parents]
            # Add intercept
            X = np.hstack((np.ones((self.num_samples, 1)), X))
            # Solve OLS
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
            rss = np.sum(residuals ** 2)
            k = len(parents)
            
        # Variance estimate
        var_hat = rss / self.num_samples
        # Avoid log(0)
        var_hat = max(var_hat, 1e-8)
        
        # Log-likelihood: ln L = -m/2 * ln(2 * pi * var_hat) - m/2
        log_likelihood = -0.5 * self.num_samples * (np.log(2 * np.pi * var_hat) + 1)
        
        # BIC = -2 * ln(L) + k * ln(m)
        bic = -2 * log_likelihood + k * np.log(self.num_samples)
        
        return bic

    def compute_graph_bic(self, adjacency_matrix: np.ndarray) -> float:
        """
        Computes the total BIC score for the entire graph.
        
        Args:
            adjacency_matrix: A binary numpy array of shape (num_vars, num_vars)
                              where A[i, j] = 1 means i -> j.
                              
        Returns:
            The total BIC score. Lower is better.
        """
        if adjacency_matrix.shape != (self.num_vars, self.num_vars):
            raise ValueError(f"Adjacency matrix must be {self.num_vars}x{self.num_vars}")
            
        total_bic = 0.0
        for v in range(self.num_vars):
            parents = np.where(adjacency_matrix[:, v] == 1)[0].tolist()
            total_bic += self.compute_node_bic(v, parents)
            
        return total_bic
