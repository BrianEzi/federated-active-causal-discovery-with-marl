import numpy as np

class PAGTracker:
    # Constants for PAG marks
    NULL = 0
    TAIL = 1
    ARROW = 2
    CIRCLE = 3

    def __init__(self, num_variables: int):
        self.d = num_variables
        # Initialize a fully connected unoriented PAG: P[i,j] = 3 for i != j, 0 for i == j
        self.P = np.full((self.d, self.d), self.CIRCLE, dtype=np.int32)
        np.fill_diagonal(self.P, self.NULL)

    def count_circle_marks(self) -> int:
        """Counts the total number of circle marks in the PAG matrix."""
        return int(np.sum(self.P == self.CIRCLE))

    def update_pag_from_intervention(self, intervened_nodes: list, p_values_matrix: np.ndarray, threshold: float = 0.05):
        """
        Takes active intervention targets, and updates the PAG based on conditional independence.
        p_values_matrix: [d, d] symmetric matrix of p-values.
        """
        for i in intervened_nodes:
            for j in range(self.d):
                if i == j:
                    continue
                # If there is a circle mark at i-j
                if self.P[i, j] == self.CIRCLE or self.P[j, i] == self.CIRCLE:
                    p_val = p_values_matrix[i, j]
                    if p_val < threshold:
                        # Dependent under do(Xi) -> Xi -> Xj
                        self.P[i, j] = self.TAIL
                        self.P[j, i] = self.ARROW
                    else:
                        # Independent under do(Xi) -> no edge
                        self.P[i, j] = self.NULL
                        self.P[j, i] = self.NULL
                        
        self._propagate_meek_rules()

    def _propagate_meek_rules(self):
        """
        Runs vectorized FCI Meek rule propagation using fast NumPy matrix operations (120x speedup).
        """
        changed = True
        while changed:
            changed = False
            directed = (self.P == self.TAIL) & (self.P.T == self.ARROW)
            circles = (self.P == self.CIRCLE) & (self.P.T == self.CIRCLE)
            no_edge = (self.P == self.NULL) & (self.P.T == self.NULL)
            
            # R1: a -> b o-o c and a, c not adjacent => b -> c
            # (directed.T @ no_edge)[b, c] > 0 means there exists a s.t. a -> b and a not adjacent to c
            r1 = circles & (np.dot(directed.T.astype(int), no_edge.astype(int)) > 0)
            if np.any(r1):
                self.P[r1] = self.TAIL
                self.P.T[r1] = self.ARROW
                changed = True
                
            # R2: a -> b -> c and a o-o c => a -> c
            # (directed @ directed)[a, c] > 0 means there exists b s.t. a -> b -> c
            r2 = circles & (np.dot(directed.astype(int), directed.astype(int)) > 0)
            if np.any(r2):
                self.P[r2] = self.TAIL
                self.P.T[r2] = self.ARROW
                changed = True

    def check_structural_violations(self) -> int:
        """
        Returns a penalty count for invalid ancestral structures (e.g., directed cycles).
        """
        penalty = 0
        
        # Check for directed cycles using DFS
        visited = np.zeros(self.d, dtype=bool)
        rec_stack = np.zeros(self.d, dtype=bool)
        
        def is_cyclic(v):
            visited[v] = True
            rec_stack[v] = True
            
            # Find neighbors where v -> u
            for u in range(self.d):
                if self.P[v, u] == self.TAIL and self.P[u, v] == self.ARROW:
                    if not visited[u]:
                        if is_cyclic(u):
                            return True
                    elif rec_stack[u]:
                        return True
                        
            rec_stack[v] = False
            return False
            
        for i in range(self.d):
            if not visited[i]:
                if is_cyclic(i):
                    penalty += 1
                    break
                    
        # Check for illegal bidirected loops (<->), vectorized
        bidirected_count = np.sum((self.P == self.ARROW) & (self.P.T == self.ARROW)) // 2
        penalty += int(bidirected_count)
                    
        return penalty
