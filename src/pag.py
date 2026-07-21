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
        Runs a simplified FCI Meek rule propagation to orient edges.
        """
        changed = True
        while changed:
            changed = False
            for b in range(self.d):
                for c in range(self.d):
                    if b == c: continue
                    
                    # R1: a -> b o-o c and a, c not adjacent => b -> c
                    if self.P[b, c] == self.CIRCLE and self.P[c, b] == self.CIRCLE:
                        for a in range(self.d):
                            if a == b or a == c: continue
                            # check a -> b
                            if self.P[a, b] == self.TAIL and self.P[b, a] == self.ARROW:
                                # check a, c not adjacent
                                if self.P[a, c] == self.NULL and self.P[c, a] == self.NULL:
                                    self.P[b, c] = self.TAIL
                                    self.P[c, b] = self.ARROW
                                    changed = True
                                    
                    # R2: a -> b -> c and a o-o c => a -> c
                    if self.P[b, c] == self.TAIL and self.P[c, b] == self.ARROW:
                        for a in range(self.d):
                            if a == b or a == c: continue
                            # check a -> b
                            if self.P[a, b] == self.TAIL and self.P[b, a] == self.ARROW:
                                # check a o-o c
                                if self.P[a, c] == self.CIRCLE and self.P[c, a] == self.CIRCLE:
                                    self.P[a, c] = self.TAIL
                                    self.P[c, a] = self.ARROW
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
                    
        # Check for illegal bidirected loops (assuming causal sufficiency, no <-> allowed)
        for i in range(self.d):
            for j in range(i+1, self.d):
                if self.P[i, j] == self.ARROW and self.P[j, i] == self.ARROW:
                    penalty += 1
                    
        return penalty
