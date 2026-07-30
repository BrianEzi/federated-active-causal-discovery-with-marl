import numpy as np

def stitch_predicted_dags(predicted_probs: dict, d: int) -> tuple[np.ndarray, bool]:
    """
    Stitches the predicted edge probabilities into a single global DAG adjacency matrix (0 or 1).
    predicted_probs maps 'agent_k' -> [d, d] matrix of edge probabilities.
    Returns (stitched_dag, has_cycle).
    """
    prob_1 = predicted_probs["agent_0"]
    prob_2 = predicted_probs["agent_1"]
    
    global_probs = np.zeros((d, d))
    
    # Agent 1 predicts edges among {0, 1, 2}
    global_probs[0, 0:3] = prob_1[0, 0:3]
    global_probs[1:3, 0] = prob_1[1:3, 0]
    
    # Agent 2 predicts edges among {1, 2, 3}
    global_probs[3, 1:4] = prob_2[3, 1:4]
    global_probs[1:3, 3] = prob_2[1:3, 3]
    
    # Overlapping boundary nodes: {1, 2}
    # Both agents predict edges between X1 (1) and X2 (2). Average their probabilities.
    global_probs[1:3, 1:3] = (prob_1[1:3, 1:3] + prob_2[1:3, 1:3]) / 2.0
    
    # Clear self-loops
    np.fill_diagonal(global_probs, 0.0)
    
    # Threshold to discrete DAG
    stitched_dag = (global_probs > 0.5).astype(np.float32)
    
    # Enforce no bidirectional edges: if i -> j and j -> i, remove both or keep the stronger one?
    # In continuous probabilities, one is usually stronger. But if thresholded, it might create a 2-cycle.
    # We will let the cycle detector catch 2-cycles and penalize them.
    has_cycle = detect_cycle(stitched_dag)
    
    return stitched_dag, has_cycle

def detect_cycle(adj: np.ndarray) -> bool:
    d = adj.shape[0]
    visited = np.zeros(d, dtype=bool)
    rec_stack = np.zeros(d, dtype=bool)
    
    def dfs(v):
        visited[v] = True
        rec_stack[v] = True
        for u in range(d):
            if adj[v, u] > 0:
                if not visited[u]:
                    if dfs(u):
                        return True
                elif rec_stack[u]:
                    return True
        rec_stack[v] = False
        return False
        
    for i in range(d):
        if not visited[i]:
            if dfs(i):
                return True
    return False
