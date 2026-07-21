import numpy as np

def compute_shd(pred_dag: np.ndarray, true_dag: np.ndarray) -> int:
    """
    Computes the Structural Hamming Distance (SHD) between two DAG adjacency matrices.
    
    This function counts:
    - Missing edges (False Negatives)
    - Extra edges (False Positives)
    - Reversed edges
    
    Args:
        pred_dag: The predicted adjacency matrix (binary).
        true_dag: The ground-truth adjacency matrix (binary).
        
    Returns:
        The Structural Hamming Distance (integer). Lower is better (0 = perfect match).
    """
    assert pred_dag.shape == true_dag.shape
    num_vars = pred_dag.shape[0]
    
    shd = 0
    
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            # Check edge existence regardless of direction
            pred_edge = max(pred_dag[i, j], pred_dag[j, i])
            true_edge = max(true_dag[i, j], true_dag[j, i])
            
            if pred_edge == 1 and true_edge == 0:
                # Extra edge
                shd += 1
            elif pred_edge == 0 and true_edge == 1:
                # Missing edge
                shd += 1
            elif pred_edge == 1 and true_edge == 1:
                # Edge exists in both, check orientation
                if pred_dag[i, j] != true_dag[i, j]:
                    # Reversed edge
                    shd += 1
                    
    return shd

def compute_overlap_metrics(pred_dag: np.ndarray, true_dag: np.ndarray, v5_idx: int, v6_idx: int):
    """
    Computes True Positive Rate (TPR) and False Discovery Rate (FDR) specifically 
    for the structural relationship between V5 and V6.
    
    Args:
        pred_dag: Predicted adjacency matrix for the local agent.
        true_dag: Ground-truth adjacency matrix for the local agent.
        v5_idx: The local index representing V5.
        v6_idx: The local index representing V6.
        
    Returns:
        tpr: True Positive Rate for the V5-V6 connection.
        fdr: False Discovery Rate for the V5-V6 connection.
    """
    tp = 0
    fp = 0
    fn = 0
    
    # Edges to check: V5 -> V6 and V6 -> V5
    edges_to_check = [(v5_idx, v6_idx), (v6_idx, v5_idx)]
    
    for u, v in edges_to_check:
        true_edge = true_dag[u, v]
        pred_edge = pred_dag[u, v]
        
        if true_edge == 1 and pred_edge == 1:
            tp += 1
        elif true_edge == 0 and pred_edge == 1:
            fp += 1
        elif true_edge == 1 and pred_edge == 0:
            fn += 1
            
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    return tpr, fdr
