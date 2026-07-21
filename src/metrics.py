import numpy as np
from typing import Dict
from src.pag import PAGTracker

def evaluate_pag_against_dag(pag_matrix: np.ndarray, true_adjacency: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a Partial Ancestral Graph (PAG) matrix against the true DAG adjacency matrix.
    
    pag_matrix: [d, d] array of PAG marks ({0: NULL, 1: TAIL, 2: ARROW, 3: CIRCLE})
    true_adjacency: [d, d] binary adjacency matrix where 1 means true edge i -> j.
    """
    d = pag_matrix.shape[0]
    
    # Extract predicted directed edges: P[i, j] == TAIL (1) and P[j, i] == ARROW (2) -> i -> j
    pred_directed = (pag_matrix == PAGTracker.TAIL) & (pag_matrix.T == PAGTracker.ARROW)
    
    # Extract predicted skeleton (any mark != NULL)
    pred_skeleton = (pag_matrix != PAGTracker.NULL)
    pred_skeleton = pred_skeleton | pred_skeleton.T
    np.fill_diagonal(pred_skeleton, False)
    
    # True skeleton
    true_skeleton = (true_adjacency > 0) | (true_adjacency.T > 0)
    np.fill_diagonal(true_skeleton, False)
    
    # Directed Edge Metrics
    true_directed = (true_adjacency > 0)
    tp_dir = np.sum(pred_directed & true_directed)
    fp_dir = np.sum(pred_directed & ~true_directed)
    fn_dir = np.sum(~pred_directed & true_directed)
    
    precision = float(tp_dir / (tp_dir + fp_dir)) if (tp_dir + fp_dir) > 0 else 0.0
    recall = float(tp_dir / (tp_dir + fn_dir)) if (tp_dir + fn_dir) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance (SHD)
    shd = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            true_ij = true_adjacency[i, j] > 0
            true_ji = true_adjacency[j, i] > 0
            
            p_ij = pag_matrix[i, j]
            p_ji = pag_matrix[j, i]
            
            # Case 1: No edge in true DAG
            if not true_ij and not true_ji:
                if p_ij != PAGTracker.NULL or p_ji != PAGTracker.NULL:
                    shd += 1.0
            # Case 2: Directed edge i -> j in true DAG
            elif true_ij:
                if p_ij == PAGTracker.NULL and p_ji == PAGTracker.NULL:
                    shd += 1.0
                elif p_ij == PAGTracker.ARROW and p_ji == PAGTracker.TAIL:
                    shd += 1.0
                elif p_ij == PAGTracker.CIRCLE or p_ji == PAGTracker.CIRCLE:
                    shd += 0.5
            # Case 3: Directed edge j -> i in true DAG
            elif true_ji:
                if p_ij == PAGTracker.NULL and p_ji == PAGTracker.NULL:
                    shd += 1.0
                elif p_ij == PAGTracker.TAIL and p_ji == PAGTracker.ARROW:
                    shd += 1.0
                elif p_ij == PAGTracker.CIRCLE or p_ji == PAGTracker.CIRCLE:
                    shd += 0.5
                    
    return {
        "shd": shd,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": float(tp_dir),
        "false_positives": float(fp_dir),
        "false_negatives": float(fn_dir)
    }
