import numpy as np
from typing import Dict

def evaluate_dag_against_true(pred_adjacency: np.ndarray, true_adjacency: np.ndarray) -> Dict[str, float]:
    """
    Evaluates the predicted DAG adjacency matrix against the true DAG adjacency matrix.
    
    pred_adjacency: [d, d] binary adjacency matrix where 1 means predicted edge i -> j.
    true_adjacency: [d, d] binary adjacency matrix where 1 means true edge i -> j.
    """
    pred_adjacency = (pred_adjacency > 0.5).astype(float)
    true_adjacency = (true_adjacency > 0.5).astype(float)
    
    tp_dir = np.sum((pred_adjacency == 1) & (true_adjacency == 1))
    fp_dir = np.sum((pred_adjacency == 1) & (true_adjacency == 0))
    fn_dir = np.sum((pred_adjacency == 0) & (true_adjacency == 1))
    
    precision = float(tp_dir / (tp_dir + fp_dir)) if (tp_dir + fp_dir) > 0 else 0.0
    recall = float(tp_dir / (tp_dir + fn_dir)) if (tp_dir + fn_dir) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance (SHD) for DAGs
    shd = float(fp_dir + fn_dir)
                    
    return {
        "shd": shd,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": float(tp_dir),
        "false_positives": float(fp_dir),
        "false_negatives": float(fn_dir)
    }
