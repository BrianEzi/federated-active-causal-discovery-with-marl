import pytest
import numpy as np
from src.pag import PAGTracker
from src.metrics import evaluate_pag_against_dag

def test_evaluate_pag_against_dag():
    d = 4
    # True DAG: 0 -> 1 -> 2 -> 3
    true_dag = np.zeros((d, d))
    true_dag[0, 1] = 1.0
    true_dag[1, 2] = 1.0
    true_dag[2, 3] = 1.0
    
    # Perfect PAG: 0 -> 1 -> 2 -> 3
    pag = np.zeros((d, d), dtype=np.int32)
    pag[0, 1] = PAGTracker.TAIL
    pag[1, 0] = PAGTracker.ARROW
    pag[1, 2] = PAGTracker.TAIL
    pag[2, 1] = PAGTracker.ARROW
    pag[2, 3] = PAGTracker.TAIL
    pag[3, 2] = PAGTracker.ARROW
    
    metrics = evaluate_pag_against_dag(pag, true_dag)
    
    assert metrics["shd"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
