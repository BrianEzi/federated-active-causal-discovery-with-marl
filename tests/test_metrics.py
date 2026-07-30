import numpy as np
import pytest
from src.metrics import evaluate_dag_against_true

def test_evaluate_dag_against_true_perfect():
    pred = np.zeros((4, 4))
    true = np.zeros((4, 4))
    pred[0, 1] = 1.0
    true[0, 1] = 1.0
    
    metrics = evaluate_dag_against_true(pred, true)
    
    assert metrics["shd"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["true_positives"] == 1.0
    assert metrics["false_positives"] == 0.0
    assert metrics["false_negatives"] == 0.0

def test_evaluate_dag_against_true_fp():
    pred = np.zeros((4, 4))
    true = np.zeros((4, 4))
    pred[0, 1] = 1.0
    
    metrics = evaluate_dag_against_true(pred, true)
    
    assert metrics["shd"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["false_positives"] == 1.0

def test_evaluate_dag_against_true_fn():
    pred = np.zeros((4, 4))
    true = np.zeros((4, 4))
    true[0, 1] = 1.0
    
    metrics = evaluate_dag_against_true(pred, true)
    
    assert metrics["shd"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["false_negatives"] == 1.0
