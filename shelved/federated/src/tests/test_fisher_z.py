import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import scipy.stats as stats
from src.scm.generator import LinearGaussianSCM
from src.scm.partition import partition_data

def partial_correlation(x, y, z):
    """
    Computes the partial correlation of x and y given a 2D array of control variables z.
    """
    if z is None or z.shape[1] == 0:
        return np.corrcoef(x, y)[0, 1]
    
    # Regress x on z
    beta_x = np.linalg.lstsq(z, x, rcond=None)[0]
    res_x = x - z @ beta_x
    
    # Regress y on z
    beta_y = np.linalg.lstsq(z, y, rcond=None)[0]
    res_y = y - z @ beta_y
    
    # Correlation between residuals
    return np.corrcoef(res_x, res_y)[0, 1]

def fisher_z_test(data, x_idx, y_idx, z_indices=None, alpha=0.05):
    """
    Performs the Fisher-Z conditional independence test for X _|_ Y | Z.
    
    Returns:
        is_independent (bool), p_value (float), test_stat (float)
    """
    n = data.shape[0]
    x = data[:, x_idx]
    y = data[:, y_idx]
    
    if z_indices is None or len(z_indices) == 0:
        z = None
        dz = 0
    else:
        z = data[:, z_indices]
        # Add intercept for regression
        z = np.hstack((np.ones((n, 1)), z))
        dz = len(z_indices)
        
    r = partial_correlation(x, y, z)
    
    # Apply Fisher Z-transformation
    # Clip r to prevent division by zero or log of negative numbers
    r = np.clip(r, -0.99999, 0.99999)
    z_stat = 0.5 * np.log((1 + r) / (1 - r))
    
    # Calculate test statistic
    w = np.sqrt(n - dz - 3) * np.abs(z_stat)
    
    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(w))
    
    is_independent = p_value > alpha
    return is_independent, p_value, w

def test_scm_v_structure():
    print("Generating SCM Data...")
    scm = LinearGaussianSCM(num_vars=10, random_seed=123)
    data = scm.generate_data(num_samples=5000, noise_std=1.0)
    
    centralized, agent1, agent2 = partition_data(data)
    
    print(f"Data shapes - Centralized: {centralized.shape}, Agent 1: {agent1.shape}, Agent 2: {agent2.shape}")
    
    print("\n--- Validating V-structure V4 -> V5 <- V6 ---")
    # Recall indices: V4 is 3, V5 is 4, V6 is 5
    v4_idx, v5_idx, v6_idx = 3, 4, 5
    
    # Test 1: V4 and V6 should be marginally independent
    is_indep_marginal, p_val_marginal, stat_marginal = fisher_z_test(
        centralized, v4_idx, v6_idx, z_indices=[]
    )
    
    print(f"Test: V4 _|_ V6")
    print(f"Statistic: {stat_marginal:.4f}, p-value: {p_val_marginal:.4e}")
    print(f"Result: {'Independent (Expected)' if is_indep_marginal else 'Dependent (Unexpected)'}")
    assert is_indep_marginal, "V4 and V6 should be marginally independent."
    
    # Test 2: V4 and V6 should be dependent conditional on V5
    is_indep_cond, p_val_cond, stat_cond = fisher_z_test(
        centralized, v4_idx, v6_idx, z_indices=[v5_idx]
    )
    
    print(f"\nTest: V4 _|_ V6 | V5")
    print(f"Statistic: {stat_cond:.4f}, p-value: {p_val_cond:.4e}")
    print(f"Result: {'Independent (Unexpected)' if is_indep_cond else 'Dependent (Expected)'}")
    assert not is_indep_cond, "V4 and V6 should be dependent conditional on V5."

    print("\nAll validation tests passed successfully!")

if __name__ == "__main__":
    test_scm_v_structure()
