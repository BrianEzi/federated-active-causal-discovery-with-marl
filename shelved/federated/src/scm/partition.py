import numpy as np

def partition_data(data: np.ndarray):
    """
    Partitions the 10-variable dataset into three federated views.
    
    Args:
        data: A numpy array of shape (num_samples, 10) representing the full dataset.
        
    Returns:
        A tuple of three numpy arrays:
        - centralized_data: The full dataset (variables 0-9).
        - agent1_data: Agent 1's view (variables 0-5).
        - agent2_data: Agent 2's view (variables 4-9).
    """
    if data.shape[1] != 10:
        raise ValueError(f"Expected 10 variables, got {data.shape[1]}")
        
    centralized_data = data.copy()
    
    # Agent 1 observes V1 to V6 (indices 0 to 5)
    agent1_data = data[:, 0:6].copy()
    
    # Agent 2 observes V5 to V10 (indices 4 to 9)
    agent2_data = data[:, 4:10].copy()
    
    return centralized_data, agent1_data, agent2_data
