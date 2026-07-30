import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_dag_to_wandb_image(adj: np.ndarray, title: str):
    """
    Renders an adjacency matrix as a directed graph image for WandB.
    """
    # Import wandb here so we don't crash if wandb isn't installed
    import wandb
    
    # Create directed graph
    G = nx.DiGraph(adj)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Position nodes in a circle for standard 4-node
    pos = nx.circular_layout(G)
    
    # Draw
    labels = {
        0: "Z1\n(A1 Pvt)",
        1: "X1\n(A1 Bndry)",
        2: "X2\n(A2 Bndry)",
        3: "Z2\n(A2 Pvt)"
    }
    nx.draw(G, pos, ax=ax, labels=labels, node_color='lightblue', 
            node_size=2000, arrowsize=20, font_size=9, font_weight='bold')
    
    ax.set_title(title)
    
    # Convert to WandB image
    wandb_img = wandb.Image(fig)
    
    # Close figure to prevent memory leak
    plt.close(fig)
    
    return wandb_img
