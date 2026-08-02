import jax
import jax.numpy as jnp
from src.types import MechanismType, SCMParams

def generate_er_dag(key: jax.Array, num_variables: int, edge_prob: float) -> jax.Array:
    """
    Generates an Erdős-Rényi DAG where edges only exist for i < j with probability p.
    Returns a strictly upper-triangular adjacency matrix [num_variables, num_variables].
    """
    # Sample Bernoulli for all entries
    adj = jax.random.bernoulli(key, p=edge_prob, shape=(num_variables, num_variables)).astype(jnp.float32)
    # Mask to keep strictly upper triangular part
    return jnp.triu(adj, k=1)

def generate_4node_topologies(key: jax.Array, force_idx: int = None, allowed_indices = None) -> tuple[jax.Array, jax.Array]:
    """
    Generates one of the 4 base topologies (and their reversed perspectives) for a 4-node federated system.
    Nodes: 0 (Z1, Agent 1 Private), 1 (X1, Agent 1 Boundary), 2 (X2, Agent 2 Boundary), 3 (Z2, Agent 2 Private).
    If allowed_indices is provided, samples uniformly from that subset.
    Returns: (adjacency matrix, topological order).
    """
    # Define adjacency matrices (adj[i, j] means i -> j)
    # 0=Z1, 1=X1, 2=X2, 3=Z2
    
    matrices = []
    orders = []
    
    def add_top(edges, order):
        adj = jnp.zeros((4, 4))
        for u, v in edges:
            adj = adj.at[u, v].set(1.0)
        matrices.append(adj)
        orders.append(jnp.array(order))

    # 1. Chain: Z1 -> X1 -> X2 -> Z2
    add_top([(0, 1), (1, 2), (2, 3)], [0, 1, 2, 3])
    
    # 1b. Reversed Chain: Z1 <- X1 <- X2 <- Z2
    add_top([(1, 0), (2, 1), (3, 2)], [3, 2, 1, 0])
    
    # 2. Collider: Z1 -> X1 <- X2 <- Z2 (X1 is collider, X2 is chain)
    add_top([(0, 1), (2, 1), (3, 2)], [0, 3, 2, 1])
    
    # 2b. Reversed Collider: Z1 -> X1 -> X2 <- Z2 (X2 is collider)
    add_top([(0, 1), (1, 2), (3, 2)], [0, 1, 3, 2])
    
    # 3. Fork: Z1 <- X1 -> X2 -> Z2 (X1 is fork)
    add_top([(1, 0), (1, 2), (2, 3)], [1, 0, 2, 3])
    
    # 3b. Reversed Fork: Z1 <- X1 <- X2 -> Z2 (X2 is fork)
    add_top([(1, 0), (2, 1), (2, 3)], [2, 1, 0, 3])
    
    # 4. Fork + Collider: Z1 -> X1 <- X2 -> Z2 (X1 is collider, X2 is fork)
    add_top([(0, 1), (2, 1), (2, 3)], [0, 2, 1, 3])
    
    # 4b. Reversed Fork + Collider: Z1 <- X1 -> X2 <- Z2 (X1 is fork, X2 is collider)
    add_top([(1, 0), (1, 2), (3, 2)], [1, 3, 0, 2])
    
    matrices = jnp.stack(matrices)
    orders = jnp.stack(orders)
    
    if force_idx is not None:
        idx = force_idx
    elif allowed_indices is not None and len(allowed_indices) > 0:
        allowed_arr = jnp.array(allowed_indices)
        sub_idx = jax.random.randint(key, (), 0, len(allowed_indices))
        idx = allowed_arr[sub_idx]
    else:
        idx = jax.random.randint(key, (), 0, 8)
    return matrices[idx], orders[idx]

def generate_scm_params(key: jax.Array, adjacency: jax.Array, mechanism_type: int) -> SCMParams:
    """
    Fills non-zero entries of the adjacency matrix with random continuous weights,
    and initializes MLP weights if mechanism is non-linear.
    """
    d = adjacency.shape[0]
    
    # 1. Generate Linear Weights
    # random continuous weights beta_ij ~ U([-2.0, -0.5] U [0.5, 2.0])
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    
    signs = jax.random.choice(k1, jnp.array([-1.0, 1.0]), shape=(d, d))
    magnitudes = jax.random.uniform(k2, shape=(d, d), minval=0.5, maxval=2.0)
    
    W = adjacency * signs * magnitudes
    
    # Transpose W so that W[i, j] is the weight of j -> i (for the apply_mechanism logic)
    # The prompt says adjacency is strictly upper triangular (i < j means i->j).
    # In apply_mechanism, W[node_idx, :] selects parents. Thus we need W[j, i] to be the weight of i->j.
    W = W.T
    
    # 2. Generate MLP Weights for NONLINEAR_ANM
    hidden_dim = 16
    # Initialize with Glorot Uniform or similar
    limit1 = jnp.sqrt(6.0 / (d + hidden_dim))
    mlp_w1 = jax.random.uniform(k3, shape=(d, d, hidden_dim), minval=-limit1, maxval=limit1)
    mlp_b1 = jnp.zeros((d, hidden_dim))
    
    limit2 = jnp.sqrt(6.0 / (hidden_dim + 1))
    mlp_w2 = jax.random.uniform(k5, shape=(d, hidden_dim, 1), minval=-limit2, maxval=limit2)
    mlp_b2 = jnp.zeros((d, 1))
    
    return SCMParams(
        W=W,
        mlp_w1=mlp_w1,
        mlp_b1=mlp_b1,
        mlp_w2=mlp_w2,
        mlp_b2=mlp_b2
    )
