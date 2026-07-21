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

def generate_ba_dag(key: jax.Array, num_variables: int, num_edges_per_node: int) -> jax.Array:
    """
    Generates a Barabási-Albert scale-free DAG using sequential preferential attachment.
    Returns a strictly upper-triangular adjacency matrix.
    """
    # Start with a small fully connected component? 
    # For a DAG, we can just attach each new node i (from m to d-1) to m previous nodes (from 0 to i-1).
    m = num_edges_per_node
    
    # We will build the transposed adjacency matrix iteratively and then transpose it,
    # or build it such that adj[i, j] means j -> i.
    # We want upper triangular, so i < j means i -> j. We will fill row by row?
    # No, it's easier to add node i and pick its parents from {0, ..., i-1}.
    # Since we want an upper triangular matrix where i < j means i->j, 
    # the parents of j are some i < j. 
    # So we will iterate j from m to num_variables - 1.
    
    adj = jnp.zeros((num_variables, num_variables))
    
    # Initialize the first m nodes as a clique or just a chain.
    # To keep it simple, let's just make the first m nodes a chain (which is a DAG).
    def init_chain(adj):
        for i in range(m - 1):
            adj = adj.at[i, i+1].set(1.0)
        return adj
    
    # Since JAX requires static shapes, we'll use scan over nodes j from m to num_variables - 1.
    # Actually, a python loop is fine since num_variables is static. But let's use scan for efficiency.
    def scan_fn(carry, j):
        adj, k = carry
        
        # Calculate degrees of nodes 0 to j-1
        # In-degree + out-degree in the DAG so far. 
        # out-degree of i: sum(adj[i, :])
        # in-degree of i: sum(adj[:, i])
        degrees = jnp.sum(adj, axis=1) + jnp.sum(adj, axis=0)
        
        # We only consider nodes 0 to j-1. Set degrees of nodes >= j to 0.
        mask = jnp.arange(num_variables) < j
        valid_degrees = degrees * mask
        
        # Add a small epsilon to avoid zero probabilities (e.g. for disconnected nodes)
        probs = (valid_degrees + 1e-5) * mask
        probs = probs / jnp.sum(probs)
        
        # Sample m parents from 0 to j-1 without replacement
        k, subk = jax.random.split(k)
        
        # Gumbel-max trick for sampling without replacement
        gumbel = jax.random.gumbel(subk, shape=(num_variables,))
        log_probs = jnp.log(probs + 1e-10)
        scores = log_probs + gumbel
        
        # We want the top m scores, but we can't easily do top_k in a jittable way dynamically?
        # Actually jax.lax.top_k works with static k.
        _, parent_indices = jax.lax.top_k(scores, m)
        
        # Set edges from parents to j
        # We want parents -> j, which means adj[parent, j] = 1.0 since parent < j (upper triangular)
        adj_new = adj.at[parent_indices, j].set(1.0)
        
        return (adj_new, k), None

    # Initial edges (first m nodes are connected in a chain to ensure non-zero degrees)
    # Actually, just connecting 0 -> 1, 1 -> 2 ... up to m-1 is enough.
    initial_adj = jnp.zeros((num_variables, num_variables))
    # We use a static loop for initialization
    for i in range(m - 1):
        initial_adj = initial_adj.at[i, i+1].set(1.0)
        
    (final_adj, _), _ = jax.lax.scan(scan_fn, (initial_adj, key), jnp.arange(m, num_variables))
    return final_adj

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
