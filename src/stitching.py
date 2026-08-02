import numpy as np

def stitch_predicted_dags(predicted_probs: dict, d: int, margin: float = 0.10) -> tuple[np.ndarray, bool]:
    """
    Stitches predicted edge probabilities into a single global DAG adjacency matrix (0 or 1).
    Applies competitive differential thresholding (prob_ij - prob_ji > margin) to mathematically
    prevent bidirectional 2-cycles and resolve uncertain edge orientations.
    predicted_probs maps 'agent_k' -> [d, d] matrix of edge probabilities.
    Returns (stitched_dag, has_cycle).
    """
    prob_1 = predicted_probs["agent_0"]
    prob_2 = predicted_probs["agent_1"]
    
    global_probs = np.zeros((d, d), dtype=np.float32)
    
    # Agent 1 predicts edges among {0, 1, 2}
    global_probs[0, 0:3] = prob_1[0, 0:3]
    global_probs[1:3, 0] = prob_1[1:3, 0]
    
    # Agent 2 predicts edges among {1, 2, 3}
    global_probs[3, 1:4] = prob_2[3, 1:4]
    global_probs[1:3, 3] = prob_2[1:3, 3]
    
    # Overlapping boundary nodes: {1, 2}
    # Both agents predict edges between X1 (1) and X2 (2). Use maximum to pool evidence.
    global_probs[1:3, 1:3] = np.maximum(prob_1[1:3, 1:3], prob_2[1:3, 1:3])
    
    # Clear self-loops
    np.fill_diagonal(global_probs, 0.0)
    
    # Pairwise competitive differential thresholding:
    # An edge i -> j is activated iff P(i -> j) > 0.5 AND P(i -> j) - P(j -> i) > margin
    diff = global_probs - global_probs.T
    stitched_dag = ((global_probs > 0.5) & (diff > margin)).astype(np.float32)
    
    # Cycle detection for higher-order cycles (k >= 3)
    has_cycle = detect_cycle(stitched_dag)
    
    return stitched_dag, has_cycle

def detect_cycle(adj: np.ndarray) -> bool:
    d = adj.shape[0]
    visited = np.zeros(d, dtype=bool)
    rec_stack = np.zeros(d, dtype=bool)
    
    def dfs(v):
        visited[v] = True
        rec_stack[v] = True
        for u in range(d):
            if adj[v, u] > 0:
                if not visited[u]:
                    if dfs(u):
                        return True
                elif rec_stack[u]:
                    return True
        rec_stack[v] = False
        return False
        
    for i in range(d):
        if not visited[i]:
            if dfs(i):
                return True
    return False


import functools
import jax
import jax.numpy as jnp
from typing import Tuple

@jax.jit
def jitted_detect_cycle(adj: jax.Array) -> jax.Array:
    """
    Detects if a directed graph contains any directed cycles using trace matrix powers.
    For d=4, has_cycle <=> sum_{k=2}^4 Tr(A^k) > 0.
    Supports single [d, d] or batched [B, d, d].
    """
    if adj.ndim == 2:
        a2 = jnp.dot(adj, adj)
        a3 = jnp.dot(a2, adj)
        a4 = jnp.dot(a3, adj)
        return (jnp.trace(a2) + jnp.trace(a3) + jnp.trace(a4)) > 0
    else:
        a2 = jnp.matmul(adj, adj)
        a3 = jnp.matmul(a2, adj)
        a4 = jnp.matmul(a3, adj)
        return (jnp.trace(a2, axis1=-2, axis2=-1) + jnp.trace(a3, axis1=-2, axis2=-1) + jnp.trace(a4, axis1=-2, axis2=-1)) > 0

@functools.partial(jax.jit, static_argnames=("d",))
def jitted_stitch_dags(
    prob_1: jax.Array, 
    prob_2: jax.Array, 
    d: int = 4, 
    margin: float = 0.10
) -> Tuple[jax.Array, jax.Array]:
    """
    Stitches predicted edge probabilities in pure JAX using pairwise differential thresholding
    and detects cycles on device.
    prob_1: [d, d] or [B, d, d]
    prob_2: [d, d] or [B, d, d]
    Returns (stitched_dag, has_cycle).
    """
    if prob_1.ndim == 2:
        g0 = jnp.zeros((d, d))
        g0 = g0.at[0, 0:3].set(prob_1[0, 0:3])
        g0 = g0.at[1:3, 0].set(prob_1[1:3, 0])
        g0 = g0.at[3, 1:4].set(prob_2[3, 1:4])
        g0 = g0.at[1:3, 3].set(prob_2[1:3, 3])
        g0 = g0.at[1:3, 1:3].set(jnp.maximum(prob_1[1:3, 1:3], prob_2[1:3, 1:3]))
        g0 = g0 * (1.0 - jnp.eye(d))
        
        diff = g0 - g0.T
        stitched_dag = jnp.where((g0 > 0.5) & (diff > margin), 1.0, 0.0)
        has_cycle = jitted_detect_cycle(stitched_dag)
        return stitched_dag, has_cycle
    else:
        B = prob_1.shape[0]
        g0 = jnp.zeros((B, d, d))
        g0 = g0.at[:, 0, 0:3].set(prob_1[:, 0, 0:3])
        g0 = g0.at[:, 1:3, 0].set(prob_1[:, 1:3, 0])
        g0 = g0.at[:, 3, 1:4].set(prob_2[:, 3, 1:4])
        g0 = g0.at[:, 1:3, 3].set(prob_2[:, 1:3, 3])
        g0 = g0.at[:, 1:3, 1:3].set(jnp.maximum(prob_1[:, 1:3, 1:3], prob_2[:, 1:3, 1:3]))
        g0 = g0 * (1.0 - jnp.eye(d)[None, :, :])
        
        diff = g0 - jnp.swapaxes(g0, -1, -2)
        stitched_dag = jnp.where((g0 > 0.5) & (diff > margin), 1.0, 0.0)
        has_cycle = jitted_detect_cycle(stitched_dag)
        return stitched_dag, has_cycle


