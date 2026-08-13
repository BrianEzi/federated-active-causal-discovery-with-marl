import jax
import jax.numpy as jnp
from typing import Tuple
from src.types import SCMConfig, SCMParams, EnvState, AgentObservation, InterventionSpec
from src.scm import sample_scm

def init_env(key: jax.Array,
             config: SCMConfig,
             adjacency: jax.Array,
             scm_params: SCMParams,
             topological_order: jax.Array,
             agent_masks: jax.Array,
             initial_budgets: jax.Array,
             capacity: int) -> EnvState:
    """
    Initializes the environment state.
    agent_masks is [K, d], identifying which variables each agent can observe/intervene on.
    capacity: size of the raw-sample buffer (max_steps * sample_count -- the exact max an
    episode can ever produce, so the buffer never needs eviction/wraparound).
    """
    return EnvState(
        true_adjacency=adjacency,
        topological_order=topological_order,
        scm_params=scm_params,
        budgets=initial_budgets,
        step_count=0,
        running_covariance=jnp.zeros((config.d, config.d)),
        total_samples=jnp.array([0.0]),
        obs_covariance=jnp.zeros((config.d, config.d)),
        int_covariance=jnp.zeros((config.d, config.d, config.d)),
        int_mask=jnp.zeros(config.d),
        running_mean=jnp.zeros(config.d),
        raw_samples=jnp.zeros((capacity, config.d)),
        raw_interv=jnp.zeros((capacity, config.d)),
        raw_count=jnp.array([0], dtype=jnp.int32)
    )

def get_observations(state: EnvState, 
                     config: SCMConfig,
                     agent_masks: jax.Array, 
                     num_samples: int, 
                     key: jax.Array) -> AgentObservation:
    """
    Computes empirical local sample covariance matrices for each agent.
    We assume this is an observational phase, so no interventions are active.
    """
    # No interventions during the observation step
    intervention_spec = InterventionSpec(
        mask=jnp.zeros(config.d),
        type=jnp.zeros(config.d, dtype=jnp.int32),
        value=jnp.zeros(config.d)
    )
    
    # Sample the SCM
    # samples shape: [num_samples, d]
    samples = sample_scm(key, state, config, num_samples, intervention_spec)
    
    # Compute global graph proxy (flattened true adjacency)
    global_proxy = state.true_adjacency.flatten()
    
    def get_agent_obs(k):
        # Mask out variables that the agent cannot see.
        mask = agent_masks[k] # [d]
        
        # Zero out unobserved variables in the samples
        masked_samples = samples * mask[None, :] # [num_samples, d]
        
        # Center the samples
        mean = jnp.mean(masked_samples, axis=0)
        centered = masked_samples - mean
        
        # Calculate empirical covariance [d, d]
        # For unobserved variables, the variance/covariance will be exactly 0.
        cov = jnp.dot(centered.T, centered) / (num_samples - 1.0)
        
        return AgentObservation(
            local_covariance=cov,
            remaining_budget=jnp.array([state.budgets[k]]),
            global_graph_proxy=global_proxy
        )
        
    # Vmap over the K agents
    return jax.vmap(get_agent_obs)(jnp.arange(config.K))

def update_running_statistics(state: EnvState, new_cov: jax.Array, num_new_samples: int) -> EnvState:
    """Updates the running covariance using a weighted average."""
    n_old = state.total_samples[0]
    n_new = float(num_new_samples)
    n_total = n_old + n_new
    
    # Weighted average of covariances
    updated_cov = (state.running_covariance * n_old + new_cov * n_new) / n_total
    
    return state.replace(
        running_covariance=updated_cov,
        total_samples=jnp.array([n_total])
    )

def step_env(state: EnvState, 
             joint_action: jax.Array, 
             cost_vector: jax.Array) -> Tuple[EnvState, jax.Array]:
    """
    Advances the environment, decays budgets, and updates step logs.
    joint_action: A representation of the joint actions.
    cost_vector: [K] array of budget costs corresponding to the joint action.
    """
    new_budgets = state.budgets - cost_vector
    
    new_state = state.replace(
        budgets=new_budgets,
        step_count=state.step_count + 1
    )
    
    reward = jnp.zeros(())
    return new_state, reward

@jax.jit
def stitch_global_covariance(local_covariances: jax.Array, 
                             agent_masks: jax.Array, 
                             sample_counts: jax.Array) -> jax.Array:
    """
    Aggregates local sample covariance matrices into an initial d x d global covariance matrix.
    Averages overlapping entries weighted by sample counts and marks unobserved entries as NaN.
    
    local_covariances: [m, d, d]
    agent_masks: [m, d]
    sample_counts: [m]
    Returns: [d, d] global covariance matrix.
    """
    # Create pairwise masks indicating which agent observed both variable i and j
    # agent_masks[k, i] * agent_masks[k, j] = 1 if agent k observes both
    pairwise_masks = agent_masks[:, :, None] * agent_masks[:, None, :] # [m, d, d]
    
    # Reshape sample counts for broadcasting
    weights = sample_counts.reshape(-1, 1, 1) # [m, 1, 1]
    
    # Total valid samples observing each pair (i, j)
    total_weights = jnp.sum(weights * pairwise_masks, axis=0) # [d, d]
    
    # Weighted sum of covariances
    sum_covariances = jnp.sum(weights * pairwise_masks * local_covariances, axis=0) # [d, d]
    
    # Avoid division by zero by safely defaulting to 1.0 where total_weight is 0
    safe_total_weights = jnp.where(total_weights > 0, total_weights, 1.0)
    
    global_cov = sum_covariances / safe_total_weights
    
    # Mark unobserved entries (where total_weights == 0) as 0.0 (nan breaks IPPO obs)
    global_cov = jnp.where(total_weights > 0, global_cov, 0.0)

    return global_cov

@jax.jit
def stitch_global_mean(local_means: jax.Array,
                       agent_masks: jax.Array,
                       sample_counts: jax.Array) -> jax.Array:
    """
    Aggregates local sample means into a global [d] mean vector, mirroring
    stitch_global_covariance's masked, sample-count-weighted-average pattern at rank 1
    instead of rank 2 -- a node observed only by one agent takes that agent's local mean
    rather than being diluted by other agents' masked-to-zero contribution.

    local_means: [m, d]
    agent_masks: [m, d]
    sample_counts: [m]
    Returns: [d] global mean vector.
    """
    weights = sample_counts.reshape(-1, 1)  # [m, 1]
    total_weights = jnp.sum(weights * agent_masks, axis=0)  # [d]
    sum_means = jnp.sum(weights * agent_masks * local_means, axis=0)  # [d]
    safe_total_weights = jnp.where(total_weights > 0, total_weights, 1.0)
    global_mean = sum_means / safe_total_weights
    global_mean = jnp.where(total_weights > 0, global_mean, 0.0)
    return global_mean
