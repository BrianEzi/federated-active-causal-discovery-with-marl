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
             initial_budgets: jax.Array) -> EnvState:
    """
    Initializes the environment state.
    agent_masks is [K, d], identifying which variables each agent can observe/intervene on.
    """
    return EnvState(
        true_adjacency=adjacency,
        topological_order=topological_order,
        scm_params=scm_params,
        budgets=initial_budgets,
        step_count=0
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

def step_env(state: EnvState, 
             joint_action: jax.Array, 
             cost_vector: jax.Array) -> Tuple[EnvState, jax.Array]:
    """
    Advances the environment, decays budgets, and updates step logs.
    joint_action: A representation of the joint actions. For Phase 1, we just apply the cost decay.
    cost_vector: [K] array of budget costs corresponding to the joint action.
    """
    new_budgets = state.budgets - cost_vector
    
    new_state = state.replace(
        budgets=new_budgets,
        step_count=state.step_count + 1
    )
    
    # Return a dummy reward for now, as the actual reward mechanism isn't specified in Phase 1
    reward = jnp.zeros(())
    return new_state, reward
