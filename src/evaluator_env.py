import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from src.scm import sample_scm

from src.types import SCMConfig, SCMParams, InterventionSpec, InterventionType, ActionCategory
from src.environment import init_env, step_env, update_running_statistics
from src.alignment import stitch_global_covariance
from src.generators import generate_4node_topologies, generate_scm_params

@jax.jit
def compute_local_covariances(samples: jax.Array, agent_masks: jax.Array) -> jax.Array:
    def _compute_single_agent(mask):
        masked_samples = samples * mask[None, :]
        mean = jnp.mean(masked_samples, axis=0)
        centered = masked_samples - mean
        N = jnp.maximum(1.0, samples.shape[0] - 1.0)
        cov = jnp.dot(centered.T, centered) / N
        return cov
    return jax.vmap(_compute_single_agent)(agent_masks)

class FederatedCausalEnv:
    def __init__(self, config: SCMConfig, action_costs: np.ndarray,
                 initial_budget: float = 10.0, sample_count: int = 100):
        self.config = config
        self.action_costs = action_costs
        self.initial_budget = initial_budget
        self.sample_count = sample_count
        
        # Agent masks for the 4-node topology (Agent 1: Z1, X1 -> 0, 1) (Agent 2: X2, Z2 -> 2, 3)
        self.agent_masks = jnp.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ])
        
        self.jax_state = None
        self.max_steps = 20
        
    def _get_obs_dict(self):
        """Constructs IPPO observations."""
        obs_dict = {}
        for k in range(self.config.K):
            # Agent sees its own running covariance mask
            mask = np.array(self.agent_masks[k])
            # To allow boundary observation, we let agents see correlations with peer boundary nodes.
            # Boundary nodes: X1 (1) and X2 (2). 
            # Agent 1 observes 0, 1, 2. Agent 2 observes 1, 2, 3.
            obs_mask = np.zeros(self.config.d)
            if k == 0:
                obs_mask[0] = 1.0; obs_mask[1] = 1.0; obs_mask[2] = 1.0
            else:
                obs_mask[1] = 1.0; obs_mask[2] = 1.0; obs_mask[3] = 1.0
                
            cov = np.array(self.jax_state.running_covariance)
            masked_cov = cov * obs_mask[:, None] * obs_mask[None, :]
            
            budget = np.array([self.jax_state.budgets[k]])
            
            # obs: flattened masked covariance + budget
            obs = np.concatenate([masked_cov.flatten(), budget])
            obs_dict[f"agent_{k}"] = obs
            
        return obs_dict
        
    def reset(self, key: jax.Array) -> Tuple[Dict[str, np.ndarray], Dict]:
        # Meta-learning: Generate random topology
        k1, k2, k3, key = jax.random.split(key, 4)
        adjacency, topo_order = generate_4node_topologies(k1)
        scm_params = generate_scm_params(k2, adjacency, int(self.config.mechanism_type))
        
        budgets = jnp.full(self.config.K, self.initial_budget)
        self.jax_state = init_env(k3, self.config, adjacency, scm_params, topo_order, self.agent_masks, budgets)
        
        # Get initial observational data
        obs_key, key = jax.random.split(key)
        obs_spec = InterventionSpec(
            mask=jnp.zeros(self.config.d),
            type=jnp.zeros(self.config.d, dtype=jnp.int32),
            value=jnp.zeros(self.config.d)
        )
        samples = sample_scm(obs_key, self.jax_state, self.config, self.sample_count, obs_spec)
        
        local_covs_jnp = compute_local_covariances(samples, self.agent_masks)
        sample_counts = jnp.full(self.config.K, float(self.sample_count))
        stitched_cov = stitch_global_covariance(local_covs_jnp, self.agent_masks, sample_counts)
        
        self.jax_state = update_running_statistics(self.jax_state, stitched_cov, self.sample_count)
        
        obs_dict = self._get_obs_dict()
            
        return obs_dict, {"true_adjacency": np.array(self.jax_state.true_adjacency)}
        
    def step(self, joint_actions: Dict[str, Tuple[int, int]], predicted_dags: Dict[str, np.ndarray], key: jax.Array) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        joint_actions: Dictionary mapping agent id to (Category, Target Node).
        predicted_dags: Dictionary mapping agent id to its predicted local DAG [d, d].
        """
        mask = np.zeros(self.config.d)
        types = np.full(self.config.d, int(InterventionType.HARD), dtype=np.int32)
        values = np.zeros(self.config.d) 
        costs = np.zeros(self.config.K)
        
        for k in range(self.config.K):
            cat, target = joint_actions[f"agent_{k}"]
            budget_k = self.jax_state.budgets[k]
            
            # Only process if agent has budget and didn't NO-OP
            if budget_k >= self.action_costs[k] and cat != ActionCategory.NOOP:
                if cat == ActionCategory.LOCAL_INTERVENTION:
                    # Verify target is local
                    if self.agent_masks[k, target] == 1.0:
                        mask[target] = 1.0
                        values[target] = 5.0
                        costs[k] = self.action_costs[k]
                elif cat == ActionCategory.PEER_REQUEST:
                    # Peer request targets boundary nodes
                    if target in [1, 2]: # Node 1 (X1) or 2 (X2)
                        mask[target] = 1.0
                        values[target] = 5.0
                        costs[k] = self.action_costs[k] # Cost deducted from requester
                        
        intervention_spec = InterventionSpec(
            mask=jnp.array(mask),
            type=jnp.array(types),
            value=jnp.array(values)
        )
        
        self.jax_state, _ = step_env(self.jax_state, jnp.array([]), jnp.array(costs))
        
        samples = sample_scm(key, self.jax_state, self.config, self.sample_count, intervention_spec)
        local_covs_jnp = compute_local_covariances(samples, self.agent_masks)
        sample_counts = jnp.full(self.config.K, float(self.sample_count))
        stitched_cov = stitch_global_covariance(local_covs_jnp, self.agent_masks, sample_counts)
        
        self.jax_state = update_running_statistics(self.jax_state, stitched_cov, self.sample_count)
        
        from src.stitching import stitch_predicted_dags
        from src.rewards import compute_ippo_rewards
        
        stitched_dag, has_cycle = stitch_predicted_dags(predicted_dags, self.config.d)
        true_dag = np.array(self.jax_state.true_adjacency)
        rewards = compute_ippo_rewards(stitched_dag, true_dag, has_cycle)
        
        terminated = False
        if self.jax_state.step_count >= self.max_steps:
            terminated = True
        if np.all(np.array(self.jax_state.budgets) <= 0):
            terminated = True
            
        obs_dict = self._get_obs_dict()
            
        return obs_dict, rewards, terminated, {"true_adjacency": np.array(self.jax_state.true_adjacency)}
