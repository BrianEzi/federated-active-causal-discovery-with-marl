import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple

from src.types import SCMConfig, SCMParams, InterventionSpec, InterventionType
from src.environment import init_env, step_env
from src.pag import PAGTracker
from src.alignment import stitch_global_covariance
from src.rewards import compute_global_reward
from src.scm import sample_scm

class FederatedCausalEnv:
    def __init__(self, config: SCMConfig, adjacency: jax.Array, scm_params: SCMParams, 
                 topological_order: jax.Array, agent_masks: jax.Array, action_costs: jax.Array):
        self.config = config
        self.adjacency = adjacency
        self.scm_params = scm_params
        self.topological_order = topological_order
        self.agent_masks = agent_masks
        self.action_costs = action_costs
        
        self.pag_tracker = None
        self.jax_state = None
        
        # Max steps per episode
        self.max_steps = 50
        
    def _get_obs_state_avail(self, local_covs_jnp, stitched_cov_np):
        """Constructs MARL observations, state, and available actions."""
        # Global state: flattened stitched cov (d^2), flattened adjacency (d^2), budgets (K)
        # Using stitched_cov_np as the proxy for global covariance
        stitched_flat = np.nan_to_num(stitched_cov_np.flatten())
        adj_flat = np.array(self.adjacency).flatten()
        budgets_np = np.array(self.jax_state.budgets)
        
        global_state = np.concatenate([stitched_flat, adj_flat, budgets_np])
        
        obs_dict = {}
        avail_dict = {}
        
        for k in range(self.config.K):
            local_flat = np.array(local_covs_jnp[k]).flatten()
            mask_np = np.array(self.agent_masks[k])
            budget = np.array([self.jax_state.budgets[k]])
            
            # obs: [local_cov, mask, budget, global_cov]
            obs = np.concatenate([local_flat, mask_np, budget, stitched_flat])
            obs_dict[f"agent_{k}"] = obs
            
            # Avail actions: 1 if mask is true and budget >= cost, NO-OP always 1
            avail = np.zeros(self.config.d + 1, dtype=np.float32)
            for i in range(self.config.d):
                if mask_np[i] > 0 and budget[0] >= self.action_costs[k]:
                    avail[i] = 1.0
            avail[self.config.d] = 1.0 # NO-OP
            avail_dict[f"agent_{k}"] = avail
            
        return obs_dict, global_state, avail_dict
        
    def reset(self, key: jax.Array) -> Tuple[Dict[str, np.ndarray], Dict]:
        budgets = jnp.full(self.config.K, 10.0) # initial budgets
        self.jax_state = init_env(key, self.config, self.adjacency, self.scm_params, self.topological_order, self.agent_masks, budgets)
        self.pag_tracker = PAGTracker(self.config.d)
        
        # Get initial observational data
        obs_key, key = jax.random.split(key)
        obs_spec = InterventionSpec(
            mask=jnp.zeros(self.config.d),
            type=jnp.zeros(self.config.d, dtype=jnp.int32),
            value=jnp.zeros(self.config.d)
        )
        samples = sample_scm(obs_key, self.jax_state, self.config, 1000, obs_spec)
        
        obs_dict = {}
        local_covs = []
        for k in range(self.config.K):
            agent_mask = self.agent_masks[k]
            masked_samples = samples * agent_mask[None, :]
            mean = jnp.mean(masked_samples, axis=0)
            centered = masked_samples - mean
            cov = jnp.dot(centered.T, centered) / 999.0
            local_covs.append(cov)
        local_covs_jnp = jnp.stack(local_covs)
            
        sample_counts = jnp.full(self.config.K, 1000.0)
        stitched_cov = stitch_global_covariance(local_covs_jnp, self.agent_masks, sample_counts)
        stitched_cov_np = np.array(stitched_cov)
        
        obs_dict, global_state, avail_dict = self._get_obs_state_avail(local_covs_jnp, stitched_cov_np)
            
        return obs_dict, {"pag": self.pag_tracker.P.copy(), "state": global_state, "avail_actions": avail_dict}
        
    def step(self, joint_actions: Dict[str, int], key: jax.Array) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        joint_actions: Dictionary mapping agent id (e.g. "agent_0") to a discrete action.
        If action < d, it corresponds to a hard intervention on that node.
        If action == d, it's a NO-OP (observe only).
        """
        action_array = np.array([joint_actions[f"agent_{k}"] for k in range(self.config.K)])
        
        mask = np.zeros(self.config.d)
        types = np.full(self.config.d, int(InterventionType.HARD), dtype=np.int32)
        values = np.zeros(self.config.d) 
        
        costs = np.zeros(self.config.K)
        intervened_nodes = []
        
        for k in range(self.config.K):
            a = action_array[k]
            if a < self.config.d:
                # Agent k intervenes on node a
                mask[a] = 1.0
                values[a] = 5.0 # hard intervene to 5.0
                costs[k] = self.action_costs[k]
                intervened_nodes.append(int(a))
                
        intervention_spec = InterventionSpec(
            mask=jnp.array(mask),
            type=jnp.array(types),
            value=jnp.array(values)
        )
        
        # Step JAX Env (decay budgets)
        self.jax_state, _ = step_env(self.jax_state, jnp.array(action_array), jnp.array(costs))
        
        # Generate batched interventional data
        samples = sample_scm(key, self.jax_state, self.config, 500, intervention_spec)
        
        local_covs = []
        obs_dict = {}
        for k in range(self.config.K):
            agent_mask = self.agent_masks[k]
            masked_samples = samples * agent_mask[None, :]
            mean = jnp.mean(masked_samples, axis=0)
            centered = masked_samples - mean
            cov = jnp.dot(centered.T, centered) / 499.0
            local_covs.append(cov)
            
        local_covs_jnp = jnp.stack(local_covs)
        
        # Stitch global covariance
        sample_counts = jnp.full(self.config.K, 500.0)
        stitched_cov = stitch_global_covariance(local_covs_jnp, self.agent_masks, sample_counts)
        stitched_cov_np = np.array(stitched_cov)
        
        # Generate proxy P-values from stitched covariance
        # High covariance means dependent (low p-value)
        p_values_matrix = np.exp(-np.abs(np.nan_to_num(stitched_cov_np)) * 10)
        
        # Update PAGTracker
        prev_circles = self.pag_tracker.count_circle_marks()
        self.pag_tracker.update_pag_from_intervention(intervened_nodes, p_values_matrix, threshold=0.05)
        curr_circles = self.pag_tracker.count_circle_marks()
        violations = self.pag_tracker.check_structural_violations()
        
        # Compute Global Reward
        reward = compute_global_reward(prev_circles, curr_circles, action_array, costs, violations)
        
        # Check termination
        terminated = False
        if curr_circles == 0:
            terminated = True
        if self.jax_state.step_count >= self.max_steps:
            terminated = True
        if np.all(np.array(self.jax_state.budgets) <= 0):
            terminated = True
            
        obs_dict, global_state, avail_dict = self._get_obs_state_avail(local_covs_jnp, stitched_cov_np)
            
        return obs_dict, reward, terminated, {"pag": self.pag_tracker.P.copy(), "state": global_state, "avail_actions": avail_dict}
