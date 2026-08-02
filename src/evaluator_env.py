import functools
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from src.scm import sample_scm, _sample_scm_jitted

from src.types import SCMConfig, SCMParams, EnvState, InterventionSpec, InterventionType, ActionCategory
from src.environment import init_env, step_env, update_running_statistics, stitch_global_covariance
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

@functools.partial(jax.jit, static_argnames=("sample_count", "d", "mech_type", "noise_type", "noise_scale"))
def jitted_initial_obs_kernel(
    key: jax.Array,
    state: EnvState,
    sample_count: int,
    agent_masks: jax.Array,
    obs_masks: jax.Array,
    d: int,
    mech_type: int,
    noise_type: int,
    noise_scale: float
) -> Tuple[EnvState, jax.Array]:
    obs_spec = InterventionSpec(
        mask=jnp.zeros(d),
        type=jnp.zeros(d, dtype=jnp.int32),
        value=jnp.zeros(d)
    )
    samples = _sample_scm_jitted(key, state, obs_spec, sample_count, d, mech_type, noise_type, noise_scale)
    
    def _compute_single_agent(m):
        masked_samples = samples * m[None, :]
        mean = jnp.mean(masked_samples, axis=0)
        centered = masked_samples - mean
        N = jnp.maximum(1.0, float(sample_count) - 1.0)
        return jnp.dot(centered.T, centered) / N
    local_covs = jax.vmap(_compute_single_agent)(agent_masks)
    
    sample_counts = jnp.full(agent_masks.shape[0], float(sample_count))
    stitched_cov = stitch_global_covariance(local_covs, agent_masks, sample_counts)
    
    n_total = float(sample_count)
    final_state = state.replace(
        running_covariance=stitched_cov,
        total_samples=jnp.array([n_total])
    )
    
    def _get_agent_obs(k):
        m = obs_masks[k]
        m_cov = stitched_cov * m[:, None] * m[None, :]
        return jnp.concatenate([m_cov.flatten(), jnp.array([final_state.budgets[k]])])
    agent_observations = jax.vmap(_get_agent_obs)(jnp.arange(agent_masks.shape[0]))
    
    return final_state, agent_observations

@functools.partial(jax.jit, static_argnames=("sample_count", "d", "mech_type", "noise_type", "noise_scale"))
def jitted_env_step_kernel(
    key: jax.Array,
    state: EnvState,
    mask: jax.Array,
    types: jax.Array,
    values: jax.Array,
    costs: jax.Array,
    sample_count: int,
    agent_masks: jax.Array,
    obs_masks: jax.Array,
    d: int,
    mech_type: int,
    noise_type: int,
    noise_scale: float
) -> Tuple[EnvState, jax.Array]:
    new_budgets = state.budgets - costs
    new_state = state.replace(
        budgets=new_budgets,
        step_count=state.step_count + 1
    )
    
    int_spec = InterventionSpec(mask=mask, type=types, value=values)
    samples = _sample_scm_jitted(key, new_state, int_spec, sample_count, d, mech_type, noise_type, noise_scale)
    
    def _compute_single_agent(m):
        masked_samples = samples * m[None, :]
        mean = jnp.mean(masked_samples, axis=0)
        centered = masked_samples - mean
        N = jnp.maximum(1.0, float(sample_count) - 1.0)
        return jnp.dot(centered.T, centered) / N
    local_covs = jax.vmap(_compute_single_agent)(agent_masks)
    
    sample_counts = jnp.full(agent_masks.shape[0], float(sample_count))
    stitched_cov = stitch_global_covariance(local_covs, agent_masks, sample_counts)
    
    n_old = new_state.total_samples[0]
    n_new = float(sample_count)
    n_total = n_old + n_new
    updated_cov = (new_state.running_covariance * n_old + stitched_cov * n_new) / n_total
    
    final_state = new_state.replace(
        running_covariance=updated_cov,
        total_samples=jnp.array([n_total])
    )
    
    def _get_agent_obs(k):
        m = obs_masks[k]
        m_cov = updated_cov * m[:, None] * m[None, :]
        return jnp.concatenate([m_cov.flatten(), jnp.array([final_state.budgets[k]])])
    agent_observations = jax.vmap(_get_agent_obs)(jnp.arange(agent_masks.shape[0]))
    
    return final_state, agent_observations

@functools.partial(jax.jit, static_argnames=("d",))
def build_intervention_spec_jitted(
    cat_0: jax.Array, target_0: jax.Array,
    cat_1: jax.Array, target_1: jax.Array,
    budgets: jax.Array,
    action_costs: jax.Array,
    agent_masks: jax.Array,
    d: int = 4
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mask = jnp.zeros(d)
    types = jnp.full(d, int(InterventionType.HARD), dtype=jnp.int32)
    values = jnp.zeros(d)
    costs = jnp.zeros(2)
    
    # Agent 0
    valid_a0_local = (budgets[0] >= action_costs[0]) & (cat_0 == int(ActionCategory.LOCAL_INTERVENTION)) & (agent_masks[0, target_0] == 1.0)
    valid_a0_peer = (budgets[0] >= action_costs[0]) & (cat_0 == int(ActionCategory.PEER_REQUEST)) & ((target_0 == 1) | (target_0 == 2))
    apply_0 = valid_a0_local | valid_a0_peer
    
    mask = jnp.where(apply_0, mask.at[target_0].set(1.0), mask)
    values = jnp.where(apply_0, values.at[target_0].set(5.0), values)
    costs = jnp.where(apply_0, costs.at[0].set(action_costs[0]), costs)
    
    # Agent 1
    valid_a1_local = (budgets[1] >= action_costs[1]) & (cat_1 == int(ActionCategory.LOCAL_INTERVENTION)) & (agent_masks[1, target_1] == 1.0)
    valid_a1_peer = (budgets[1] >= action_costs[1]) & (cat_1 == int(ActionCategory.PEER_REQUEST)) & ((target_1 == 1) | (target_1 == 2))
    apply_1 = valid_a1_local | valid_a1_peer
    
    mask = jnp.where(apply_1, mask.at[target_1].set(1.0), mask)
    values = jnp.where(apply_1, values.at[target_1].set(5.0), values)
    costs = jnp.where(apply_1, costs.at[1].set(action_costs[1]), costs)
    
    return mask, types, values, costs

class FederatedCausalEnv:
    def __init__(self, config: SCMConfig, action_costs: np.ndarray,
                 initial_budget: float = 10.0, sample_count: int = 100, fixed_graph: bool = False,
                 max_steps: int = 20, boundary_margin: float = 0.10, normalize_rewards: bool = True):
        self.config = config
        self.action_costs = action_costs
        self.initial_budget = initial_budget
        self.sample_count = sample_count
        self.fixed_graph = fixed_graph
        self.max_steps = max_steps
        self.boundary_margin = boundary_margin
        self.normalize_rewards = normalize_rewards
        
        # Agent masks for the 4-node topology (Agent 1: Z1, X1 -> 0, 1) (Agent 2: X2, Z2 -> 2, 3)
        self.agent_masks = jnp.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0]
        ])
        
        # Observation masks for Agent 1 (observes 0, 1, 2) and Agent 2 (observes 1, 2, 3)
        self.obs_masks = jnp.array([
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0]
        ])
        
        self.local_masks = [self.agent_masks[k] for k in range(self.config.K)]
        self.boundary_mask = jnp.array([0.0, 1.0, 1.0, 0.0])
        self.edge_masks = [
            jnp.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=jnp.float32),
            jnp.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]], dtype=jnp.float32)
        ]
        self.obs_dim = self.config.d * self.config.d + 1
        
        self.jax_state = None
        self._agent_observations = None
        self._fixed_topology_cache = None

        
    def _get_obs_dict(self):
        """Constructs IPPO observations."""
        if self._agent_observations is not None:
            return {
                f"agent_{k}": np.array(self._agent_observations[k])
                for k in range(self.config.K)
            }
        obs_dict = {}
        for k in range(self.config.K):
            obs_mask = np.array(self.obs_masks[k])
            cov = np.array(self.jax_state.running_covariance)
            masked_cov = cov * obs_mask[:, None] * obs_mask[None, :]
            budget = np.array([self.jax_state.budgets[k]])
            obs = np.concatenate([masked_cov.flatten(), budget])
            obs_dict[f"agent_{k}"] = obs
        return obs_dict
        
    def reset(self, key: jax.Array, force_idx: int = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        # Meta-learning: Generate random topology
        k1, k2, k3, key = jax.random.split(key, 4)
        
        if self.fixed_graph:
            if self._fixed_topology_cache is None:
                adjacency, topo_order = generate_4node_topologies(jax.random.PRNGKey(42), force_idx=force_idx)
                scm_params = generate_scm_params(jax.random.PRNGKey(43), adjacency, int(self.config.mechanism_type))
                self._fixed_topology_cache = (adjacency, topo_order, scm_params)
            else:
                adjacency, topo_order, scm_params = self._fixed_topology_cache
        else:
            adjacency, topo_order = generate_4node_topologies(k1, force_idx=force_idx)
            scm_params = generate_scm_params(k2, adjacency, int(self.config.mechanism_type))
        
        budgets = jnp.full(self.config.K, self.initial_budget)
        self.jax_state = init_env(k3, self.config, adjacency, scm_params, topo_order, self.agent_masks, budgets)
        
        # Get initial observational data using fast JIT kernel
        obs_key, key = jax.random.split(key)
        self.jax_state, self._agent_observations = jitted_initial_obs_kernel(
            obs_key, self.jax_state, self.sample_count, self.agent_masks, self.obs_masks,
            int(self.config.d), int(self.config.mechanism_type), int(self.config.noise_type), float(self.config.noise_scale)
        )
        
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
                        
        self.jax_state, self._agent_observations = jitted_env_step_kernel(
            key, self.jax_state, jnp.array(mask), jnp.array(types), jnp.array(values), jnp.array(costs),
            self.sample_count, self.agent_masks, self.obs_masks,
            int(self.config.d), int(self.config.mechanism_type), int(self.config.noise_type), float(self.config.noise_scale)
        )
        
        from src.stitching import stitch_predicted_dags
        from src.rewards import compute_ippo_rewards
        
        stitched_dag, has_cycle = stitch_predicted_dags(predicted_dags, self.config.d, margin=self.boundary_margin)
        true_dag = np.array(self.jax_state.true_adjacency)
        norm_factor = float(self.max_steps) if self.normalize_rewards else 1.0
        rewards = compute_ippo_rewards(stitched_dag, true_dag, has_cycle, max_steps=norm_factor)
        
        terminated = bool(self.jax_state.step_count >= self.max_steps or np.all(np.array(self.jax_state.budgets) <= 0))
            
        obs_dict = self._get_obs_dict()
        return obs_dict, rewards, terminated, {"true_adjacency": true_dag}

    def step_jitted(
        self,
        cat_0: jax.Array, target_0: jax.Array, graph_pred_0: jax.Array,
        cat_1: jax.Array, target_1: jax.Array, graph_pred_1: jax.Array,
        key: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, bool, jax.Array]:
        """
        Pure JAX step executing intervention building, SCM sampling,
        DAG stitching, and reward computation entirely on GPU.
        """
        from src.stitching import jitted_stitch_dags
        from src.rewards import jitted_compute_ippo_rewards
        
        costs_vec = jnp.array(self.action_costs)
        mask, types, values, costs = build_intervention_spec_jitted(
            cat_0, target_0, cat_1, target_1, self.jax_state.budgets, costs_vec, self.agent_masks, int(self.config.d)
        )
        
        k_step, key = jax.random.split(key)
        self.jax_state, self._agent_observations = jitted_env_step_kernel(
            k_step, self.jax_state, mask, types, values, costs,
            self.sample_count, self.agent_masks, self.obs_masks,
            int(self.config.d), int(self.config.mechanism_type), int(self.config.noise_type), float(self.config.noise_scale)
        )
        
        stitched_dag, has_cycle = jitted_stitch_dags(graph_pred_0, graph_pred_1, int(self.config.d), margin=self.boundary_margin)
        norm_factor = float(self.max_steps) if self.normalize_rewards else 1.0
        r0, r1 = jitted_compute_ippo_rewards(stitched_dag, self.jax_state.true_adjacency, has_cycle, max_steps=norm_factor)
        
        terminated = bool(self.jax_state.step_count >= self.max_steps or (self.jax_state.budgets[0] <= 0 and self.jax_state.budgets[1] <= 0))
        return self._agent_observations, r0, r1, terminated, stitched_dag


