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

@jax.jit
def compute_invariance_asymmetry_matrix(
    obs_cov: jax.Array,    # [d, d]
    int_cov: jax.Array,    # [d, d, d]
    int_mask: jax.Array,   # [d]
    eps: float = 1e-5
) -> jax.Array:
    """
    Computes A_{i, j} = |1 - Var(X_j | do(X_i)) / Var_obs(X_j)| - |1 - Var(X_i | do(X_j)) / Var_obs(X_i)|
    Returns: [d, d] matrix where A[i, j] > 0 indicates empirical evidence for i -> j.
    """
    var_obs = jnp.diag(obs_cov) + eps
    var_int = jnp.diagonal(int_cov, axis1=1, axis2=2) # [d, d]
    ratio = var_int / var_obs[None, :]
    shift = jnp.abs(1.0 - ratio) * int_mask[:, None]
    asymmetry = shift - shift.T
    return asymmetry

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
        obs_covariance=stitched_cov,
        int_covariance=jnp.zeros((d, d, d)),
        int_mask=jnp.zeros(d),
        total_samples=jnp.array([n_total])
    )
    
    asymmetry = compute_invariance_asymmetry_matrix(final_state.obs_covariance, final_state.int_covariance, final_state.int_mask)
    
    def _get_agent_obs(k):
        m = obs_masks[k]
        m_cov_obs = final_state.obs_covariance * m[:, None] * m[None, :]
        m_cov_run = stitched_cov * m[:, None] * m[None, :]
        m_asym = asymmetry * m[:, None] * m[None, :]
        return jnp.concatenate([
            m_cov_obs.flatten(),
            m_cov_run.flatten(),
            m_asym.flatten(),
            jnp.array([final_state.budgets[k]])
        ])
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
) -> Tuple[EnvState, jax.Array, jax.Array]:
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
    
    # Compute intrinsic information gain: Frobenius norm of covariance shift projected on observable mask
    diff_cov = stitched_cov - state.running_covariance
    def _compute_info_gain(m):
        m_diff = diff_cov * m[:, None] * m[None, :]
        frob_norm = jnp.sqrt(jnp.sum(m_diff ** 2) + 1e-8)
        num_obs = jnp.sum(m)
        return frob_norm / jnp.maximum(1.0, num_obs)
    info_gains = jax.vmap(_compute_info_gain)(obs_masks)
    
    n_old = new_state.total_samples[0]
    n_new = float(sample_count)
    n_total = n_old + n_new
    updated_cov = (new_state.running_covariance * n_old + stitched_cov * n_new) / n_total
    
    is_intervened = mask > 0.5
    updated_int_cov = jnp.where(is_intervened[:, None, None], stitched_cov[None, :, :], new_state.int_covariance)
    updated_int_mask = jnp.maximum(new_state.int_mask, is_intervened.astype(float))
    
    final_state = new_state.replace(
        running_covariance=updated_cov,
        int_covariance=updated_int_cov,
        int_mask=updated_int_mask,
        total_samples=jnp.array([n_total])
    )
    
    asymmetry = compute_invariance_asymmetry_matrix(final_state.obs_covariance, final_state.int_covariance, final_state.int_mask)
    
    def _get_agent_obs(k):
        m = obs_masks[k]
        m_cov_obs = final_state.obs_covariance * m[:, None] * m[None, :]
        m_cov_run = updated_cov * m[:, None] * m[None, :]
        m_asym = asymmetry * m[:, None] * m[None, :]
        return jnp.concatenate([
            m_cov_obs.flatten(),
            m_cov_run.flatten(),
            m_asym.flatten(),
            jnp.array([final_state.budgets[k]])
        ])
    agent_observations = jax.vmap(_get_agent_obs)(jnp.arange(agent_masks.shape[0]))
    
    return final_state, agent_observations, info_gains

@functools.partial(jax.jit, static_argnames=("d", "intervention_type"))
def build_intervention_spec_jitted(
    cat_0: jax.Array, target_0: jax.Array,
    cat_1: jax.Array, target_1: jax.Array,
    budgets: jax.Array,
    action_costs: jax.Array,
    agent_masks: jax.Array,
    d: int = 4,
    intervention_type: int = int(InterventionType.SOFT_SHIFT),
    shift_val: float = 2.0
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mask = jnp.zeros(d)
    types = jnp.full(d, intervention_type, dtype=jnp.int32)
    values = jnp.zeros(d)
    costs = jnp.zeros(2)
    
    # Agent 0
    valid_a0_local = (budgets[0] >= action_costs[0]) & (cat_0 == int(ActionCategory.LOCAL_INTERVENTION)) & (agent_masks[0, target_0] == 1.0)
    valid_a0_peer = (budgets[0] >= action_costs[0]) & (cat_0 == int(ActionCategory.PEER_REQUEST)) & ((target_0 == 1) | (target_0 == 2))
    apply_0 = valid_a0_local | valid_a0_peer
    
    mask = jnp.where(apply_0, mask.at[target_0].set(1.0), mask)
    values = jnp.where(apply_0, values.at[target_0].set(shift_val), values)
    costs = jnp.where(apply_0, costs.at[0].set(action_costs[0]), costs)
    
    # Agent 1
    valid_a1_local = (budgets[1] >= action_costs[1]) & (cat_1 == int(ActionCategory.LOCAL_INTERVENTION)) & (agent_masks[1, target_1] == 1.0)
    valid_a1_peer = (budgets[1] >= action_costs[1]) & (cat_1 == int(ActionCategory.PEER_REQUEST)) & ((target_1 == 1) | (target_1 == 2))
    apply_1 = valid_a1_local | valid_a1_peer
    
    mask = jnp.where(apply_1, mask.at[target_1].set(1.0), mask)
    values = jnp.where(apply_1, values.at[target_1].set(shift_val), values)
    costs = jnp.where(apply_1, costs.at[1].set(action_costs[1]), costs)
    
    return mask, types, values, costs

class FederatedCausalEnv:
    def __init__(self, config: SCMConfig, action_costs: np.ndarray,
                 initial_budget: float = 10.0, sample_count: int = 100, fixed_graph: bool = False,
                 max_steps: int = 20, boundary_margin: float = 0.10, normalize_rewards: bool = True,
                 intrinsic_coef: float = 0.05, intervention_type: str = "soft_shift",
                 shift_val: float = 2.0, estimator_type: str = "analytic",
                 freeze_graph_estimator: bool = True, obs_feedback: bool = True,
                 impact_coef: float = 0.0, reward_density: str = "dense"):
        self.config = config
        self.action_costs = action_costs
        self.initial_budget = initial_budget
        self.sample_count = sample_count
        self.fixed_graph = fixed_graph
        self.max_steps = max_steps
        self.boundary_margin = boundary_margin
        self.normalize_rewards = normalize_rewards
        self.intrinsic_coef = intrinsic_coef
        self.intervention_type = intervention_type
        self.shift_val = shift_val
        self.estimator_type = estimator_type
        self.freeze_graph_estimator = freeze_graph_estimator
        self.obs_feedback = obs_feedback
        self.impact_coef = impact_coef
        self.reward_density = reward_density
        
        self.int_type_code = int(InterventionType.HARD if intervention_type == "hard" else InterventionType.SOFT_SHIFT)
        
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
        
        self.last_predicted_dag = np.full((self.config.d, self.config.d), 0.5, dtype=np.float32)
        np.fill_diagonal(self.last_predicted_dag, 0.0)
        self.prev_shd = None
        
        # Attempt to load AVICI if requested
        self.avici_model = None
        if self.estimator_type == "avici":
            try:
                import avici
                self.avici_model = avici.load_pretrained(download="scm-v0")
            except Exception as e:
                print(f"Notice: AVICI model unavailable ({e}), falling back to Analytic Invariance Scorer.")
                self.estimator_type = "analytic"
        
        # Calculate dynamic obs_dim:
        # Base: 3 * d * d + 1 (obs_cov + run_cov + asym + budget)
        # If obs_feedback: + d * d (predicted local DAG slice flattened)
        self.obs_dim = 3 * self.config.d * self.config.d + 1 + (self.config.d * self.config.d if obs_feedback else 0)
        
        self.jax_state = None
        self._agent_observations = None
        self._fixed_topology_cache = None

    def predict_graph_hypothesis(self, obs_cov: np.ndarray, run_cov: np.ndarray, asym: np.ndarray) -> np.ndarray:
        """
        Stage 2 Inference Engine: Predicts continuous edge probability matrix [d, d].
        """
        d = self.config.d
        if self.avici_model is not None and self.estimator_type == "avici":
            try:
                # Synthesize dataset summary or pass running covariance
                prob = np.array(self.avici_model(x=run_cov))
                np.fill_diagonal(prob, 0.0)
                return prob
            except Exception:
                pass
                
        # Fast Analytic Invariance Estimator
        # Skeleton: S_ij = 0.5 * (|run_cov_ij| + |obs_cov_ij|)
        # Orientation: O_ij = 2.0 * asym_ij
        S = 0.5 * (np.abs(run_cov) + np.abs(obs_cov))
        O = 2.0 * asym
        logits = S + O
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))
        np.fill_diagonal(prob, 0.0)
        return prob.astype(np.float32)

    def _get_obs_dict(self):
        """Constructs IPPO observations with optional DAG observation feedback."""
        obs_dict = {}
        asymmetry = np.array(compute_invariance_asymmetry_matrix(
            self.jax_state.obs_covariance, self.jax_state.int_covariance, self.jax_state.int_mask
        ))
        obs_cov = np.array(self.jax_state.obs_covariance)
        run_cov = np.array(self.jax_state.running_covariance)
        
        for k in range(self.config.K):
            m = np.array(self.obs_masks[k])
            m_obs_cov = obs_cov * m[:, None] * m[None, :]
            m_run_cov = run_cov * m[:, None] * m[None, :]
            m_asym = asymmetry * m[:, None] * m[None, :]
            budget = np.array([float(self.jax_state.budgets[k])])
            
            parts = [m_obs_cov.flatten(), m_run_cov.flatten(), m_asym.flatten()]
            if self.obs_feedback:
                m_pred_dag = self.last_predicted_dag * m[:, None] * m[None, :]
                parts.append(m_pred_dag.flatten())
            parts.append(budget)
            
            obs_dict[f"agent_{k}"] = np.concatenate(parts).astype(np.float32)
        return obs_dict
        
    def reset(self, key: jax.Array, force_idx: int = None, allowed_topologies = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        # Meta-learning: Generate random topology
        k1, k2, k3, key = jax.random.split(key, 4)
        
        if self.fixed_graph:
            if self._fixed_topology_cache is None:
                adjacency, topo_order = generate_4node_topologies(jax.random.PRNGKey(42), force_idx=force_idx, allowed_indices=allowed_topologies)
                scm_params = generate_scm_params(jax.random.PRNGKey(43), adjacency, int(self.config.mechanism_type))
                self._fixed_topology_cache = (adjacency, topo_order, scm_params)
            else:
                adjacency, topo_order, scm_params = self._fixed_topology_cache
        else:
            adjacency, topo_order = generate_4node_topologies(k1, force_idx=force_idx, allowed_indices=allowed_topologies)
            scm_params = generate_scm_params(k2, adjacency, int(self.config.mechanism_type))
        
        budgets = jnp.full(self.config.K, self.initial_budget)
        self.jax_state = init_env(k3, self.config, adjacency, scm_params, topo_order, self.agent_masks, budgets)
        
        # Reset graph hypothesis and SHD tracking
        self.last_predicted_dag = np.full((self.config.d, self.config.d), 0.5, dtype=np.float32)
        np.fill_diagonal(self.last_predicted_dag, 0.0)
        self.prev_shd = None
        
        # Get initial observational data using fast JIT kernel
        obs_key, key = jax.random.split(key)
        self.jax_state, self._agent_observations = jitted_initial_obs_kernel(
            obs_key, self.jax_state, self.sample_count, self.agent_masks, self.obs_masks,
            int(self.config.d), int(self.config.mechanism_type), int(self.config.noise_type), float(self.config.noise_scale)
        )
        
        # Initial Stage 2 Prediction
        asym = np.array(compute_invariance_asymmetry_matrix(
            self.jax_state.obs_covariance, self.jax_state.int_covariance, self.jax_state.int_mask
        ))
        self.last_predicted_dag = self.predict_graph_hypothesis(
            np.array(self.jax_state.obs_covariance), np.array(self.jax_state.running_covariance), asym
        )
        
        obs_dict = self._get_obs_dict()
        return obs_dict, {"true_adjacency": np.array(self.jax_state.true_adjacency)}
        
    def step(self, joint_actions: Dict[str, Tuple[int, int]], predicted_dags: Dict[str, np.ndarray], key: jax.Array) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        joint_actions: Dictionary mapping agent id to (Category, Target Node).
        predicted_dags: Optional Dictionary mapping agent id to predicted local DAG [d, d].
        """
        mask = np.zeros(self.config.d)
        types = np.full(self.config.d, self.int_type_code, dtype=np.int32)
        values = np.zeros(self.config.d) 
        costs = np.zeros(self.config.K)
        
        for k in range(self.config.K):
            cat, target = joint_actions[f"agent_{k}"]
            budget_k = self.jax_state.budgets[k]
            
            # Only process if agent has budget and didn't NO-OP
            if budget_k >= self.action_costs[k] and cat != ActionCategory.NOOP:
                if cat == ActionCategory.LOCAL_INTERVENTION:
                    if self.agent_masks[k, target] == 1.0:
                        mask[target] = 1.0
                        values[target] = self.shift_val
                        costs[k] = self.action_costs[k]
                elif cat == ActionCategory.PEER_REQUEST:
                    if target in [1, 2]:
                        mask[target] = 1.0
                        values[target] = self.shift_val
                        costs[k] = self.action_costs[k]
                        
        self.jax_state, self._agent_observations, info_gains = jitted_env_step_kernel(
            key, self.jax_state, jnp.array(mask), jnp.array(types), jnp.array(values), jnp.array(costs),
            self.sample_count, self.agent_masks, self.obs_masks,
            int(self.config.d), int(self.config.mechanism_type), int(self.config.noise_type), float(self.config.noise_scale)
        )
        
        asym = np.array(compute_invariance_asymmetry_matrix(
            self.jax_state.obs_covariance, self.jax_state.int_covariance, self.jax_state.int_mask
        ))
        
        # Stage 2 Per-Step Graph Prediction
        if predicted_dags is not None and len(predicted_dags) > 0:
            from src.stitching import stitch_predicted_dags
            stitched_dag, has_cycle = stitch_predicted_dags(predicted_dags, self.config.d, margin=self.boundary_margin)
            self.last_predicted_dag = stitched_dag
        else:
            self.last_predicted_dag = self.predict_graph_hypothesis(
                np.array(self.jax_state.obs_covariance), np.array(self.jax_state.running_covariance), asym
            )
            has_cycle = False
            stitched_dag = (self.last_predicted_dag > 0.5).astype(np.float32)

        true_dag = np.array(self.jax_state.true_adjacency)
        norm_factor = float(self.max_steps) if self.normalize_rewards else 1.0
        ig_dict = {"agent_0": float(info_gains[0]), "agent_1": float(info_gains[1])}
        
        # Compute interventional impact scores (number of non-zero variance shift entries)
        impact_a0 = float(np.sum(np.abs(asym[0:3, 0:3]) > 0.1))
        impact_a1 = float(np.sum(np.abs(asym[1:4, 1:4]) > 0.1))
        impact_dict = {"agent_0": impact_a0, "agent_1": impact_a1}
        
        from src.metrics import evaluate_dag_against_true
        eval_metrics = evaluate_dag_against_true(stitched_dag, true_dag)
        curr_shd = float(eval_metrics["shd"])
        
        terminated = bool(self.jax_state.step_count >= self.max_steps or np.all(np.array(self.jax_state.budgets) <= 0))
        
        diff = np.abs(stitched_dag - true_dag)
        e1 = float(np.sum(diff[0, :]) + np.sum(diff[:, 0]) + diff[1, 2] + diff[2, 1])
        e2 = float(np.sum(diff[3, :]) + np.sum(diff[:, 3]) + diff[1, 2] + diff[2, 1])
        
        from src.rewards import compute_ippo_rewards
        rewards = compute_ippo_rewards(
            stitched_dag, true_dag, has_cycle, 
            max_steps=norm_factor, 
            info_gains=ig_dict, 
            intrinsic_coef=self.intrinsic_coef,
            impact_scores=impact_dict,
            impact_coef=self.impact_coef,
            reward_density=self.reward_density,
            is_terminal=terminated,
            prev_shd=self.prev_shd
        )
        self.prev_shd = (e1, e2)
            
        obs_dict = self._get_obs_dict()
        return obs_dict, rewards, terminated, {
            "true_adjacency": true_dag, "info_gains": ig_dict,
            "impact_scores": impact_dict, "shd": curr_shd,
            "predicted_dag": self.last_predicted_dag
        }

    def step_jitted(
        self,
        cat_0: jax.Array, target_0: jax.Array, graph_pred_0: jax.Array,
        cat_1: jax.Array, target_1: jax.Array, graph_pred_1: jax.Array,
        key: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, bool, jax.Array, jax.Array]:
        """
        Pure JAX step executing intervention building, SCM sampling,
        DAG stitching, and reward computation entirely on GPU.
        Returns: (agent_observations, r0, r1, terminated, stitched_dag, info_gains)
        """
        from src.stitching import jitted_stitch_dags
        from src.rewards import jitted_compute_ippo_rewards
        
        costs_vec = jnp.array(self.action_costs)
        mask, types, values, costs = build_intervention_spec_jitted(
            cat_0, target_0, cat_1, target_1, self.jax_state.budgets, costs_vec, self.agent_masks, int(self.config.d),
            self.int_type_code, self.shift_val
        )
        
        k_step, key = jax.random.split(key)
        self.jax_state, self._agent_observations, info_gains = jitted_env_step_kernel(
            k_step, self.jax_state, mask, types, values, costs,
            self.sample_count, self.agent_masks, self.obs_masks,
            int(self.config.d), int(self.config.mechanism_type), int(self.config.noise_type), float(self.config.noise_scale)
        )
        
        stitched_dag, has_cycle = jitted_stitch_dags(graph_pred_0, graph_pred_1, int(self.config.d), margin=self.boundary_margin)
        norm_factor = float(self.max_steps) if self.normalize_rewards else 1.0
        r0, r1 = jitted_compute_ippo_rewards(
            stitched_dag, self.jax_state.true_adjacency, has_cycle, 
            max_steps=norm_factor, 
            info_gains=info_gains, 
            intrinsic_coef=self.intrinsic_coef
        )
        
        terminated = bool(self.jax_state.step_count >= self.max_steps or (self.jax_state.budgets[0] <= 0 and self.jax_state.budgets[1] <= 0))
        return self._agent_observations, r0, r1, terminated, stitched_dag, info_gains


