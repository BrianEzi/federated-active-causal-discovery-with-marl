import jax
import jax.numpy as jnp
import numpy as np
import json
from typing import Dict, Any

from src.types import SCMConfig, STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK, compute_edge_authority_masks, ActionCategory
from src.evaluator_env import FederatedCausalEnv
from src.marl.ppo_agent import IPPOActor, IPPORNNActor, InductiveIPPOActor, InductiveIPPORNNActor

def run_evaluation_suite(
    actor: Any,
    actor_params: Any,
    config: SCMConfig,
    action_costs: np.ndarray,
    initial_budget: float,
    use_rnn: bool = False,
    temperature: float = 0.0,
    boundary_margin: float = 0.10,
    seed: int = 42,
    ucb_coef: float = 0.0,
    uncertainty_coef: float = 0.0,
    oracle_agreement: bool = True,
    estimator_type: str = "analytic",
    intervention_type: str = "soft_shift",
    obs_feedback: bool = True,
    sample_count: int = 100,
    avici_max_context: int = 400,
    running_cov_ema_alpha: float = 0.3
) -> Dict[str, Any]:
    """
    Evaluates the trained agents on all 8 possible 4-node topologies.
    Supports both single and disjoint agent parameter lists, with optional low-temperature stochastic sampling
    and differential boundary margin thresholding.
    Returns a detailed execution trace.

    oracle_agreement: when True, additionally scores every step's chosen intervention
    target against what an information-optimal oracle would have picked, given the exact
    Bayes-optimal posterior over the 8 known candidate topologies (see
    src/marl/oracle_policy.py). Adds per-step oracle fields to the trace plus an
    episode-level summary -- a direct measure of *policy* quality that doesn't depend on
    the estimator actually converging, unlike SHD.
    """
    trace = {
        "metadata": {
            "temperature": float(temperature),
            "seed": int(seed),
            "initial_budget": float(initial_budget),
            "boundary_margin": float(boundary_margin),
            "ucb_coef": float(ucb_coef),
            "uncertainty_coef": float(uncertainty_coef),
            "estimator_type": estimator_type,
            "intervention_type": intervention_type,
            "use_rnn": bool(use_rnn)
        }
    }
    actor_apply = jax.jit(actor.apply)
    
    local_masks = [STANDARD_LOCAL_MASKS[0], STANDARD_LOCAL_MASKS[1]]
    boundary_mask = STANDARD_BOUNDARY_MASK
    edge_masks = compute_edge_authority_masks(STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK)
    
    from src.marl.ppo_agent import mask_invalid_targets, compute_ucb_bonus, compute_uncertainty_bonus
    from src.stitching import stitch_predicted_dags
    from src.metrics import evaluate_dag_against_true

    if oracle_agreement:
        from src.generators import get_all_4node_topologies
        from src.marl.bayes_optimal_estimator import compute_hypothesis_posterior
        from src.marl.oracle_policy import score_agent_action
        oracle_adjacencies = np.array(get_all_4node_topologies()[0])

    # Run evaluation on all 8 graphs
    for graph_idx in range(8):
        # We don't want fixed_graph to cache one, we want to force it per episode
        env = FederatedCausalEnv(config, action_costs, initial_budget=initial_budget, fixed_graph=False,
                                  boundary_margin=boundary_margin, ucb_coef=ucb_coef,
                                  uncertainty_coef=uncertainty_coef, estimator_type=estimator_type,
                                  intervention_type=intervention_type, obs_feedback=obs_feedback,
                                  sample_count=sample_count, avici_max_context=avici_max_context,
                                  running_cov_ema_alpha=running_cov_ema_alpha)
        local_masks = env.local_masks
        boundary_mask = env.boundary_mask
        key = jax.random.PRNGKey(seed + graph_idx)
        
        obs_dict, info = env.reset(key, force_idx=graph_idx)
        true_adj = info["true_adjacency"]
        
        if use_rnn:
            actor_states = {
                f"agent_{k}": IPPORNNActor.initial_state(1, 64)
                for k in range(config.K)
            }
            
        episode_trace = {
            "graph_idx": graph_idx,
            "true_adj": true_adj.tolist(),
            "steps": []
        }
        
        done = False
        step = 0
        
        while not done:
            step_trace = {
                "step": step,
                "budgets": env.jax_state.budgets.tolist(),
                "actions": {},
                "rewards": {}
            }
            
            joint_actions = {}

            # Same target-selection bonus applied at training time (see
            # src/marl/ppo_agent.py::compute_ucb_bonus / compute_uncertainty_bonus and
            # train.py's rollout loop) -- computed once per step, added to both agents'
            # target_logits before masking/sampling, so frozen evaluation exercises the
            # same policy definition it was trained under, not a train/eval mismatch.
            ucb_bonus = compute_ucb_bonus(env.jax_state.node_intervention_counts, env.jax_state.step_count, ucb_coef)
            uncertainty_bonus = compute_uncertainty_bonus(
                jnp.array(env.last_predicted_dag), jnp.array(env.structural_mask), uncertainty_coef
            )
            target_bonus = ucb_bonus + uncertainty_bonus

            if oracle_agreement:
                # Exact Bayes-optimal posterior over the 8 candidate topologies given the
                # data actually collected so far this episode -- the same computation the
                # bayes_optimal estimator uses, here purely as an analysis reference (it
                # never touches the policy's inputs or the reward).
                n_valid = int(env.jax_state.raw_count[0])
                if n_valid > 0:
                    oracle_posterior = compute_hypothesis_posterior(
                        np.array(env.jax_state.raw_samples[:n_valid], dtype=np.float64),
                        np.array(env.jax_state.raw_interv[:n_valid], dtype=np.float64),
                        oracle_adjacencies, float(config.noise_scale)
                    )
                else:
                    oracle_posterior = np.full(oracle_adjacencies.shape[0], 1.0 / oracle_adjacencies.shape[0])

            for k in range(config.K):
                obs = jnp.expand_dims(obs_dict[f"agent_{k}"], axis=0)
                params_k = actor_params[k] if isinstance(actor_params, list) else actor_params

                if use_rnn:
                    (cat_logits, target_logits), next_state = actor_apply(params_k, obs, actor_states[f"agent_{k}"])
                    actor_states[f"agent_{k}"] = next_state
                else:
                    cat_logits, target_logits = actor_apply(params_k, obs)
                target_logits = target_logits + target_bonus[None, :]

                local_mask = local_masks[k]
                    
                # Action selection: Greedy deterministic or temperature-controlled stochastic
                if temperature <= 0.0:
                    cat_action = int(jnp.argmax(cat_logits, axis=-1)[0])
                    masked_target_logits = mask_invalid_targets(jnp.array([cat_action]), target_logits, jnp.maximum(local_mask, boundary_mask))
                    target_action = int(jnp.argmax(masked_target_logits, axis=-1)[0])
                else:
                    cat_probs = np.array(jax.nn.softmax(cat_logits[0] / temperature))
                    cat_action = int(np.random.choice(len(cat_probs), p=cat_probs))
                    masked_target_logits = mask_invalid_targets(jnp.array([cat_action]), target_logits, jnp.maximum(local_mask, boundary_mask))
                    tgt_probs = np.array(jax.nn.softmax(masked_target_logits[0] / temperature))
                    if np.isnan(tgt_probs).any():
                        target_action = 0
                    else:
                        target_action = int(np.random.choice(len(tgt_probs), p=tgt_probs))
                
                joint_actions[f"agent_{k}"] = (cat_action, target_action)

                step_trace["actions"][f"agent_{k}"] = {"cat": cat_action, "target": target_action}

                if oracle_agreement:
                    step_trace["actions"][f"agent_{k}"]["oracle"] = score_agent_action(
                        cat_action, target_action, oracle_posterior, oracle_adjacencies,
                        valid_mask=np.array(jnp.maximum(local_mask, boundary_mask))
                    )

                step_trace["rewards"] = {k: float(v) for k, v in rewards.items()} if 'rewards' in locals() else {}

            next_obs, rewards, done, info = env.step(joint_actions, predicted_dags=None, key=key)
            
            step_trace["rewards"] = {k: float(v) for k, v in rewards.items()}
            
            stitched_dag = env.last_predicted_dag
            eval_metrics = evaluate_dag_against_true(stitched_dag, true_adj)
            
            step_trace["stitched_dag"] = stitched_dag.tolist()
            step_trace["shd"] = float(eval_metrics["shd"])
            
            episode_trace["steps"].append(step_trace)
            obs_dict = next_obs
            step += 1

        if oracle_agreement:
            scored = [a["oracle"] for s in episode_trace["steps"] for a in s["actions"].values() if "oracle" in a]
            if scored:
                episode_trace["oracle_summary"] = {
                    "scored_steps": len(scored),
                    "optimal_rate": float(np.mean([x["is_optimal"] for x in scored])),
                    "mean_normalized_score": float(np.mean([x["normalized_score"] for x in scored])),
                    "mean_regret": float(np.mean([x["regret"] for x in scored])),
                }
            else:
                episode_trace["oracle_summary"] = {
                    "scored_steps": 0,
                    "optimal_rate": None,
                    "mean_normalized_score": None,
                    "mean_regret": None,
                }

        trace[f"graph_{graph_idx}"] = episode_trace

    return trace

def evaluate_checkpoint(
    ckpt_path: str = "checkpoints/best_ippo_params.pkl",
    config: SCMConfig = None,
    action_costs: np.ndarray = None,
    initial_budget: float = 20.0,
    temperature: float = 0.0,
    boundary_margin: float = 0.10,
    seed: int = 42,
    oracle_agreement: bool = True
) -> Dict[str, Any]:
    """
    Loads checkpoint, dynamically detects architecture (Feedforward vs RNN, Baseline vs Inductive),
    and executes evaluation across all 8 graph topologies.
    """
    import os
    import pickle
    import haiku as hk
    from src.types import MechanismType, NoiseType

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_path}' not found. "
            "Please ensure training has completed and saved best_ippo_params.pkl before evaluating."
        )

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    if "actor_list" in ckpt:
        actor_params = ckpt["actor_list"]
    elif "actor" in ckpt:
        actor_params = [ckpt["actor"], ckpt["actor"]]
    else:
        raise KeyError(f"Unexpected checkpoint format in {ckpt_path}. Keys: {list(ckpt.keys())}")

    # Inspect first parameter key to detect RNN vs Feedforward vs Inductive
    first_param_key = list(actor_params[0].keys())[0] if len(actor_params) > 0 and len(actor_params[0]) > 0 else ""
    use_rnn = ckpt.get("use_rnn", "ippornn" in first_param_key.lower() or "rnn" in first_param_key.lower())
    use_inductive = ckpt.get("use_inductive_graph_head", "inductive" in first_param_key.lower())
    d = ckpt.get("d", 4)
    # Default 0.0 (no bonus) for checkpoints saved before the UCB/uncertainty fix existed --
    # they weren't trained with one, so applying a nonzero bonus at eval time would itself
    # be a train/eval mismatch for those older checkpoints.
    ucb_coef = ckpt.get("ucb_coef", 0.0)
    uncertainty_coef = ckpt.get("uncertainty_coef", 0.0)

    # Environment-construction parameters the checkpoint was actually trained under.
    # IMPORTANT BUG NOTE (found 2026-08-14): before these keys existed, this function
    # silently fell back to FederatedCausalEnv's own defaults (estimator_type="analytic",
    # intervention_type="soft_shift") and SCMConfig's default noise_scale=1.0, regardless
    # of what the checkpoint was actually trained with -- meaning every frozen-eval trace
    # produced before this fix (including the earlier "hard intervention re-check" finding
    # in docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md) silently evaluated under those
    # defaults instead. The get()-fallbacks below intentionally preserve that old
    # (incorrect) behavior *only* for checkpoints that predate this fix and have no other
    # way to recover their true training config -- they are a compatibility shim, not a
    # claim that "analytic"/"soft_shift"/1.0 were actually used to train those checkpoints.
    estimator_type = ckpt.get("estimator_type", "analytic")
    intervention_type = ckpt.get("intervention_type", "soft_shift")
    noise_scale = ckpt.get("noise_scale", 1.0)
    mechanism_type = ckpt.get("mechanism_type", "LINEAR")
    K = ckpt.get("K", 2)
    sample_count = ckpt.get("sample_count", 100)
    ckpt_obs_feedback = ckpt.get("obs_feedback", True)
    avici_max_context = ckpt.get("avici_max_context", 400)
    running_cov_ema_alpha = ckpt.get("running_cov_ema_alpha", 0.3)
    mechanism_type_int = int(MechanismType.LINEAR) if mechanism_type == "LINEAR" else int(MechanismType.NONLINEAR_ANM)

    if config is None:
        config = SCMConfig(d=d, K=K, mechanism_type=mechanism_type_int, noise_type=int(NoiseType.GAUSSIAN),
                            noise_scale=noise_scale)
    if action_costs is None:
        action_costs = np.array([1.0, 1.0])

    if use_inductive:
        if use_rnn:
            def forward(obs, state):
                return InductiveIPPORNNActor(d=d)(obs, state)
        else:
            def forward(obs):
                return InductiveIPPOActor(d=d)(obs)
    else:
        if use_rnn:
            def forward(obs, state):
                return IPPORNNActor(d=d)(obs, state)
        else:
            def forward(obs):
                return IPPOActor(d=d)(obs)

    actor_trans = hk.without_apply_rng(hk.transform(forward))

    return run_evaluation_suite(
        actor=actor_trans,
        actor_params=actor_params,
        config=config,
        action_costs=action_costs,
        initial_budget=initial_budget,
        use_rnn=use_rnn,
        temperature=temperature,
        boundary_margin=boundary_margin,
        seed=seed,
        ucb_coef=ucb_coef,
        uncertainty_coef=uncertainty_coef,
        oracle_agreement=oracle_agreement,
        estimator_type=estimator_type,
        intervention_type=intervention_type,
        obs_feedback=ckpt_obs_feedback,
        sample_count=sample_count,
        avici_max_context=avici_max_context,
        running_cov_ema_alpha=running_cov_ema_alpha
    )

