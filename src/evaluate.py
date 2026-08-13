import jax
import jax.numpy as jnp
import numpy as np
import json
from typing import Dict, Any

from src.types import SCMConfig, STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK, compute_edge_authority_masks
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
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates the trained agents on all 8 possible 4-node topologies.
    Supports both single and disjoint agent parameter lists, with optional low-temperature stochastic sampling
    and differential boundary margin thresholding.
    Returns a detailed execution trace.
    """
    trace = {
        "metadata": {
            "temperature": float(temperature),
            "seed": int(seed),
            "initial_budget": float(initial_budget),
            "boundary_margin": float(boundary_margin),
            "use_rnn": bool(use_rnn)
        }
    }
    actor_apply = jax.jit(actor.apply)
    
    local_masks = [STANDARD_LOCAL_MASKS[0], STANDARD_LOCAL_MASKS[1]]
    boundary_mask = STANDARD_BOUNDARY_MASK
    edge_masks = compute_edge_authority_masks(STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK)
    
    from src.marl.ppo_agent import mask_invalid_targets
    from src.stitching import stitch_predicted_dags
    from src.metrics import evaluate_dag_against_true
    
    # Run evaluation on all 8 graphs
    for graph_idx in range(8):
        # We don't want fixed_graph to cache one, we want to force it per episode
        env = FederatedCausalEnv(config, action_costs, initial_budget=initial_budget, fixed_graph=False, boundary_margin=boundary_margin)
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
            
            for k in range(config.K):
                obs = jnp.expand_dims(obs_dict[f"agent_{k}"], axis=0)
                params_k = actor_params[k] if isinstance(actor_params, list) else actor_params
                
                if use_rnn:
                    (cat_logits, target_logits), next_state = actor_apply(params_k, obs, actor_states[f"agent_{k}"])
                    actor_states[f"agent_{k}"] = next_state
                else:
                    cat_logits, target_logits = actor_apply(params_k, obs)
                
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
            
        trace[f"graph_{graph_idx}"] = episode_trace
        
    return trace

def evaluate_checkpoint(
    ckpt_path: str = "checkpoints/best_ippo_params.pkl",
    config: SCMConfig = None,
    action_costs: np.ndarray = None,
    initial_budget: float = 20.0,
    temperature: float = 0.0,
    boundary_margin: float = 0.10,
    seed: int = 42
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

    if config is None:
        config = SCMConfig(d=d, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
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
        seed=seed
    )

