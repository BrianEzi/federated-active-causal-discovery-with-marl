import jax
import jax.numpy as jnp
import numpy as np
import json
from typing import Dict, Any

from src.types import SCMConfig
from src.evaluator_env import FederatedCausalEnv
from src.marl.ppo_agent import IPPOActor, IPPORNNActor

def run_evaluation_suite(
    actor: Any,
    actor_params: Any,
    config: SCMConfig,
    action_costs: np.ndarray,
    initial_budget: float,
    use_rnn: bool = False,
    temperature: float = 0.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates the trained agents on all 8 possible 4-node topologies.
    Supports both single and disjoint agent parameter lists, with optional low-temperature stochastic sampling.
    Returns a detailed execution trace.
    """
    trace = {}
    actor_apply = jax.jit(actor.apply)
    
    local_masks = [jnp.array([1.0, 1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0, 1.0])]
    boundary_mask = jnp.array([0.0, 1.0, 1.0, 0.0])
    edge_masks = [
        jnp.maximum(jnp.outer(local_masks[0], local_masks[0]), jnp.outer(boundary_mask, boundary_mask)),
        jnp.maximum(jnp.outer(local_masks[1], local_masks[1]), jnp.outer(boundary_mask, boundary_mask))
    ]
    
    from src.marl.ppo_agent import mask_invalid_targets
    from src.stitching import stitch_predicted_dags
    from src.metrics import evaluate_dag_against_true
    
    # Run evaluation on all 8 graphs
    for graph_idx in range(8):
        # We don't want fixed_graph to cache one, we want to force it per episode
        env = FederatedCausalEnv(config, action_costs, initial_budget=initial_budget, fixed_graph=False)
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
                "graph_preds": {}
            }
            
            joint_actions = {}
            predicted_dags = {}
            
            for k in range(config.K):
                obs = jnp.expand_dims(obs_dict[f"agent_{k}"], axis=0)
                params_k = actor_params[k] if isinstance(actor_params, list) else actor_params
                
                if use_rnn:
                    (cat_logits, target_logits, graph_logits), next_state = actor_apply(params_k, obs, actor_states[f"agent_{k}"])
                    actor_states[f"agent_{k}"] = next_state
                else:
                    cat_logits, target_logits, graph_logits = actor_apply(params_k, obs)
                
                local_mask = local_masks[k]
                    
                # Action selection: Greedy deterministic or temperature-controlled stochastic
                if temperature <= 0.0:
                    cat_action = int(jnp.argmax(cat_logits, axis=-1)[0])
                    masked_target_logits = mask_invalid_targets(jnp.array([cat_action]), target_logits, local_mask, boundary_mask)
                    target_action = int(jnp.argmax(masked_target_logits, axis=-1)[0])
                else:
                    cat_probs = np.array(jax.nn.softmax(cat_logits[0] / temperature))
                    cat_action = int(np.random.choice(len(cat_probs), p=cat_probs))
                    masked_target_logits = mask_invalid_targets(jnp.array([cat_action]), target_logits, local_mask, boundary_mask)
                    tgt_probs = np.array(jax.nn.softmax(masked_target_logits[0] / temperature))
                    if np.isnan(tgt_probs).any():
                        target_action = 0
                    else:
                        target_action = int(np.random.choice(len(tgt_probs), p=tgt_probs))
                
                graph_pred = jax.nn.sigmoid(graph_logits[0]) * edge_masks[k]
                
                joint_actions[f"agent_{k}"] = (cat_action, target_action)
                predicted_dags[f"agent_{k}"] = np.array(graph_pred)
                
                step_trace["actions"][f"agent_{k}"] = {"cat": cat_action, "target": target_action}
                step_trace["graph_preds"][f"agent_{k}"] = np.array(graph_pred).tolist()
                
            next_obs, rewards, done, info = env.step(joint_actions, predicted_dags, key)
            
            step_trace["rewards"] = {k: float(v) for k, v in rewards.items()}
            
            stitched_dag, _ = stitch_predicted_dags(predicted_dags, config.d)
            eval_metrics = evaluate_dag_against_true(stitched_dag, true_adj)
            
            step_trace["stitched_dag"] = stitched_dag.tolist()
            step_trace["shd"] = float(eval_metrics["shd"])
            
            episode_trace["steps"].append(step_trace)
            obs_dict = next_obs
            step += 1
            
        trace[f"graph_{graph_idx}"] = episode_trace
        
    return trace
