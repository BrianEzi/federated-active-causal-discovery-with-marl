"""Single-Agent Benchmark Evaluation Suite.

Evaluates:
1. Information-Optimal Oracle Policy on all 8 graphs with full authority (K=1).
2. Random and Round-Robin Baselines.
3. Trained Single-Agent PPO policy under deterministic greedy evaluation (temp=0.0).
"""
import argparse
import json
import os
import numpy as np
import jax
import jax.numpy as jnp

from src.types import SCMConfig, MechanismType, NoiseType, ActionCategory
from src.evaluator_env import FederatedCausalEnv
from src.generators import get_all_4node_topologies
from src.marl.oracle_policy import oracle_best_targets, score_agent_action
from src.marl.bayes_optimal_estimator import compute_hypothesis_posterior
from src.metrics import evaluate_dag_against_true


def run_oracle_single_agent_benchmark(budget=5.0, action_cost=1.0, max_steps=10):
    print("================================================================================")
    print("       SINGLE-AGENT THEORETICAL CEILING: INFORMATION-OPTIMAL ORACLE             ")
    print("================================================================================")
    
    adjacencies, _ = get_all_4node_topologies()
    candidate_adjacencies = np.array(adjacencies)
    H, d, _ = candidate_adjacencies.shape
    
    names = [
        "Graph 0: Chain (Z1 -> X1 -> X2 -> Z2)",
        "Graph 1: Rev Chain (Z1 <- X1 <- X2 <- Z2)",
        "Graph 2: Collider (Z1 -> X1 <- X2 <- Z2)",
        "Graph 3: Rev Collider (Z1 -> X1 -> X2 <- Z2)",
        "Graph 4: Fork (Z1 <- X1 -> X2 -> Z2)",
        "Graph 5: Rev Fork (Z1 <- X1 <- X2 -> Z2)",
        "Graph 6: Fork+Collider (Z1 -> X1 <- X2 -> Z2)",
        "Graph 7: Rev Fork+Collider (Z1 <- X1 -> X2 <- Z2)"
    ]
    
    results = []
    
    for graph_idx in range(8):
        config = SCMConfig(d=4, K=1, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN), noise_scale=0.1)
        action_costs = np.array([action_cost])
        env = FederatedCausalEnv(config, action_costs, initial_budget=budget, fixed_graph=True, estimator_type="bayes_optimal")
        
        obs_dict, info = env.reset(jax.random.PRNGKey(graph_idx * 100), force_idx=graph_idx)
        true_adj = info["true_adjacency"]
        
        step = 0
        done = False
        step_history = []
        
        while not done and step < max_steps:
            n_valid = int(env.jax_state.raw_count[0])
            if n_valid > 0:
                posterior = compute_hypothesis_posterior(
                    np.array(env.jax_state.raw_samples[:n_valid], dtype=np.float64),
                    np.array(env.jax_state.raw_interv[:n_valid], dtype=np.float64),
                    candidate_adjacencies, float(config.noise_scale)
                )
            else:
                posterior = np.full(H, 1.0 / H)
                
            scores, best_targets = oracle_best_targets(posterior, candidate_adjacencies, valid_mask=[1, 1, 1, 1])
            best_score = float(np.max(scores))
            
            # If information remains, intervene on best target; otherwise NOOP
            if best_score > 1e-4:
                cat_action = int(ActionCategory.INTERVENE)
                target_action = int(np.where(best_targets)[0][0])
            else:
                cat_action = int(ActionCategory.NOOP)
                target_action = 0
                
            joint_actions = {"agent_0": (cat_action, target_action)}
            next_obs, rewards, done, step_info = env.step(joint_actions, predicted_dags=None, key=jax.random.PRNGKey(step * 77))
            
            curr_shd = float(step_info["shd"])
            score_data = score_agent_action(cat_action, target_action, posterior, candidate_adjacencies, valid_mask=[1, 1, 1, 1])
            
            step_history.append({
                "step": step,
                "action": f"{'INTERVENE(do(X' + str(target_action) + '))' if cat_action == 0 else 'NOOP'}",
                "shd": curr_shd,
                "score_data": score_data
            })
            step += 1
            if curr_shd == 0 and cat_action == int(ActionCategory.NOOP):
                break
                
        final_shd = step_history[-1]["shd"]
        interventions_used = sum(1 for s in step_history if "INTERVENE" in s["action"])
        remaining_b = float(env.jax_state.budgets[0])
        
        results.append({
            "graph": names[graph_idx],
            "final_shd": final_shd,
            "interventions": interventions_used,
            "steps": len(step_history),
            "remaining_budget": remaining_b,
            "history": step_history
        })
        
        print(f"\n[{names[graph_idx]}]")
        print(f"  Final SHD: {final_shd:.1f} | Interventions Used: {interventions_used} | Remaining Budget: {remaining_b:.1f}/{budget}")
        for s in step_history:
            print(f"    Step {s['step']}: {s['action']:<25} -> SHD={s['shd']:.1f} ({s['score_data']['action_type']})")
            
    print("\n--------------------------------------------------------------------------------")
    print(f"ORACLE BENCHMARK SUMMARY (K=1, Budget={budget}):")
    print(f"  Mean Final SHD: {np.mean([r['final_shd'] for r in results]):.2f}")
    print(f"  Success Rate (SHD=0): {np.mean([r['final_shd'] == 0 for r in results]):.1%}")
    print(f"  Mean Interventions Used: {np.mean([r['interventions'] for r in results]):.2f}")
    print(f"  Mean Budget Preserved: {np.mean([r['remaining_budget'] for r in results]):.2f}/{budget}")
    print("--------------------------------------------------------------------------------\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--action_cost", type=float, default=1.0)
    args = parser.parse_args()
    
    run_oracle_single_agent_benchmark(budget=args.budget, action_cost=args.action_cost)
