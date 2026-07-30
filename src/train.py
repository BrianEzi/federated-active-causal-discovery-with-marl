import argparse
import os
import time
import random
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from src.types import SCMConfig, MechanismType, NoiseType
from src.evaluator_env import FederatedCausalEnv
from src.marl.ppo_agent import IPPOActor, IPPOCritic, IPPORNNActor, IPPORNNCritic, mask_invalid_targets
from src.marl.ppo_trainer import IPPOTrainer, RolloutBuffer, compute_gae
from src.baselines import RandomAgent, RoundRobinAgent
from src.metrics import evaluate_dag_against_true

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Active Causal Discovery IPPO Trainer")
    parser.add_argument("--agent_type", type=str, default="ippo", choices=["ippo", "random", "round_robin"])
    
    parser.add_argument("--num_variables", "-d", type=int, default=4)
    parser.add_argument("--num_agents", "-K", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--initial_budget", type=float, default=20.0)
    parser.add_argument("--action_cost", type=float, default=1.0)
    parser.add_argument("--sample_count", type=int, default=100)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--mechanism_type", type=str, default="LINEAR", choices=["LINEAR", "NONLINEAR_ANM"])
    
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--graph_coef", type=float, default=1.0)
    parser.add_argument("--eval_freq", type=int, default=10)
    
    parser.add_argument("--use_rnn", action="store_true")
    parser.add_argument("--fixed_graph", action="store_true")
    parser.add_argument("--save_file", action="store_true")
    
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="federated-causal-ippo")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.use_wandb and WANDB_AVAILABLE:
        name = args.run_name or f"{args.agent_type}{'_rnn' if args.use_rnn else ''}{'_fixed' if args.fixed_graph else ''}_d{args.num_variables}"
        wandb.init(project=args.wandb_project, name=name, config=vars(args))
            
    print(f"=== Starting Training Session ===")
    print(f"Config: agent={args.agent_type}, d={args.num_variables}, Episodes={args.num_episodes}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    action_costs = jnp.full(args.num_agents, args.action_cost)
    config = SCMConfig(
        d=args.num_variables,
        K=args.num_agents,
        mechanism_type=int(MechanismType.LINEAR if args.mechanism_type == "LINEAR" else MechanismType.NONLINEAR_ANM),
        noise_type=int(NoiseType.GAUSSIAN),
        noise_scale=args.noise_scale
    )
    
    env = FederatedCausalEnv(config, action_costs, initial_budget=args.initial_budget, sample_count=args.sample_count, fixed_graph=args.fixed_graph)
    env.max_steps = args.max_steps
    
    if args.agent_type == "ippo":
        def make_actor():
            def forward(obs, state=None): 
                if state is not None: return IPPORNNActor(d=args.num_variables)(obs, state)
                return IPPOActor(d=args.num_variables)(obs)
            return hk.without_apply_rng(hk.transform(forward))
            
        def make_critic():
            def forward(obs, state=None): 
                if state is not None: return IPPORNNCritic()(obs, state)
                return IPPOCritic()(obs)
            return hk.without_apply_rng(hk.transform(forward))
            
        actor_trans = make_actor()
        critic_trans = make_critic()
        
        actor_lr = args.learning_rate if args.learning_rate != 3e-4 else args.actor_lr
        critic_lr = args.learning_rate if args.learning_rate != 3e-4 else args.critic_lr
        trainer = IPPOTrainer(actor_trans, critic_trans, actor_lr=actor_lr, critic_lr=critic_lr,
                              entropy_coef=args.entropy_coef, graph_coef=args.graph_coef, use_rnn=args.use_rnn)
                              
        # Init params
        dummy_obs = jnp.zeros((1, args.num_variables * args.num_variables + 1))
        k1, k2, key = jax.random.split(key, 3)
        if args.use_rnn:
            dummy_state_a = IPPORNNActor.initial_state(1)
            dummy_state_c = IPPORNNCritic.initial_state(1)
            actor_params = actor_trans.init(k1, dummy_obs, dummy_state_a)
            critic_params = critic_trans.init(k2, dummy_obs, dummy_state_c)
        else:
            actor_params = actor_trans.init(k1, dummy_obs)
            critic_params = critic_trans.init(k2, dummy_obs)
            
        actor_opt = trainer.actor_opt.init(actor_params)
        critic_opt = trainer.critic_opt.init(critic_params)
        
        buffers = [RolloutBuffer() for _ in range(args.num_agents)]
    else:
        agents = [RandomAgent(i, args.num_variables) if args.agent_type == "random" else RoundRobinAgent(i, args.num_variables) for i in range(args.num_agents)]
    
    best_shd = 999.0
    best_f1 = -1.0
    
    all_metrics_history = []
    
    for episode in range(1, args.num_episodes + 1):
        k_ep, key = jax.random.split(key)
        obs_dict, info = env.reset(k_ep)
        true_adj = info["true_adjacency"]
        
        if args.agent_type == "ippo" and args.use_rnn:
            actor_states = [IPPORNNActor.initial_state(1) for _ in range(args.num_agents)]
            critic_states = [IPPORNNCritic.initial_state(1) for _ in range(args.num_agents)]
            
        done = False
        ep_reward = 0.0
        
        while not done:
            joint_actions = {}
            predicted_dags = {}
            
            if args.agent_type == "ippo":
                # IPPO Act
                for k in range(args.num_agents):
                    obs_k = jnp.array([obs_dict[f"agent_{k}"]])
                    if args.use_rnn:
                        (cat_logits, target_logits, graph_logits), actor_states[k] = actor_trans.apply(actor_params, obs_k, actor_states[k])
                        val_batch, critic_states[k] = critic_trans.apply(critic_params, obs_k, critic_states[k])
                        val = val_batch[0]
                    else:
                        cat_logits, target_logits, graph_logits = actor_trans.apply(actor_params, obs_k)
                        val = critic_trans.apply(critic_params, obs_k)[0]
                    
                    # Action selection
                    cat_dist = jax.nn.softmax(cat_logits[0])
                    cat = int(np.random.choice(3, p=np.array(cat_dist)))
                    
                    if k == 0:
                        mask = jnp.array([1.0, 1.0, 1.0, 0.0])
                    else:
                        mask = jnp.array([0.0, 1.0, 1.0, 1.0])
                        
                    masked_targets = mask_invalid_targets(jnp.array([cat]), target_logits, mask)[0]
                    tgt_dist = jax.nn.softmax(masked_targets)
                    # Safe fallback if all nan
                    if jnp.isnan(tgt_dist).any():
                        target = 0
                        tgt_lp = 0.0
                    else:
                        target = int(np.random.choice(args.num_variables, p=np.array(tgt_dist)))
                        tgt_lp = jnp.log(tgt_dist[target])
                        
                    cat_lp = jnp.log(cat_dist[cat])
                    log_prob = cat_lp + tgt_lp
                    
                    graph_pred = jax.nn.sigmoid(graph_logits[0])
                    
                    # Apply hard mask to prevent cross-domain edge predictions
                    edge_mask = jnp.outer(mask, mask)
                    graph_pred = graph_pred * edge_mask
                    
                    joint_actions[f"agent_{k}"] = (cat, target)
                    predicted_dags[f"agent_{k}"] = np.array(graph_pred)
                    
                    buffers[k].add(obs=obs_k[0], cat_actions=cat, target_actions=target, 
                                   values=val, log_probs=log_prob, graph_preds=graph_pred,
                                   rewards=0.0, dones=False) # rewards/dones updated after step
            else:
                for k in range(args.num_agents):
                    act, g_pred = agents[k].act(obs_dict[f"agent_{k}"])
                    joint_actions[f"agent_{k}"] = act
                    predicted_dags[f"agent_{k}"] = g_pred
                    
            k_step, key = jax.random.split(key)
            next_obs_dict, rewards, done, _ = env.step(joint_actions, predicted_dags, k_step)
            
            if args.agent_type == "ippo":
                for k in range(args.num_agents):
                    buffers[k].data["rewards"][-1] = rewards[f"agent_{k}"]
                    buffers[k].data["dones"][-1] = done
            
            obs_dict = next_obs_dict
            ep_reward += sum(rewards.values())
            
        # End of episode update
        if args.agent_type == "ippo":
            actor_loss = 0.0
            critic_loss = 0.0
            graph_loss = 0.0
            entropy = 0.0
            for k in range(args.num_agents):
                # Calculate GAE
                b = buffers[k].get_batches()
                advs, rets = compute_gae(b["rewards"], b["values"], b["dones"])
                b["advantages"] = advs
                b["returns"] = rets
                
                # Mask out unobservable nodes for the graph BCE loss
                if k == 0:
                    observed_mask = jnp.array([1.0, 1.0, 1.0, 0.0])
                else:
                    observed_mask = jnp.array([0.0, 1.0, 1.0, 1.0])
                
                # Update
                actor_params, critic_params, actor_opt, critic_opt, metrics = trainer.update_step(
                    actor_params, critic_params, actor_opt, critic_opt, b, true_adj, observed_mask
                )
                actor_loss += metrics["actor_loss"]
                critic_loss += metrics["critic_loss"]
                graph_loss += metrics["graph_loss"]
                entropy += metrics["entropy"]
                buffers[k].reset()
                
        # Evaluate Stitched DAG (final step)
        from src.stitching import stitch_predicted_dags
        final_dag, _ = stitch_predicted_dags(predicted_dags, args.num_variables)
        eval_metrics = evaluate_dag_against_true(final_dag, true_adj)
        
        log_data = {
            "train/episode": episode,
            "train/episode_reward": ep_reward,
            "eval/shd": eval_metrics["shd"],
            "eval/f1": eval_metrics["f1"],
        }
        for k in range(args.num_agents):
            log_data[f"agent_{k}_budget"] = env.jax_state.budgets[k]
            
        if args.agent_type == "ippo":
            log_data["train/actor_loss"] = actor_loss / args.num_agents
            log_data["train/critic_loss"] = critic_loss / args.num_agents
            log_data["train/graph_loss"] = graph_loss / args.num_agents
            log_data["train/entropy"] = entropy / args.num_agents
            
        all_metrics_history.append(log_data.copy())
            
        if args.use_wandb and WANDB_AVAILABLE:
            if episode % args.eval_freq == 0:
                from src.visualizations import plot_dag_to_wandb_image
                log_data["eval/true_dag_img"] = plot_dag_to_wandb_image(true_adj, f"True DAG (Ep {episode})")
                log_data["eval/pred_dag_img"] = plot_dag_to_wandb_image(final_dag, f"Predicted DAG (Ep {episode})")
                
            wandb.log(log_data)
            
        if episode % 10 == 0:
            print(f"[Episode {episode}] Reward: {ep_reward:.2f} | SHD: {eval_metrics['shd']:.2f} | F1: {eval_metrics['f1']:.2f}")

        # Checkpoint Best Model
        if args.agent_type == "ippo":
            current_shd = eval_metrics["shd"]
            current_f1 = eval_metrics["f1"]
            is_best = False
            
            if current_shd < best_shd:
                is_best = True
            elif current_shd == best_shd and current_f1 > best_f1:
                is_best = True
                
            if is_best:
                best_shd = current_shd
                best_f1 = current_f1
                import pickle
                os.makedirs("checkpoints", exist_ok=True)
                with open("checkpoints/best_ippo_params.pkl", "wb") as f:
                    pickle.dump({"actor": actor_params, "critic": critic_params}, f)

    if args.save_file:
        import pandas as pd
        df = pd.DataFrame(all_metrics_history)
        df.to_csv("training_metrics.csv", index=False)
        print("Saved metrics to training_metrics.csv")
        
    if args.agent_type == "ippo":
        print("Running post-training evaluation suite on the BEST model...")
        import pickle
        try:
            with open("checkpoints/best_ippo_params.pkl", "rb") as f:
                ckpt = pickle.load(f)
                actor_params = ckpt["actor"]
                critic_params = ckpt["critic"]
                print("Successfully loaded best_ippo_params.pkl for evaluation.")
        except Exception as e:
            print("Could not load checkpoint, evaluating final model instead.")
            
        from src.evaluate import run_evaluation_suite
        import json
        trace = run_evaluation_suite(
            actor=actor_trans,
            actor_params=actor_params,
            config=config,
            action_costs=action_costs,
            initial_budget=args.initial_budget,
            use_rnn=args.use_rnn
        )
        with open("evaluation_trace.json", "w") as f:
            json.dump(trace, f, indent=2)
        print("Saved evaluation trace to evaluation_trace.json")
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.save("evaluation_trace.json")
            print("Uploaded evaluation trace to WandB.")

if __name__ == "__main__":
    main()
