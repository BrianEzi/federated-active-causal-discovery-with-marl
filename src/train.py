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
from src.marl.ppo_agent import IPPOActor, IPPOCritic, IPPORNNActor, IPPORNNCritic, mask_invalid_targets, sample_actions_jitted
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
    parser.add_argument("--fixed_graph", type=int, nargs='?', const=-1, default=None, help="Pass an int (0-7) to fix to a specific topology, or pass without int to fix a random topology.")
    parser.add_argument("--save_file", action="store_true")
    
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="federated-causal-ippo")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: --use_wandb was passed, but the wandb library is not installed! WandB metrics will not be logged.")
        else:
            is_fixed = args.fixed_graph is not None
            name = args.run_name or f"{args.agent_type}{'_rnn' if args.use_rnn else ''}{'_fixed' if is_fixed else ''}_d{args.num_variables}"
            wandb.init(project=args.wandb_project, name=name, config=vars(args))
            if wandb.run:
                print(f"WandB successfully initialized! View live run at: {wandb.run.url}")
            
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
    
    is_fixed = args.fixed_graph is not None
    fixed_idx = args.fixed_graph if (is_fixed and args.fixed_graph >= 0) else None
    
    env = FederatedCausalEnv(config, action_costs, initial_budget=args.initial_budget, sample_count=args.sample_count, fixed_graph=is_fixed)
    env.max_steps = args.max_steps
    
    if args.agent_type == "ippo":
        if args.use_rnn:
            def make_actor():
                def forward(obs, state): return IPPORNNActor(d=args.num_variables)(obs, state)
                return hk.without_apply_rng(hk.transform(forward))
                
            def make_critic():
                def forward(obs, state): return IPPORNNCritic()(obs, state)
                return hk.without_apply_rng(hk.transform(forward))
        else:
            def make_actor():
                def forward(obs): return IPPOActor(d=args.num_variables)(obs)
                return hk.without_apply_rng(hk.transform(forward))
                
            def make_critic():
                def forward(obs): return IPPOCritic()(obs)
                return hk.without_apply_rng(hk.transform(forward))
            
        actor_trans = make_actor()
        critic_trans = make_critic()
        
        actor_apply = jax.jit(actor_trans.apply)
        critic_apply = jax.jit(critic_trans.apply)

        
        local_masks = [jnp.array([1.0, 1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0, 1.0])]
        boundary_mask = jnp.array([0.0, 1.0, 1.0, 0.0])
        observed_masks = [jnp.array([1.0, 1.0, 1.0, 0.0]), jnp.array([0.0, 1.0, 1.0, 1.0])]
        edge_masks = [
            jnp.maximum(jnp.outer(local_masks[0], local_masks[0]), jnp.outer(boundary_mask, boundary_mask)),
            jnp.maximum(jnp.outer(local_masks[1], local_masks[1]), jnp.outer(boundary_mask, boundary_mask))
        ]
        
        actor_lr = args.learning_rate if args.learning_rate != 3e-4 else args.actor_lr
        critic_lr = args.learning_rate if args.learning_rate != 3e-4 else args.critic_lr
        trainer = IPPOTrainer(actor_trans, critic_trans, actor_lr=actor_lr, critic_lr=critic_lr,
                              entropy_coef=args.entropy_coef, graph_coef=args.graph_coef, use_rnn=args.use_rnn)
                              
        # Initialize Disjoint parameters and optimizers per agent
        actor_params_list = []
        critic_params_list = []
        actor_opts = []
        critic_opts = []
        
        dummy_obs = jnp.zeros((1, args.num_variables * args.num_variables + 1))
        for k in range(args.num_agents):
            k1, k2, key = jax.random.split(key, 3)
            if args.use_rnn:
                dummy_state_a = IPPORNNActor.initial_state(1)
                dummy_state_c = IPPORNNCritic.initial_state(1)
                a_p = actor_trans.init(k1, dummy_obs, dummy_state_a)
                c_p = critic_trans.init(k2, dummy_obs, dummy_state_c)
            else:
                a_p = actor_trans.init(k1, dummy_obs)
                c_p = critic_trans.init(k2, dummy_obs)
            actor_params_list.append(a_p)
            critic_params_list.append(c_p)
            actor_opts.append(trainer.actor_opt.init(a_p))
            critic_opts.append(trainer.critic_opt.init(c_p))
        
        buffers = [RolloutBuffer() for _ in range(args.num_agents)]
    else:
        agents = [RandomAgent(i, args.num_variables) if args.agent_type == "random" else RoundRobinAgent(i, args.num_variables) for i in range(args.num_agents)]
    
    best_shd = 999.0
    best_f1 = -1.0
    
    all_metrics_history = []
    
    for episode in range(1, args.num_episodes + 1):
        k_ep, key = jax.random.split(key)
        obs_dict, info = env.reset(k_ep, force_idx=fixed_idx)
        true_adj = info["true_adjacency"]
        
        if args.agent_type == "ippo" and args.use_rnn:
            actor_states = [IPPORNNActor.initial_state(1) for _ in range(args.num_agents)]
            critic_states = [IPPORNNCritic.initial_state(1) for _ in range(args.num_agents)]
            
        done = False
        ep_reward = 0.0
        final_dag = None
        
        while not done:
            if args.agent_type == "ippo":
                # Agent 0
                k0_act, key = jax.random.split(key)
                obs_0 = jnp.expand_dims(env._agent_observations[0], 0)
                if args.use_rnn:
                    (cat_l0, tgt_l0, gr_l0), actor_states[0] = actor_apply(actor_params_list[0], obs_0, actor_states[0])
                    v0, critic_states[0] = critic_apply(critic_params_list[0], obs_0, critic_states[0])
                    val_0 = v0[0]
                else:
                    cat_l0, tgt_l0, gr_l0 = actor_apply(actor_params_list[0], obs_0)
                    val_0 = critic_apply(critic_params_list[0], obs_0)[0]
                c0, t0, lp0, gp0 = sample_actions_jitted(cat_l0[0], tgt_l0[0], gr_l0[0], local_masks[0], boundary_mask, edge_masks[0], k0_act)
                
                # Agent 1
                k1_act, key = jax.random.split(key)
                obs_1 = jnp.expand_dims(env._agent_observations[1], 0)
                if args.use_rnn:
                    (cat_l1, tgt_l1, gr_l1), actor_states[1] = actor_apply(actor_params_list[1], obs_1, actor_states[1])
                    v1, critic_states[1] = critic_apply(critic_params_list[1], obs_1, critic_states[1])
                    val_1 = v1[0]
                else:
                    cat_l1, tgt_l1, gr_l1 = actor_apply(actor_params_list[1], obs_1)
                    val_1 = critic_apply(critic_params_list[1], obs_1)[0]
                c1, t1, lp1, gp1 = sample_actions_jitted(cat_l1[0], tgt_l1[0], gr_l1[0], local_masks[1], boundary_mask, edge_masks[1], k1_act)
                
                k_step, key = jax.random.split(key)
                agent_obs, r0, r1, done, final_dag = env.step_jitted(c0, t0, gp0, c1, t1, gp1, k_step)
                
                buffers[0].add(obs=obs_0[0], cat_actions=c0, target_actions=t0, values=val_0, log_probs=lp0, graph_preds=gp0, rewards=r0, dones=done)
                buffers[1].add(obs=obs_1[0], cat_actions=c1, target_actions=t1, values=val_1, log_probs=lp1, graph_preds=gp1, rewards=r1, dones=done)
                ep_reward += float(r0 + r1)
            else:
                joint_actions = {}
                predicted_dags = {}
                for k in range(args.num_agents):
                    act, g_pred = agents[k].act(obs_dict[f"agent_{k}"])
                    joint_actions[f"agent_{k}"] = act
                    predicted_dags[f"agent_{k}"] = g_pred
                    
                k_step, key = jax.random.split(key)
                next_obs_dict, rewards, done, _ = env.step(joint_actions, predicted_dags, k_step)
                obs_dict = next_obs_dict
                ep_reward += sum(rewards.values())
                
                from src.stitching import stitch_predicted_dags
                final_dag, _ = stitch_predicted_dags(predicted_dags, args.num_variables)
            
        # End of episode update
        if args.agent_type == "ippo":
            actor_loss = 0.0
            critic_loss = 0.0
            graph_loss = 0.0
            entropy = 0.0
            per_agent_metrics = {}
            for k in range(args.num_agents):
                # Calculate GAE and pad batches to static shape max_steps to prevent XLA recompilations
                b = buffers[k].get_batches(max_size=args.max_steps)
                advs, rets = compute_gae(b["rewards"], b["values"], b["dones"])
                b["advantages"] = advs
                b["returns"] = rets
                
                # Update agent k's private parameters strictly on its own buffer
                a_p, c_p, a_opt, c_opt, metrics = trainer.update_step(
                    actor_params_list[k], critic_params_list[k], actor_opts[k], critic_opts[k], b, true_adj, observed_masks[k]
                )
                actor_params_list[k] = a_p
                critic_params_list[k] = c_p
                actor_opts[k] = a_opt
                critic_opts[k] = c_opt
                
                per_agent_metrics[k] = metrics
                actor_loss += float(metrics["actor_loss"])
                critic_loss += float(metrics["critic_loss"])
                graph_loss += float(metrics["graph_loss"])
                entropy += float(metrics["entropy"])
                buffers[k].reset()
                
        # Evaluate Stitched DAG (final step)
        eval_metrics = evaluate_dag_against_true(np.array(final_dag), true_adj)
        
        log_data = {
            "train/episode": int(episode),
            "train/episode_reward": float(ep_reward),
            "eval/shd": float(eval_metrics["shd"]),
            "eval/f1": float(eval_metrics["f1"]),
        }
        for k in range(args.num_agents):
            log_data[f"agent_{k}_budget"] = float(env.jax_state.budgets[k])
            
        if args.agent_type == "ippo":
            log_data["train/actor_loss"] = float(actor_loss / args.num_agents)
            log_data["train/critic_loss"] = float(critic_loss / args.num_agents)
            log_data["train/graph_loss"] = float(graph_loss / args.num_agents)
            log_data["train/entropy"] = float(entropy / args.num_agents)
            for k in range(args.num_agents):
                log_data[f"train/agent_{k}_actor_loss"] = float(per_agent_metrics[k]["actor_loss"])
                log_data[f"train/agent_{k}_critic_loss"] = float(per_agent_metrics[k]["critic_loss"])
                log_data[f"train/agent_{k}_graph_loss"] = float(per_agent_metrics[k]["graph_loss"])
                log_data[f"train/agent_{k}_entropy"] = float(per_agent_metrics[k]["entropy"])
            
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
                    pickle.dump({"actor_list": actor_params_list, "critic_list": critic_params_list}, f)

    if args.save_file:
        try:
            import pandas as pd
            df = pd.DataFrame(all_metrics_history)
            df.to_csv("training_metrics.csv", index=False)
        except ImportError:
            import csv
            if all_metrics_history:
                keys = list(all_metrics_history[0].keys())
                with open("training_metrics.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(all_metrics_history)
        print("Saved metrics to training_metrics.csv")
        
    if args.agent_type == "ippo":
        print("Running post-training evaluation suite on the BEST model...")
        import pickle
        try:
            with open("checkpoints/best_ippo_params.pkl", "rb") as f:
                ckpt = pickle.load(f)
                if "actor_list" in ckpt:
                    eval_actor_params = ckpt["actor_list"]
                else:
                    eval_actor_params = [ckpt["actor"] for _ in range(args.num_agents)]
                print("Successfully loaded best_ippo_params.pkl for evaluation.")
        except Exception as e:
            print("Could not load checkpoint, evaluating final model instead.")
            eval_actor_params = actor_params_list
            
        from src.evaluate import run_evaluation_suite
        import json
        trace = run_evaluation_suite(
            actor=actor_trans,
            actor_params=eval_actor_params,
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
            
    if args.use_wandb and WANDB_AVAILABLE and wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()
