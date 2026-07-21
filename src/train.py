import argparse
import os
import time
import numpy as np
import jax
import jax.numpy as jnp

from src.types import SCMConfig, MechanismType, NoiseType
from src.generators import generate_er_dag, generate_ba_dag, generate_scm_params
from src.evaluator_env import FederatedCausalEnv
from src.marl.agent import MLPAgent, RNNAgent, CausalTransformerAgent
from src.marl.mixer import QMIXMixer
from src.marl.trainer import QMIXTrainer
from src.marl.buffer import TrajectoryBuffer
from src.metrics import evaluate_pag_against_dag

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Active Causal Discovery MARL Trainer")
    # Architecture Choice
    parser.add_argument("--agent_type", type=str, default="mlp", choices=["mlp", "rnn", "transformer"],
                        help="Decentralized agent architecture: mlp, rnn (GRU), or transformer (Causal Self-Attention)")
    
    # Environment & SCM Hyperparameters
    parser.add_argument("--num_variables", "-d", type=int, default=5, help="Number of nodes in SCM graph")
    parser.add_argument("--num_agents", "-K", type=int, default=2, help="Number of decentralized agents")
    parser.add_argument("--graph_type", type=str, default="ER", choices=["ER", "BA"], help="Graph generator type")
    parser.add_argument("--edge_prob", type=float, default=0.5, help="ER edge probability")
    parser.add_argument("--ba_edges", type=int, default=1, help="BA edges per node")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--initial_budget", type=float, default=10.0, help="Initial intervention budget per agent")
    parser.add_argument("--action_cost", type=float, default=0.05, help="Cost per intervention action")
    parser.add_argument("--sample_count", type=int, default=500, help="Interventional sample count per step")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="SCM observational noise scale")
    parser.add_argument("--mechanism_type", type=str, default="LINEAR", choices=["LINEAR", "NONLINEAR_ANM"], help="SCM mechanism type")
    parser.add_argument("--noise_type", type=str, default="GAUSSIAN", choices=["GAUSSIAN", "GUMBEL", "UNIFORM"], help="SCM noise distribution")
    
    # Reward Shaping Hyperparameters
    parser.add_argument("--circle_reward", type=float, default=10.0, help="Reward per resolved circle mark")
    parser.add_argument("--noop_penalty", type=float, default=0.5, help="Penalty applied when ALL agents NO-OP while circles remain")
    parser.add_argument("--violation_penalty", type=float, default=20.0, help="Penalty per structural PAG violation")
    
    # Exploration & QMIX Hyperparameters
    parser.add_argument("--num_episodes", type=int, default=100, help="Total training episodes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for QMIX TD loss")
    parser.add_argument("--buffer_capacity", type=int, default=100, help="Replay buffer episode capacity")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Optax Adam")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Agent hidden dimension")
    parser.add_argument("--mixer_hidden_dim", type=int, default=32, help="QMIX hypernetwork hidden dimension")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum exploration epsilon")
    parser.add_argument("--epsilon_decay_frac", type=float, default=0.8, help="Fraction of episodes over which epsilon decays")
    parser.add_argument("--target_update_freq", type=int, default=10, help="Episodes between target network updates")
    parser.add_argument("--eval_freq", type=int, default=10, help="Episodes between evaluation runs")
    
    # WandB Settings
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="federated-causal-marl", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/username")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name override")
    parser.add_argument("--seed", type=int, default=42, help="Random PRNG seed")
    
    return parser.parse_args()

def create_agent_masks(num_variables: int, num_agents: int) -> jax.Array:
    """Creates overlapping variable jurisdiction masks for K agents."""
    masks = np.zeros((num_agents, num_variables), dtype=np.float32)
    vars_per_agent = max(2, int(np.ceil(num_variables * 0.7)))
    for k in range(num_agents):
        start_idx = (k * (num_variables - vars_per_agent)) // max(1, num_agents - 1)
        end_idx = min(num_variables, start_idx + vars_per_agent)
        masks[k, start_idx:end_idx] = 1.0
        
    return jnp.array(masks)

def evaluate_agent(env, trainer, train_state, key, agent_type="mlp", num_eval_episodes=3):
    """Performs deterministic evaluation rollouts and logs structural graph metrics."""
    shd_list, f1_list, prec_list, rec_list, return_list = [], [], [], [], []
    K = env.config.K
    
    for ep in range(num_eval_episodes):
        obs_dict, info = env.reset(jax.random.fold_in(key, ep + 999))
        done = False
        step_count = 0
        ep_reward = 0.0
        
        if agent_type == "rnn":
            rnn_carry = trainer.agent.initialize_carry((K,))
        elif agent_type == "transformer":
            obs_history = []
        
        while not done and step_count < env.max_steps:
            obs_array = jnp.stack([obs_dict[f"agent_{k}"] for k in range(K)])
            avail_array = np.array([info["avail_actions"][f"agent_{k}"] for k in range(K)])
            
            # Greedy action selection (epsilon = 0.0)
            if agent_type == "mlp":
                q_values_jnp = trainer.agent.apply(train_state.params['agent'], obs_array)
            elif agent_type == "rnn":
                rnn_carry, q_values_jnp = trainer.agent.apply(train_state.params['agent'], rnn_carry, obs_array)
            elif agent_type == "transformer":
                obs_history.append(obs_array)
                seq_jnp = jnp.array(obs_history).transpose(1, 0, 2)
                q_seq = trainer.agent.apply(train_state.params['agent'], seq_jnp)
                q_values_jnp = q_seq[:, -1, :]
                
            joint_actions = {}
            for k in range(K):
                q_k = np.array(q_values_jnp[k])
                q_k[avail_array[k] == 0] = -1e9
                greedy_action = int(np.argmax(q_k))
                joint_actions[f"agent_{k}"] = greedy_action
                
            obs_dict, reward, done, info = env.step(joint_actions, jax.random.fold_in(key, step_count))
            ep_reward += reward
            step_count += 1
            
        pag_matrix = info["pag"]
        true_dag = np.array(env.adjacency)
        metrics = evaluate_pag_against_dag(pag_matrix, true_dag)
        
        shd_list.append(metrics["shd"])
        f1_list.append(metrics["f1"])
        prec_list.append(metrics["precision"])
        rec_list.append(metrics["recall"])
        return_list.append(ep_reward)
        
    return {
        "eval/mean_return": float(np.mean(return_list)),
        "eval/mean_shd": float(np.mean(shd_list)),
        "eval/mean_f1": float(np.mean(f1_list)),
        "eval/mean_precision": float(np.mean(prec_list)),
        "eval/mean_recall": float(np.mean(rec_list)),
    }

def main():
    args = parse_args()
    
    # Format a descriptive WandB run name reflecting parameters and model choice
    descriptive_run_name = (
        args.run_name or 
        f"qmix_{args.agent_type}_d{args.num_variables}_K{args.num_agents}_{args.graph_type}_"
        f"lr{args.lr}_cost{args.action_cost}_cr{args.circle_reward}_noop{args.noop_penalty}_s{args.seed}"
    )
    
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("[Warning] wandb package not found. Continuing without wandb logging.")
            args.use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=descriptive_run_name,
                config=vars(args)
            )
            
    print(f"=== Starting Training Session ===")
    print(f"Run Name: {descriptive_run_name}")
    print(f"Config: agent={args.agent_type}, d={args.num_variables}, K={args.num_agents}, Graph={args.graph_type}, Episodes={args.num_episodes}")
    
    key = jax.random.PRNGKey(args.seed)
    k1, k2, key = jax.random.split(key, 3)
    
    d = args.num_variables
    K = args.num_agents
    
    mech_enum = MechanismType.LINEAR if args.mechanism_type == "LINEAR" else MechanismType.NONLINEAR_ANM
    noise_enum = getattr(NoiseType, args.noise_type)
    
    # 1. Generate synthetic graph & SCM parameters
    if args.graph_type == "ER":
        adj = generate_er_dag(k1, d, edge_prob=args.edge_prob)
    else:
        adj = generate_ba_dag(k1, d, num_edges_per_node=args.ba_edges)
        
    scm_params = generate_scm_params(k2, adj, mech_enum)
    topological_order = jnp.arange(d)
    agent_masks = create_agent_masks(d, K)
    action_costs = jnp.full(K, args.action_cost)
    
    config = SCMConfig(
        d=d,
        K=K,
        mechanism_type=int(mech_enum),
        noise_type=int(noise_enum),
        noise_scale=args.noise_scale
    )
    
    # 2. Initialize Environment
    env = FederatedCausalEnv(
        config, adj, scm_params, topological_order, agent_masks, action_costs,
        initial_budget=args.initial_budget,
        sample_count=args.sample_count,
        circle_reward=args.circle_reward,
        noop_penalty=args.noop_penalty,
        violation_penalty=args.violation_penalty
    )
    env.max_steps = args.max_steps
    
    # 3. Initialize Agent Architecture (MLP, RNN, or CausalTransformer)
    obs_dim = 2 * d * d + d + 1
    state_dim = d * d * 2 + K
    num_actions = d + 1
    
    if args.agent_type == "mlp":
        agent = MLPAgent(num_actions=num_actions, hidden_dim=args.hidden_dim)
    elif args.agent_type == "rnn":
        agent = RNNAgent(num_actions=num_actions, hidden_dim=args.hidden_dim)
    elif args.agent_type == "transformer":
        agent = CausalTransformerAgent(num_actions=num_actions, hidden_dim=args.hidden_dim, num_heads=2)
    else:
        raise ValueError(f"Unknown agent_type: {args.agent_type}")
        
    mixer = QMIXMixer(hidden_dim=args.mixer_hidden_dim)
    trainer = QMIXTrainer(agent, mixer, lr=args.lr, gamma=args.gamma, agent_type=args.agent_type)
    
    k_init, key = jax.random.split(key)
    train_state, target_state = trainer.init_state(k_init, obs_dim, state_dim, K, max_steps=args.max_steps)
    
    buffer = TrajectoryBuffer(
        capacity=args.buffer_capacity,
        max_steps=args.max_steps,
        state_dim=state_dim,
        obs_dim=obs_dim,
        num_agents=K,
        num_actions=num_actions
    )
    
    best_eval_f1 = -1.0
    
    # 4. Main Training Loop
    for episode in range(1, args.num_episodes + 1):
        epsilon = trainer.get_epsilon(
            episode, args.num_episodes, 
            start=args.epsilon_start, 
            min_eps=args.epsilon_min, 
            decay_frac=args.epsilon_decay_frac
        )
        k_ep, key = jax.random.split(key)
        
        obs_dict, info = env.reset(k_ep)
        done = False
        step_count = 0
        ep_reward = 0.0
        
        if args.agent_type == "rnn":
            rnn_carry = agent.initialize_carry((K,))
        elif args.agent_type == "transformer":
            obs_history = []
            
        ep_states, ep_obs, ep_acts, ep_rews, ep_dones, ep_avail = [], [], [], [], [], []
        
        while not done and step_count < args.max_steps:
            state = info["state"]
            obs = np.array([obs_dict[f"agent_{k}"] for k in range(K)])
            avail = np.array([info["avail_actions"][f"agent_{k}"] for k in range(K)])
            
            # Epsilon-greedy action selection
            actions = []
            if np.random.rand() < epsilon:
                for k in range(K):
                    valid_actions = np.where(avail[k] == 1.0)[0]
                    a = np.random.choice(valid_actions)
                    actions.append(a)
            else:
                if args.agent_type == "mlp":
                    q_values_jnp = trainer.agent.apply(train_state.params['agent'], obs)
                elif args.agent_type == "rnn":
                    rnn_carry, q_values_jnp = trainer.agent.apply(train_state.params['agent'], rnn_carry, obs)
                elif args.agent_type == "transformer":
                    obs_history.append(obs)
                    seq_jnp = jnp.array(obs_history).transpose(1, 0, 2)
                    q_seq = trainer.agent.apply(train_state.params['agent'], seq_jnp)
                    q_values_jnp = q_seq[:, -1, :]
                    
                q_values_np = np.array(q_values_jnp)
                for k in range(K):
                    q_k = q_values_np[k].copy()
                    q_k[avail[k] == 0] = -1e9
                    a = int(np.argmax(q_k))
                    actions.append(a)
                    
            joint_actions = {f"agent_{k}": actions[k] for k in range(K)}
            
            k_step, key = jax.random.split(key)
            next_obs_dict, reward, done, info = env.step(joint_actions, k_step)
            
            ep_states.append(state)
            ep_obs.append(obs)
            ep_acts.append(actions)
            ep_rews.append([reward])
            ep_dones.append([done])
            ep_avail.append(avail)
            
            obs_dict = next_obs_dict
            ep_reward += reward
            step_count += 1
            
        buffer.add_episode({
            'states': ep_states,
            'observations': ep_obs,
            'actions': ep_acts,
            'rewards': ep_rews,
            'dones': ep_dones,
            'avail_actions': ep_avail
        })
        
        # Train Step
        td_loss = 0.0
        if buffer.size >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            train_state, loss_val = trainer.train_step(train_state, target_state, batch)
            td_loss = float(loss_val)
            
        # Target Network Update
        if episode % args.target_update_freq == 0:
            target_state = target_state.replace(params=train_state.params)
            
        # Logging & Evaluation
        log_data = {
            "train/episode": episode,
            "train/episode_reward": ep_reward,
            "train/td_loss": td_loss,
            "train/epsilon": epsilon,
            "train/remaining_circles": env.pag_tracker.count_circle_marks()
        }
        
        if episode % args.eval_freq == 0:
            k_eval, key = jax.random.split(key)
            eval_metrics = evaluate_agent(env, trainer, train_state, k_eval, agent_type=args.agent_type)
            log_data.update(eval_metrics)
            
            print(f"[{args.agent_type.upper()} | Episode {episode}/{args.num_episodes}] "
                  f"Reward: {ep_reward:.2f} | Loss: {td_loss:.4f} | "
                  f"Eval F1: {eval_metrics['eval/mean_f1']:.3f} | SHD: {eval_metrics['eval/mean_shd']:.2f}")
                  
            if eval_metrics['eval/mean_f1'] > best_eval_f1:
                best_eval_f1 = eval_metrics['eval/mean_f1']
                os.makedirs("checkpoints", exist_ok=True)
                np.savez(f"checkpoints/best_{args.agent_type}_params.npz", params=train_state.params)
                
        if args.use_wandb:
            wandb.log(log_data)
            
    print(f"=== Training Complete! Best {args.agent_type.upper()} Eval F1: {best_eval_f1:.3f} ===")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
