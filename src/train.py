import argparse
import os
import time
import random
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from src.types import SCMConfig, MechanismType, NoiseType, STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK, STANDARD_OBS_MASKS, ActionCategory
from src.evaluator_env import FederatedCausalEnv
from src.marl.ppo_agent import IPPOActor, IPPOCritic, IPPORNNActor, IPPORNNCritic, InductiveIPPOActor, InductiveIPPORNNActor, mask_invalid_targets, sample_actions_jitted
from src.marl.ppo_trainer import IPPOTrainer, RolloutBuffer, compute_gae
from src.baselines import RandomAgent, RoundRobinAgent, VanillaAgent
from src.metrics import evaluate_dag_against_true
from src.episode_metrics import gaussian_entropy, shd_trajectory_auc, shd_reduction_auc, normalized_target_entropy


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def parse_topology_list(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return tuple(int(x) for x in val)
    cleaned = str(val).replace("[", "").replace("]", "").replace(",", " ").strip()
    if not cleaned:
        return None
    return tuple(int(x) for x in cleaned.split())

def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Active Causal Discovery IPPO Trainer: Multi-Agent RL for Decentralized SCM Reconstruction"
    )
    
    # ---------------------------------------------------------
    # Agent & Architecture Configuration
    # ---------------------------------------------------------
    # Specifies the agent algorithm: Disjoint IPPO (RL) or heuristic baselines (Random / Round-Robin / Vanilla)
    parser.add_argument(
        "--agent_type", type=str, default="ippo", choices=["ippo", "random", "round_robin", "vanilla"],
        help="Algorithm type: 'ippo' (Independent PPO), 'random' (random uniform actions), 'round_robin' (cyclic), or 'vanilla' (flat 4-action discrete baseline)"
    )
    
    # ---------------------------------------------------------
    # Environment & SCM Parameters
    # ---------------------------------------------------------
    # Total number of causal variables d across all agent subdomains (e.g. 4 for [Z1, X1, X2, Z2])
    parser.add_argument(
        "--num_variables", "-d", type=int, default=4,
        help="Total number of variables in the ground-truth SCM graph (default: 4)"
    )
    # Number of sovereign federated agents K partitioning the causal graph
    parser.add_argument(
        "--num_agents", "-K", type=int, default=2,
        help="Number of independent federated agents (default: 2)"
    )
    # Maximum interaction steps (rollout horizon) allowed per episode
    parser.add_argument(
        "--max_steps", type=int, default=20,
        help="Maximum rollout horizon / steps per training episode (default: 20)"
    )
    # Starting intervention / sampling budget allocated to each agent per episode
    parser.add_argument(
        "--initial_budget", type=float, default=20.0,
        help="Initial budget allocated to each agent at the start of each episode (default: 20.0)"
    )
    # Budget cost deducted whenever an agent performs a local intervention or peer interventional request
    parser.add_argument(
        "--action_cost", type=float, default=1.0,
        help="Budget cost deducted per active intervention or peer request (default: 1.0)"
    )
    # Number of observational / interventional data samples drawn from the SCM per environment step
    parser.add_argument(
        "--sample_count", type=int, default=100,
        help="Number of SCM data points sampled per environment step for covariance estimation (default: 100)"
    )
    # Standard deviation sigma of additive Gaussian noise added to SCM structural equations
    parser.add_argument(
        "--noise_scale", type=float, default=0.1,
        help="Standard deviation of exogenous Gaussian noise in SCM equations (default: 0.1)"
    )
    # Functional form of causal mechanisms: Linear Gaussian or Nonlinear Additive Noise Model
    parser.add_argument(
        "--mechanism_type", type=str, default="LINEAR", choices=["LINEAR", "NONLINEAR_ANM"],
        help="SCM mechanism function type: 'LINEAR' or 'NONLINEAR_ANM' (default: LINEAR)"
    )
    
    # ---------------------------------------------------------
    # PPO Reinforcement Learning Hyperparameters
    # ---------------------------------------------------------
    # Total number of training episodes for the multi-agent policy loop
    parser.add_argument(
        "--num_episodes", type=int, default=5000,
        help="Total number of training episodes to run (default: 5000)"
    )
    # PPO mini-batch size / rollout buffer capacity for policy gradient updates
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Mini-batch size for PPO policy and value network updates (default: 16)"
    )
    # Global default learning rate fallback if specific actor/critic LRs are not set
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4,
        help="Default global Adam learning rate (default: 3e-4)"
    )
    # Learning rate specifically for the actor (policy) network optimization
    parser.add_argument(
        "--actor_lr", type=float, default=3e-4,
        help="Adam learning rate for agent Actor policy heads (default: 3e-4)"
    )
    # Learning rate specifically for the critic (value function baseline) network optimization
    parser.add_argument(
        "--critic_lr", type=float, default=1e-3,
        help="Adam learning rate for Critic value heads (default: 1e-3)"
    )
    # Entropy bonus coefficient (c_ent) added to loss to encourage action exploration and prevent premature policy collapse
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01,
        help="Entropy regularization bonus weight to encourage exploration (default: 0.01)"
    )
    # Evaluation frequency: interval of episodes between computing full metrics, logging, and DAG visualizations
    parser.add_argument(
        "--eval_freq", type=int, default=10,
        help="Frequency (in episodes) to evaluate stitched DAG, log metrics, and generate graph plots (default: 10)"
    )
    
    # ---------------------------------------------------------
    # Network Architecture & Graph Evaluation Mode
    # ---------------------------------------------------------
    # Equips actor and critic networks with GRU recurrent memory for tracking trajectory history.
    # Default True: each episode is a sequence of up to max_steps interventions where every step's
    # observation is only the *current* covariance/mask state with no explicit memory of earlier
    # steps in the episode, so a feedforward MLP re-decides from scratch each step. An RNN carries
    # that within-episode history forward instead.
    parser.add_argument(
        "--use_rnn", action="store_true", default=True,
        help="Enable recurrent GRU layers in Actor and Critic networks to handle partially observable histories (default: True)"
    )
    parser.add_argument(
        "--no_rnn", action="store_false", dest="use_rnn",
        help="Disable recurrent GRU layers and use plain feedforward Actor/Critic networks instead"
    )
    # Fixes the ground truth DAG topology: pass int 0-7 for specific topology, flag without int for fixed random, or omit for dynamic multi-topology sampling
    parser.add_argument(
        "--fixed_graph", type=int, nargs='?', const=-1, default=None,
        help="Fix training to a specific topology index (0-7), or pass flag alone to fix a random topology; omit to train dynamically across all topologies"
    )
    # Directory path where model checkpoints are saved
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory where best model checkpoints (.pkl) are saved (default: 'checkpoints')"
    )
    # Output directory where metrics CSV and evaluation traces are saved
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory where training_metrics.csv and evaluation_trace.json are saved (default: '.')"
    )
    # Boundary margin: confidence difference threshold for resolving bidirectional edge predictions without cycles
    parser.add_argument(
        "--boundary_margin", type=float, default=0.10,
        help="Confidence margin delta for pairwise differential edge orientation in DAG stitching (default: 0.10)"
    )
    # Intrinsic curiosity reward scaling factor for interventional covariance shift (Solution 2)
    parser.add_argument(
        "--intrinsic_coef", type=float, default=0.05,
        help="Curiosity reward scaling factor for interventional covariance shift (default: 0.05)"
    )
    # When enabled, normalizes step rewards by max_steps to decouple cumulative episode return from trajectory horizon
    parser.add_argument(
        "--normalize_rewards", action="store_true", default=True,
        help="Normalize per-step rewards by max_steps to eliminate horizon-induced return variance (default: True)"
    )
    parser.add_argument(
        "--no_normalize_rewards", action="store_false", dest="normalize_rewards",
        help="Disable reward normalization and use raw unnormalized cumulative step penalties"
    )
    # NOTE: the Anti-Symmetric Tournament Inductive Graph Head was removed from the actor
    # networks in the ActionCategory INTERVENE/NOOP collapse refactor -- InductiveIPPOActor
    # is now architecturally equivalent to IPPOActor. This flag is currently a no-op, kept
    # only so existing scripts/notebooks and saved checkpoints (which store this flag's
    # value in their metadata) keep working.
    parser.add_argument(
        "--use_inductive_graph_head", action="store_true", default=True,
        help="[Currently a no-op -- graph head removed] Selects InductiveIPPOActor, which is architecturally identical to IPPOActor (default: True)"
    )
    parser.add_argument(
        "--no_inductive_graph_head", action="store_false", dest="use_inductive_graph_head",
        help="[Currently a no-op -- graph head removed] Selects the plain IPPOActor class"
    )
    # Sampling temperature for post-training evaluation across topologies
    parser.add_argument(
        "--eval_temperature", type=float, default=0.0,
        help="Sampling temperature for post-training evaluation trace (0.0 for deterministic, >0.0 for stochastic) (default: 0.0)"
    )
    # When enabled, saves trained model checkpoints (best_ippo_params.pkl), CSV metrics, and JSON evaluation traces to disk
    parser.add_argument(
        "--save_file", action="store_true", default=True,
        help="Save best model weights (.pkl), training history (.csv), and post-training evaluation trace (.json) to disk (default: True)"
    )
    parser.add_argument(
        "--no_save_file", action="store_false", dest="save_file",
        help="Disable saving model weights, metrics CSV, and evaluation trace to disk"
    )
    # Custom subset of topologies to sample during training (e.g. --allowed_topologies 0,1 or 0,2,6)
    parser.add_argument(
        "--allowed_topologies", type=parse_topology_list, default=None,
        help="Comma or space separated subset of topology indices (0-7) to train on (e.g., '0,1' or '0,2,6'). Default: all topologies."
    )
    # ---------------------------------------------------------
    # Curriculum Learning (Solution 3)
    # ---------------------------------------------------------
    # When enabled, trains agents progressively through 3 stages of MEC graph complexity
    parser.add_argument(
        "--curriculum", action="store_true", default=True,
        help="Enable 3-stage topology curriculum learning across training episodes (default: True)"
    )
    parser.add_argument(
        "--no_curriculum", action="store_false", dest="curriculum",
        help="Disable 3-stage topology curriculum learning"
    )
    parser.add_argument(
        "--curriculum_stage1_ratio", type=float, default=0.20,
        help="Fraction of total training episodes dedicated to Stage 1 (Graph 0 only) (default: 0.20)"
    )
    parser.add_argument(
        "--curriculum_stage2_ratio", type=float, default=0.30,
        help="Fraction of total training episodes dedicated to Stage 2 (Graphs 0 & 1 MEC pair) (default: 0.30)"
    )

    
    # ---------------------------------------------------------
    # Ablation Study Suite Flags
    # ---------------------------------------------------------
    parser.add_argument(
        "--estimator_type", type=str, default="avici",
        choices=["analytic", "avici", "learned", "bayes_optimal"],
        help="Stage 2 graph prediction engine: 'avici' (pretrained AVICI scm-v0 checkpoint, frozen -- "
             "default; reaches ~99%% SHD=0 from early training with no memorization risk since it "
             "cannot adapt, see docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md), 'analytic' (frozen "
             "Analytic Invariance Scorer), 'learned' (small trainable edge-scorer updated online "
             "via supervised BCE against the true adjacency each step -- see src/marl/graph_estimator.py), "
             "or 'bayes_optimal' (exact posterior over the 8 known candidate topologies -- a comparison "
             "ceiling, not a realistic deployable estimator; see src/marl/bayes_optimal_estimator.py)"
    )
    parser.add_argument(
        "--avici_max_context", type=int, default=400,
        help="Max number of most-recent raw-sample rows fed to AVICI per call (only relevant "
             "when --estimator_type avici). Bounds per-call compute cost and avoids per-step "
             "JIT recompilation from a constantly-growing input shape (default: 400, i.e. the "
             "most recent 4 steps' worth of samples at the default sample_count=100)"
    )
    parser.add_argument(
        "--intervention_type", type=str, default="hard", choices=["soft_shift", "hard"],
        help="Intervention mechanism: 'hard' (do(X_i = c) -- default; the classical 'perfect intervention' "
             "most identifiability theory assumes, and empirically a much stronger structure-learning signal, "
             "see docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md) or 'soft_shift' (do(X_i := f_i + e_i + delta_i), "
             "a weaker but more realistic 'imperfect intervention')"
    )
    parser.add_argument(
        "--soft_shift_val", type=float, default=2.0,
        help="Shift mean magnitude mu_delta for soft shift interventions (default: 2.0)"
    )
    parser.add_argument(
        "--freeze_graph_estimator", type=str, default="true", choices=["true", "false"],
        help="Whether to freeze Stage 2 graph estimator weights (default: 'true')"
    )
    parser.add_argument(
        "--obs_feedback", type=str, default="true", choices=["true", "false"],
        help="Whether to feed local predicted DAG slice back into agent observations (default: 'true')"
    )
    parser.add_argument(
        "--impact_coef", type=float, default=0.0,
        help="Scaling factor for downstream interventional impact bonus (default: 0.0)"
    )
    parser.add_argument(
        "--reward_density", type=str, default="sparse", choices=["dense", "sparse"],
        help="Reward density: 'sparse' (terminal episode-end SHD penalty -- default; the 18-run "
             "diagnostic matrix found no meaningful difference vs dense, see investigation doc) or "
             "'dense' (step-wise SHD reduction)"
    )
    
    # ---------------------------------------------------------
    # Experiment Tracking & Reproducibility
    # ---------------------------------------------------------
    # When enabled, logs live metric curves, evaluation traces, and DAG visual images to Weights & Biases
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable live metrics, artifact tracking, and graph visualization uploads to Weights & Biases"
    )
    # Weights & Biases project name under which this run will be registered
    parser.add_argument(
        "--wandb_project", type=str, default="federated-causal-ippo",
        help="Target Weights & Biases project name (default: 'federated-causal-ippo')"
    )
    # Custom display name for this run in WandB (if None, auto-generated based on hyperparameter config)
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Custom run name for Weights & Biases logging (auto-generated if None)"
    )
    # Master pseudo-random number generator (PRNG) seed for JAX and NumPy reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for JAX PRNG and environment initialization (default: 42)"
    )
    
    return parser.parse_args()

def get_curriculum_topologies(
    episode: int, 
    total_episodes: int, 
    stage1_ratio: float = 0.20, 
    stage2_ratio: float = 0.30
) -> tuple[tuple[int, ...], int]:
    """
    Returns (allowed_topology_indices, curriculum_stage) based on current episode:
    - Stage 1 (0 to stage1_episodes): (0,) (Base forward chain)
    - Stage 2 (stage1_episodes to stage1_episodes + stage2_episodes): (0, 1) (Chain MEC pair: Forward vs Reverse)
    - Stage 3 (stage2_episodes to end): (0, 1, 2, 3, 4, 5, 6, 7) (Full 8 topologies)
    """
    s1_end = max(1, int(total_episodes * stage1_ratio))
    s2_end = max(s1_end + 1, int(total_episodes * (stage1_ratio + stage2_ratio)))
    
    if episode <= s1_end:
        return (0,), 1
    elif episode <= s2_end:
        return (0, 1), 2
    else:
        return tuple(range(8)), 3

def main():
    args = parse_args()
    
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: --use_wandb was passed, but the wandb library is not installed! WandB metrics will not be logged.")
        else:
            try:
                is_fixed = args.fixed_graph is not None
                name = args.run_name or f"{args.agent_type}{'_rnn' if args.use_rnn else ''}{'_fixed' if is_fixed else ''}_d{args.num_variables}"
                wandb.init(project=args.wandb_project, name=name, config=vars(args))
                if wandb.run:
                    print(f"WandB successfully initialized! View live run at: {wandb.run.url}")
            except Exception as e:
                print(f"WARNING: WandB initialization failed ({e}). Continuing training locally.")
                args.use_wandb = False
            
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
    
    freeze_estimator = args.freeze_graph_estimator.lower() == "true"
    enable_obs_feedback = args.obs_feedback.lower() == "true"
    
    env = FederatedCausalEnv(
        config, action_costs, 
        initial_budget=args.initial_budget, 
        sample_count=args.sample_count, 
        fixed_graph=is_fixed,
        max_steps=args.max_steps,
        boundary_margin=args.boundary_margin,
        normalize_rewards=args.normalize_rewards,
        intrinsic_coef=args.intrinsic_coef,
        intervention_type=args.intervention_type,
        shift_val=args.soft_shift_val,
        estimator_type=args.estimator_type,
        freeze_graph_estimator=freeze_estimator,
        obs_feedback=enable_obs_feedback,
        impact_coef=args.impact_coef,
        reward_density=args.reward_density,
        avici_max_context=args.avici_max_context
    )
    
    if args.agent_type == "ippo":
        if args.use_inductive_graph_head:
            if args.use_rnn:
                def make_actor():
                    def forward(obs, state): return InductiveIPPORNNActor(d=args.num_variables)(obs, state)
                    return hk.without_apply_rng(hk.transform(forward))
            else:
                def make_actor():
                    def forward(obs): return InductiveIPPOActor(d=args.num_variables)(obs)
                    return hk.without_apply_rng(hk.transform(forward))
        else:
            if args.use_rnn:
                def make_actor():
                    def forward(obs, state): return IPPORNNActor(d=args.num_variables)(obs, state)
                    return hk.without_apply_rng(hk.transform(forward))
            else:
                def make_actor():
                    def forward(obs): return IPPOActor(d=args.num_variables)(obs)
                    return hk.without_apply_rng(hk.transform(forward))
                
        if args.use_rnn:
            def make_critic():
                def forward(obs, state): return IPPORNNCritic()(obs, state)
                return hk.without_apply_rng(hk.transform(forward))
        else:
            def make_critic():
                def forward(obs): return IPPOCritic()(obs)
                return hk.without_apply_rng(hk.transform(forward))
            
        actor_trans = make_actor()
        critic_trans = make_critic()
        
        actor_apply = jax.jit(actor_trans.apply)
        critic_apply = jax.jit(critic_trans.apply)

        
        local_masks = [STANDARD_LOCAL_MASKS[0], STANDARD_LOCAL_MASKS[1]]
        boundary_mask = STANDARD_BOUNDARY_MASK
        observed_masks = [STANDARD_OBS_MASKS[0], STANDARD_OBS_MASKS[1]]
        valid_intervention_masks = [jnp.maximum(local_masks[0], boundary_mask), jnp.maximum(local_masks[1], boundary_mask)]

        actor_lr = args.learning_rate if args.learning_rate != 3e-4 else args.actor_lr
        critic_lr = args.learning_rate if args.learning_rate != 3e-4 else args.critic_lr
        trainer = IPPOTrainer(actor_trans, critic_trans, actor_lr=actor_lr, critic_lr=critic_lr,
                              entropy_coef=args.entropy_coef, use_rnn=args.use_rnn,
                              total_episodes=args.num_episodes, normalize_rewards=args.normalize_rewards, max_steps=float(args.max_steps))
                              
        # Initialize Disjoint parameters and optimizers per agent
        actor_params_list = []
        critic_params_list = []
        actor_opts = []
        critic_opts = []
        
        obs_dim = env.obs_dim
        dummy_obs = jnp.zeros((1, obs_dim))
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
        if args.agent_type == "random":
            agents = [RandomAgent(i, args.num_variables) for i in range(args.num_agents)]
        elif args.agent_type == "vanilla":
            agents = [VanillaAgent(i, args.num_variables) for i in range(args.num_agents)]
        else:
            agents = [RoundRobinAgent(i, args.num_variables) for i in range(args.num_agents)]
    
    best_shd = 999.0
    best_f1 = -1.0
    
    all_metrics_history = []
    
    for episode in range(1, args.num_episodes + 1):
        k_ep, key = jax.random.split(key)
        
        if args.curriculum and fixed_idx is None:
            allowed_topos, curr_stage = get_curriculum_topologies(
                episode, args.num_episodes, args.curriculum_stage1_ratio, args.curriculum_stage2_ratio
            )
        elif args.allowed_topologies is not None and fixed_idx is None:
            allowed_topos = args.allowed_topologies
            curr_stage = 0
        else:
            allowed_topos = None
            curr_stage = 0
            
        obs_dict, info = env.reset(k_ep, force_idx=fixed_idx, allowed_topologies=allowed_topos)
        true_adj = info["true_adjacency"]

        if args.agent_type == "ippo" and args.use_rnn:
            actor_states = [IPPORNNActor.initial_state(1) for _ in range(args.num_agents)]
            critic_states = [IPPORNNCritic.initial_state(1) for _ in range(args.num_agents)]

        done = False
        ep_reward = 0.0
        ep_info_gain_0 = 0.0
        ep_info_gain_1 = 0.0
        ep_steps = 0
        final_dag = None

        # Agent-vs-estimator-learning metrics (see src/episode_metrics.py) -- accumulated
        # per step, reduced to episode-level scalars after the loop. All purely diagnostic;
        # none feed back into the reward.
        initial_shd = float(evaluate_dag_against_true(np.array(env.last_predicted_dag), np.array(true_adj))["shd"])
        shd_trajectory = [initial_shd]
        cumulative_interventions = 0
        interventions_to_zero = None
        sum_positive_delta = {0: 0.0, 1: 0.0}
        redundant_steps = 0
        node_intervention_counts = {i: 0 for i in range(args.num_variables)}
        ep_impact_sum = {0: 0.0, 1: 0.0}
        ep_asym_mag_sum = 0.0
        max_shd_possible = float(np.sum(env.structural_mask))
        entropy_before = gaussian_entropy(np.array(env.jax_state.running_covariance), args.num_variables)
        
        while not done:
            if args.agent_type == "ippo":
                # Agent 0
                k0_act, key = jax.random.split(key)
                obs_0 = jnp.expand_dims(obs_dict["agent_0"], 0)
                if args.use_rnn:
                    (cat_l0, tgt_l0), actor_states[0] = actor_apply(actor_params_list[0], obs_0, actor_states[0])
                    v0, critic_states[0] = critic_apply(critic_params_list[0], obs_0, critic_states[0])
                    val_0 = v0[0]
                else:
                    cat_l0, tgt_l0 = actor_apply(actor_params_list[0], obs_0)
                    val_0 = critic_apply(critic_params_list[0], obs_0)[0]
                c0, t0, lp0 = sample_actions_jitted(cat_l0[0], tgt_l0[0], valid_intervention_masks[0], k0_act)
                
                # Agent 1
                k1_act, key = jax.random.split(key)
                obs_1 = jnp.expand_dims(obs_dict["agent_1"], 0)
                if args.use_rnn:
                    (cat_l1, tgt_l1), actor_states[1] = actor_apply(actor_params_list[1], obs_1, actor_states[1])
                    v1, critic_states[1] = critic_apply(critic_params_list[1], obs_1, critic_states[1])
                    val_1 = v1[0]
                else:
                    cat_l1, tgt_l1 = actor_apply(actor_params_list[1], obs_1)
                    val_1 = critic_apply(critic_params_list[1], obs_1)[0]
                c1, t1, lp1 = sample_actions_jitted(cat_l1[0], tgt_l1[0], valid_intervention_masks[1], k1_act)
                
                joint_actions = {
                    "agent_0": (int(c0), int(t0)),
                    "agent_1": (int(c1), int(t1))
                }
                
                k_step, key = jax.random.split(key)
                next_obs_dict, rewards, done, step_info = env.step(joint_actions, predicted_dags=None, key=k_step)
                
                r0 = rewards["agent_0"]
                r1 = rewards["agent_1"]
                final_dag = step_info.get("predicted_dag", env.last_predicted_dag)
                info_gains = step_info.get("info_gains", {"agent_0": 0.0, "agent_1": 0.0})
                
                buffers[0].add(obs=obs_0[0], cat_actions=c0, target_actions=t0, values=val_0, log_probs=lp0, rewards=r0, dones=done)
                buffers[1].add(obs=obs_1[0], cat_actions=c1, target_actions=t1, values=val_1, log_probs=lp1, rewards=r1, dones=done)
                obs_dict = next_obs_dict
                ep_reward += float(r0 + r1)
                ep_info_gain_0 += float(info_gains["agent_0"])
                ep_info_gain_1 += float(info_gains["agent_1"])
                ep_steps += 1

                # Agent-vs-estimator-learning metric accumulation (see src/episode_metrics.py)
                did_intervene = {0: int(c0) == int(ActionCategory.INTERVENE), 1: int(c1) == int(ActionCategory.INTERVENE)}
                cumulative_interventions += int(did_intervene[0]) + int(did_intervene[1])
                step_shd = float(step_info["shd"])
                shd_trajectory.append(step_shd)
                if interventions_to_zero is None and step_shd == 0.0:
                    interventions_to_zero = cumulative_interventions
                shd_delta = step_info.get("shd_delta", {"agent_0": 0.0, "agent_1": 0.0})
                if did_intervene[0]:
                    sum_positive_delta[0] += max(0.0, float(shd_delta["agent_0"]))
                if did_intervene[1]:
                    sum_positive_delta[1] += max(0.0, float(shd_delta["agent_1"]))
                if did_intervene[0] and did_intervene[1] and int(t0) == int(t1):
                    redundant_steps += 1
                if did_intervene[0]:
                    node_intervention_counts[int(t0)] = node_intervention_counts.get(int(t0), 0) + 1
                if did_intervene[1]:
                    node_intervention_counts[int(t1)] = node_intervention_counts.get(int(t1), 0) + 1
                impact_scores = step_info.get("impact_scores", {"agent_0": 0.0, "agent_1": 0.0})
                ep_impact_sum[0] += float(impact_scores["agent_0"])
                ep_impact_sum[1] += float(impact_scores["agent_1"])
                if "asym_matrix" in step_info:
                    ep_asym_mag_sum += float(np.mean(np.abs(step_info["asym_matrix"])))
            else:
                joint_actions = {}
                predicted_dags = {}
                for k in range(args.num_agents):
                    act, g_pred = agents[k].act(obs_dict[f"agent_{k}"])
                    joint_actions[f"agent_{k}"] = act
                    predicted_dags[f"agent_{k}"] = g_pred
                    
                k_step, key = jax.random.split(key)
                next_obs_dict, rewards, done, step_info = env.step(joint_actions, predicted_dags, k_step)
                obs_dict = next_obs_dict
                ep_reward += sum(rewards.values())
                if "info_gains" in step_info:
                    ep_info_gain_0 += float(step_info["info_gains"]["agent_0"])
                    ep_info_gain_1 += float(step_info["info_gains"]["agent_1"])
                ep_steps += 1
                
                from src.stitching import stitch_predicted_dags
                final_dag, _ = stitch_predicted_dags(predicted_dags, args.num_variables)
            
        # End of episode update
        if args.agent_type == "ippo":
            actor_loss = 0.0
            critic_loss = 0.0
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
                    actor_params_list[k], critic_params_list[k], actor_opts[k], critic_opts[k], b, valid_intervention_masks[k]
                )
                actor_params_list[k] = a_p
                critic_params_list[k] = c_p
                actor_opts[k] = a_opt
                critic_opts[k] = c_opt
                
                per_agent_metrics[k] = metrics
                actor_loss += float(metrics["actor_loss"])
                critic_loss += float(metrics["critic_loss"])
                entropy += float(metrics["entropy"])
                buffers[k].reset()
                
        # Evaluate Stitched DAG (final step)
        eval_metrics = evaluate_dag_against_true(np.array(final_dag), true_adj)
        
        log_data = {
            "train/episode": int(episode),
            "train/curriculum_stage": int(curr_stage),
            "train/episode_reward": float(ep_reward),
            "train/info_gain_a0": float(ep_info_gain_0 / max(1, ep_steps)),
            "train/info_gain_a1": float(ep_info_gain_1 / max(1, ep_steps)),
            "eval/shd": float(eval_metrics["shd"]),
            "eval/f1": float(eval_metrics["f1"]),
        }
        for k in range(args.num_agents):
            log_data[f"agent_{k}_budget"] = float(env.jax_state.budgets[k])

        if args.agent_type == "ippo":
            entropy_after = gaussian_entropy(np.array(env.jax_state.running_covariance), args.num_variables)
            log_data.update({
                "eval/interventions_to_shd0": float(interventions_to_zero) if interventions_to_zero is not None else float("nan"),
                "eval/reached_shd0": int(interventions_to_zero is not None),
                "eval/shd_auc_normalized": shd_trajectory_auc(shd_trajectory, max_shd_possible),
                "eval/shd_reduction_auc_normalized": shd_reduction_auc(shd_trajectory, max_shd_possible),
                "eval/orientation_precision_a0": sum_positive_delta[0] / max(1, cumulative_interventions),
                "eval/orientation_precision_a1": sum_positive_delta[1] / max(1, cumulative_interventions),
                "eval/entropy_gain_episode": entropy_before - entropy_after,
                "eval/impact_score_a0": ep_impact_sum[0] / max(1, ep_steps),
                "eval/impact_score_a1": ep_impact_sum[1] / max(1, ep_steps),
                "eval/asym_magnitude_mean": ep_asym_mag_sum / max(1, ep_steps),
                "eval/redundant_interventions": redundant_steps,
                "eval/redundancy_rate": redundant_steps / max(1, cumulative_interventions),
                "eval/node_coverage": len([v for v in node_intervention_counts.values() if v > 0]) / args.num_variables,
                "eval/target_entropy_normalized": normalized_target_entropy(node_intervention_counts, args.num_variables),
            })
            # Boundary-specific coverage -- the more precise "is federation actually
            # working" signal than overall coverage, since private nodes never require
            # coordination at all (derived from STANDARD_BOUNDARY_MASK, not hardcoded).
            boundary_nodes = [i for i in range(args.num_variables) if float(STANDARD_BOUNDARY_MASK[i]) > 0.5]
            boundary_covered = len([n for n in boundary_nodes if node_intervention_counts.get(n, 0) > 0])
            log_data["eval/boundary_node_coverage"] = boundary_covered / max(1, len(boundary_nodes))


        if args.agent_type == "ippo":
            log_data["train/actor_loss"] = float(actor_loss / args.num_agents)
            log_data["train/critic_loss"] = float(critic_loss / args.num_agents)
            log_data["train/entropy"] = float(entropy / args.num_agents)
            for k in range(args.num_agents):
                log_data[f"train/agent_{k}_actor_loss"] = float(per_agent_metrics[k]["actor_loss"])
                log_data[f"train/agent_{k}_critic_loss"] = float(per_agent_metrics[k]["critic_loss"])
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
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                ckpt_file = os.path.join(args.checkpoint_dir, "best_ippo_params.pkl")
                with open(ckpt_file, "wb") as f:
                    pickle.dump({
                        "actor_list": actor_params_list,
                        "critic_list": critic_params_list,
                        "use_rnn": args.use_rnn,
                        "use_inductive_graph_head": args.use_inductive_graph_head,
                        "d": args.num_variables
                    }, f)

    if args.save_file:
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_file = os.path.join(args.output_dir, "training_metrics.csv")
        try:
            import pandas as pd
            df = pd.DataFrame(all_metrics_history)
            df.to_csv(metrics_file, index=False)
        except ImportError:
            import csv
            if all_metrics_history:
                keys = list(all_metrics_history[0].keys())
                with open(metrics_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(all_metrics_history)
        print(f"Saved metrics to {metrics_file}")
        
    if args.agent_type == "ippo":
        print("Running post-training evaluation suite on the BEST model...")
        from src.evaluate import evaluate_checkpoint
        import json
        
        ckpt_file = os.path.join(args.checkpoint_dir, "best_ippo_params.pkl")
        os.makedirs(args.output_dir, exist_ok=True)
        trace_file = os.path.join(args.output_dir, "evaluation_trace.json")
        
        try:
            trace = evaluate_checkpoint(
                ckpt_path=ckpt_file,
                config=config,
                action_costs=action_costs,
                initial_budget=args.initial_budget,
                temperature=args.eval_temperature,
                boundary_margin=args.boundary_margin,
                seed=args.seed
            )
            with open(trace_file, "w") as f:
                json.dump(trace, f, indent=2)
            print(f"Saved evaluation trace to {trace_file}")
            
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.save(trace_file)
                print("Uploaded evaluation trace to WandB.")
        except Exception as e:
            print(f"Post-training evaluation encountered an issue: {e}")
            
    if args.use_wandb and WANDB_AVAILABLE and wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()
