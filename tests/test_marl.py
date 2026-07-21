import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.marl.agent import MLPAgent
from src.marl.mixer import QMIXMixer
from src.marl.trainer import QMIXTrainer
from src.types import SCMConfig, MechanismType, NoiseType
from src.generators import generate_er_dag, generate_scm_params
from src.evaluator_env import FederatedCausalEnv
from src.marl.buffer import TrajectoryBuffer

def test_qmix_monotonicity():
    """Test 1: Assert QMIXMixer satisfies non-negativity constraint."""
    K = 3
    state_dim = 10
    mixer = QMIXMixer(hidden_dim=32)
    key = jax.random.PRNGKey(0)
    
    # Init mixer params
    dummy_q = jnp.zeros(K)
    dummy_state = jnp.zeros(state_dim)
    params = mixer.init(key, dummy_q, dummy_state)
    
    # We want to check d(Q_tot) / d(Q_k) >= 0 for all K.
    q_vals = jax.random.normal(key, (K,))
    state_vals = jax.random.normal(jax.random.fold_in(key, 1), (state_dim,))
    
    def q_tot_fn(q):
        return mixer.apply(params, q, state_vals)[0]
        
    grad_fn = jax.grad(q_tot_fn)
    grads = grad_fn(q_vals)
    
    # Assert all gradients are strictly non-negative
    assert jnp.all(grads >= -1e-6), f"Gradients are not positive: {grads}"

def test_qmix_trainer_loop():
    """Test 2: Perform 5 training iterations of QMIXTrainer and verify parameter updates complete without shape errors."""
    d = 4
    K = 2
    
    config = SCMConfig(
        d=d,
        K=K,
        mechanism_type=MechanismType.LINEAR,
        noise_type=NoiseType.GAUSSIAN,
        noise_scale=0.1
    )
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    adj = generate_er_dag(k1, d, edge_prob=0.8)
    scm_params = generate_scm_params(k2, adj, MechanismType.LINEAR)
    topological_order = jnp.arange(d)
    
    agent_masks = jnp.array([
        [1., 1., 1., 0.],
        [0., 1., 1., 1.]
    ])
    action_costs = jnp.array([1.0, 1.0])
    
    env = FederatedCausalEnv(config, adj, scm_params, topological_order, agent_masks, action_costs)
    
    obs_dim = 2 * d * d + d + 1
    state_dim = d * d * 2 + K
    num_actions = d + 1
    max_steps = 10
    env.max_steps = max_steps
    
    agent = MLPAgent(num_actions=num_actions, hidden_dim=32)
    mixer = QMIXMixer(hidden_dim=32)
    trainer = QMIXTrainer(agent, mixer, lr=0.01)
    
    train_state, target_state = trainer.init_state(key, obs_dim, state_dim, K)
    
    buffer = TrajectoryBuffer(capacity=10, max_steps=max_steps, state_dim=state_dim, 
                              obs_dim=obs_dim, num_agents=K, num_actions=num_actions)
                              
    # Gather 2 episodes
    for ep in range(2):
        obs_dict, info = env.reset(jax.random.fold_in(key, ep + 100))
        done = False
        
        ep_states = []
        ep_obs = []
        ep_acts = []
        ep_rews = []
        ep_dones = []
        ep_avail = []
        
        step_count = 0
        while not done and step_count < max_steps:
            state = info["state"]
            obs = np.array([obs_dict[f"agent_{k}"] for k in range(K)])
            avail = np.array([info["avail_actions"][f"agent_{k}"] for k in range(K)])
            
            # Random valid actions
            actions = []
            for k in range(K):
                valid = np.where(avail[k] == 1.0)[0]
                a = np.random.choice(valid)
                actions.append(a)
                
            joint_actions = {f"agent_{k}": actions[k] for k in range(K)}
            
            next_obs_dict, reward, done, info = env.step(joint_actions, key)
            
            ep_states.append(state)
            ep_obs.append(obs)
            ep_acts.append(actions)
            ep_rews.append([reward])
            ep_dones.append([done])
            ep_avail.append(avail)
            
            obs_dict = next_obs_dict
            step_count += 1
            
        buffer.add_episode({
            'states': ep_states,
            'observations': ep_obs,
            'actions': ep_acts,
            'rewards': ep_rews,
            'dones': ep_dones,
            'avail_actions': ep_avail
        })
        
    batch = buffer.sample(batch_size=2)
    
    # Run 5 training steps
    losses = []
    for i in range(5):
        train_state, loss = trainer.train_step(train_state, target_state, batch)
        losses.append(float(loss))
            
    # Verify no shape errors occurred and loss is calculated
    assert len(losses) == 5
    assert losses[0] >= 0.0
