import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import Tuple, Dict, Any
from functools import partial

from src.marl.agent import mask_q_values
from src.marl.mixer import QMIXMixer

class QMIXTrainer:
    def __init__(self, agent: Any, mixer: QMIXMixer, lr: float = 1e-3, gamma: float = 0.99, agent_type: str = "mlp"):
        self.agent = agent
        self.mixer = mixer
        self.gamma = gamma
        self.lr = lr
        self.agent_type = agent_type.lower()
        
    def init_state(self, key: jax.Array, obs_dim: int, state_dim: int, K: int, max_steps: int = 20) -> Tuple[TrainState, TrainState]:
        k1, k2 = jax.random.split(key)
        
        if self.agent_type == "mlp":
            dummy_obs = jnp.zeros((obs_dim,))
            agent_params = self.agent.init(k1, dummy_obs)
        elif self.agent_type == "rnn":
            dummy_carry = jnp.zeros((self.agent.hidden_dim,))
            dummy_obs = jnp.zeros((obs_dim,))
            agent_params = self.agent.init(k1, dummy_carry, dummy_obs)
        elif self.agent_type == "transformer":
            dummy_seq = jnp.zeros((1, max_steps, obs_dim))
            agent_params = self.agent.init(k1, dummy_seq)
        else:
            raise ValueError(f"Unknown agent_type: {self.agent_type}")
            
        dummy_q_chosen = jnp.zeros((K,))
        dummy_state = jnp.zeros((state_dim,))
        mixer_params = self.mixer.init(k2, dummy_q_chosen, dummy_state)
        
        params = {'agent': agent_params, 'mixer': mixer_params}
        
        tx = optax.adam(self.lr)
        state = TrainState.create(apply_fn=None, params=params, tx=tx)
        target_state = TrainState.create(apply_fn=None, params=params, tx=optax.sgd(0.0))
        return state, target_state
        
    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state: TrainState, target_state: TrainState, batch: Dict[str, jax.Array]) -> Tuple[TrainState, float]:
        """
        batch arrays:
        states: [B, T, state_dim]
        observations: [B, T, K, obs_dim]
        actions: [B, T, K]
        rewards: [B, T, 1]
        dones: [B, T, 1]
        avail_actions: [B, T, K, num_actions]
        """
        def loss_fn(params):
            B, T, K, obs_dim = batch['observations'].shape
            
            if self.agent_type == "mlp":
                obs_flat = batch['observations'].reshape(-1, obs_dim)
                q_values_flat = self.agent.apply(params['agent'], obs_flat)
                q_values = q_values_flat.reshape(B, T, K, -1)
                
                target_q_flat = self.agent.apply(target_state.params['agent'], obs_flat)
                target_q_values = target_q_flat.reshape(B, T, K, -1)
                
            elif self.agent_type == "rnn":
                obs_t = batch['observations'].transpose(1, 0, 2, 3).reshape(T, B * K, obs_dim)
                init_carry = jnp.zeros((B * K, self.agent.hidden_dim))
                
                def scan_online(carry, obs):
                    new_carry, q_vals = self.agent.apply(params['agent'], carry, obs)
                    return new_carry, q_vals
                    
                def scan_target(carry, obs):
                    new_carry, q_vals = self.agent.apply(target_state.params['agent'], carry, obs)
                    return new_carry, q_vals
                    
                _, q_seq = jax.lax.scan(scan_online, init_carry, obs_t)
                q_values = q_seq.reshape(T, B, K, -1).transpose(1, 0, 2, 3)
                
                _, target_q_seq = jax.lax.scan(scan_target, init_carry, obs_t)
                target_q_values = target_q_seq.reshape(T, B, K, -1).transpose(1, 0, 2, 3)
                
            elif self.agent_type == "transformer":
                obs_seq = batch['observations'].transpose(0, 2, 1, 3).reshape(B * K, T, obs_dim)
                q_seq = self.agent.apply(params['agent'], obs_seq)
                q_values = q_seq.reshape(B, K, T, -1).transpose(0, 2, 1, 3)
                
                target_q_seq = self.agent.apply(target_state.params['agent'], obs_seq)
                target_q_values = target_q_seq.reshape(B, K, T, -1).transpose(0, 2, 1, 3)

            # Select chosen Q-values
            chosen_q = jnp.take_along_axis(q_values, jnp.expand_dims(batch['actions'], axis=-1), axis=-1).squeeze(-1)
            
            # Compute online Q_tot
            states_flat = batch['states'].reshape(B * T, -1)
            chosen_q_flat = chosen_q.reshape(B * T, K)
            q_tot_flat = self.mixer.apply(params['mixer'], chosen_q_flat, states_flat)
            q_tot = q_tot_flat.reshape(B, T, 1)
            
            # Double Q-learning target values
            online_q_values_masked = mask_q_values(q_values, batch['avail_actions'])
            best_actions = jnp.argmax(online_q_values_masked, axis=-1, keepdims=True)
            
            target_chosen_q = jnp.take_along_axis(target_q_values, best_actions, axis=-1).squeeze(-1)
            target_chosen_q_flat = target_chosen_q.reshape(B * T, K)
            
            target_q_tot_flat = self.mixer.apply(target_state.params['mixer'], target_chosen_q_flat, states_flat)
            target_q_tot = target_q_tot_flat.reshape(B, T, 1)
            
            # Compute TD Error over transitions
            targets = batch['rewards'][:, :-1] + self.gamma * (1.0 - batch['dones'][:, :-1].astype(jnp.float32)) * target_q_tot[:, 1:]
            predictions = q_tot[:, :-1]
            
            td_error = predictions - jax.lax.stop_gradient(targets)
            loss = jnp.mean(td_error ** 2)
            
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def get_epsilon(self, episode: int, total_episodes: int, start: float = 1.0, min_eps: float = 0.05, decay_frac: float = 0.8) -> float:
        """Computes linear epsilon decay over a specified fraction of total episodes."""
        decay_episodes = max(1, int(total_episodes * decay_frac))
        progress = min(1.0, episode / decay_episodes)
        epsilon = start - (start - min_eps) * progress
        return max(min_eps, epsilon)
