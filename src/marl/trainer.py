import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import Tuple, Dict
from functools import partial

from src.marl.agent import MLPAgent, mask_q_values
from src.marl.mixer import QMIXMixer

class QMIXTrainer:
    def __init__(self, agent: MLPAgent, mixer: QMIXMixer, lr: float = 1e-3, gamma: float = 0.99):
        self.agent = agent
        self.mixer = mixer
        self.gamma = gamma
        self.lr = lr
        
    def init_state(self, key: jax.Array, obs_dim: int, state_dim: int, K: int) -> Tuple[TrainState, TrainState]:
        k1, k2 = jax.random.split(key)
        
        # We use shared parameters for all agents.
        dummy_obs = jnp.zeros((obs_dim,))
        agent_params = self.agent.init(k1, dummy_obs)
        
        dummy_q_chosen = jnp.zeros((K,))
        dummy_state = jnp.zeros((state_dim,))
        mixer_params = self.mixer.init(k2, dummy_q_chosen, dummy_state)
        
        params = {'agent': agent_params, 'mixer': mixer_params}
        
        tx = optax.adam(self.lr)
        state = TrainState.create(
            apply_fn=None,
            params=params,
            tx=tx,
        )
        target_state = TrainState.create(
            apply_fn=None,
            params=params,
            tx=optax.sgd(0.0), # No optimizer needed for target network
        )
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
            
            # Forward pass online network
            obs_flat = batch['observations'].reshape(-1, obs_dim)
            q_values_flat = self.agent.apply(params['agent'], obs_flat)
            q_values = q_values_flat.reshape(B, T, K, -1)
            
            # Select chosen Q-values
            chosen_q = jnp.take_along_axis(q_values, jnp.expand_dims(batch['actions'], axis=-1), axis=-1).squeeze(-1) # [B, T, K]
            
            # Compute online Q_tot
            states_flat = batch['states'].reshape(B * T, -1)
            chosen_q_flat = chosen_q.reshape(B * T, K)
            q_tot_flat = self.mixer.apply(params['mixer'], chosen_q_flat, states_flat)
            q_tot = q_tot_flat.reshape(B, T, 1)
            
            # Forward pass target network
            target_q_values_flat = self.agent.apply(target_state.params['agent'], obs_flat)
            target_q_values = target_q_values_flat.reshape(B, T, K, -1)
            
            # Double Q-learning target values
            online_q_values_masked = mask_q_values(q_values, batch['avail_actions'])
            best_actions = jnp.argmax(online_q_values_masked, axis=-1, keepdims=True)
            
            target_chosen_q = jnp.take_along_axis(target_q_values, best_actions, axis=-1).squeeze(-1)
            target_chosen_q_flat = target_chosen_q.reshape(B * T, K)
            
            target_q_tot_flat = self.mixer.apply(target_state.params['mixer'], target_chosen_q_flat, states_flat)
            target_q_tot = target_q_tot_flat.reshape(B, T, 1)
            
            # Compute TD Error over transitions
            # y_t = R_t + gamma * (1 - d_t) * Q_tot(t+1)
            targets = batch['rewards'][:, :-1] + self.gamma * (1.0 - batch['dones'][:, :-1].astype(jnp.float32)) * target_q_tot[:, 1:]
            predictions = q_tot[:, :-1]
            
            td_error = predictions - jax.lax.stop_gradient(targets)
            loss = jnp.mean(td_error ** 2)
            
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    def get_epsilon(self, episode: int, total_episodes: int) -> float:
        # Linear decay from 1.0 to 0.05
        epsilon = 1.0 - (1.0 - 0.05) * (episode / total_episodes)
        return max(0.05, epsilon)
