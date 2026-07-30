import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import Dict, Any, Tuple

class RolloutBuffer:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.data = {
            "obs": [], "cat_actions": [], "target_actions": [], 
            "rewards": [], "dones": [], "values": [], 
            "log_probs": [], "graph_preds": []
        }
        
    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)
            
    def get_batches(self):
        return {k: jnp.array(v) for k, v in self.data.items()}

def compute_gae(rewards: jax.Array, values: jax.Array, dones: jax.Array, gamma: float = 0.99, lam: float = 0.95):
    """Computes Generalized Advantage Estimation."""
    advs = jnp.zeros_like(rewards)
    
    def scan_fn(carry, transition):
        r, v, nv, d = transition
        gae = carry
        delta = r + gamma * nv * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        return gae, gae
        
    next_values = jnp.append(values[1:], 0.0)
    transitions = (rewards[::-1], values[::-1], next_values[::-1], dones[::-1])
    
    _, advs_rev = jax.lax.scan(scan_fn, 0.0, transitions)
    advs = advs_rev[::-1]
    
    returns = advs + values
    return advs, returns

class IPPOTrainer:
    def __init__(self, actor_transform, critic_transform, 
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3, 
                 clip_eps: float = 0.2, entropy_coef: float = 0.01, graph_coef: float = 0.5):
        self.actor = actor_transform
        self.critic = critic_transform
        
        self.actor_opt = optax.adam(learning_rate=actor_lr)
        self.critic_opt = optax.adam(learning_rate=critic_lr)
        
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.graph_coef = graph_coef
        
    def loss_fn(self, actor_params, critic_params, batch: Dict[str, jax.Array], true_adj: jax.Array):
        """Computes the PPO loss for a single agent."""
        obs = batch["obs"]
        cat_acts = batch["cat_actions"]
        tgt_acts = batch["target_actions"]
        old_log_probs = batch["log_probs"]
        advs = batch["advantages"]
        returns = batch["returns"]
        # Normalize advantages
        advs = (advs - jnp.mean(advs)) / (jnp.std(advs) + 1e-8)
        
        # 1. Critic Loss
        v_preds = self.critic.apply(critic_params, obs)
        critic_loss = jnp.mean((v_preds - returns) ** 2)
        
        # 2. Actor Loss
        cat_logits, target_logits, graph_logits = self.actor.apply(actor_params, obs)
        
        # Action log probs
        cat_dist = jax.nn.log_softmax(cat_logits)
        tgt_dist = jax.nn.log_softmax(target_logits)
        
        # Gather chosen action probs
        cat_lp = jax.vmap(lambda p, a: p[a])(cat_dist, cat_acts)
        tgt_lp = jax.vmap(lambda p, a: p[a])(tgt_dist, tgt_acts)
        
        # Joint log prob
        new_log_probs = cat_lp + tgt_lp
        
        ratio = jnp.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advs
        clipped = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
        
        actor_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        
        # 3. Entropy Bonus
        entropy = -jnp.mean(jnp.sum(jnp.exp(cat_dist) * cat_dist, axis=-1) + 
                            jnp.sum(jnp.exp(tgt_dist) * tgt_dist, axis=-1))
        
        # 4. Graph Supervised Loss (BCE against true DAG for fast convergence)
        # Using optax.sigmoid_binary_cross_entropy
        true_adj_batch = jnp.tile(true_adj[None, :, :], (obs.shape[0], 1, 1))
        graph_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(graph_logits, true_adj_batch))
        total_actor_loss = actor_loss - self.entropy_coef * entropy + self.graph_coef * graph_loss
        total_loss = total_actor_loss + critic_loss
        return total_loss, {"actor_loss": actor_loss, "entropy": entropy, "graph_loss": graph_loss, "critic_loss": critic_loss}
        
    import functools
    @functools.partial(jax.jit, static_argnums=(0,))
    def update_step(self, actor_params, critic_params, actor_opt_state, critic_opt_state, batch, true_adj):
        # Compute losses and gradients
        (total_loss, metrics), (a_grads, c_grads) = jax.value_and_grad(self.loss_fn, argnums=(0, 1), has_aux=True)(
            actor_params, critic_params, batch, true_adj
        )
        
        # Apply updates
        a_updates, new_actor_opt = self.actor_opt.update(a_grads, actor_opt_state)
        new_actor_params = optax.apply_updates(actor_params, a_updates)
        
        c_updates, new_critic_opt = self.critic_opt.update(c_grads, critic_opt_state)
        new_critic_params = optax.apply_updates(critic_params, c_updates)
        
        return new_actor_params, new_critic_params, new_actor_opt, new_critic_opt, metrics
