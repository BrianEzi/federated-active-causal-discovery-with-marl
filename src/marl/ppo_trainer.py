import functools
import jax
import jax.numpy as jnp
import optax
import haiku as hk
from typing import Dict, Any, Tuple
from src.types import STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK, compute_edge_authority_mask
from src.marl.ppo_agent import mask_invalid_targets

class RolloutBuffer:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.data = {
            "obs": [], "cat_actions": [], "target_actions": [],
            "rewards": [], "dones": [], "values": [],
            "log_probs": [], "ucb_bonus": []
        }
        
    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)
            
    def get_batches(self, max_size: int = None):
        """Returns batches as jnp arrays, optionally padded to max_size with valid_mask."""
        curr_len = len(self.data["obs"])
        if max_size is None or curr_len == 0 or curr_len == max_size:
            batches = {k: jnp.array(v) for k, v in self.data.items()}
            batches["valid_mask"] = jnp.ones(curr_len, dtype=jnp.float32)
            return batches
            
        pad_len = max_size - curr_len
        batches = {}
        for k, v in self.data.items():
            arr = jnp.array(v)
            pad_shape = (pad_len,) + arr.shape[1:]
            pad_val = jnp.zeros(pad_shape, dtype=arr.dtype)
            batches[k] = jnp.concatenate([arr, pad_val], axis=0)
            
        valid_mask = jnp.concatenate([jnp.ones(curr_len, dtype=jnp.float32), jnp.zeros(pad_len, dtype=jnp.float32)], axis=0)
        batches["valid_mask"] = valid_mask
        return batches

@jax.jit
def compute_gae(rewards: jax.Array, values: jax.Array, dones: jax.Array, gamma: float = 0.99, lam: float = 0.95):
    """
    Computes Generalized Advantage Estimation.
    Returns RAW (unnormalized) advantages and returns. Advantages must only be normalized
    downstream over the *valid* (unpadded) transitions -- see IPPOTrainer.loss_fn, which
    already does this correctly for the policy-gradient ratio. Normalizing here (before
    returns = advs + values is computed) would make the critic's regression target contain
    an artificially unit-variance term independent of critic quality, putting a hard floor
    under critic_loss that no amount of training can cross; it would also let zero-padded
    trailing transitions (from RolloutBuffer.get_batches's static-shape padding) skew the
    normalization statistics for real transitions in early-terminated episodes.
    """
    def scan_fn(carry, transition):
        r, v, nv, d = transition
        gae = carry
        delta = r + gamma * nv * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
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
                 clip_eps: float = 0.2, entropy_coef: float = 0.01,
                 use_rnn: bool = False, total_episodes: int = 5000, 
                 normalize_rewards: bool = True, max_steps: float = 20.0):
        self.actor = actor_transform
        self.critic = critic_transform
        self.use_rnn = use_rnn
        self.total_episodes = total_episodes
        self.normalize_rewards = normalize_rewards
        self.max_steps = max_steps
        
        warmup_steps = int(total_episodes * 0.05)
        decay_steps = max(1, total_episodes - warmup_steps)
        
        actor_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=actor_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=actor_lr * 0.1
        )
        critic_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=critic_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=critic_lr * 0.1
        )
        
        self.actor_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=actor_schedule)
        )
        self.critic_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=critic_schedule)
        )
        
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        
    def loss_fn(self, actor_params, critic_params, batch: Dict[str, jax.Array], valid_intervention_mask: jax.Array):
        """
        Computes the PPO loss for a single agent.
        valid_intervention_mask: [d] mask of nodes this agent may target on INTERVENE, identical
        to what was passed to sample_actions_jitted at rollout time. Required so target_logits
        gets masked identically here as it was at rollout -- otherwise old_log_probs (computed
        rollout-side from masked logits) and new_log_probs (recomputed here) diverge purely from
        the masking asymmetry, corrupting the PPO ratio even when the policy hasn't changed.
        """
        obs = batch["obs"]
        cat_acts = batch["cat_actions"]
        tgt_acts = batch["target_actions"]
        old_log_probs = batch["log_probs"]
        advs = batch["advantages"]
        returns = batch["returns"]
        ucb_bonus = batch["ucb_bonus"]
        valid_mask = batch.get("valid_mask", jnp.ones(obs.shape[0], dtype=jnp.float32))
        valid_count = jnp.maximum(1.0, jnp.sum(valid_mask))
        
        # Normalize advantages strictly over valid transitions
        adv_mean = jnp.sum(advs * valid_mask) / valid_count
        adv_var = jnp.sum(((advs - adv_mean) ** 2) * valid_mask) / valid_count
        advs = ((advs - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)) * valid_mask
        
        # 1. Critic Loss
        if self.use_rnn:
            # Manual unroll over time dimension using scan
            def scan_critic(state, x):
                v, next_state = self.critic.apply(critic_params, x, state)
                return next_state, v
            obs_t = jnp.expand_dims(obs, axis=1)
            h0 = jnp.zeros((1, 64))
            _, v_preds = jax.lax.scan(scan_critic, h0, obs_t)
            v_preds = jnp.squeeze(v_preds, axis=1)
        else:
            v_preds = self.critic.apply(critic_params, obs)
            
        value_scale = (self.max_steps ** 2) if self.normalize_rewards else 1.0
        critic_loss = (jnp.sum(((v_preds - returns) ** 2) * valid_mask) / valid_count) * value_scale
        
        # 2. Actor Loss
        if self.use_rnn:
            def scan_actor(state, x):
                outs, next_state = self.actor.apply(actor_params, x, state)
                return next_state, outs
            obs_t = jnp.expand_dims(obs, axis=1)
            h0 = jnp.zeros((1, 64))
            _, outs = jax.lax.scan(scan_actor, h0, obs_t)
            cat_logits, target_logits = outs
            cat_logits = jnp.squeeze(cat_logits, axis=1)
            target_logits = jnp.squeeze(target_logits, axis=1)
        else:
            cat_logits, target_logits = self.actor.apply(actor_params, obs)
        
        # Action log probs -- add the same per-transition UCB bonus that was applied at
        # rollout time (see train.py's rollout loop and compute_ucb_bonus's docstring)
        # before masking, exactly mirroring rollout, so old_log_probs and new_log_probs
        # are computed under the same distribution (see valid_intervention_mask docstring
        # above -- the UCB bonus is a second, equally real source of the same train/eval
        # masking-mismatch risk if it were applied inconsistently here).
        target_logits = target_logits + ucb_bonus
        masked_target_logits = mask_invalid_targets(cat_acts, target_logits, valid_intervention_mask)
        cat_dist = jax.nn.log_softmax(cat_logits)
        tgt_dist = jax.nn.log_softmax(masked_target_logits)
        
        # Gather chosen action probs
        cat_lp = jax.vmap(lambda p, a: p[a])(cat_dist, cat_acts)
        tgt_lp = jax.vmap(lambda p, a: p[a])(tgt_dist, tgt_acts)
        
        # Joint log prob
        new_log_probs = cat_lp + tgt_lp
        
        ratio = jnp.exp(new_log_probs - old_log_probs)
        unclipped = ratio * advs
        clipped = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
        
        actor_loss = -jnp.sum(jnp.minimum(unclipped, clipped) * valid_mask) / valid_count
        
        # 3. Entropy Bonus
        entropy = -jnp.sum((jnp.sum(jnp.exp(cat_dist) * cat_dist, axis=-1) + 
                            jnp.sum(jnp.exp(tgt_dist) * tgt_dist, axis=-1)) * valid_mask) / valid_count
        
        total_actor_loss = actor_loss - self.entropy_coef * entropy
        total_loss = total_actor_loss + critic_loss
        return total_loss, {"actor_loss": actor_loss, "entropy": entropy, "critic_loss": critic_loss}
        
    @functools.partial(jax.jit, static_argnums=(0,))
    def update_step(self, actor_params, critic_params, actor_opt_state, critic_opt_state, batch, valid_intervention_mask):
        # Compute losses and gradients
        (total_loss, metrics), (a_grads, c_grads) = jax.value_and_grad(self.loss_fn, argnums=(0, 1), has_aux=True)(
            actor_params, critic_params, batch, valid_intervention_mask
        )
        
        # Apply updates
        a_updates, new_actor_opt = self.actor_opt.update(a_grads, actor_opt_state)
        new_actor_params = optax.apply_updates(actor_params, a_updates)
        
        c_updates, new_critic_opt = self.critic_opt.update(c_grads, critic_opt_state)
        new_critic_params = optax.apply_updates(critic_params, c_updates)
        
        return new_actor_params, new_critic_params, new_actor_opt, new_critic_opt, metrics
