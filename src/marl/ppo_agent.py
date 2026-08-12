import jax
import jax.numpy as jnp
import haiku as hk
from typing import Tuple, Dict
from src.types import ActionCategory

class IPPOActor(hk.Module):
    def __init__(self, d: int, embed_dim: int = 32, hidden_dim: int = 64, name: str = None):
        super().__init__(name=name)
        self.d = d
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        action_hidden = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.hidden_dim),
            jax.nn.relu
        ])(obs)
        
        cat_logits = hk.Linear(2)(action_hidden)
        target_logits = hk.Linear(self.d)(action_hidden)
        
        return cat_logits, target_logits

class IPPOCritic(hk.Module):
    def __init__(self, hidden_dim: int = 64, name: str = None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim

    def __call__(self, obs: jax.Array) -> jax.Array:
        v = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(1)
        ])(obs)
        return jnp.squeeze(v, axis=-1)

class IPPORNNActor(hk.Module):
    def __init__(self, d: int, embed_dim: int = 32, hidden_dim: int = 64, name: str = None):
        super().__init__(name=name)
        self.d = d
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def __call__(self, obs: jax.Array, state: jax.Array) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        cov_flat = obs[:, : self.d * self.d]
        cov = jnp.reshape(cov_flat, (-1, self.d, self.d))
        
        node_embeddings = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.embed_dim)
        ])(cov)
        
        global_rep = hk.Flatten()(node_embeddings)
        
        gru = hk.GRU(self.hidden_dim)
        rnn_out, next_state = gru(global_rep, state)
        
        action_hidden = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu
        ])(rnn_out)
        
        cat_logits = hk.Linear(2)(action_hidden)
        target_logits = hk.Linear(self.d)(action_hidden)
        
        return (cat_logits, target_logits), next_state

    @staticmethod
    def initial_state(batch_size: int, hidden_dim: int = 64) -> jax.Array:
        return jnp.zeros((batch_size, hidden_dim))

class IPPORNNCritic(hk.Module):
    def __init__(self, hidden_dim: int = 64, name: str = None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim

    def __call__(self, obs: jax.Array, state: jax.Array) -> Tuple[jax.Array, jax.Array]:
        gru = hk.GRU(self.hidden_dim)
        rnn_out, next_state = gru(obs, state)
        
        v = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(1)
        ])(rnn_out)
        return jnp.squeeze(v, axis=-1), next_state

    @staticmethod
    def initial_state(batch_size: int, hidden_dim: int = 64) -> jax.Array:
        return jnp.zeros((batch_size, hidden_dim))

class InductiveIPPOActor(hk.Module):
    def __init__(self, d: int, embed_dim: int = 32, hidden_dim: int = 64, gamma: float = 2.0, name: str = None):
        super().__init__(name=name)
        self.d = d
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        d2 = self.d * self.d
        if obs.shape[-1] >= 3 * d2 + 1:
            cov_obs_flat = obs[:, :d2]
        else:
            cov_obs_flat = obs[:, :d2]

        cov_obs = jnp.reshape(cov_obs_flat, (-1, self.d, self.d))

        node_embeddings = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.embed_dim)
        ])(cov_obs)

        global_rep = hk.Flatten()(node_embeddings)
        action_hidden = hk.Sequential([
            hk.Linear(self.hidden_dim), jax.nn.relu,
            hk.Linear(self.hidden_dim), jax.nn.relu
        ])(global_rep)
        
        cat_logits = hk.Linear(2)(action_hidden)
        target_logits = hk.Linear(self.d)(action_hidden)

        return cat_logits, target_logits

class InductiveIPPORNNActor(hk.Module):
    def __init__(self, d: int, embed_dim: int = 32, hidden_dim: int = 64, gamma: float = 2.0, name: str = None):
        super().__init__(name=name)
        self.d = d
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def __call__(self, obs: jax.Array, state: jax.Array) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        d2 = self.d * self.d
        if obs.shape[-1] >= 3 * d2 + 1:
            cov_obs_flat = obs[:, :d2]
        else:
            cov_obs_flat = obs[:, :d2]

        cov_obs = jnp.reshape(cov_obs_flat, (-1, self.d, self.d))

        node_embeddings = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.embed_dim)
        ])(cov_obs)

        global_rep = hk.Flatten()(node_embeddings)
        
        gru = hk.GRU(self.hidden_dim)
        rnn_out, next_state = gru(global_rep, state)
        
        action_hidden = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu
        ])(rnn_out)
        
        cat_logits = hk.Linear(2)(action_hidden)
        target_logits = hk.Linear(self.d)(action_hidden)

        return (cat_logits, target_logits), next_state

    @staticmethod
    def initial_state(batch_size: int, hidden_dim: int = 64) -> jax.Array:
        return jnp.zeros((batch_size, hidden_dim))


def mask_invalid_targets(cat_action: jax.Array, target_logits: jax.Array, valid_intervention_mask: jax.Array) -> jax.Array:
    is_intervene = (cat_action == int(ActionCategory.INTERVENE))[:, None]
    
    masked_logits = jnp.where((is_intervene * valid_intervention_mask[None, :]) > 0.5, target_logits, -1e9)
    return masked_logits

@jax.jit
def sample_actions_jitted(
    cat_logits: jax.Array,
    target_logits: jax.Array,
    valid_intervention_mask: jax.Array,
    key: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    is_batched = (cat_logits.ndim > 1)
    k1, k2 = jax.random.split(key)

    if is_batched:
        B = cat_logits.shape[0]
        cat = jax.random.categorical(k1, cat_logits)
        masked_target_logits = mask_invalid_targets(cat, target_logits, valid_intervention_mask)
        safe_masked = jnp.where(jnp.isnan(masked_target_logits), -1e9, masked_target_logits)
        target = jax.random.categorical(k2, safe_masked)
        
        cat_lp = jax.nn.log_softmax(cat_logits)[jnp.arange(B), cat]
        target_lp = jax.nn.log_softmax(safe_masked)[jnp.arange(B), target]
        total_lp = cat_lp + target_lp
        
        return cat, target, total_lp
    else:
        cat = jax.random.categorical(k1, cat_logits)
        masked_target_logits = mask_invalid_targets(jnp.array([cat]), target_logits[None, :], valid_intervention_mask)[0]
        safe_masked = jnp.where(jnp.isnan(masked_target_logits), -1e9, masked_target_logits)
        target = jax.random.categorical(k2, safe_masked)
        
        cat_lp = jax.nn.log_softmax(cat_logits)[cat]
        target_lp = jax.nn.log_softmax(safe_masked)[target]
        total_lp = cat_lp + target_lp
        
        return cat, target, total_lp
