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

    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        obs: [batch_size, obs_dim]
        Returns:
            cat_logits: [batch_size, 3]
            target_logits: [batch_size, d]
            graph_logits: [batch_size, d, d]
        """
        # obs is flattened masked covariance (d*d) + budget (1)
        cov_flat = obs[:, :-1]
        cov = jnp.reshape(cov_flat, (-1, self.d, self.d))
        
        # 1. Node Embeddings
        # For each node i, its features could be its row and column in the covariance matrix
        # cov is symmetric, so row is enough.
        node_features = cov # [batch, d, d], where node i's features is cov[:, i, :]
        
        # We apply a shared MLP to each node's features to get its embedding
        # hk.Linear applied to [batch, d, d] broadcasts over the first two dims
        node_embeddings = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.embed_dim)
        ])(node_features) # [batch, d, embed_dim]
        
        # 2. Action Head (Macro + Micro)
        # Flatten embeddings to get a global representation for action selection
        global_rep = hk.Flatten()(node_embeddings)
        
        action_hidden = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.hidden_dim),
            jax.nn.relu
        ])(global_rep)
        
        cat_logits = hk.Linear(3)(action_hidden) # [batch, 3]
        target_logits = hk.Linear(self.d)(action_hidden) # [batch, d]
        
        # 3. Graph Head (Shared Edge Scorer)
        # We want to score every directed pair (i, j)
        # Expand embeddings to form all pairs: [batch, d, d, 2 * embed_dim]
        emb_expand_i = jnp.expand_dims(node_embeddings, axis=2) # [batch, d, 1, embed_dim]
        emb_expand_j = jnp.expand_dims(node_embeddings, axis=1) # [batch, 1, d, embed_dim]
        
        emb_expand_i = jnp.tile(emb_expand_i, (1, 1, self.d, 1))
        emb_expand_j = jnp.tile(emb_expand_j, (1, self.d, 1, 1))
        
        pairs = jnp.concatenate([emb_expand_i, emb_expand_j], axis=-1) # [batch, d, d, 2*embed_dim]
        
        edge_scorer = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(1)
        ])
        
        graph_logits = edge_scorer(pairs) # [batch, d, d, 1]
        graph_logits = jnp.squeeze(graph_logits, axis=-1) # [batch, d, d]
        
        return cat_logits, target_logits, graph_logits

class IPPOCritic(hk.Module):
    def __init__(self, hidden_dim: int = 64, name: str = None):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim

    def __call__(self, obs: jax.Array) -> jax.Array:
        """
        obs: [batch_size, obs_dim]
        Returns: [batch_size]
        """
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

    def __call__(self, obs: jax.Array, state: jax.Array) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        """
        obs: [batch_size, obs_dim]
        state: [batch_size, hidden_dim]
        """
        cov_flat = obs[:, :-1]
        cov = jnp.reshape(cov_flat, (-1, self.d, self.d))
        node_features = cov
        
        node_embeddings = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.embed_dim)
        ])(node_features)
        
        global_rep = hk.Flatten()(node_embeddings)
        
        # RNN Core
        gru = hk.GRU(self.hidden_dim)
        rnn_out, next_state = gru(global_rep, state)
        
        action_hidden = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu
        ])(rnn_out)
        
        cat_logits = hk.Linear(3)(action_hidden)
        target_logits = hk.Linear(self.d)(action_hidden)
        
        emb_expand_i = jnp.expand_dims(node_embeddings, axis=2)
        emb_expand_j = jnp.expand_dims(node_embeddings, axis=1)
        emb_expand_i = jnp.tile(emb_expand_i, (1, 1, self.d, 1))
        emb_expand_j = jnp.tile(emb_expand_j, (1, self.d, 1, 1))
        
        # Also broadcast the RNN output to the pairs so graph prediction depends on memory
        rnn_expand = jnp.expand_dims(jnp.expand_dims(rnn_out, axis=1), axis=1)
        rnn_expand = jnp.tile(rnn_expand, (1, self.d, self.d, 1))
        
        pairs = jnp.concatenate([emb_expand_i, emb_expand_j, rnn_expand], axis=-1)
        
        edge_scorer = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(1)
        ])
        
        graph_logits = edge_scorer(pairs)
        graph_logits = jnp.squeeze(graph_logits, axis=-1)
        
        return (cat_logits, target_logits, graph_logits), next_state

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


def mask_invalid_targets(cat_action: jax.Array, target_logits: jax.Array, local_ownership_mask: jax.Array, peer_boundary_mask: jax.Array = None) -> jax.Array:
    """
    Masks out invalid target logits based on the chosen category action.
    cat_action: [batch_size] (0: Local, 1: Peer, 2: NOOP)
    target_logits: [batch_size, d]
    local_ownership_mask: [d] (1 if owned by agent, 0 otherwise)
    peer_boundary_mask: [d] (1 if boundary node requestable via peer, 0 otherwise)
    """
    local_valid = local_ownership_mask[None, :] # [1, d]
    if peer_boundary_mask is not None:
        peer_valid = peer_boundary_mask[None, :] # [1, d]
    else:
        peer_valid = 1.0 - local_valid
    
    is_local = (cat_action == int(ActionCategory.LOCAL_INTERVENTION))[:, None]
    is_peer = (cat_action == int(ActionCategory.PEER_REQUEST))[:, None]
    
    valid_mask = is_local * local_valid + is_peer * peer_valid
    
    # Apply large negative value to invalid targets
    masked_logits = jnp.where(valid_mask > 0.5, target_logits, -1e9)
    return masked_logits


@jax.jit
def sample_actions_jitted(
    cat_logits: jax.Array,
    target_logits: jax.Array,
    graph_logits: jax.Array,
    local_ownership_mask: jax.Array,
    peer_boundary_mask: jax.Array,
    edge_mask: jax.Array,
    key: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Samples category and target actions directly on GPU without host-device synchronization.
    Supports single observation [3], [d], [d, d] or batched [B, 3], [B, d], [B, d, d].
    """
    is_batched = (cat_logits.ndim > 1)
    k1, k2 = jax.random.split(key)
    
    if is_batched:
        B = cat_logits.shape[0]
        cat = jax.random.categorical(k1, cat_logits) # [B]
        masked_target_logits = mask_invalid_targets(cat, target_logits, local_ownership_mask, peer_boundary_mask) # [B, d]
        safe_masked = jnp.where(jnp.isnan(masked_target_logits), -1e9, masked_target_logits)
        target = jax.random.categorical(k2, safe_masked) # [B]
        
        cat_lp = jax.nn.log_softmax(cat_logits)[jnp.arange(B), cat]
        target_lp = jax.nn.log_softmax(safe_masked)[jnp.arange(B), target]
        total_lp = cat_lp + target_lp
        
        graph_pred = jax.nn.sigmoid(graph_logits) * edge_mask[None, :, :]
        return cat, target, total_lp, graph_pred
    else:
        cat = jax.random.categorical(k1, cat_logits) # scalar
        masked_target_logits = mask_invalid_targets(jnp.array([cat]), target_logits[None, :], local_ownership_mask, peer_boundary_mask)[0]
        safe_masked = jnp.where(jnp.isnan(masked_target_logits), -1e9, masked_target_logits)
        target = jax.random.categorical(k2, safe_masked) # scalar
        
        cat_lp = jax.nn.log_softmax(cat_logits)[cat]
        target_lp = jax.nn.log_softmax(safe_masked)[target]
        total_lp = cat_lp + target_lp
        
        graph_pred = jax.nn.sigmoid(graph_logits) * edge_mask
        return cat, target, total_lp, graph_pred

