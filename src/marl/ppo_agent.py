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
        # Encode the full dynamic observation vector (obs_cov + run_cov + asym + counts + budget)
        obs_features = hk.Sequential([
            hk.Linear(self.hidden_dim),
            jax.nn.relu,
            hk.Linear(self.hidden_dim),
            jax.nn.relu
        ])(obs)
        
        gru = hk.GRU(self.hidden_dim)
        rnn_out, next_state = gru(obs_features, state)
        
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
    """
    NOTE: The Skew-Symmetric Tournament graph head this class was originally built around
    was removed when the action space collapsed to INTERVENE/NOOP (graph_logits output
    dropped entirely). This class is currently architecturally identical to IPPOActor;
    kept as a distinct name only for CLI/checkpoint backward compatibility.
    """
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
    """NOTE: see InductiveIPPOActor -- the graph head is likewise removed here."""
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


@jax.jit
def compute_ucb_bonus(visits: jax.Array, step_count: jax.Array, c: float) -> jax.Array:
    """UCB-style exploration bonus per node: c * sqrt(log(t+1) / (visits+1)).

    `visits` [d]: per-node intervention counts so far this episode
    (EnvState.node_intervention_counts, shared across both agents). `step_count`: current
    episode step index (EnvState.step_count). `c`: exploration coefficient (--ucb_coef).

    Added to an agent's target_logits *before* masking/sampling, at both training-time
    action selection and eval time (never just grafted on post-hoc), so the trained
    policy is learned with this structural exploration bias already shaping which
    targets get reinforced -- see docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md's
    greedy-policy-collapse fix. A direct function of the agent's own action history, not
    environment response, so it doesn't inherit the running-covariance-style saturation
    that made purely reactive signals converge to a near-fixed-point under a repeated
    action.
    """
    t = jnp.asarray(step_count, dtype=jnp.float32)
    return c * jnp.sqrt(jnp.log(t + 1.0) / (visits + 1.0))


@jax.jit
def compute_uncertainty_bonus(predicted_dag: jax.Array, structural_mask: jax.Array, c: float) -> jax.Array:
    """Uncertainty-driven exploration bonus per node -- Track B (see
    docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md's greedy-policy-collapse follow-up).

    The UCB visit-count bonus (compute_ucb_bonus) increased target diversity
    dramatically but reached0 fell to 0% -- it diversified "for its own sake" via raw
    visit counts, blind to which node's data would actually resolve remaining structural
    uncertainty for the specific topology in play. This version is tied directly to the
    Stage-2 estimator's own edge-confidence instead: for each candidate edge (i, j), a
    predicted probability p near 0.5 means the estimator is still unsure whether that
    edge exists; p near 0 or 1 means it's confident. Per-edge uncertainty is
    `1 - |2p - 1|` (0 at p=0/1, 1 at p=0.5), masked to only structurally-possible edges.
    A node's bonus is the sum of uncertainty over every edge touching it (both as source
    and target) -- nodes sitting at the center of unresolved edges get the biggest push.

    `predicted_dag` [d, d]: the current predicted edge-probability matrix
    (FederatedCausalEnv.last_predicted_dag -- initialized to 0.5 everywhere at reset,
    i.e. maximal uncertainty before any data exists). `structural_mask` [d, d]: zeroes
    out structurally-impossible edges. `c`: exploration coefficient (--uncertainty_coef).

    Added to target_logits the same way and at the same points as compute_ucb_bonus
    (before masking/sampling, at both training-time action selection and eval time) --
    the two bonuses are summed, not mutually exclusive, though --ucb_coef defaults to
    0.0 now given its standalone result.
    """
    edge_uncertainty = (1.0 - jnp.abs(2.0 * predicted_dag - 1.0)) * structural_mask
    node_uncertainty = jnp.sum(edge_uncertainty, axis=1) + jnp.sum(edge_uncertainty, axis=0)
    return c * node_uncertainty


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
