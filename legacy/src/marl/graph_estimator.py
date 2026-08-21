"""Trainable Stage-2 graph structure estimator.

Restores a gradient-based structure-learning signal without coupling it into the
intervention-policy actor. The pre-refactor architecture had a "shared edge scorer"
graph head living *inside* the actor, trained via a BCE loss added directly to the
actor's RL loss (see docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md) -- effective, but
it meant the RL policy's own representations were entangled with structure prediction,
which the later two-stage refactor deliberately decoupled.

This module keeps that decoupling: GraphEstimatorNet is a separate small network with
its own parameters and optimizer, structurally similar to the old edge scorer, but
never touched by the actor's gradients. It is trained online via supervised BCE
against the ground-truth adjacency matrix (masked by the same domain-privacy
structural mask used elsewhere) using the same per-step covariance/asymmetry features
the frozen analytic estimator already uses -- giving `predict_graph_hypothesis` a
learned, continuously-improving estimator option (`estimator_type == "learned"`)
alongside the existing frozen "analytic" and pretrained "avici" options.
"""
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Tuple


class GraphEstimatorNet(hk.Module):
    def __init__(self, d: int, embed_dim: int = 16, hidden_dim: int = 32, name: str = None):
        super().__init__(name=name)
        self.d = d
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def __call__(self, obs_cov: jax.Array, run_cov: jax.Array, asym: jax.Array) -> jax.Array:
        # Each node's features are its rows across the three [d, d] statistics.
        node_features = jnp.concatenate([obs_cov, run_cov, asym], axis=-1)  # [d, 3d]
        node_embeddings = hk.Sequential([
            hk.Linear(self.hidden_dim), jax.nn.relu, hk.Linear(self.embed_dim)
        ])(node_features)  # [d, embed_dim]

        emb_i = jnp.tile(node_embeddings[:, None, :], (1, self.d, 1))
        emb_j = jnp.tile(node_embeddings[None, :, :], (self.d, 1, 1))
        pairs = jnp.concatenate([emb_i, emb_j], axis=-1)  # [d, d, 2*embed_dim]

        edge_scorer = hk.Sequential([hk.Linear(self.hidden_dim), jax.nn.relu, hk.Linear(1)])
        graph_logits = jnp.squeeze(edge_scorer(pairs), axis=-1)  # [d, d]
        return graph_logits


def init_graph_estimator(rng: jax.Array, d: int):
    def forward(obs_cov, run_cov, asym):
        return GraphEstimatorNet(d=d)(obs_cov, run_cov, asym)
    transformed = hk.without_apply_rng(hk.transform(forward))
    dummy = jnp.zeros((d, d))
    params = transformed.init(rng, dummy, dummy, dummy)
    return transformed, params


def make_graph_estimator_fns(transformed, optimizer: optax.GradientTransformation, structural_mask: jax.Array):
    def loss_fn(params, obs_cov, run_cov, asym, true_adj):
        logits = transformed.apply(params, obs_cov, run_cov, asym)
        bce = optax.sigmoid_binary_cross_entropy(logits, true_adj)
        masked_bce = bce * structural_mask
        valid_count = jnp.maximum(1.0, jnp.sum(structural_mask))
        return jnp.sum(masked_bce) / valid_count

    @jax.jit
    def update_step(params, opt_state, obs_cov, run_cov, asym, true_adj):
        loss, grads = jax.value_and_grad(loss_fn)(params, obs_cov, run_cov, asym, true_adj)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def predict(params, obs_cov, run_cov, asym):
        logits = transformed.apply(params, obs_cov, run_cov, asym)
        return jax.nn.sigmoid(logits)

    return update_step, predict
