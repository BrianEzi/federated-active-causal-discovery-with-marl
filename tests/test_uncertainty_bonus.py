import numpy as np
import jax.numpy as jnp

from src.marl.ppo_agent import compute_uncertainty_bonus
from src.types import compute_global_structural_mask, STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK


def _mask():
    return jnp.array(np.array(compute_global_structural_mask(STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK)))


def test_maximally_uncertain_prediction_gives_nonzero_bonus_everywhere_touched():
    """predicted_dag = 0.5 everywhere is the reset-time default (see
    FederatedCausalEnv.last_predicted_dag) -- maximal uncertainty, should give the
    largest possible per-edge bonus (1.0 per structurally-valid edge)."""
    d = 4
    predicted = jnp.full((d, d), 0.5)
    mask = _mask()
    bonus = compute_uncertainty_bonus(predicted, mask, c=1.0)
    # Every node touching at least one structurally-valid edge should have bonus > 0.
    touches_any_edge = (np.array(mask).sum(axis=0) + np.array(mask).sum(axis=1)) > 0
    for i in range(d):
        if touches_any_edge[i]:
            assert bonus[i] > 0.0


def test_confident_prediction_gives_much_smaller_bonus_than_uncertain():
    """predicted_dag near 0 or 1 everywhere (estimator is sure) should give a much
    smaller bonus than the maximally-uncertain (p=0.5 everywhere) case -- not
    necessarily ~0 in absolute terms, since a node touching several edges accumulates
    small per-edge residuals (e.g. a boundary node touching ~6 edges at p=0.99 still
    sums to ~0.12), but the *relative* drop from full uncertainty must be large."""
    d = 4
    mask = _mask()
    rng = np.random.default_rng(0)
    confident = np.where(rng.uniform(size=(d, d)) > 0.5, 0.99, 0.01)
    bonus_confident = compute_uncertainty_bonus(jnp.array(confident), mask, c=1.0)
    bonus_uncertain = compute_uncertainty_bonus(jnp.full((d, d), 0.5), mask, c=1.0)
    touches_any_edge = (np.array(mask).sum(axis=0) + np.array(mask).sum(axis=1)) > 0
    for i in range(d):
        if touches_any_edge[i]:
            assert bonus_confident[i] < 0.15 * bonus_uncertain[i]


def test_structural_mask_zeroes_out_impossible_edges():
    """A node with only structurally-impossible edges (e.g. two private nodes that can
    never connect directly) should get zero bonus regardless of predicted_dag's value
    there, since the mask should zero those entries before summing."""
    d = 4
    # Z1 (0) and Z2 (3) can never have a direct edge under this project's structural
    # mask -- force maximal "uncertainty" there and confirm it contributes nothing.
    predicted = jnp.zeros((d, d))
    predicted = predicted.at[0, 3].set(0.5).at[3, 0].set(0.5)
    mask = _mask()
    assert np.array(mask)[0, 3] == 0.0  # sanity check on the mask itself
    bonus = compute_uncertainty_bonus(predicted, mask, c=1.0)
    # Node 0 and node 3 have no OTHER edges set uncertain here, so their bonus should
    # be exactly 0 -- the masked-out Z1<->Z2 uncertainty must not leak through.
    assert float(bonus[0]) == 0.0
    assert float(bonus[3]) == 0.0


def test_node_with_more_uncertain_edges_gets_higher_bonus():
    d = 4
    mask = _mask()
    predicted_uniform_uncertain = jnp.full((d, d), 0.5) * mask
    predicted_one_confident = predicted_uniform_uncertain.at[0, 1].set(0.99).at[1, 0].set(0.01)
    bonus_uncertain = compute_uncertainty_bonus(predicted_uniform_uncertain, mask, c=1.0)
    bonus_one_confident = compute_uncertainty_bonus(predicted_one_confident, mask, c=1.0)
    # Node 0's touching edge (0,1) became confident -- its bonus should strictly drop.
    assert float(bonus_one_confident[0]) < float(bonus_uncertain[0])


def test_zero_coefficient_disables_it():
    d = 4
    predicted = jnp.full((d, d), 0.5)
    mask = _mask()
    bonus = compute_uncertainty_bonus(predicted, mask, c=0.0)
    assert np.allclose(np.array(bonus), 0.0)
