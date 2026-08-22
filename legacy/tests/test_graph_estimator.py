import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from legacy.src.marl.graph_estimator import init_graph_estimator, make_graph_estimator_fns
from legacy.src.types import STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK, compute_global_structural_mask


def _make_fns(d=4):
    rng = jax.random.PRNGKey(0)
    transformed, params = init_graph_estimator(rng, d)
    structural_mask = compute_global_structural_mask(STANDARD_LOCAL_MASKS, STANDARD_BOUNDARY_MASK)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    update_step, predict = make_graph_estimator_fns(transformed, optimizer, structural_mask)
    return params, opt_state, update_step, predict, structural_mask


def test_predict_returns_valid_probabilities_shaped_dxd():
    params, _, _, predict, _ = _make_fns(d=4)
    rng = np.random.default_rng(0)
    obs_cov = rng.normal(size=(4, 4))
    run_cov = rng.normal(size=(4, 4))
    asym = rng.normal(size=(4, 4))
    probs = np.array(predict(params, jnp.array(obs_cov), jnp.array(run_cov), jnp.array(asym)))
    assert probs.shape == (4, 4)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_update_step_reduces_loss_on_a_fixed_example():
    params, opt_state, update_step, predict, structural_mask = _make_fns(d=4)
    rng = np.random.default_rng(1)
    obs_cov = jnp.array(rng.normal(size=(4, 4)))
    run_cov = jnp.array(rng.normal(size=(4, 4)))
    asym = jnp.array(rng.normal(size=(4, 4)))
    true_adj = jnp.array(np.array(structural_mask))  # target: predict the mask itself, a fixed learnable pattern

    losses = []
    for _ in range(50):
        params, opt_state, loss = update_step(params, opt_state, obs_cov, run_cov, asym, true_adj)
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_update_step_only_trains_within_structural_mask():
    params, opt_state, update_step, predict, structural_mask = _make_fns(d=4)
    mask_np = np.array(structural_mask)
    assert mask_np[0, 3] == 0.0, "sanity: Z1<->Z2 (private-private) must be forbidden by the mask"
    rng = np.random.default_rng(2)
    obs_cov = jnp.array(rng.normal(size=(4, 4)))
    run_cov = jnp.array(rng.normal(size=(4, 4)))
    asym = jnp.array(rng.normal(size=(4, 4)))
    # Adversarial label: true_adj says edge (0,3) exists, which the mask forbids.
    true_adj = jnp.zeros((4, 4)).at[0, 3].set(1.0)

    for _ in range(50):
        params, opt_state, _ = update_step(params, opt_state, obs_cov, run_cov, asym, true_adj)

    probs = np.array(predict(params, obs_cov, run_cov, asym))
    # predict_graph_hypothesis multiplies by structural_mask afterwards, but the
    # network's own training signal for the forbidden cell should stay near its
    # (near-0.5, untrained) initialization rather than being pulled toward 1.0,
    # since the masked BCE loss never penalizes that cell.
    assert probs[0, 3] < 0.9, "forbidden edge was pulled toward the adversarial label despite masking"


def test_evaluator_env_learned_estimator_predicts_and_updates():
    from legacy.src.evaluator_env import FederatedCausalEnv
    from legacy.src.types import SCMConfig, MechanismType, NoiseType

    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    env = FederatedCausalEnv(config=config, action_costs=np.array([1.0, 1.0]), estimator_type="learned")
    assert env.estimator_type == "learned"
    assert env.graph_estimator_params is not None

    rng = np.random.default_rng(3)
    obs_cov = rng.normal(size=(4, 4))
    run_cov = rng.normal(size=(4, 4))
    asym = rng.normal(size=(4, 4))
    true_adj = np.array(env.structural_mask)  # arbitrary but structurally valid target

    pred_before = env.predict_graph_hypothesis(obs_cov, run_cov, asym)
    assert pred_before.shape == (4, 4)
    assert np.all(pred_before >= 0.0) and np.all(pred_before <= 1.0)

    params_before = env.graph_estimator_params
    loss = env.update_graph_estimator(obs_cov, run_cov, asym, true_adj)
    assert np.isfinite(loss)
    assert env.graph_estimator_params is not params_before, "params object should be replaced after an update"
