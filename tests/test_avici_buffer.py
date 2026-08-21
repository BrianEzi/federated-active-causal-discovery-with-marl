import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import MagicMock
from legacy.src.types import SCMConfig, MechanismType, NoiseType, ActionCategory
from legacy.src.evaluator_env import FederatedCausalEnv
from legacy.src.environment import stitch_global_mean


@pytest.fixture
def base_config():
    return SCMConfig(
        d=4,
        K=2,
        mechanism_type=int(MechanismType.LINEAR),
        noise_type=int(NoiseType.GAUSSIAN),
        noise_scale=0.1
    )


def test_stitch_global_mean_matches_hand_computed_average():
    # Agent 0 observes nodes [0,1], agent 1 observes nodes [2,3]; disjoint, no overlap.
    agent_masks = jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    local_means = jnp.array([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]])
    sample_counts = jnp.array([10.0, 10.0])
    result = np.array(stitch_global_mean(local_means, agent_masks, sample_counts))
    # Disjoint masks, equal weights -> each entry just takes whichever agent observed it.
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0], atol=1e-6)


def test_stitch_global_mean_weights_overlapping_observations():
    # Both agents observe node 0 with different sample counts and different local means.
    agent_masks = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    local_means = jnp.array([[2.0, 0.0], [4.0, 0.0]])
    sample_counts = jnp.array([10.0, 30.0])
    result = np.array(stitch_global_mean(local_means, agent_masks, sample_counts))
    expected_node0 = (2.0 * 10.0 + 4.0 * 30.0) / 40.0
    np.testing.assert_allclose(result[0], expected_node0, atol=1e-6)


def test_raw_sample_buffer_capacity_matches_max_steps_plus_reset_block(base_config):
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0, sample_count=50, max_steps=5)
    key = jax.random.PRNGKey(42)
    env.reset(key)
    # +1 block: the observational reset phase writes one sample_count-sized block before
    # any of the max_steps step-kernel writes happen.
    expected_capacity = (env.max_steps + 1) * env.sample_count
    assert env.jax_state.raw_samples.shape == (expected_capacity, env.config.d)
    assert env.jax_state.raw_interv.shape == (expected_capacity, env.config.d)
    assert int(env.jax_state.raw_count[0]) == env.sample_count


def test_raw_sample_buffer_never_overflows_across_full_episode(base_config):
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=1000.0, sample_count=20, max_steps=5)
    key = jax.random.PRNGKey(1)
    obs, info = env.reset(key)
    capacity = env.jax_state.raw_samples.shape[0]

    joint_actions = {
        "agent_0": (int(ActionCategory.INTERVENE), 0),
        "agent_1": (int(ActionCategory.NOOP), 0)
    }
    for _ in range(env.max_steps):
        key, step_key = jax.random.split(key)
        obs, rewards, done, info = env.step(joint_actions, predicted_dags=None, key=step_key)
        assert int(env.jax_state.raw_count[0]) <= capacity
        if done:
            break

    # After a full episode: reset block + every step taken.
    assert int(env.jax_state.raw_count[0]) == capacity


def test_raw_interv_labels_reflect_actual_interventions(base_config):
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0, sample_count=30, max_steps=5)
    key = jax.random.PRNGKey(7)
    env.reset(key)

    # Reset-phase block (first sample_count rows) must be all-zero (purely observational).
    reset_block = np.array(env.jax_state.raw_interv[:env.sample_count])
    assert np.all(reset_block == 0.0)

    # Agent 0 intervenes on node 0 this step -> the next sample_count-row block's column 0
    # should be entirely 1.0, and every other column should be entirely 0.0 (no other node
    # was under active intervention this step).
    joint_actions = {
        "agent_0": (int(ActionCategory.INTERVENE), 0),
        "agent_1": (int(ActionCategory.NOOP), 0)
    }
    env.step(joint_actions, predicted_dags=None, key=key)
    step_block = np.array(env.jax_state.raw_interv[env.sample_count:2 * env.sample_count])
    assert np.all(step_block[:, 0] == 1.0)
    assert np.all(step_block[:, 1:] == 0.0)


def test_avici_branch_uses_real_data_not_synthetic(base_config, monkeypatch):
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0, sample_count=30, max_steps=5)
    key = jax.random.PRNGKey(3)
    env.reset(key)

    mock_avici = MagicMock(return_value=np.zeros((4, 4)))
    env.avici_model = mock_avici
    env.estimator_type = "avici"

    joint_actions = {
        "agent_0": (int(ActionCategory.INTERVENE), 0),
        "agent_1": (int(ActionCategory.NOOP), 0)
    }
    env.step(joint_actions, predicted_dags=None, key=key)

    assert mock_avici.called
    call_kwargs = mock_avici.call_args.kwargs
    assert "interv" in call_kwargs and call_kwargs["interv"] is not None
    # Real per-sample intervention labels, not the old interv=None -- must contain
    # nonzero entries after a step with an active intervention.
    assert np.any(call_kwargs["interv"] != 0.0)

    # x must match the real accumulated raw-sample buffer, not a freshly-drawn synthetic
    # Gaussian sample (the original bug).
    n_valid = int(env.jax_state.raw_count[0])
    expected_x = np.array(env.jax_state.raw_samples[:n_valid])
    np.testing.assert_allclose(call_kwargs["x"], expected_x, atol=1e-5)
