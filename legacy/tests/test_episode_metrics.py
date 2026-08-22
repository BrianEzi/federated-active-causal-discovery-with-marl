import numpy as np
import pytest
from scipy.stats import multivariate_normal
from legacy.src.episode_metrics import gaussian_entropy, shd_trajectory_auc, shd_reduction_auc, normalized_target_entropy
from legacy.src.types import SCMConfig, MechanismType, NoiseType, ActionCategory
from legacy.src.evaluator_env import FederatedCausalEnv
import jax
import jax.numpy as jnp


def test_gaussian_entropy_matches_scipy_reference():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    cov = A @ A.T + np.eye(4) * 0.5  # guaranteed PD
    expected = multivariate_normal(mean=np.zeros(4), cov=cov).entropy()
    result = gaussian_entropy(cov, d=4, eps=0.0)
    assert result == pytest.approx(expected, abs=1e-6)


def test_gaussian_entropy_handles_zero_covariance_via_eps_floor():
    cov = np.zeros((4, 4))
    result = gaussian_entropy(cov, d=4)
    assert np.isfinite(result)


def test_shd_trajectory_auc_matches_hand_computed_trapezoidal_sum():
    trajectory = [4.0, 4.0, 2.0, 0.0, 0.0]
    max_shd = 12.0
    expected_raw = np.trapz(trajectory)  # (4+4)/2 + (4+2)/2 + (2+0)/2 + (0+0)/2 = 4+3+1+0 = 8
    expected_normalized = expected_raw / (max_shd * (len(trajectory) - 1))
    result = shd_trajectory_auc(trajectory, max_shd)
    assert result == pytest.approx(expected_normalized, abs=1e-8)


def test_shd_trajectory_auc_lower_for_faster_convergence():
    fast = [4.0, 0.0, 0.0, 0.0, 0.0]
    slow = [4.0, 4.0, 4.0, 4.0, 0.0]
    assert shd_trajectory_auc(fast, 12.0) < shd_trajectory_auc(slow, 12.0)


def test_shd_reduction_auc_is_zero_when_no_reduction_occurs():
    flat = [4.0, 4.0, 4.0, 4.0]
    assert shd_reduction_auc(flat, 12.0) == pytest.approx(0.0, abs=1e-8)


def test_shd_reduction_auc_positive_when_shd_improves():
    improving = [4.0, 2.0, 0.0, 0.0]
    assert shd_reduction_auc(improving, 12.0) > 0.0


def test_normalized_target_entropy_uniform_distribution_near_one():
    counts = {0: 5, 1: 5, 2: 5, 3: 5}
    result = normalized_target_entropy(counts, d=4)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_normalized_target_entropy_single_node_is_zero():
    counts = {0: 20, 1: 0, 2: 0, 3: 0}
    result = normalized_target_entropy(counts, d=4)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_normalized_target_entropy_no_interventions_is_zero_not_undefined():
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    assert normalized_target_entropy(counts, d=4) == 0.0


@pytest.fixture
def base_config():
    return SCMConfig(
        d=4, K=2,
        mechanism_type=int(MechanismType.LINEAR),
        noise_type=int(NoiseType.GAUSSIAN),
        noise_scale=0.1
    )


def test_step_info_exposes_shd_delta_and_asym_matrix(base_config):
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0)
    key = jax.random.PRNGKey(42)
    env.reset(key)

    joint_actions = {
        "agent_0": (int(ActionCategory.INTERVENE), 0),
        "agent_1": (int(ActionCategory.NOOP), 0)
    }
    _, _, _, info = env.step(joint_actions, predicted_dags=None, key=key)

    assert "shd_delta" in info
    assert "agent_0" in info["shd_delta"] and "agent_1" in info["shd_delta"]
    # First step of the episode: no prior SHD to compare against, so delta is the
    # documented 0.0 sentinel, not a crash/None.
    assert info["shd_delta"]["agent_0"] == 0.0
    assert info["shd_delta"]["agent_1"] == 0.0

    assert "asym_matrix" in info
    assert np.array(info["asym_matrix"]).shape == (4, 4)


def test_redundant_same_node_intervention_produces_a_shd_delta_but_costs_both_agents(base_config):
    """Regression test for the coordination/redundancy scenario: both agents can legally
    target the same boundary node (X1, index 1) in the same step. Confirms this doesn't
    crash and both agents' budgets are charged even though only one effective intervention
    reaches the SCM."""
    action_costs = jnp.array([0.5, 0.5])
    env = FederatedCausalEnv(base_config, action_costs, initial_budget=10.0)
    key = jax.random.PRNGKey(42)
    obs, _ = env.reset(key)

    joint_actions = {
        "agent_0": (int(ActionCategory.INTERVENE), 1),
        "agent_1": (int(ActionCategory.INTERVENE), 1)
    }
    next_obs, rewards, done, info = env.step(joint_actions, predicted_dags=None, key=key)

    # Both agents pay their intervention cost despite targeting the same node.
    assert next_obs["agent_0"][-1] == 9.5
    assert next_obs["agent_1"][-1] == 9.5
