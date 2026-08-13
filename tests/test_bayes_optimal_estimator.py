import jax
import numpy as np

from src.generators import get_all_4node_topologies, generate_4node_topologies, generate_scm_params
from src.environment import init_env
from src.types import SCMConfig, MechanismType, NoiseType, InterventionSpec, InterventionType
from src.scm import sample_scm
from src.marl.bayes_optimal_estimator import compute_hypothesis_posterior, bayes_optimal_predict


def test_get_all_4node_topologies_matches_individual_force_idx_calls():
    """Regression test for the generators.py refactor: get_all_4node_topologies (used by
    the Bayes-optimal estimator) must return exactly the same 8 candidates, in the same
    order, as generate_4node_topologies(force_idx=i) for each i."""
    all_adj, all_orders = get_all_4node_topologies()
    assert all_adj.shape == (8, 4, 4)
    assert all_orders.shape == (8, 4)
    for i in range(8):
        adj_i, order_i = generate_4node_topologies(jax.random.PRNGKey(0), force_idx=i)
        assert np.allclose(np.array(adj_i), np.array(all_adj[i]))
        assert np.allclose(np.array(order_i), np.array(all_orders[i]))
    # Every candidate is a spanning tree over 4 nodes: exactly 3 edges each.
    assert np.all(np.sum(np.array(all_adj), axis=(1, 2)) == 3)


def test_zero_samples_gives_uniform_posterior():
    all_adj, _ = get_all_4node_topologies()
    prob, posterior = bayes_optimal_predict(
        raw_samples=np.zeros((0, 4)), raw_interv=np.zeros((0, 4)),
        n_valid=0, candidate_adjacencies=all_adj, noise_scale=0.1
    )
    assert np.allclose(posterior, 1.0 / 8.0)
    assert prob.shape == (4, 4)


def test_posterior_is_a_valid_probability_distribution():
    all_adj, _ = get_all_4node_topologies()
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(50, 4))
    interv = np.zeros((50, 4))
    posterior = compute_hypothesis_posterior(samples, interv, all_adj, noise_scale=0.1)
    assert posterior.shape == (8,)
    assert np.all(posterior >= 0.0)
    assert np.isclose(np.sum(posterior), 1.0)


def _generate_real_samples_under_hypothesis(hyp_idx: int, noise_scale: float, seed: int):
    """Builds a real EnvState for a known hypothesis and draws samples under a mix of
    observational and hard-interventional conditions, using the actual SCM simulation
    code (src/scm.py) -- not a hand-rolled synthetic generator -- so this is a genuine
    end-to-end check of whether the estimator can recover a real generative process."""
    key = jax.random.PRNGKey(seed)
    k_params, k_env, *step_keys = jax.random.split(key, 10)

    all_adj, all_orders = get_all_4node_topologies()
    adjacency = all_adj[hyp_idx]
    order = all_orders[hyp_idx]

    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR),
                        noise_type=int(NoiseType.GAUSSIAN), noise_scale=noise_scale)
    scm_params = generate_scm_params(k_params, adjacency, int(MechanismType.LINEAR))

    agent_masks = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    budgets = np.array([20.0, 20.0])
    state = init_env(k_env, config, adjacency, scm_params, order, agent_masks, budgets, capacity=1)

    sample_count = 150
    all_samples, all_interv = [], []

    # One observational batch, then a hard intervention on each node in turn -- enough
    # coverage for every node's parents (under the true hypothesis) to be identifiable.
    conditions = [None, 0, 1, 2, 3]
    for i, target in enumerate(conditions):
        mask = np.zeros(4)
        types = np.full(4, int(InterventionType.HARD), dtype=np.int32)
        values = np.zeros(4)
        if target is not None:
            mask[target] = 1.0
            values[target] = 1.5  # arbitrary hard-set value
        spec = InterventionSpec(mask=mask, type=types, value=values)
        samples = np.array(sample_scm(step_keys[i], state, config, sample_count, spec))
        all_samples.append(samples)
        all_interv.append(np.broadcast_to(mask, (sample_count, 4)).copy())

    return np.concatenate(all_samples, axis=0), np.concatenate(all_interv, axis=0), all_adj


def test_posterior_concentrates_on_the_true_hypothesis_given_real_scm_data():
    """The core sanity check: fed genuine data generated from a KNOWN hypothesis (via the
    real SCM code), the estimator should assign that hypothesis by far the highest
    posterior probability, given enough samples and a small noise scale."""
    noise_scale = 0.1
    for true_idx in [0, 2, 5, 7]:  # spot-check a few of the 8, not just index 0
        samples, interv, all_adj = _generate_real_samples_under_hypothesis(true_idx, noise_scale, seed=true_idx)
        posterior = compute_hypothesis_posterior(samples, interv, all_adj, noise_scale)
        assert np.argmax(posterior) == true_idx, (
            f"hypothesis {true_idx}: posterior did not peak at the true hypothesis "
            f"(got argmax={np.argmax(posterior)}, posterior={posterior})"
        )
        assert posterior[true_idx] > 0.9, (
            f"hypothesis {true_idx}: true-hypothesis posterior only {posterior[true_idx]:.3f}, "
            f"expected strong concentration given {samples.shape[0]} samples at noise_scale={noise_scale}"
        )


def test_bayes_optimal_predict_output_is_a_valid_convex_combination():
    all_adj, _ = get_all_4node_topologies()
    samples, interv, _ = _generate_real_samples_under_hypothesis(3, 0.1, seed=42)
    prob, posterior = bayes_optimal_predict(samples, interv, samples.shape[0], all_adj, noise_scale=0.1)
    assert prob.shape == (4, 4)
    assert np.all(prob >= -1e-6) and np.all(prob <= 1.0 + 1e-6)
    assert np.isclose(np.sum(posterior), 1.0)
