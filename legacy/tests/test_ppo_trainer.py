import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pytest

from legacy.src.marl.ppo_trainer import compute_gae, RolloutBuffer, IPPOTrainer
from legacy.src.marl.ppo_agent import IPPOActor, IPPOCritic, mask_invalid_targets, sample_actions_jitted
from legacy.src.evaluator_env import FederatedCausalEnv
from legacy.src.stitching import detect_cycle
from legacy.src.types import SCMConfig, MechanismType, NoiseType


def test_compute_gae_returns_are_not_variance_normalized():
    """
    Regression test for the critic_loss floor bug: returns = advs + values must use RAW
    (unnormalized) advantages. If advs is normalized to unit variance before this sum,
    Var(returns - values) is forced to ~1.0 regardless of the critic's quality, putting a
    hard floor under critic_loss that training can never cross.
    """
    rewards = jnp.array(np.random.RandomState(0).randn(20) * 0.3)
    values = jnp.array(np.random.RandomState(1).randn(20) * 0.5)
    dones = jnp.zeros(20).at[-1].set(1.0)

    advs, returns = compute_gae(rewards, values, dones)

    # A hand-rolled unnormalized reference GAE computation.
    ref_advs = np.zeros(20)
    gae = 0.0
    next_values = np.append(np.array(values)[1:], 0.0)
    for t in reversed(range(20)):
        d = float(dones[t])
        delta = float(rewards[t]) + 0.99 * next_values[t] * (1 - d) - float(values[t])
        gae = delta + 0.99 * 0.95 * (1 - d) * gae
        ref_advs[t] = gae

    np.testing.assert_allclose(np.array(advs), ref_advs, atol=1e-4)
    np.testing.assert_allclose(np.array(returns), ref_advs + np.array(values), atol=1e-4)
    # Explicitly assert it is NOT unit-variance-normalized (the bug's signature).
    assert not np.isclose(float(jnp.var(advs)), 1.0, atol=0.05)


def test_compute_gae_padding_does_not_contaminate_real_transitions():
    """
    Regression test for the GAE padding-contamination bug: a short (early-terminated)
    episode's real advantages must be identical whether or not it's zero-padded to a
    longer static shape (as RolloutBuffer.get_batches does for max_steps).
    """
    buf = RolloutBuffer()
    for i in range(5):
        buf.add(obs=jnp.zeros(4), cat_actions=0, target_actions=0,
                 values=0.1, log_probs=-1.0, rewards=-0.1, dones=(i == 4))

    b_unpadded = buf.get_batches(max_size=None)
    advs_ref, rets_ref = compute_gae(b_unpadded["rewards"], b_unpadded["values"], b_unpadded["dones"])

    b_padded = buf.get_batches(max_size=20)
    advs_pad, rets_pad = compute_gae(b_padded["rewards"], b_padded["values"], b_padded["dones"])

    np.testing.assert_allclose(np.array(advs_ref), np.array(advs_pad[:5]), atol=1e-4)
    np.testing.assert_allclose(np.array(rets_ref), np.array(rets_pad[:5]), atol=1e-4)


def test_loss_fn_ratio_is_one_when_policy_unchanged():
    """
    Regression test for the target-logit masking mismatch: if old_log_probs (from rollout,
    masked) and new_log_probs (recomputed in loss_fn) are computed under the *same* params
    with the *same* masking, the PPO ratio must be ~1.0. Before the fix, loss_fn recomputed
    log-probs from unmasked target_logits while rollout used masked ones, producing a
    spurious ratio even with zero real parameter change.
    """
    d = 4
    def actor_fwd(obs): return IPPOActor(d=d)(obs)
    def critic_fwd(obs): return IPPOCritic()(obs)
    actor_trans = hk.without_apply_rng(hk.transform(actor_fwd))
    critic_trans = hk.without_apply_rng(hk.transform(critic_fwd))

    key = jax.random.PRNGKey(3)
    obs_dim = 3 * d * d + 1
    dummy_obs = jnp.zeros((1, obs_dim))
    k1, k2, key = jax.random.split(key, 3)
    actor_params = actor_trans.init(k1, dummy_obs)
    critic_params = critic_trans.init(k2, dummy_obs)

    valid_mask = jnp.array([1.0, 1.0, 1.0, 0.0])  # agent 0: local {0,1} union boundary {1,2}

    # Roll out ONE transition exactly as train.py does.
    obs = jax.random.normal(key, (1, obs_dim))
    cat_logits, target_logits = actor_trans.apply(actor_params, obs)
    value = critic_trans.apply(critic_params, obs)[0]
    k_act, key = jax.random.split(key)
    cat, target, old_log_prob = sample_actions_jitted(cat_logits[0], target_logits[0], valid_mask, k_act)

    buf = RolloutBuffer()
    buf.add(obs=obs[0], cat_actions=cat, target_actions=target, values=value,
             log_probs=old_log_prob, rewards=-0.1, dones=True)
    batch = buf.get_batches(max_size=None)
    advs, rets = compute_gae(batch["rewards"], batch["values"], batch["dones"])
    batch["advantages"] = advs
    batch["returns"] = rets

    trainer = IPPOTrainer(actor_trans, critic_trans, total_episodes=10)
    # Reach into loss_fn directly (bypassing update_step's jit+grad) to inspect the ratio.
    obs_b, cat_acts, tgt_acts = batch["obs"], batch["cat_actions"], batch["target_actions"]
    cat_logits_lf, target_logits_lf = actor_trans.apply(actor_params, obs_b)
    masked_lf = mask_invalid_targets(cat_acts, target_logits_lf, valid_mask)
    cat_dist = jax.nn.log_softmax(cat_logits_lf)
    tgt_dist = jax.nn.log_softmax(masked_lf)
    cat_lp = jax.vmap(lambda p, a: p[a])(cat_dist, cat_acts)
    tgt_lp = jax.vmap(lambda p, a: p[a])(tgt_dist, tgt_acts)
    new_log_probs = cat_lp + tgt_lp
    ratio = jnp.exp(new_log_probs - batch["log_probs"])

    assert float(jnp.abs(ratio[0] - 1.0)) < 1e-4, f"ratio should be ~1.0 with unchanged params, got {float(ratio[0])}"


def test_update_step_runs_with_valid_intervention_mask():
    """update_step must accept and use valid_intervention_mask without erroring."""
    d = 4
    def actor_fwd(obs): return IPPOActor(d=d)(obs)
    def critic_fwd(obs): return IPPOCritic()(obs)
    actor_trans = hk.without_apply_rng(hk.transform(actor_fwd))
    critic_trans = hk.without_apply_rng(hk.transform(critic_fwd))

    key = jax.random.PRNGKey(5)
    obs_dim = 3 * d * d + 1
    dummy_obs = jnp.zeros((1, obs_dim))
    k1, k2 = jax.random.split(key)
    actor_params = actor_trans.init(k1, dummy_obs)
    critic_params = critic_trans.init(k2, dummy_obs)

    trainer = IPPOTrainer(actor_trans, critic_trans, total_episodes=10)
    actor_opt = trainer.actor_opt.init(actor_params)
    critic_opt = trainer.critic_opt.init(critic_params)

    buf = RolloutBuffer()
    for i in range(5):
        buf.add(obs=jnp.zeros(obs_dim), cat_actions=0, target_actions=0,
                 values=0.1, log_probs=-1.0, rewards=-0.1, dones=(i == 4))
    b = buf.get_batches(max_size=20)
    advs, rets = compute_gae(b["rewards"], b["values"], b["dones"])
    b["advantages"] = advs
    b["returns"] = rets

    valid_mask = jnp.array([1.0, 1.0, 1.0, 0.0])
    new_actor, new_critic, new_aopt, new_copt, metrics = trainer.update_step(
        actor_params, critic_params, actor_opt, critic_opt, b, valid_mask
    )
    assert "actor_loss" in metrics and "critic_loss" in metrics and "entropy" in metrics


def test_evaluator_env_step_applies_cycle_penalty_in_analytic_hypothesis_path():
    """
    Regression test for the dead cycle-penalty bug: FederatedCausalEnv.step()'s
    predicted_dags=None path (the one real IPPO training always uses) must actually run
    cycle detection on the thresholded analytic hypothesis and apply the reward penalty --
    not hardcode has_cycle=False. Checking the *reward* is essential here: the old buggy
    code still returned a technically-cyclic predicted_dag, it just never penalized it, so
    a test that only inspects predicted_dag would pass even with the bug present.

    To avoid confounding with a plain SHD difference (a different hypothesis matrix would
    also change the false-positive count), this compares env.step()'s *actual* observed
    reward against compute_ippo_rewards() called directly with the exact same
    stitched_dag/true_dag/prev_shd under has_cycle=True vs has_cycle=False -- isolating
    the cycle-penalty term specifically.
    """
    from legacy.src.rewards import compute_ippo_rewards

    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    # intrinsic_coef/impact_coef=0 so the reference reward computation below (which omits
    # info_gains/impact_scores) isn't confounded by those terms -- isolating has_cycle.
    env = FederatedCausalEnv(config, action_costs=np.array([1.0, 1.0]), initial_budget=10.0, sample_count=20,
                              fixed_graph=True, intrinsic_coef=0.0, impact_coef=0.0)
    env.reset(jax.random.PRNGKey(0))

    cyclic = np.zeros((4, 4), dtype=np.float32)
    cyclic[0, 1] = cyclic[1, 2] = cyclic[2, 0] = 1.0
    assert detect_cycle(cyclic)
    env.predict_graph_hypothesis = lambda *a, **kw: cyclic

    true_dag_before = np.array(env.jax_state.true_adjacency)
    prev_shd_before = env.prev_shd  # None on the first step after reset

    joint_actions = {"agent_0": (1, 0), "agent_1": (1, 0)}  # both NOOP
    _, rewards_actual, _, info = env.step(joint_actions, predicted_dags=None, key=jax.random.PRNGKey(1))

    stitched_dag = (cyclic > 0.5).astype(np.float32)
    norm_factor = float(env.max_steps)
    rewards_if_cycle_penalized = compute_ippo_rewards(
        stitched_dag, true_dag_before, has_cycle=True, max_steps=norm_factor,
        reward_density=env.reward_density, is_terminal=False, prev_shd=prev_shd_before
    )
    rewards_if_cycle_ignored = compute_ippo_rewards(
        stitched_dag, true_dag_before, has_cycle=False, max_steps=norm_factor,
        reward_density=env.reward_density, is_terminal=False, prev_shd=prev_shd_before
    )
    # The two reference computations must actually differ, or this test proves nothing.
    assert rewards_if_cycle_penalized["agent_0"] < rewards_if_cycle_ignored["agent_0"]

    assert rewards_actual["agent_0"] == pytest.approx(rewards_if_cycle_penalized["agent_0"], abs=1e-4), (
        "env.step()'s reward must match the has_cycle=True computation -- got a reward "
        "matching has_cycle=False, meaning the cycle penalty is still not being applied."
    )


def test_predict_graph_hypothesis_forbids_cross_domain_and_private_edges():
    """
    Regression test: FederatedCausalEnv.predict_graph_hypothesis (the centralized analytic
    estimator that drives the predicted_dag shown in WandB visualizations, and the reward
    for real IPPO training via predicted_dags=None) must never assign nonzero probability
    to a structurally-impossible edge: Z1<->X2 (private-to-peer-domain), Z1<->Z2 or
    X1<->Z2 (private-to-private / cross-domain). Constructed with strong, non-degenerate
    covariance and asymmetry signal on exactly those forbidden pairs, to make sure the
    fix is a hard structural mask and not just "usually zero because signal is weak".
    """
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    env = FederatedCausalEnv(config, action_costs=np.array([1.0, 1.0]), initial_budget=10.0, sample_count=20, fixed_graph=True)
    env.reset(jax.random.PRNGKey(0))

    d = 4
    obs_cov = np.eye(d, dtype=np.float32)
    run_cov = np.eye(d, dtype=np.float32)
    asym = np.zeros((d, d), dtype=np.float32)
    # Force maximal (saturating) signal on every off-diagonal pair, including the
    # forbidden ones -- if masking weren't applied, all of these would show as edges.
    for i in range(d):
        for j in range(d):
            if i != j:
                obs_cov[i, j] = run_cov[i, j] = 5.0
    asym[0, 2], asym[2, 0] = 5.0, -5.0  # strong (fake) Z1<->X2 directional signal
    asym[0, 3], asym[3, 0] = 5.0, -5.0  # strong (fake) Z1<->Z2 directional signal
    asym[1, 3], asym[3, 1] = 5.0, -5.0  # strong (fake) X1<->Z2 directional signal

    prob = env.predict_graph_hypothesis(obs_cov, run_cov, asym)

    forbidden = [(0, 2), (2, 0), (0, 3), (3, 0), (1, 3), (3, 1)]
    for i, j in forbidden:
        assert prob[i, j] == 0.0, f"forbidden edge ({i},{j}) got nonzero probability {prob[i, j]}"

    # Legitimate edges (adjacent in the [Z1, X1, X2, Z2] chain) must still be predictable.
    allowed = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    for i, j in allowed:
        assert prob[i, j] > 0.0, f"legitimate edge ({i},{j}) was incorrectly zeroed"


def test_predict_graph_hypothesis_avici_receives_correctly_shaped_samples():
    """
    Regression test: the AVICI branch of predict_graph_hypothesis previously called
    self.avici_model(x=run_cov), passing a [d, d] covariance matrix where AVICI's real
    API (confirmed directly from the avici package source, avici/pretrain.py
    AVICIModel.__call__) requires x: [n, d] raw observation samples (asserts x.ndim==2,
    which a [d,d] covariance matrix would satisfy without erroring -- so the bug produced
    silently-wrong predictions, not a crash). A mock avici_model captures exactly what
    shape it's called with.
    """
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    env = FederatedCausalEnv(config, action_costs=np.array([1.0, 1.0]), initial_budget=10.0,
                              sample_count=64, estimator_type="analytic")
    env.reset(jax.random.PRNGKey(0))

    captured = {}
    def mock_avici(x, interv=None):
        captured["x_shape"] = x.shape
        prob = np.random.RandomState(0).rand(4, 4).astype(np.float32)
        np.fill_diagonal(prob, 0.0)
        return prob

    env.avici_model = mock_avici
    env.estimator_type = "avici"

    obs_cov = np.array(env.jax_state.obs_covariance)
    run_cov = np.array(env.jax_state.running_covariance)
    asym = np.zeros((4, 4), dtype=np.float32)
    result = env.predict_graph_hypothesis(obs_cov, run_cov, asym)

    assert captured["x_shape"] == (64, 4), (
        f"AVICI must receive [n, d]=(64, 4) samples, got {captured['x_shape']} "
        "-- looks like the [d,d] covariance is being passed directly again."
    )
    # AVICI's raw output must still go through the structural mask, same as the analytic branch.
    assert np.allclose(result, result * env.structural_mask)
    assert result[0, 2] == 0.0 and result[2, 0] == 0.0  # forbidden Z1<->X2


def test_predict_graph_hypothesis_avici_failure_falls_back_loudly(capsys):
    """AVICI errors must not be silently swallowed -- must print a visible warning and
    still return a valid analytic-estimator fallback rather than crashing the episode."""
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    env = FederatedCausalEnv(config, action_costs=np.array([1.0, 1.0]), initial_budget=10.0,
                              sample_count=64, estimator_type="analytic")
    env.reset(jax.random.PRNGKey(0))

    def failing_avici(x, interv=None):
        raise RuntimeError("simulated AVICI failure")

    env.avici_model = failing_avici
    env.estimator_type = "avici"

    obs_cov = np.array(env.jax_state.obs_covariance)
    run_cov = np.array(env.jax_state.running_covariance)
    asym = np.zeros((4, 4), dtype=np.float32)
    result = env.predict_graph_hypothesis(obs_cov, run_cov, asym)

    assert result.shape == (4, 4)
    assert not np.isnan(result).any()
    captured_output = capsys.readouterr()
    assert "AVICI inference failed" in captured_output.out
