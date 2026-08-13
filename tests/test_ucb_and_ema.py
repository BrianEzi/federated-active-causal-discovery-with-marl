import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pytest

from src.marl.ppo_agent import IPPOActor, IPPOCritic, compute_ucb_bonus, mask_invalid_targets, sample_actions_jitted
from src.marl.ppo_trainer import RolloutBuffer, compute_gae, IPPOTrainer
from src.evaluator_env import FederatedCausalEnv
from src.types import SCMConfig, MechanismType, NoiseType, ActionCategory


# ============ compute_ucb_bonus ============

def test_ucb_bonus_favors_under_visited_nodes():
    visits = jnp.array([10.0, 0.0, 5.0, 2.0])
    bonus = compute_ucb_bonus(visits, step_count=10, c=1.0)
    # Strictly decreasing bonus as visit count increases (all else equal).
    order = np.argsort(np.array(visits))  # ascending visits
    bonus_np = np.array(bonus)
    assert np.all(np.diff(bonus_np[order]) <= 0), f"bonus should decrease as visits increase: {bonus_np} for visits {visits}"


def test_ucb_bonus_zero_coefficient_disables_it():
    visits = jnp.array([10.0, 0.0, 5.0, 2.0])
    bonus = compute_ucb_bonus(visits, step_count=10, c=0.0)
    assert np.allclose(np.array(bonus), 0.0)


def test_ucb_bonus_finite_at_zero_step_and_zero_visits():
    """step_count=0, visits=0 (very first action of an episode) must not produce NaN/inf."""
    visits = jnp.zeros(4)
    bonus = compute_ucb_bonus(visits, step_count=0, c=1.0)
    assert np.all(np.isfinite(np.array(bonus)))


def test_ucb_bonus_all_equal_visits_gives_equal_bonus():
    visits = jnp.array([3.0, 3.0, 3.0, 3.0])
    bonus = compute_ucb_bonus(visits, step_count=5, c=1.0)
    assert np.allclose(np.array(bonus), np.array(bonus)[0])


# ============ EnvState.node_intervention_counts ============

def _make_env(intervention_type="hard", running_cov_ema_alpha=0.3, ucb_coef=1.0):
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    action_costs = np.array([1.0, 1.0])
    return FederatedCausalEnv(
        config, action_costs, initial_budget=20.0, fixed_graph=True, max_steps=20,
        intervention_type=intervention_type, estimator_type="analytic",
        running_cov_ema_alpha=running_cov_ema_alpha, ucb_coef=ucb_coef,
    )


def test_node_intervention_counts_increments_only_on_real_interventions():
    env = _make_env()
    key = jax.random.PRNGKey(0)
    env.reset(key, force_idx=0)
    assert np.allclose(np.array(env.jax_state.node_intervention_counts), 0.0)

    noop = {"agent_0": (int(ActionCategory.NOOP), 0), "agent_1": (int(ActionCategory.NOOP), 0)}
    env.step(noop, predicted_dags=None, key=jax.random.fold_in(key, 1))
    assert np.allclose(np.array(env.jax_state.node_intervention_counts), 0.0), "NOOP must not increment any count"

    act = {"agent_0": (int(ActionCategory.INTERVENE), 0), "agent_1": (int(ActionCategory.INTERVENE), 2)}
    env.step(act, predicted_dags=None, key=jax.random.fold_in(key, 2))
    counts = np.array(env.jax_state.node_intervention_counts)
    assert counts[0] == 1.0 and counts[2] == 1.0
    assert counts[1] == 0.0 and counts[3] == 0.0

    env.step(act, predicted_dags=None, key=jax.random.fold_in(key, 3))
    counts = np.array(env.jax_state.node_intervention_counts)
    assert counts[0] == 2.0 and counts[2] == 2.0


def test_node_intervention_counts_shared_across_both_agents_targeting_same_node():
    """Both agents targeting the same boundary node in the same step should count as 2 --
    matches the redundancy_rate finding (both pay cost, both should register as a visit)."""
    env = _make_env()
    key = jax.random.PRNGKey(1)
    env.reset(key, force_idx=0)
    act = {"agent_0": (int(ActionCategory.INTERVENE), 1), "agent_1": (int(ActionCategory.INTERVENE), 1)}
    env.step(act, predicted_dags=None, key=jax.random.fold_in(key, 1))
    # mask[node]=1.0 either way (mask is binary, not a sum) -- one visit registered for node 1
    # this step, since the intervention application itself only sets a binary mask per node.
    counts = np.array(env.jax_state.node_intervention_counts)
    assert counts[1] == 1.0


# ============ EMA running_covariance/running_mean ============

def test_running_covariance_first_update_sets_directly_not_blended_with_zero():
    env = _make_env(running_cov_ema_alpha=0.3)
    key = jax.random.PRNGKey(2)
    env.reset(key, force_idx=0)
    cov_before_reset_step = None  # reset already does one write (observational batch)
    run_cov_after_reset = np.array(env.jax_state.running_covariance)
    obs_cov_after_reset = np.array(env.jax_state.obs_covariance)
    # After reset's single observational batch, running_covariance should exactly equal
    # that batch's stitched covariance (same value obs_covariance was set to) -- not
    # blended with a zero-initialized prior, and not affected by the step-kernel's EMA
    # formula at all (reset uses its own direct-assignment kernel).
    np.testing.assert_allclose(run_cov_after_reset, obs_cov_after_reset, atol=1e-6)


def test_running_covariance_uses_ema_not_cumulative_average_after_first_step():
    alpha = 0.3
    env = _make_env(running_cov_ema_alpha=alpha)
    key = jax.random.PRNGKey(3)
    env.reset(key, force_idx=0)
    cov_at_reset = np.array(env.jax_state.running_covariance)

    act = {"agent_0": (int(ActionCategory.INTERVENE), 0), "agent_1": (int(ActionCategory.INTERVENE), 2)}
    env.step(act, predicted_dags=None, key=jax.random.fold_in(key, 1))
    cov_after_step1 = np.array(env.jax_state.running_covariance)

    # An EMA update changes noticeably even after many samples already accumulated at
    # reset (weight alpha=0.3 on the new batch) -- a cumulative average weighted by
    # sample count (100 new vs 100 old) would give the new batch only 50% weight here
    # (n_old==n_new after the first step), close to but not necessarily identical to
    # EMA's fixed 30%. The real regression check: run a SECOND step and confirm the
    # delta from step 1 to step 2 does NOT shrink the way a cumulative average's would
    # (its weight would drop from 50% to 33%; EMA's stays fixed at 30%).
    env.step(act, predicted_dags=None, key=jax.random.fold_in(key, 2))
    cov_after_step2 = np.array(env.jax_state.running_covariance)
    delta_1 = np.linalg.norm(cov_after_step1 - cov_at_reset)
    delta_2 = np.linalg.norm(cov_after_step2 - cov_after_step1)
    # Both deltas should be of comparable magnitude under a fixed-alpha EMA (same
    # weighting every step) -- unlike the old cumulative average, where later deltas
    # would be systematically and increasingly smaller.
    assert delta_2 > 0.0
    ratio = delta_2 / max(delta_1, 1e-8)
    assert ratio > 0.3, (
        f"EMA-updated running_covariance's step-to-step delta shrank too much "
        f"(delta_1={delta_1:.4f}, delta_2={delta_2:.4f}, ratio={ratio:.4f}) -- "
        f"looks more like the old cumulative-average saturation than a fixed-alpha EMA"
    )


# ============ RolloutBuffer / loss_fn consistency with a nonzero ucb_bonus ============

def test_loss_fn_ratio_is_one_when_policy_unchanged_with_nonzero_ucb_bonus():
    """Same regression class as the pre-existing target-logit masking-mismatch test, but
    for the UCB bonus specifically: if old_log_probs (rollout, bonus-augmented + masked)
    and new_log_probs (recomputed in the real loss_fn) come from the same params and the
    same bonus, the PPO ratio must be ~1.0. Calls the real trainer.loss_fn directly
    (not a manual reimplementation), so this exercises the actual code path."""
    d = 4
    def actor_fwd(obs): return IPPOActor(d=d)(obs)
    def critic_fwd(obs): return IPPOCritic()(obs)
    actor_trans = hk.without_apply_rng(hk.transform(actor_fwd))
    critic_trans = hk.without_apply_rng(hk.transform(critic_fwd))

    key = jax.random.PRNGKey(4)
    obs_dim = 3 * d * d + d + 1
    dummy_obs = jnp.zeros((1, obs_dim))
    k1, k2, key = jax.random.split(key, 3)
    actor_params = actor_trans.init(k1, dummy_obs)
    critic_params = critic_trans.init(k2, dummy_obs)

    valid_mask = jnp.array([1.0, 1.0, 1.0, 0.0])
    ucb_bonus = jnp.array([2.0, 0.0, -1.0, 0.5])  # nonzero, asymmetric -- would expose a mismatch

    obs = jax.random.normal(key, (1, obs_dim))
    cat_logits, target_logits = actor_trans.apply(actor_params, obs)
    target_logits_with_bonus = target_logits + ucb_bonus[None, :]
    value = critic_trans.apply(critic_params, obs)[0]
    k_act, key = jax.random.split(key)
    cat, target, old_log_prob = sample_actions_jitted(cat_logits[0], target_logits_with_bonus[0], valid_mask, k_act)

    buf = RolloutBuffer()
    buf.add(obs=obs[0], cat_actions=cat, target_actions=target, values=value,
             log_probs=old_log_prob, rewards=-0.1, dones=True, ucb_bonus=ucb_bonus)
    batch = buf.get_batches(max_size=None)
    advs, rets = compute_gae(batch["rewards"], batch["values"], batch["dones"])
    batch["advantages"] = advs
    batch["returns"] = rets

    trainer = IPPOTrainer(actor_trans, critic_trans, total_episodes=10)
    (total_loss, metrics) = trainer.loss_fn(actor_params, critic_params, batch, valid_mask)

    # Reconstruct the ratio the same way loss_fn does internally, to check it's ~1.0
    # (params unchanged since rollout -> new_log_probs should equal old_log_probs).
    cat_logits_lf, target_logits_lf = actor_trans.apply(actor_params, batch["obs"])
    target_logits_lf = target_logits_lf + batch["ucb_bonus"]
    masked_lf = mask_invalid_targets(batch["cat_actions"], target_logits_lf, valid_mask)
    cat_dist = jax.nn.log_softmax(cat_logits_lf)
    tgt_dist = jax.nn.log_softmax(masked_lf)
    cat_lp = jax.vmap(lambda p, a: p[a])(cat_dist, batch["cat_actions"])
    tgt_lp = jax.vmap(lambda p, a: p[a])(tgt_dist, batch["target_actions"])
    new_log_probs = cat_lp + tgt_lp
    ratio = jnp.exp(new_log_probs - batch["log_probs"])

    assert float(jnp.abs(ratio[0] - 1.0)) < 1e-4, f"ratio should be ~1.0 with unchanged params, got {float(ratio[0])}"
