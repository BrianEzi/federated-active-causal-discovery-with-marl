import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from src.types import SCMConfig, MechanismType, NoiseType, ActionCategory, SINGLE_AGENT_LOCAL_MASKS, SINGLE_AGENT_OBS_MASKS
from src.evaluator_env import FederatedCausalEnv
from src.marl.ppo_agent import IPPORNNActor, IPPOActor
from src.rewards import compute_ippo_rewards
from src.marl.oracle_policy import score_agent_action, oracle_best_targets
from src.generators import get_all_4node_topologies


def test_single_agent_env_init_and_step():
    config = SCMConfig(d=4, K=1, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN), noise_scale=0.1)
    action_costs = np.array([1.0])
    env = FederatedCausalEnv(config, action_costs, initial_budget=5.0, fixed_graph=True)
    
    key = jax.random.PRNGKey(42)
    obs_dict, info = env.reset(key, force_idx=0)
    
    assert "agent_0" in obs_dict
    assert "agent_1" not in obs_dict
    assert obs_dict["agent_0"].shape == (env.obs_dim,)
    assert env.agent_masks.shape == (1, 4)
    
    # Step 1: Intervene on Node 1 (X1)
    joint_actions = {"agent_0": (int(ActionCategory.INTERVENE), 1)}
    next_obs, rewards, done, step_info = env.step(joint_actions, predicted_dags=None, key=jax.random.PRNGKey(101))
    
    assert "agent_0" in rewards
    assert not done
    assert env.jax_state.budgets[0] == 4.0  # 5.0 - 1.0


def test_single_agent_actor_forward_pass():
    d = 4
    obs_dim = 3 * d * d + d + 1  # 53 dims
    dummy_obs = jnp.zeros((2, obs_dim))
    
    # Test Feedforward Actor
    def ff_forward(obs):
        return IPPOActor(d=d)(obs)
    ff_trans = jax.jit(hk.without_apply_rng(hk.transform(ff_forward)).apply)
    params = hk.transform(ff_forward).init(jax.random.PRNGKey(0), dummy_obs)
    cat_logits, target_logits = ff_trans(params, dummy_obs)
    
    assert cat_logits.shape == (2, 2)
    assert target_logits.shape == (2, 4)
    
    # Test RNN Actor
    def rnn_forward(obs, state):
        return IPPORNNActor(d=d)(obs, state)
    rnn_trans = jax.jit(hk.without_apply_rng(hk.transform(rnn_forward)).apply)
    init_state = IPPORNNActor.initial_state(2)
    params_rnn = hk.transform(rnn_forward).init(jax.random.PRNGKey(0), dummy_obs, init_state)
    (cat_l, tgt_l), next_state = rnn_trans(params_rnn, dummy_obs, init_state)
    
    assert cat_l.shape == (2, 2)
    assert tgt_l.shape == (2, 4)
    assert next_state.shape == init_state.shape


def test_single_agent_oracle_scoring():
    adjacencies, _ = get_all_4node_topologies()
    candidate_adjacencies = np.array(adjacencies)
    
    # 1. Prior is uniform over 8 graphs
    unif_posterior = np.full(8, 1.0 / 8.0)
    scores, best_targets = oracle_best_targets(unif_posterior, candidate_adjacencies, valid_mask=[1, 1, 1, 1])
    
    # Opening move on boundary nodes (1 or 2) should be optimal
    score_x1 = score_agent_action(int(ActionCategory.INTERVENE), 1, unif_posterior, candidate_adjacencies, valid_mask=[1, 1, 1, 1])
    assert score_x1["is_optimal"] == 1.0
    assert score_x1["action_type"] == "intervene_optimal"
    
    # Playing NOOP at the start should be premature / suboptimal
    score_noop_start = score_agent_action(int(ActionCategory.NOOP), 0, unif_posterior, candidate_adjacencies, valid_mask=[1, 1, 1, 1])
    assert score_noop_start["is_optimal"] == 0.0
    assert score_noop_start["action_type"] == "noop_premature"
    
    # 2. Posterior is fully resolved (only Graph 0 is plausible)
    resolved_posterior = np.zeros(8)
    resolved_posterior[0] = 1.0
    
    # When resolved, NOOP must be the only optimal action
    score_noop_resolved = score_agent_action(int(ActionCategory.NOOP), 0, resolved_posterior, candidate_adjacencies, valid_mask=[1, 1, 1, 1])
    assert score_noop_resolved["is_optimal"] == 1.0
    assert score_noop_resolved["action_type"] == "noop_correct"
    
    # Intervening when resolved should be penalized as unnecessary
    score_int_resolved = score_agent_action(int(ActionCategory.INTERVENE), 1, resolved_posterior, candidate_adjacencies, valid_mask=[1, 1, 1, 1])
    assert score_int_resolved["is_optimal"] == 0.0
    assert score_int_resolved["action_type"] == "intervene_unnecessary"


def test_single_agent_rewards():
    true_dag = np.zeros((4, 4))
    true_dag[0, 1] = 1.0
    true_dag[1, 2] = 1.0
    true_dag[2, 3] = 1.0
    
    # Perfect prediction (SHD=0)
    pred_perfect = true_dag.copy()
    r_perfect = compute_ippo_rewards(
        pred_perfect, true_dag, has_cycle=False,
        num_agents=1, holding_bonus=0.05,
        is_terminal=True, remaining_budgets={"agent_0": 4.0}, budget_bonus_coef=0.1
    )
    # Holding bonus (0.05) + Terminal budget payoff (0.1 * 4.0 = 0.4)
    assert r_perfect["agent_0"] == pytest.approx(0.45)
    
    # Imperfect prediction (SHD=3)
    pred_bad = np.zeros((4, 4))
    r_bad = compute_ippo_rewards(
        pred_bad, true_dag, has_cycle=False,
        num_agents=1, holding_bonus=0.05,
        is_terminal=True, remaining_budgets={"agent_0": 4.0}, budget_bonus_coef=0.1
    )
    # -3.0 edge penalty, no holding bonus, no budget bonus
    assert r_bad["agent_0"] == pytest.approx(-3.0)
