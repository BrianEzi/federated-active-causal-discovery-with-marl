import numpy as np
import jax
import pytest

from legacy.src.types import ActionCategory, SCMConfig, MechanismType, NoiseType
from legacy.src.baselines import RandomAgent, RoundRobinAgent, VanillaAgent, estimate_graph_from_obs
from legacy.src.evaluator_env import FederatedCausalEnv

ALL_AGENT_CLASSES = [RandomAgent, RoundRobinAgent, VanillaAgent]

def _dummy_obs(d: int = 4) -> np.ndarray:
    d2 = d * d
    return np.zeros(3 * d2 + 1, dtype=np.float32)

@pytest.mark.parametrize("agent_cls", ALL_AGENT_CLASSES)
@pytest.mark.parametrize("agent_id", [0, 1])
def test_baseline_agents_act_returns_valid_action_category(agent_cls, agent_id):
    """
    Regression test: baselines.py must only ever reference the current 2-category
    ActionCategory (INTERVENE, NOOP). Previously RandomAgent/RoundRobinAgent/VanillaAgent
    still referenced the removed LOCAL_INTERVENTION/PEER_REQUEST members and crashed with
    AttributeError on the very first act() call.
    """
    agent = agent_cls(agent_id=agent_id, d=4)
    obs = _dummy_obs()
    for _ in range(20):
        (cat, target), graph_pred = agent.act(obs)
        assert cat in (int(ActionCategory.INTERVENE), int(ActionCategory.NOOP))
        assert 0 <= target <= 3
        assert graph_pred.shape == (4, 4)

def test_random_agent_intervene_targets_within_authority():
    agent = RandomAgent(agent_id=0, d=4)
    obs = _dummy_obs()
    seen_targets = set()
    for _ in range(200):
        (cat, target), _ = agent.act(obs)
        if cat == int(ActionCategory.INTERVENE):
            seen_targets.add(target)
    # Agent 0's valid targets: local domain {0, 1} union boundary {1, 2} = {0, 1, 2}.
    assert seen_targets.issubset({0, 1, 2})
    assert 3 not in seen_targets  # Z2 (agent 1's private node) must never be targetable

def test_round_robin_agent_cycles_through_all_valid_targets():
    agent = RoundRobinAgent(agent_id=0, d=4)
    obs = _dummy_obs()
    targets = []
    for _ in range(6):
        (cat, target), _ = agent.act(obs)
        assert cat == int(ActionCategory.INTERVENE)
        targets.append(target)
    # {0, 1} local ∪ {1, 2} boundary = {0, 1, 2}, cycled deterministically, twice over.
    assert targets == [0, 1, 2, 0, 1, 2]

def test_vanilla_agent_action_space_combinations():
    agent = VanillaAgent(agent_id=0, d=4)
    obs = _dummy_obs()
    seen = set()
    for _ in range(500):
        (cat, target), _ = agent.act(obs)
        seen.add((cat, target))
    expected = {
        (int(ActionCategory.INTERVENE), 0),  # Z1
        (int(ActionCategory.INTERVENE), 1),  # X1
        (int(ActionCategory.INTERVENE), 2),  # peer boundary X2
        (int(ActionCategory.NOOP), 0),
    }
    assert seen == expected

def test_estimate_graph_from_obs_enforces_edge_authority_domain():
    """estimate_graph_from_obs must never let a private node claim an edge into the
    peer's domain, even when strongly (spuriously) correlated with it."""
    d = 4
    cov_obs = np.zeros((d, d))
    np.fill_diagonal(cov_obs, 1.0)
    cov_obs[0, 2] = cov_obs[2, 0] = 0.9  # Z1 <-> X2: forbidden cross-domain correlation
    cov_obs[0, 1] = cov_obs[1, 0] = 0.9  # Z1 <-> X1: legitimate local correlation
    cov_obs[1, 2] = cov_obs[2, 1] = 0.9  # X1 <-> X2: legitimate boundary correlation
    asym = np.zeros((d, d))
    obs = np.concatenate([cov_obs.flatten(), cov_obs.flatten(), asym.flatten(), [10.0]])

    pred_0 = estimate_graph_from_obs(obs, d, agent_id=0, threshold=0.25)
    assert pred_0[0, 2] == 0.0 and pred_0[2, 0] == 0.0
    assert pred_0[0, 1] > 0.0 or pred_0[1, 0] > 0.0
    assert pred_0[1, 2] > 0.0 or pred_0[2, 1] > 0.0

@pytest.mark.parametrize("agent_cls", ALL_AGENT_CLASSES)
def test_baseline_agent_full_env_step_integration(agent_cls):
    """
    Integration-level regression test: exercises the exact call path that crashed
    (src.train.py's baseline-agent branch calling FederatedCausalEnv.step with joint
    actions from RandomAgent/RoundRobinAgent/VanillaAgent). A unit-level mock would not
    have caught the original AttributeError, since it only reproduces with the real
    ActionCategory enum flowing all the way through env.step's validity checks.
    """
    config = SCMConfig(d=4, K=2, mechanism_type=int(MechanismType.LINEAR), noise_type=int(NoiseType.GAUSSIAN))
    env = FederatedCausalEnv(config, action_costs=np.array([1.0, 1.0]), initial_budget=10.0, sample_count=20, fixed_graph=True)
    key = jax.random.PRNGKey(0)
    obs_dict, info = env.reset(key)

    agents = [agent_cls(agent_id=k, d=4) for k in range(2)]
    joint_actions = {}
    predicted_dags = {}
    for k in range(2):
        act, g_pred = agents[k].act(obs_dict[f"agent_{k}"])
        joint_actions[f"agent_{k}"] = act
        predicted_dags[f"agent_{k}"] = g_pred

    k_step, key = jax.random.split(key)
    obs_dict, rewards, done, step_info = env.step(joint_actions, predicted_dags, k_step)

    assert "agent_0" in rewards and "agent_1" in rewards
    assert isinstance(done, bool)
