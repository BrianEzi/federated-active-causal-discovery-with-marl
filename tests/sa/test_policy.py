"""Tests for the PPO agent.

Mechanics only -- whether it *learns well* is what `scripts/run_experiment.py` measures
against the pinned criteria, and that is not something a unit test should assert.

`test_observation_features_are_on_a_comparable_scale` encodes a real bug: the budget
feature was a raw count sitting at 20.0 while posterior entries averaged 0.04, a ~500x
mismatch that saturated the tanh trunk and drowned out the belief the agent acts on.
"""
import numpy as np
import pytest
import torch

from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig
from sa.graphs import build_graph_space
from sa.policy import (
    ActorCritic,
    PerNodeActorCritic,
    PPOAgent,
    PPOConfig,
    action_to_env,
)


@pytest.fixture(scope="module")
def space3():
    return build_graph_space(3)


@pytest.fixture(scope="module")
def agent(space3):
    return PPOAgent(EnvConfig(d=3, n_obs=200, budget=10),
                    PPOConfig(total_episodes=64, episodes_per_update=32, seed=0),
                    space=space3)


# --- action mapping -------------------------------------------------------------------

def test_last_action_index_means_pass():
    assert action_to_env(3, 3) == PASS_ACTION
    assert action_to_env(0, 3) == 0
    assert action_to_env(2, 3) == 2


def test_action_space_has_one_slot_per_node_plus_pass(agent):
    assert agent.n_actions == agent.d + 1


# --- network --------------------------------------------------------------------------

def test_network_starts_near_uniform():
    """Small final-layer weights matter: a sharp initial policy never explores."""
    net = ActorCritic(obs_dim=10, n_actions=4)
    logits, _ = net(torch.zeros(10))
    assert float(logits.detach().abs().max()) < 0.1


def test_forward_shapes():
    net = ActorCritic(obs_dim=7, n_actions=4)
    logits, value = net(torch.zeros((5, 7)))
    assert logits.shape == (5, 4) and value.shape == (5,)


# --- observations -----------------------------------------------------------------------

def test_observation_features_are_on_a_comparable_scale(space3):
    """Regression test on the scale mismatch that stopped learning entirely."""
    env = CausalDiscoveryEnv(EnvConfig(d=3, n_obs=200, budget=20), space=space3)
    env.reset(seed=0)
    for kind in ("posterior", "edge_marginals"):
        obs = env.observation(kind)
        assert obs.min() >= -1e-9 and obs.max() <= 1.0 + 1e-9, (
            f"{kind}: features outside [0, 1] ({obs.min():.3f} to {obs.max():.3f}); the "
            f"budget feature was previously a raw count of 20 alongside ~0.04 probabilities."
        )


def test_agent_observation_dim_matches_the_environment(space3):
    for kind in ("posterior", "edge_marginals"):
        a = PPOAgent(EnvConfig(d=3, n_obs=200), PPOConfig(observation=kind, seed=0),
                     space=space3)
        assert a.obs_dim == a.env.observation_dim[kind]


# --- acting ---------------------------------------------------------------------------

def test_deterministic_action_is_reproducible(agent):
    obs = np.zeros(agent.obs_dim, dtype=np.float32)
    first, _, _ = agent.act(obs, deterministic=True)
    second, _, _ = agent.act(obs, deterministic=True)
    assert first == second


def test_sampled_actions_vary(agent):
    obs = np.zeros(agent.obs_dim, dtype=np.float32)
    torch.manual_seed(0)
    actions = {agent.act(obs)[0] for _ in range(60)}
    assert len(actions) > 1, "a near-uniform initial policy must not be deterministic"


def test_as_policy_returns_valid_environment_actions(agent, space3):
    env = CausalDiscoveryEnv(agent.env_config, space=space3)
    result = env.reset(seed=0)
    policy = agent.as_policy(deterministic=True)
    action = policy(env, result)
    assert action == PASS_ACTION or 0 <= action < agent.d


# --- training loop ----------------------------------------------------------------------

def test_training_runs_and_records_history(agent):
    history = agent.train()
    assert history and {"episodes", "solve_rate", "entropy"} <= set(history[0])
    assert np.isfinite(history[-1]["entropy"])


def test_advantages_are_normalised_and_returns_are_not(agent):
    """Returns must stay on the reward scale: normalising them would corrupt the value
    targets, a bug this project has hit before."""
    rewards = np.array([0.1, 0.2, 1.0, -0.05], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    advantages, returns = agent._advantages(rewards, values, dones)
    assert abs(float(advantages.mean())) < 1e-5
    assert abs(float(advantages.std()) - 1.0) < 1e-3
    assert float(returns.max()) > 1.0  # not squashed to zero mean


def test_advantages_do_not_bootstrap_across_episode_boundaries(agent):
    """A terminal step's advantage must not borrow value from the next episode."""
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    done_then_not = agent._advantages(rewards, values, np.array([1.0, 0.0], dtype=np.float32))[1]
    assert done_then_not[0] == pytest.approx(1.0)


# --- the two diagnostic levers ---------------------------------------------------------
#
# Both exist to attack one measured failure: the agent learns not to pass within ~1500
# episodes and then never learns where to intervene, ending at exactly random-policy cost.

def test_removing_pass_shrinks_the_action_space(space3):
    agent = PPOAgent(EnvConfig(d=3, n_obs=200),
                     PPOConfig(allow_pass=False, seed=0), space=space3)
    assert agent.n_actions == agent.d, "pass should be gone, leaving one action per node"


def test_an_agent_without_pass_never_emits_the_pass_action(space3):
    """The under-acting criterion is satisfied by CONSTRUCTION for such an agent, so it
    must not be read as evidence of good behaviour -- this test pins the reason why."""
    agent = PPOAgent(EnvConfig(d=3, n_obs=200),
                     PPOConfig(allow_pass=False, seed=0), space=space3)
    env = CausalDiscoveryEnv(agent.env_config, space=space3)
    env.reset(seed=0)
    policy = agent.as_policy(deterministic=False)
    torch.manual_seed(0)
    assert all(policy(env, None) != PASS_ACTION for _ in range(60))


def test_shaping_potential_is_normalised_and_ordered(space3):
    """phi in [-1, 0], and a sharper belief must score higher than a flat one."""
    agent = PPOAgent(EnvConfig(d=3, n_obs=200), PPOConfig(seed=0), space=space3)

    class R:
        pass
    flat, sharp = R(), R()
    flat.posterior = np.full(space3.n_dags, 1.0 / space3.n_dags)
    sharp.posterior = np.zeros(space3.n_dags); sharp.posterior[4] = 1.0

    assert agent._potential(flat) == pytest.approx(-1.0)
    assert agent._potential(sharp) == pytest.approx(0.0)
    assert -1.0 <= agent._potential(flat) <= agent._potential(sharp) <= 0.0


def test_shaping_telescopes_to_zero_over_an_episode(space3):
    """The property that makes potential-based shaping policy-invariant: summed
    undiscounted over an episode ending in a terminal state, the shaping contributes
    phi(terminal) - phi(start) = -phi(start), never a payoff that depends on the ROUTE
    taken. A shaping term that failed this could invent a better-looking policy that the
    unshaped objective would not prefer (Ng, Harada & Russell 1999)."""
    agent = PPOAgent(EnvConfig(d=3, n_obs=200),
                     PPOConfig(gamma=1.0, shaping_coef=1.0, seed=0), space=space3)

    class R:
        pass
    potentials = []
    for mass in (0.2, 0.5, 0.9):
        r = R(); r.posterior = np.full(space3.n_dags, (1 - mass) / (space3.n_dags - 1))
        r.posterior[0] = mass
        potentials.append(agent._potential(r))

    # gamma = 1, phi(terminal) = 0
    total = sum(b - a for a, b in zip(potentials, potentials[1:] + [0.0]))
    assert total == pytest.approx(-potentials[0])


def test_shaping_off_by_default_leaves_rewards_untouched(space3):
    assert PPOConfig().shaping_coef == 0.0
    assert PPOConfig().allow_pass is True


def test_intervention_counts_can_be_added_to_the_observation(space3):
    """The agent cannot otherwise tell which nodes it has already targeted."""
    plain = CausalDiscoveryEnv(EnvConfig(d=3, n_obs=200), space=space3)
    with_counts = CausalDiscoveryEnv(EnvConfig(d=3, n_obs=200, include_counts=True),
                                     space=space3)
    for env in (plain, with_counts):
        env.reset(seed=0)
    assert with_counts.observation_dim["edge_marginals"] == \
        plain.observation_dim["edge_marginals"] + 3

    for kind in ("posterior", "edge_marginals"):
        assert len(with_counts.observation(kind)) == with_counts.observation_dim[kind]

    # The counts must actually move, and stay in [0, 1] like every other feature.
    before = with_counts.observation("edge_marginals").copy()
    with_counts.step(1)
    after = with_counts.observation("edge_marginals")
    assert not np.array_equal(before[-3:], after[-3:]), "counts did not update"
    assert after.min() >= -1e-9 and after.max() <= 1.0 + 1e-9


# --- the permutation-equivariant architecture -------------------------------------------

def _flat_from_matrix(matrix, budget, counts=None):
    """Pack a [d, d] marginal matrix into the flat observation layout."""
    d = matrix.shape[0]
    off = ~np.eye(d, dtype=bool)
    parts = [matrix[off], np.array([budget])]
    if counts is not None:
        parts.append(counts)
    return np.concatenate(parts).astype(np.float32)


def test_pernode_rebuilds_the_marginal_matrix_correctly():
    """Node i's neighbour pairs must be (i->j, j->i) for its own row and column."""
    d = 4
    net = PerNodeActorCritic(d, hidden=8, include_counts=False, allow_pass=True)
    matrix = np.arange(d * d, dtype=np.float32).reshape(d, d)
    np.fill_diagonal(matrix, 0.0)
    obs = torch.as_tensor(_flat_from_matrix(matrix, 0.5)).unsqueeze(0)

    pairs = net._neighbour_pairs(obs)[0]
    assert pairs.shape == (d, d - 1, 2)
    for i in range(d):
        others = np.arange(d)[np.arange(d) != i]
        np.testing.assert_allclose(pairs[i, :, 0].numpy(), matrix[i, others], atol=1e-5)
        np.testing.assert_allclose(pairs[i, :, 1].numpy(), matrix[others, i], atol=1e-5)


def test_pernode_is_permutation_equivariant():
    """Relabel the nodes and the logits must permute with them.

    This is the property the whole architecture exists for, and it is true of the oracle:
    node identity carries no information, only structure does. The flat MLP cannot express
    it, which is why it must learn each node's scorer separately.
    """
    d, perm = 4, np.array([2, 0, 3, 1])
    torch.manual_seed(0)
    net = PerNodeActorCritic(d, hidden=16, include_counts=False, allow_pass=True)
    rng = np.random.default_rng(0)

    matrix = rng.random((d, d)).astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    permuted = matrix[np.ix_(perm, perm)]

    with torch.no_grad():
        base, base_value = net(torch.as_tensor(_flat_from_matrix(matrix, 0.5)))
        other, other_value = net(torch.as_tensor(_flat_from_matrix(permuted, 0.5)))

    # Node logits permute; the pass logit (last) and the value are invariant.
    np.testing.assert_allclose(other[:d].numpy(), base[:d].numpy()[perm], atol=1e-5)
    assert float(other[d]) == pytest.approx(float(base[d]), abs=1e-5)
    assert float(other_value) == pytest.approx(float(base_value), abs=1e-5)


def test_pernode_parameter_count_does_not_grow_with_d():
    """One shared scorer serves every node, so the same model form carries to d=6."""
    sizes = [sum(p.numel() for p in
                 PerNodeActorCritic(d, hidden=32).parameters()) for d in (4, 5, 6)]
    # Only the input width (2(d-1)+1) changes, so growth is linear and small -- not the
    # quadratic blow-up of a dense layer over d(d-1) inputs mapping to d outputs.
    assert sizes[2] - sizes[1] == sizes[1] - sizes[0]


def test_pernode_shapes_match_the_flat_network(space3):
    for allow_pass in (True, False):
        agent = PPOAgent(EnvConfig(d=3, n_obs=200),
                         PPOConfig(observation="edge_marginals", arch="pernode",
                                   allow_pass=allow_pass, seed=0), space=space3)
        agent.env.reset(seed=0)
        obs = agent.env.observation("edge_marginals")
        logits, value = agent.net(torch.as_tensor(obs, dtype=torch.float32))
        assert logits.shape == (agent.n_actions,)
        assert value.shape == ()


def test_pernode_rejects_the_posterior_observation(space3):
    with pytest.raises(ValueError, match="edge_marginals"):
        PPOAgent(EnvConfig(d=3, n_obs=200),
                 PPOConfig(observation="posterior", arch="pernode", seed=0), space=space3)
