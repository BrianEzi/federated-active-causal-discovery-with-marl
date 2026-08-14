import numpy as np

from src.marl.oracle_policy import (
    compute_reachability, expected_discrimination, oracle_best_targets, score_agent_choice
)
from src.generators import get_all_4node_topologies


def test_compute_reachability_on_a_simple_chain():
    # 0 -> 1 -> 2 -> 3
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 2] = adj[2, 3] = 1.0
    reach = compute_reachability(adj)
    assert list(reach[0]) == [False, True, True, True]
    assert list(reach[1]) == [False, False, True, True]
    assert list(reach[3]) == [False, False, False, False]  # sink: reaches nothing


def test_compute_reachability_on_a_fork():
    # 1 -> 0, 1 -> 2, 2 -> 3  (fork at node 1)
    adj = np.zeros((4, 4))
    adj[1, 0] = adj[1, 2] = adj[2, 3] = 1.0
    reach = compute_reachability(adj)
    assert reach[1, 0] and reach[1, 2] and reach[1, 3]  # node 1 reaches everything downstream
    assert not reach[0].any()  # node 0 is a sink


def test_fully_concentrated_posterior_gives_zero_score_everywhere():
    """If only one hypothesis remains possible, no intervention can discriminate between
    hypotheses (there's nothing left to discriminate) -- every node's score must be 0."""
    all_adj, _ = get_all_4node_topologies()
    H = all_adj.shape[0]
    posterior = np.zeros(H)
    posterior[3] = 1.0
    scores = expected_discrimination(posterior, np.array(all_adj))
    assert np.allclose(scores, 0.0)


def test_uniform_posterior_over_real_topologies_gives_positive_scores():
    """With genuine uncertainty over the real 8-topology hypothesis space and a uniform
    prior, at least some nodes should have positive discriminating power (the 8
    topologies really do differ in their reachability structure)."""
    all_adj, _ = get_all_4node_topologies()
    H = all_adj.shape[0]
    posterior = np.full(H, 1.0 / H)
    scores = expected_discrimination(posterior, np.array(all_adj))
    assert np.all(scores >= 0.0)
    assert np.any(scores > 0.0)


def test_discrimination_score_prefers_the_node_where_hypotheses_disagree():
    """Hand-constructed: two hypotheses that agree on node 0's reachability but disagree
    on node 1's -- node 1 must score higher than node 0."""
    d = 4
    h1 = np.zeros((d, d)); h1[0, 2] = 1.0; h1[1, 3] = 1.0   # 0->2, 1->3
    h2 = np.zeros((d, d)); h2[0, 2] = 1.0; h2[1, 2] = 1.0   # 0->2, 1->2 (node 1's reach differs from h1)
    candidates = np.stack([h1, h2])
    posterior = np.array([0.5, 0.5])
    scores = expected_discrimination(posterior, candidates)
    assert scores[1] > scores[0]
    assert np.isclose(scores[0], 0.0)  # node 0's reachability is identical under both


def test_oracle_best_targets_respects_valid_mask():
    all_adj, _ = get_all_4node_topologies()
    H = all_adj.shape[0]
    posterior = np.full(H, 1.0 / H)
    scores = expected_discrimination(posterior, np.array(all_adj))
    best_node_unrestricted = int(np.argmax(scores))

    # Restrict to exclude the unrestricted-best node -- oracle must pick among what's left.
    valid_mask = np.ones(4)
    valid_mask[best_node_unrestricted] = 0.0
    _, best_targets = oracle_best_targets(posterior, np.array(all_adj), valid_mask=valid_mask)
    assert not best_targets[best_node_unrestricted]
    assert best_targets.sum() >= 1


def test_score_agent_choice_optimal_and_suboptimal():
    all_adj, _ = get_all_4node_topologies()
    H = all_adj.shape[0]
    posterior = np.full(H, 1.0 / H)
    scores = expected_discrimination(posterior, np.array(all_adj))
    best_node = int(np.argmax(scores))
    worst_node = int(np.argmin(scores))

    result_best = score_agent_choice(best_node, posterior, np.array(all_adj))
    assert result_best["is_optimal"] == 1.0
    assert result_best["regret"] == 0.0
    assert np.isclose(result_best["normalized_score"], 1.0)

    if scores[worst_node] < scores[best_node]:
        result_worst = score_agent_choice(worst_node, posterior, np.array(all_adj))
        assert result_worst["is_optimal"] == 0.0
        assert result_worst["regret"] > 0.0
        assert result_worst["normalized_score"] < 1.0


def test_score_agent_choice_degenerate_case_all_options_equally_uninformative():
    """When the posterior is fully concentrated (best_score ~ 0, every option equally
    useless), the agent must not be penalized for any particular choice."""
    all_adj, _ = get_all_4node_topologies()
    H = all_adj.shape[0]
    posterior = np.zeros(H)
    posterior[0] = 1.0
    result = score_agent_choice(2, posterior, np.array(all_adj))
    assert result["is_optimal"] == 1.0
    assert np.isclose(result["normalized_score"], 1.0)
    assert result["regret"] == 0.0
