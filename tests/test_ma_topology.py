"""The two-agent node partition, its edge mask, and the confounding measurement.

The confounding tests come first and are the most detailed, because that measurement is
what block 5 uses to decide whether per-agent DAG posteriors are defensible at all. A bug
that *understated* confounding would produce a comfortable number and quietly license a
misspecified model for the rest of the project.
"""
import numpy as np
import pytest

from ma.confounding import ambiguity_location, is_confounded, latent_projection_pairs, \
    measure_topology
from ma.topology import T1, T2, T3, TOPOLOGIES, Topology, edge_class, masked_indices
from sa.graphs import build_graph_space, is_acyclic


@pytest.fixture(scope="module")
def space():
    return build_graph_space(6, fast=True)


# --------------------------------------------------------------------------------------
# The mask
# --------------------------------------------------------------------------------------

def test_cross_private_edges_are_forbidden_in_both_directions():
    """The defining constraint: neither agent can ever observe such an edge, so no data
    from anyone bears on it and it would be permanently unidentifiable."""
    allowed = T1.allowed_edges()
    for u in T1.a_private:
        for v in T1.b_private:
            assert not allowed[u, v] and not allowed[v, u]


def test_boundary_and_interior_edges_remain_allowed():
    allowed = T1.allowed_edges()
    assert allowed[0, 1] and allowed[2, 3]          # interior, within one agent
    assert allowed[0, 4] and allowed[4, 0]          # private <-> exposed
    assert allowed[4, 5] and allowed[5, 4]          # exposed <-> exposed


def test_no_self_loops_are_allowed():
    for topology in TOPOLOGIES:
        assert not np.any(np.diag(topology.allowed_edges()))


def test_t3_removes_private_parents_of_exposed_nodes():
    """T3's whole purpose. If this constraint were not applied, T3 would be T1 and the
    comparison in block 5 would be vacuous."""
    allowed = T3.allowed_edges()
    for u in T3.a_private + T3.b_private:
        for v in T3.exposed:
            assert not allowed[u, v]
            assert allowed[v, u], "exposed -> private must remain, or T3 disconnects"


def test_sampled_graphs_are_acyclic_and_respect_the_mask():
    """Acyclicity comes free from the topological-order construction -- there is no
    rejection step that could distort the prior."""
    rng = np.random.default_rng(0)
    for topology in TOPOLOGIES:
        forbidden = ~topology.allowed_edges()
        for _ in range(200):
            adjacency = topology.sample_dag(rng, p=0.5)
            assert is_acyclic(adjacency)
            assert not (np.asarray(adjacency)[forbidden] > 0).any()


def test_masked_indices_agree_with_the_generator():
    """Two independent routes to "which graphs does this topology permit": filtering the
    enumerated space, and sampling. Every sampled graph must be in the filtered set."""
    d = 6
    space = build_graph_space(d, fast=True)
    indices = set(masked_indices(space, T1).tolist())
    lookup = {space.dags[i].tobytes(): i for i in indices}
    rng = np.random.default_rng(1)
    for _ in range(100):
        adjacency = T1.sample_dag(rng, p=0.5).astype(space.dags.dtype)
        assert adjacency.tobytes() in lookup


def test_agents_see_their_own_private_nodes_and_the_exposed_ones():
    assert T1.observed_by("A") == (0, 1, 4, 5)
    assert T1.observed_by("B") == (2, 3, 4, 5)
    assert T1.hidden_from("A") == (2, 3)
    assert T1.hidden_from("B") == (0, 1)


def test_both_agents_may_intervene_on_the_exposed_nodes():
    """Shared authority is deliberate -- it is the surface coordination happens on."""
    shared = set(T1.may_intervene_on("A")) & set(T1.may_intervene_on("B"))
    assert shared == set(T1.exposed)


def test_edge_class_labels_the_boundary_correctly():
    assert edge_class(T1, 0, 1) == "interior"
    assert edge_class(T1, 0, 4) == "private_exposed"
    assert edge_class(T1, 4, 5) == "exposed_exposed"
    assert edge_class(T1, 0, 2) == "cross_private"


# --------------------------------------------------------------------------------------
# Latent confounding
# --------------------------------------------------------------------------------------

def test_a_hidden_node_with_two_observed_children_confounds():
    adjacency = np.zeros((6, 6), dtype=np.int8)
    adjacency[2, 4] = adjacency[2, 5] = 1
    assert latent_projection_pairs(adjacency, (0, 1, 4, 5), (2, 3)) == [(4, 5)]


def test_confounding_is_detected_through_a_hidden_intermediate():
    """`2 -> 3 -> 4` and `2 -> 5`: node 2 has only ONE observed child, so a
    children-only test would miss this. It is still a confounder of 4 and 5."""
    adjacency = np.zeros((6, 6), dtype=np.int8)
    adjacency[2, 3] = adjacency[3, 4] = adjacency[2, 5] = 1
    assert latent_projection_pairs(adjacency, (0, 1, 4, 5), (2, 3)) == [(4, 5)]


def test_a_hidden_node_with_one_observed_descendant_does_not_confound():
    adjacency = np.zeros((6, 6), dtype=np.int8)
    adjacency[2, 4] = 1
    assert latent_projection_pairs(adjacency, (0, 1, 4, 5), (2, 3)) == []
    assert not is_confounded(adjacency, (0, 1, 4, 5), (2, 3))


def test_an_observed_common_cause_does_not_confound():
    """Confounding is about *hidden* common causes. An observed one is just a fork the
    agent can condition on, and counting it would inflate the measurement."""
    adjacency = np.zeros((6, 6), dtype=np.int8)
    adjacency[0, 4] = adjacency[0, 5] = 1
    assert latent_projection_pairs(adjacency, (0, 1, 4, 5), (2, 3)) == []


def test_a_path_through_an_observed_node_does_not_confound():
    """`2 -> 4 -> 5` gives 2 one observed child; 5 is reached only *through* the observed
    node 4, which blocks the latent path."""
    adjacency = np.zeros((6, 6), dtype=np.int8)
    adjacency[2, 4] = adjacency[4, 5] = 1
    assert latent_projection_pairs(adjacency, (0, 1, 4, 5), (2, 3)) == []


def test_t3_is_confounding_free_by_construction(space):
    """T3's justification. If this ever fails, T3 has no reason to exist."""
    result = measure_topology(space, T3)
    assert result["confounded_either"] == 0.0
    assert result["mean_bidirected"] == 0.0


def test_t1_confounding_is_measured_not_assumed(space):
    """No threshold is asserted -- the value is the finding. What is pinned is that the
    measurement is well formed and that T1 differs from T3, which is the comparison block 5
    exists to make."""
    result = measure_topology(space, T1, max_graphs=20_000)
    assert 0.0 <= result["confounded_either"] <= 1.0
    assert result["confounded_either"] >= result["confounded_a"]
    assert result["confounded_either"] > 0.0, "T1 without confounding would make T3 pointless"


def test_the_two_agents_are_symmetric_under_t1(space):
    """A and B have identical roles in T1, so any asymmetry in the measured rates is a bug
    in the measurement rather than a property of the topology."""
    result = measure_topology(space, T1, max_graphs=20_000)
    assert result["confounded_a"] == pytest.approx(result["confounded_b"], abs=0.02)


# --------------------------------------------------------------------------------------
# Where the ambiguity sits
# --------------------------------------------------------------------------------------

def test_ambiguity_shares_are_a_distribution(space):
    result = ambiguity_location(space, T1, max_graphs=20_000)
    assert sum(result["ambiguous_edge_shares"].values()) == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= result["singleton_fraction"] <= 1.0


def test_ambiguity_never_lands_on_a_forbidden_edge(space):
    """A forbidden edge is absent from every graph, so it cannot vary within a class. A
    non-zero count here would mean the mask and the class grouping disagree."""
    result = ambiguity_location(space, T3, max_graphs=20_000)
    assert "cross_private" not in result["ambiguous_edge_counts"]
