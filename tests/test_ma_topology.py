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
from ma.topology import two_agent, T1, T2, T3, TOPOLOGIES, Topology, edge_class, masked_indices
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
    for u in T1.private[0]:
        for v in T1.private[1]:
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
    for u in T3.private[0] + T3.private[1]:
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
    assert T1.observed_by(0) == (0, 1, 4, 5)
    assert T1.observed_by(1) == (2, 3, 4, 5)
    assert T1.hidden_from(0) == (2, 3)
    assert T1.hidden_from(1) == (0, 1)


def test_both_agents_may_intervene_on_the_exposed_nodes():
    """Shared authority is deliberate -- it is the surface coordination happens on."""
    shared = set(T1.may_intervene_on(0)) & set(T1.may_intervene_on(1))
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

# --------------------------------------------------------------------------------------
# n agents (2026-08-22)
# --------------------------------------------------------------------------------------
#
# The two-agent tests above are the n=2 case and still pass unchanged, which is the gate on
# the refactor. These cover what is genuinely new: three or more agents, overlapping
# visibility, and the edge rule that replaces "no cross-private edges".


def test_the_new_rule_reproduces_the_old_one_exactly_at_two_agents():
    """THE REFACTOR GATE. The jointly-visible rule and the cross-private rule must agree
    bit for bit under a disjoint partition, or every two-agent number ever measured is
    measured on a different hypothesis space than the one now generated."""
    for a, b, exposed, t3 in [((0, 1), (2, 3), (4, 5), False),
                              ((0,), (1,), (2, 3, 4), False),
                              ((0, 1), (2, 3), (4, 5), True)]:
        topo = two_agent("x", a, b, exposed, exposed_have_no_private_parents=t3)
        d = topo.d
        old = ~np.eye(d, dtype=bool)
        for u in a:
            for v in b:
                old[u, v] = old[v, u] = False
        if t3:
            for u in tuple(a) + tuple(b):
                for v in exposed:
                    old[u, v] = False
        assert np.array_equal(topo.allowed_edges(), old), (a, b, exposed, t3)


def test_three_agents_forbid_every_cross_private_edge():
    topo = Topology("3x1", private=((0,), (1,), (2,)), exposed=(3, 4))
    allowed = topo.allowed_edges()
    assert topo.n_agents == 3 and topo.agents == (0, 1, 2)
    for u, v in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
        assert not allowed[u, v], f"{u}->{v} is private-to-private across agents"
    # ...but every private-exposed and exposed-exposed pair survives
    assert allowed[0, 3] and allowed[3, 0] and allowed[3, 4]


def test_hidden_from_is_the_union_of_every_other_private_block():
    """At n>2 a single clamp cleans only PART of what is hidden. This is the multi-private
    case the environment currently refuses, and it arrives at three agents even with one
    private node each -- see the spec's section 4."""
    topo = Topology("3x1", private=((0,), (1,), (2,)), exposed=(3, 4))
    assert topo.hidden_from(0) == (1, 2)
    assert topo.hidden_from(1) == (0, 2)
    assert len(topo.hidden_from(0)) > 1, (
        "the reason per-block confounding subsets are a BLOCKER for n >= 3, not a nicety")


def test_an_edge_no_single_agent_can_see_is_forbidden_under_overlap():
    """THE COUNTEREXAMPLE that retired the old rule.

    Node 3 is visible to agents 0 and 2 but not to agent 1, so it is private to NOBODY.
    The old rule -- "no edge between two nodes private to different agents" -- therefore
    permitted 3 -> 1, an edge that no single agent observes. Confinement breaks on exactly
    that edge. The jointly-visible rule forbids it because no agent's window contains both
    endpoints.
    """
    topo = Topology(
        "overlap",
        private=((0,), (1,), (2,)),
        exposed=(4,),
        visibility=(frozenset({0}), frozenset({1}), frozenset({2}),
                    frozenset({0, 2}), frozenset({0, 1, 2})),
    )
    allowed = topo.allowed_edges()
    assert not allowed[3, 1] and not allowed[1, 3], (
        "node 3 is invisible to agent 1 and node 1 is invisible to agents 0 and 2, so no "
        "single agent observes both -- the edge is unlearnable by anyone")
    assert allowed[3, 0] and allowed[0, 3], "agent 0 sees both 0 and 3"
    assert allowed[3, 2] and allowed[2, 3], "agent 2 sees both 2 and 3"
    assert allowed[0, 4] and allowed[1, 4], "everyone sees the exposed node"


def test_every_allowed_edge_has_a_witness_agent():
    """The rule stated as a property rather than as a table: for every permitted edge there
    must EXIST an agent observing both endpoints, and for every forbidden one there must be
    none. Checked exhaustively over several shapes."""
    shapes = [
        Topology("2x2", private=((0, 1), (2, 3)), exposed=(4, 5)),
        Topology("3x1", private=((0,), (1,), (2,)), exposed=(3, 4)),
        Topology("4x1", private=((0,), (1,), (2,), (3,)), exposed=(4, 5)),
        Topology("3x2", private=((0, 1), (2, 3), (4, 5)), exposed=(6,)),
    ]
    for topo in shapes:
        windows = [set(topo.observed_by(a)) for a in topo.agents]
        allowed = topo.allowed_edges()
        for u in range(topo.d):
            for v in range(topo.d):
                witness = any({u, v} <= w for w in windows)
                expected = witness and u != v
                assert allowed[u, v] == expected, (topo.name, u, v)


def test_sampled_graphs_at_n_agents_never_violate_the_mask():
    topo = Topology("3x1", private=((0,), (1,), (2,)), exposed=(3, 4))
    forbidden = ~topo.allowed_edges()
    rng = np.random.default_rng(0)
    for _ in range(200):
        adjacency = topo.sample_dag(rng, p=0.5)
        assert not (adjacency[forbidden] > 0).any()
        assert is_acyclic(adjacency)


def test_a_malformed_partition_is_rejected_at_construction():
    """Loudly, at build time. A node in two blocks, or a gap in the numbering, silently
    changes which columns an agent sees -- and that failure would present as a leak."""
    with pytest.raises(ValueError, match="more than one block"):
        Topology("dup", private=((0,), (0,)), exposed=(1, 2))
    with pytest.raises(ValueError, match="0..d-1"):
        Topology("gap", private=((0,), (2,)), exposed=(3,))
    with pytest.raises(ValueError, match="visibility"):
        Topology("short", private=((0,), (1,)), exposed=(2,),
                 visibility=(frozenset({0}), frozenset({1})))


def test_there_is_no_a_private_shim():
    """Deliberate. A property returning private[0] would keep every stale caller working at
    two agents and silently mean the wrong thing at five. Call sites break loudly instead."""
    assert not hasattr(T1, "a_private") and not hasattr(T1, "b_private")
