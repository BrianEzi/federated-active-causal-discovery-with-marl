"""Latent projections, and the structural claim the two-agent belief design rests on.

The load-bearing test here is `test_confounding_is_confined_to_the_shared_set`. If it
ever fails, an agent's belief can no longer be represented as "a DAG over my window plus
a flag per shared pair", the score stops decomposing, and the subset DP does not carry
over to the two-agent case. See docs/MA_DESIGN.md section 3.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.projection import common_source_pairs
from ma.projection import (
    BIDIRECTED,
    DIRECTED,
    ancestor_matrix,
    bidirected_pairs,
    d_separated,
    latent_projection,
)
from ma.topology import T1, Topology, two_agent
from ma.graphs import build_graph_space

T112 = two_agent("t112", a_private=(0,), b_private=(1,), exposed=(2, 3))
T113 = two_agent("t113", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


def masked_dags(topology):
    space = build_graph_space(topology.d)
    forbidden = ~topology.allowed_edges()
    dags = np.asarray(space.dags) > 0.5
    keep = np.flatnonzero(~dags[:, forbidden].any(axis=1))
    return space, keep


# -- the primitives ---------------------------------------------------------------

def test_ancestor_matrix_is_the_transitive_closure():
    adj = np.zeros((4, 4), dtype=int)
    adj[0, 1] = adj[1, 2] = 1
    anc = ancestor_matrix(adj)
    assert anc[0, 1] and anc[1, 2] and anc[0, 2]
    assert not anc[2, 0] and not anc[0, 3]
    assert not anc[0, 0], "a node must not be its own ancestor"


def test_d_separation_on_the_textbook_triples():
    # chain 0 -> 1 -> 2
    chain = np.zeros((3, 3), dtype=int); chain[0, 1] = chain[1, 2] = 1
    assert not d_separated(chain, 0, 2, [])
    assert d_separated(chain, 0, 2, [1])

    # fork 1 -> 0, 1 -> 2
    fork = np.zeros((3, 3), dtype=int); fork[1, 0] = fork[1, 2] = 1
    assert not d_separated(fork, 0, 2, [])
    assert d_separated(fork, 0, 2, [1])

    # collider 0 -> 1 <- 2: the orientation reverses
    coll = np.zeros((3, 3), dtype=int); coll[0, 1] = coll[2, 1] = 1
    assert d_separated(coll, 0, 2, [])
    assert not d_separated(coll, 0, 2, [1])


def test_projection_onto_everything_returns_the_dag():
    adj = np.zeros((3, 3), dtype=int); adj[0, 1] = adj[2, 1] = 1
    proj = latent_projection(adj, [0, 1, 2])
    assert proj[0, 1] == DIRECTED and proj[2, 1] == DIRECTED
    assert proj[0, 2] == 0 and proj[2, 0] == 0


def test_a_hidden_common_cause_becomes_a_bidirected_edge():
    # 2 is hidden and causes both 0 and 1, which are otherwise unrelated.
    adj = np.zeros((3, 3), dtype=int); adj[2, 0] = adj[2, 1] = 1
    proj = latent_projection(adj, [0, 1])
    assert proj[0, 1] == BIDIRECTED and proj[1, 0] == BIDIRECTED


def test_a_hidden_mediator_becomes_a_directed_edge_not_a_bidirected_one():
    # 0 -> 2 -> 1 with 2 hidden: 0 is still an ancestor of 1, so the edge is directed.
    adj = np.zeros((3, 3), dtype=int); adj[0, 2] = adj[2, 1] = 1
    proj = latent_projection(adj, [0, 1])
    assert proj[0, 1] == DIRECTED
    assert proj[1, 0] == 0


# -- the claim --------------------------------------------------------------------

@pytest.mark.parametrize("topology", [T112, T113], ids=["(1,1,2)", "(1,1,3)"])
@pytest.mark.slow
def test_confounding_is_confined_to_the_shared_set(topology):
    """No bidirected edge may touch a private node, over EVERY legal graph.

    Reason it must hold: a bidirected edge between u and v in A's projection needs a
    common cause among the nodes A cannot see, which is exactly `Z_B`; and no `z_B` may
    point into `Z_A`, because cross-private edges are forbidden. So both endpoints lie
    in `X`. This test is the exhaustive check of that argument.
    """
    space, keep = masked_dags(topology)
    shared = set(topology.exposed)
    for t in keep:
        adjacency = np.asarray(space.dags[t], dtype=np.int8)
        for agent in (0, 1):
            for u, v in bidirected_pairs(adjacency, topology.observed_by(agent)):
                assert u in shared and v in shared, (
                    f"bidirected edge {u}<->{v} touches a private node "
                    f"in graph {t} for agent {agent}"
                )


def test_confinement_also_holds_with_two_private_nodes_each():
    """`(2,2,2)` is the first case where an agent has two private nodes that could in
    principle be confounded with each other. Sampled -- d=6 masked is far too large to
    run brute-force separation over exhaustively."""
    space, keep = masked_dags(T1)
    rng = np.random.default_rng(0)
    shared = set(T1.exposed)
    for t in rng.choice(keep, 2000, replace=False):
        adjacency = np.asarray(space.dags[t], dtype=np.int8)
        for agent in (0, 1):
            for u, v in bidirected_pairs(adjacency, T1.observed_by(agent)):
                assert u in shared and v in shared


# -- the correction to the section 3 numbers --------------------------------------

@pytest.mark.parametrize("topology", [T112, T113], ids=["(1,1,2)", "(1,1,3)"])
@pytest.mark.slow
def test_the_section_3_metric_overcounts_and_the_excess_is_ancestral(topology):
    """`ma.projection.latent_projection_pairs` flags any pair with a hidden common
    source. That is a superset of the true bidirected edges: if the two nodes are also
    ancestrally related, the MAG carries a *directed* edge there instead, and the agent's
    DAG model is not wrong about that pair at all.

    Recorded as a test rather than a comment because the section 3 figures (22.7%, 43.8%)
    were read as confounding rates and are not.
    """
    space, keep = masked_dags(topology)
    excess = 0
    for t in keep:
        adjacency = np.asarray(space.dags[t], dtype=np.int8)
        anc = ancestor_matrix(adjacency)
        for agent in (0, 1):
            observed = topology.observed_by(agent)
            proxy = {tuple(p) for p in common_source_pairs(
                adjacency, observed, topology.hidden_from(agent))}
            true = set(bidirected_pairs(adjacency, observed))
            assert true <= proxy, "the proxy must be a superset, not merely different"
            for u, v in proxy - true:
                assert anc[u, v] or anc[v, u], (
                    f"pair {u},{v} flagged by the proxy, not bidirected, and NOT "
                    f"ancestrally related -- the overcount has another cause"
                )
                excess += 1
    assert excess > 0, "if there is no excess the two metrics agree and the note is moot"


def test_the_two_confounding_criteria_diverge_and_the_difference_is_recorded():
    """`common_source_pairs` and `bidirected_pairs` are NOT interchangeable.

    Merged into one module on 2026-08-23; deliberately NOT unified into one function. This
    test is the record of why, so a later reader does not "simplify" one into the other.

    Graph: hidden node 0 causes both 1 and 2, AND there is a real edge 1 -> 2.

        0 -> 1,  0 -> 2,  1 -> 2       observed = (1, 2),  hidden = (0,)

    `common_source_pairs` returns (1, 2): node 0 is a hidden common source, which is true.

    `bidirected_pairs` does NOT: in the MAG, a bidirected edge requires that NEITHER node
    is an ancestor of the other, and here 1 IS an ancestor of 2 via the real edge. The MAG
    carries 1 -> 2. That is the textbook definition (Richardson & Spirtes 2002) and it is
    what env._confounded_positions scores against.

    Neither is wrong. They answer different questions:
      common_source_pairs   is my DAG model misspecified here?      -> yes
      bidirected_pairs      does the MAG carry a bidirected edge?   -> no
    """
    adjacency = np.zeros((3, 3), dtype=int)
    adjacency[0, 1] = adjacency[0, 2] = adjacency[1, 2] = 1

    assert common_source_pairs(adjacency, (1, 2), (0,)) == [(1, 2)]
    assert bidirected_pairs(adjacency, (1, 2)) == ()

    # And where there is NO real edge between them, the two agree.
    no_edge = np.zeros((3, 3), dtype=int)
    no_edge[0, 1] = no_edge[0, 2] = 1
    assert common_source_pairs(no_edge, (1, 2), (0,)) == [(1, 2)]
    assert bidirected_pairs(no_edge, (1, 2)) == ((1, 2),)
