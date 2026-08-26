"""Scale-free generation, and what it changes about the FEDERATED structure.

Erdos-Renyi was inherited from the Bayesian DP this project began with, whose prior had to
match its generator. `prior_p` is read by `Topology.sample_dag` and by nothing else -- no
belief engine consumes it -- so the constraint was a hangover rather than a requirement, and
dropping it introduces no misspecification for either engine now in use.

The tests below check the two things that matter: the generator still obeys the visibility
mask and produces DAGs, and it actually produces HUBS -- because a scale-free generator that
merely added edges would look identical on any density-blind measurement.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.graphs import is_acyclic
from ma.projection import BIDIRECTED, latent_projection
from ma.topology import ER, SF, federated_topology


def _sample(topology, model, reps, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [topology.sample_dag(rng, model=model, **kw) for _ in range(reps)]


def test_scale_free_graphs_are_acyclic_and_obey_the_mask():
    topology = federated_topology(4, 2, 4)
    forbidden = ~topology.allowed_edges()
    for graph in _sample(topology, SF, 100, m=2):
        assert not (graph & forbidden).any(), "an edge crossed the visibility boundary"
        assert np.all(np.diag(graph) == 0)
        assert is_acyclic(graph)


def test_every_node_but_the_first_gets_a_parent_where_one_is_allowed():
    """Preferential attachment is what keeps the graph connected: each node takes up to `m`
    parents from those before it. Under Erdos-Renyi at low `p` a node can be isolated, and
    an isolated node is a window with nothing to discover."""
    topology = federated_topology(4, 1, 3)
    for graph in _sample(topology, SF, 50, m=1):
        parentless = [v for v in range(topology.d) if graph[:, v].sum() == 0]
        # Only nodes with no ALLOWED predecessor may be parentless.
        for v in parentless:
            allowed_before = topology.allowed_edges()[:, v].sum()
            assert allowed_before == 0 or graph[:, v].sum() == 0


def test_scale_free_has_a_heavier_degree_tail_than_erdos_renyi_at_the_SAME_density():
    """The comparison that matters. At matched edge counts the means are equal by
    construction, so any difference is in the SHAPE of the degree distribution."""
    topology = federated_topology(4, 2, 4)
    scale_free = _sample(topology, SF, 200, m=2)
    density = float(np.mean([g.sum() for g in scale_free]))
    # Bisect Erdos-Renyi's p to the same expected edge count.
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        sample = _sample(topology, ER, 80, p=mid)
        lo, hi = (mid, hi) if np.mean([g.sum() for g in sample]) < density else (lo, mid)
    random_graphs = _sample(topology, ER, 200, p=(lo + hi) / 2)

    assert abs(np.mean([g.sum() for g in random_graphs]) - density) < 0.1 * density
    sf_tail = np.percentile(np.concatenate([g.sum(axis=1) for g in scale_free]), 95)
    er_tail = np.percentile(np.concatenate([g.sum(axis=1) for g in random_graphs]), 95)
    assert sf_tail > er_tail, (sf_tail, er_tail)


def test_hubs_make_the_hidden_common_cause_signature_common():
    """The reason this generator earns its place, rather than a realism argument.

    A hidden node parenting three visible ones projects to a bidirected TRIANGLE, which is
    the signature of a single hidden cause and the structure the thesis is about. Under
    Erdos-Renyi it is rare enough that measuring it required constructing the graph by hand
    (2026-08-26: two triangles in 180 windows). A hub private to one agent parents many
    shared variables at once, so under scale-free the signature arises on its own.
    """
    topology = federated_topology(4, 2, 4)

    def triangles_per_window(graphs):
        total = windows = 0
        for graph in graphs:
            for agent in topology.agents:
                mag = latent_projection(graph, tuple(topology.observed_by(agent)))
                k = mag.shape[0]
                windows += 1
                bidirected = {(u, v) for u in range(k) for v in range(u + 1, k)
                              if mag[u, v] == BIDIRECTED}
                for u in range(k):
                    for v in range(u + 1, k):
                        for w in range(v + 1, k):
                            total += {(u, v), (u, w), (v, w)} <= bidirected
        return total / max(windows, 1)

    scale_free = triangles_per_window(_sample(topology, SF, 150, m=2))
    # p chosen at the matched density measured for m=2 on this topology (~18.8 edges).
    random_graphs = triangles_per_window(_sample(topology, ER, 150, p=0.45))
    assert scale_free > 2 * random_graphs, (scale_free, random_graphs)


def test_an_unknown_model_is_refused_rather_than_silently_defaulted():
    topology = federated_topology(3, 1, 3)
    with pytest.raises(ValueError, match="model must be one of"):
        topology.sample_dag(np.random.default_rng(0), model="barabasi")
