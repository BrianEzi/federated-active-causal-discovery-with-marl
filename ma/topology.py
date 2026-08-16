"""Who owns which node, which edges may exist, and what each agent can see.

The specification agreed on 2026-08-15 (see docs/MULTI_AGENT_DESIGN.md and the overnight
plan): six nodes, two private to each agent, two exposed and shared.

**Edges between the two private sets are forbidden.** Neither agent can ever observe such
an edge -- agent A never sees B's private columns and vice versa -- so no data from anyone
bears on it and it would be permanently unidentifiable. Allowing it would make the global
graph unrecoverable *by construction*, which is a much worse problem than a hard one.

**The prior carries the same mask as the generator.** A generator that forbids cross-private
edges paired with a prior that allows them is a misspecification, and it would surface later
as systematic overconfidence that looks exactly like an estimator bug. Same discipline as
the single-agent environment.

**Generation is a mask, not a procedure.** Draw a random permutation as a topological order
and include each allowed forward pair with probability `p`. Acyclicity is then free, there
is no rejection sampling to distort the prior, and there is no repair step where two
separately-drawn DAGs are stitched at the boundary. This is the "generate two DAGs then
connect them" idea expressed so that it cannot go wrong.

Authority: each agent may intervene on its own private nodes **and on the exposed nodes**.
Shared authority over the exposed nodes is deliberate -- it is the surface on which
coordination and contention actually happen, and removing it would remove the problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Topology:
    """A node partition and the edge mask it implies."""

    name: str
    a_private: Tuple[int, ...]
    b_private: Tuple[int, ...]
    exposed: Tuple[int, ...]
    # T3's defining constraint: exposed nodes may not have private parents at all, which
    # removes latent confounding by construction -- at a cost in realism that has to be
    # argued for rather than assumed. See `ma/confounding.py`.
    exposed_have_no_private_parents: bool = False

    @property
    def d(self) -> int:
        return len(self.a_private) + len(self.b_private) + len(self.exposed)

    def observed_by(self, agent: str) -> Tuple[int, ...]:
        """Columns an agent sees: its own private nodes plus the exposed ones."""
        private = self.a_private if agent == "A" else self.b_private
        return tuple(sorted(private + self.exposed))

    def hidden_from(self, agent: str) -> Tuple[int, ...]:
        """The other agent's private nodes -- invisible, but causally active."""
        return tuple(sorted(self.b_private if agent == "A" else self.a_private))

    def may_intervene_on(self, agent: str) -> Tuple[int, ...]:
        """Own private nodes plus the exposed ones. Shared exposed authority is the point."""
        private = self.a_private if agent == "A" else self.b_private
        return tuple(sorted(private + self.exposed))

    def allowed_edges(self) -> np.ndarray:
        """`[d, d]` bool: may edge `i -> j` exist at all?

        Diagonal is False (no self-loops). Cross-private pairs are False in both
        directions. Under `exposed_have_no_private_parents`, private-to-exposed is also
        False.
        """
        d = self.d
        allowed = ~np.eye(d, dtype=bool)
        for u in self.a_private:
            for v in self.b_private:
                allowed[u, v] = allowed[v, u] = False
        if self.exposed_have_no_private_parents:
            private = self.a_private + self.b_private
            for u in private:
                for v in self.exposed:
                    allowed[u, v] = False
        return allowed

    def sample_dag(self, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
        """Draw one DAG: random topological order, allowed forward pairs with prob `p`."""
        d = self.d
        allowed = self.allowed_edges()
        order = rng.permutation(d)
        adjacency = np.zeros((d, d), dtype=np.int8)
        for i in range(d):
            for j in range(i + 1, d):
                u, v = int(order[i]), int(order[j])
                if allowed[u, v] and rng.random() < p:
                    adjacency[u, v] = 1
        return adjacency


# The three candidates compared in block 5. T1 is the agreed default; T2 widens the
# boundary; T3 exists to test whether removing latent confounding by construction is worth
# what it costs.
T1 = Topology("T1", a_private=(0, 1), b_private=(2, 3), exposed=(4, 5))
T2 = Topology("T2", a_private=(0,), b_private=(1,), exposed=(2, 3, 4, 5))
T3 = Topology("T3", a_private=(0, 1), b_private=(2, 3), exposed=(4, 5),
              exposed_have_no_private_parents=True)

TOPOLOGIES = (T1, T2, T3)


def masked_indices(space, topology: Topology) -> np.ndarray:
    """Indices of the DAGs in an enumerated `GraphSpace` that respect the mask.

    Vectorised over the whole space: at d=6 there are 3,781,503 graphs and a Python loop
    over them costs minutes.
    """
    forbidden = ~topology.allowed_edges()
    if not forbidden.any():
        return np.arange(space.n_dags)
    dags = np.asarray(space.dags) > 0.5
    violates = dags[:, forbidden].any(axis=1)
    return np.flatnonzero(~violates)


def edge_class(topology: Topology, u: int, v: int) -> str:
    """Where an edge sits relative to the federation boundary.

    `interior`         -- both endpoints private to the same agent.
    `private_exposed`  -- one private endpoint, one exposed. The boundary proper.
    `exposed_exposed`  -- both exposed; visible to both agents, owned by neither.
    """
    exposed = set(topology.exposed)
    a, b = set(topology.a_private), set(topology.b_private)
    in_exposed = (u in exposed, v in exposed)
    if all(in_exposed):
        return "exposed_exposed"
    if not any(in_exposed):
        return "interior" if ({u, v} <= a or {u, v} <= b) else "cross_private"
    return "private_exposed"


def edge_class_matrix(topology: Topology) -> np.ndarray:
    """`[d, d]` array of the labels above, for vectorised classification."""
    d = topology.d
    out = np.empty((d, d), dtype=object)
    for u in range(d):
        for v in range(d):
            out[u, v] = "self" if u == v else edge_class(topology, u, v)
    return out
