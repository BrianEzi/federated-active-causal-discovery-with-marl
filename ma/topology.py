"""Who owns which node, which edges may exist, and what each agent can see.

Generalised to `n` agents on 2026-08-22 (`docs/N_AGENT_REFACTOR_SPEC.md` section 3). The
two-agent form is the `n = 2` case of this one, not a separate thing.

**An edge may exist only if some agent observes BOTH of its endpoints.** This is the rule,
and it replaces "no edge between two nodes private to different agents". The two coincide
exactly under a disjoint partition, so nothing changes at two agents -- but the old rule is
too permissive the moment visibility overlaps, and a counterexample was found immediately:
a node visible to agents {0, 2} is private to nobody, so the old rule permitted it to parent
a node private to agent 1, an edge NO agent can see. That single edge breaks confinement.

The rule is also the more principled statement. An edge no one can observe is not learnable
by anyone, so admitting it to the hypothesis space adds structure that no data can bear on,
and makes the global graph unrecoverable *by construction* -- a much worse problem than a
hard one.

**The prior carries the same mask as the generator.** A generator that forbids an edge
paired with a prior that allows it is a misspecification, and it would surface later as
systematic overconfidence that looks exactly like an estimator bug. Same discipline as the
single-agent environment.

**Generation is a mask, not a procedure.** Draw a random permutation as a topological order
and include each allowed forward pair with probability `p`. Acyclicity is then free, there
is no rejection sampling to distort the prior, and there is no repair step where separately
drawn DAGs are stitched at a boundary. This is the "generate a DAG per agent then connect
them" idea expressed so that it cannot go wrong.

**Authority**: each agent may intervene on its own private nodes **and on the exposed
nodes**. Shared authority over the exposed nodes is deliberate -- it is the surface on which
coordination and contention actually happen, and removing it would remove the problem.

**Agents are integers `0 .. n-1`.** The strings "A" and "B" were fine for two and become
noise at five. `ma/env.py` still keys its dicts by string and translates at the boundary;
that is the next step of the refactor, not this one.

**No compatibility shim for `a_private` / `b_private`, deliberately.** A property returning
`private[0]` would keep every stale caller working at two agents and silently mean the wrong
thing at five, which is exactly the class of failure this project has been bitten by twice
(a budget that quietly changed meaning, a clean-rule that quietly could not fire). Call
sites break loudly instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Sequence, Tuple

import numpy as np

# How the generating DAG is drawn. See `Topology.sample_dag` for why this is a choice at
# all: the Erdos-Renyi assumption came from a Bayesian prior that no engine in use reads.
ER = "er"
SF = "sf"
GRAPH_MODELS = (ER, SF)


@dataclass(frozen=True)
class Topology:
    """A node partition over `n` agents, and the edge mask it implies.

    `private[i]` is agent `i`'s private nodes; `exposed` is visible to everyone. Nodes are
    `0 .. d-1` and every node must appear exactly once across `private` and `exposed`.
    """

    name: str
    private: Tuple[Tuple[int, ...], ...]
    exposed: Tuple[int, ...]
    # T3's defining constraint: exposed nodes may not have private parents at all, which
    # removes latent confounding by construction -- at a cost in realism that has to be
    # argued for rather than assumed. See `ma/confounding.py`.
    exposed_have_no_private_parents: bool = False
    # Explicit per-node visibility, superseding private/exposed when given. This is what
    # makes OVERLAPPING visibility expressible -- a node seen by agents {0, 2} but not 1 --
    # which is the case that exposed the old edge rule. Unused by the current topologies;
    # `observed_by` reads it when present.
    visibility: Optional[Tuple[FrozenSet[int], ...]] = None

    def __post_init__(self):
        seen = [node for block in self.private for node in block] + list(self.exposed)
        if len(seen) != len(set(seen)):
            raise ValueError(f"{self.name}: a node appears in more than one block: {seen}")
        if self.visibility is None:
            # private + exposed IS the partition, so it must tile 0..d-1 exactly.
            if sorted(seen) != list(range(len(seen))):
                raise ValueError(f"{self.name}: nodes must be exactly 0..d-1 with no gaps, "
                                 f"got {sorted(seen)}")
            return
        # With explicit visibility, THAT is the partition and `d` comes from it. A node may
        # legitimately belong to no private block and not be exposed either -- visible to
        # some agents but not all -- which is the whole point of the overlap case and the
        # thing the old cross-private edge rule could not express.
        d = len(self.visibility)
        if any(node >= d or node < 0 for node in seen):
            raise ValueError(f"{self.name}: visibility has {d} nodes but a block names "
                             f"{sorted(seen)}")
        if any(not who for who in self.visibility):
            raise ValueError(f"{self.name}: a node visible to NO agent cannot be modelled -- "
                             f"no data bears on it, so it is not identifiable by anyone")

    # -- shape ---------------------------------------------------------------------------

    @property
    def d(self) -> int:
        if self.visibility is not None:
            return len(self.visibility)
        return sum(len(block) for block in self.private) + len(self.exposed)

    @property
    def n_agents(self) -> int:
        return len(self.private)

    @property
    def agents(self) -> Tuple[int, ...]:
        return tuple(range(self.n_agents))

    # -- visibility ----------------------------------------------------------------------

    def observed_by(self, agent: int) -> Tuple[int, ...]:
        """Columns an agent sees: its own private nodes plus the exposed ones."""
        if self.visibility is not None:
            return tuple(sorted(n for n, who in enumerate(self.visibility) if agent in who))
        return tuple(sorted(self.private[agent] + self.exposed))

    def hidden_from(self, agent: int) -> Tuple[int, ...]:
        """Everything this agent cannot see -- invisible, but causally active.

        At `n > 2` this is the UNION of every other agent's private nodes, so a single
        clamp cleans only PART of it. That is the multi-private case the environment
        currently refuses, and it becomes unavoidable at three agents even with one private
        node each. See the spec's section 4.
        """
        observed = set(self.observed_by(agent))
        return tuple(n for n in range(self.d) if n not in observed)

    def may_intervene_on(self, agent: int) -> Tuple[int, ...]:
        """Own private nodes plus the exposed ones. Shared exposed authority is the point."""
        if self.visibility is not None:
            return self.observed_by(agent)
        return tuple(sorted(self.private[agent] + self.exposed))

    # -- the mask ------------------------------------------------------------------------

    def allowed_edges(self) -> np.ndarray:
        """`[d, d]` bool: may edge `i -> j` exist at all?

        THE JOINTLY-VISIBLE RULE: an edge is allowed only where some single agent observes
        both endpoints. Diagonal is False (no self-loops). Under
        `exposed_have_no_private_parents`, private-to-exposed is additionally False.
        """
        d = self.d
        allowed = np.zeros((d, d), dtype=bool)
        for agent in self.agents:
            seen = np.array(self.observed_by(agent), dtype=int)
            allowed[np.ix_(seen, seen)] = True
        np.fill_diagonal(allowed, False)
        if self.exposed_have_no_private_parents:
            private = [n for block in self.private for n in block]
            for u in private:
                for v in self.exposed:
                    allowed[u, v] = False
        return allowed

    def sample_dag(self, rng: np.random.Generator, p: float = 0.5,
                   model: str = ER, m: int = 2) -> np.ndarray:
        """Draw one DAG under `model`, respecting the jointly-visible mask.

        `er` -- random topological order, each allowed forward pair included with
        probability `p`. Every node is exchangeable, so degrees concentrate and there are
        no hubs.

        `sf` -- scale-free by preferential attachment along the same random topological
        order: each node takes up to `m` parents from the nodes before it, chosen with
        probability proportional to their out-degree so far (plus one, so a node with no
        children yet can still be chosen). Acyclicity is free for the same reason it is
        free under `er` -- edges only ever run forward along the order.

        WHY THIS MATTERS HERE AND NOT ONLY AS REALISM. Erdos-Renyi was inherited from the
        Bayesian DP this project started with, whose prior had to match the generator.
        Neither engine now in use reads `prior_p` -- it is consumed by this method and
        nowhere else -- so the constraint was a hangover, not a requirement.

        What changes is the FEDERATED structure, not just the degree distribution. Under
        `er` every private node is a weak, interchangeable confounder of its agent's
        partners. Under `sf` a private node can be a HUB, parenting many shared variables
        at once -- which is precisely the single-hidden-cause-of-many-children pattern that
        projects to a bidirected CLIQUE in every other agent's window. Hubs therefore make
        the structure this project measures more common rather than incidental, and they
        make it matter more who clamps what: clamping a hub cleans a great deal for a
        partner, clamping a leaf cleans almost nothing. Under `er` those two moves are
        nearly the same move.
        """
        if model not in GRAPH_MODELS:
            raise ValueError(f"model must be one of {GRAPH_MODELS}, got {model!r}")
        d = self.d
        allowed = self.allowed_edges()
        order = rng.permutation(d)
        adjacency = np.zeros((d, d), dtype=np.int8)
        if model == ER:
            for i in range(d):
                for j in range(i + 1, d):
                    u, v = int(order[i]), int(order[j])
                    if allowed[u, v] and rng.random() < p:
                        adjacency[u, v] = 1
            return adjacency

        out_degree = np.zeros(d, dtype=float)
        for j in range(1, d):
            v = int(order[j])
            candidates = [int(order[i]) for i in range(j) if allowed[int(order[i]), v]]
            if not candidates:
                continue
            # +1 so a childless node is reachable; without it the first edge could never
            # form and the whole construction would stall at zero.
            weights = out_degree[candidates] + 1.0
            take = min(m, len(candidates))
            chosen = rng.choice(candidates, size=take, replace=False,
                                p=weights / weights.sum())
            for u in np.atleast_1d(chosen):
                adjacency[int(u), v] = 1
                out_degree[int(u)] += 1.0
        return adjacency


def two_agent(name: str, a_private: Sequence[int], b_private: Sequence[int],
              exposed: Sequence[int], exposed_have_no_private_parents: bool = False):
    """Build a two-agent topology from the old field names.

    A CONSTRUCTOR, not a compatibility shim: it produces a normal n-agent `Topology` and
    the result has no `a_private` attribute. It exists because the two-agent case is written
    out dozens of times across scripts and tests, and `private=((0,), (1,))` is easy to get
    subtly wrong by hand -- `private=(0, 1)` is a plausible typo that would read as two
    agents with a nonsense block each.
    """
    return Topology(name=name,
                    private=(tuple(a_private), tuple(b_private)),
                    exposed=tuple(exposed),
                    exposed_have_no_private_parents=exposed_have_no_private_parents)


def federated_topology(n_agents: int, private_size: int = 1, n_shared: int = 3) -> Topology:
    """`n_agents` with `private_size` private nodes each, over `n_shared` shared ones.

    THE ONE PLACE this family is built. It was written out inline in four scripts, each
    hard-coding one private node per agent and three shared -- so "scale the private set"
    meant editing four call sites in step, and a topology named `T_4agent_1each` while
    carrying two private nodes each is exactly the kind of quiet mislabelling that makes a
    results file unreadable a week later. The name is derived, so it cannot drift.

    Window size is `private_size + n_shared`, and that is what the version space costs
    3^(edges) in -- not `d`. See `cb/versionspace.py`: k <= 6 is the usable range.
    """
    if n_agents < 1 or private_size < 1 or n_shared < 1:
        raise ValueError(f"federated_topology({n_agents}, {private_size}, {n_shared}): "
                         f"all three must be at least 1")
    private = tuple(tuple(range(i * private_size, (i + 1) * private_size))
                    for i in range(n_agents))
    base = n_agents * private_size
    return Topology(name=f"T_{n_agents}agent_{private_size}each_{n_shared}shared",
                    private=private,
                    exposed=tuple(range(base, base + n_shared)))


# The three candidates compared in block 5. T1 is the agreed default; T2 widens the
# boundary; T3 exists to test whether removing latent confounding by construction is worth
# what it costs.
T1 = two_agent("T1", a_private=(0, 1), b_private=(2, 3), exposed=(4, 5))
T2 = two_agent("T2", a_private=(0,), b_private=(1,), exposed=(2, 3, 4, 5))
T3 = two_agent("T3", a_private=(0, 1), b_private=(2, 3), exposed=(4, 5),
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

    `interior`         -- both endpoints private to the SAME agent.
    `private_exposed`  -- one private endpoint, one exposed. The boundary proper.
    `exposed_exposed`  -- both exposed; visible to every agent, owned by none.
    `cross_private`    -- private to DIFFERENT agents. Forbidden by the mask; the label
                          exists so that a violation can be named rather than merely
                          rejected.
    """
    exposed = set(topology.exposed)
    in_exposed = (u in exposed, v in exposed)
    if all(in_exposed):
        return "exposed_exposed"
    if not any(in_exposed):
        owner = {n: i for i, block in enumerate(topology.private) for n in block}
        return "interior" if owner.get(u) == owner.get(v) else "cross_private"
    return "private_exposed"


def edge_class_matrix(topology: Topology) -> np.ndarray:
    """`[d, d]` array of the labels above, for vectorised classification."""
    d = topology.d
    out = np.empty((d, d), dtype=object)
    for u in range(d):
        for v in range(d):
            out[u, v] = "self" if u == v else edge_class(topology, u, v)
    return out
