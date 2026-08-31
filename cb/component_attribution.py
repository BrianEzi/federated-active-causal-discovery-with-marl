"""Attribution past k=12: the candidate set factored over CONNECTED COMPONENTS.

WHY NOT PER-PAIR, which is what this was going to be. `cb/factored_attribution.py` scales the
STRUCTURE half and leaves ownership enumerated, so it is bounded by the attribution space
itself: 5 / 35 / 482 / 8.4e10 / 8.9e15 hypotheses at k = 4 / 8 / 12 / 20 / 30. The obvious
next move -- factor ownership per pair, as `cb/factored.py` factors marks per pair -- does not
work, and the reason is visible in `consistent_with_partner`:

    for group in groups:                       # RULE 2, ATOMICITY
        hit = set(group.pairs()) & moved
        if hit and hit != pairs:
            return False
        if group.owner == owner:
            covered |= pairs
    return bool(moved & covered)               # RULE 1, LOCAL DISTURBANCE

Rule 2 refutes a candidate that lets PART of a clique move. Refuting requires knowing which
pairs share a latent -- the clique structure -- which is precisely the joint fact a per-pair
belief cannot hold. Rule 1 is a disjunction over the groups an owner has. Dropping either is
not an option: the docstring of `consistent_with_partner` records that deleting rule 1 took
`right` from 72 to 0 over 162 groups, because atomicity alone never refutes enough to reach
bar 1.0.

WHAT DOES FACTOR, EXACTLY. Rule 2 is UNARY at the granularity of a group -- the test on each
group names no other group -- so it decomposes at any granularity. Rule 1 is the only joint
constraint, and it couples groups only through their OWNER. Meanwhile the candidate set
itself already factors over the connected components of the bidirected graph: owner sets are
assigned per pair, cliques never span components, and the coverage check is per pair. So

    attributions_for(pairs, owners) == PRODUCT over components of attributions_for(c, owners)

exactly -- verified on 300 random pair sets at 2-4 owners, and pinned in
`tests/crosscheck/test_component_attribution.py`. Enumerating per component therefore loses
NOTHING about the space, and the cost falls from (2^(n-1) - 1)^P in the total settled-pair
count to a SUM of (2^(n-1) - 1)^Pc over components. At 3 agents with 16 settled pairs in
components of at most 3, that is 43 million against roughly 300, and the product is never
materialised.

WHAT IS APPROXIMATED, AND IT IS ONE THING. Rule 1 can name pairs in several components at
once, and "at least one of the actor's groups explains something that moved" is then a clause
across components that a product cannot represent. It is applied by UNIT PROPAGATION to a
fixpoint: let C_j be the components still holding a candidate that satisfies message j.

    |C_j| == 1   the clause is unit -- filter that component. EXACT.
    |C_j| == 0   no assignment satisfies it -- drop the message, exactly as the enumerated
                 backend drops a message that would empty the candidate set.
    |C_j| >= 2   skip, and re-test after every other filtering, because pruning elsewhere can
                 make it unit later.

WHY THAT IS SOUND. The represented belief is the product, which is a SUPERSET of the true
surviving set: candidates are removed only when no global survivor could contain them. A
superset preserves both ends of the bar-1.0 test -- frequency 1.0 still means genuinely
forced, frequency 0.0 still means genuinely refuted -- so this belief can be LESS decided
than the enumerated one but never differently decided and never wrong. The `|C_j| >= 2` skips
are the entire cost, and the crosscheck reports how many decisions they cost rather than
leaving it to be argued.

THE OTHER GAIN, which matters more than the speed. `max_attribution_pairs` is a GLOBAL cap:
past the fifth settled pair every further pair is held out of scope, so most of a large
window is unattributable by construction. Here the cap is PER COMPONENT, and a sparse window
fits entirely -- scope grows to nearly the whole window rather than to five pairs of it.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

from cb.attribution import LatentGroup, attributions_for, consistent_with_partner
from cb.factored_attribution import FactoredAttributedBackend

Pair = Tuple[int, int]


def connected_components(pairs: Sequence[Pair]) -> Tuple[Tuple[Pair, ...], ...]:
    """The bidirected graph's connected components, as pair lists, deterministically ordered.

    Deterministic because the components index the belief's candidate lists and the pruning
    replays the whole message log against them -- an ordering that depended on set iteration
    would make the belief depend on hash order rather than on evidence.
    """
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in pairs:
        a, b = find(u), find(v)
        if a != b:
            parent[a] = b
    buckets: Dict[int, List[Pair]] = {}
    for pair in pairs:
        buckets.setdefault(find(pair[0]), []).append((min(pair), max(pair)))
    return tuple(tuple(sorted(group)) for group in
                 sorted(buckets.values(), key=lambda g: sorted(g)))


class ComponentAttributedBelief:
    """Per-component candidate lists, read as one belief.

    Frequencies are EXACT for the product this belief represents: a group lives in exactly
    one component, and the components are independent, so its global frequency equals its
    frequency within its own component. No sampling, no bound, no approximation on this side.
    """

    def __init__(self, structure, components, k: int, scope=()):
        self.k = int(k)
        self.adjacency = structure.adjacency
        self.directed = structure.directed
        self.bidirected = structure.bidirected
        # components: ((pairs, candidates), ...) -- candidates is a tuple of attributions,
        # each a tuple of LatentGroup whose pairs lie inside that component.
        self.components = tuple((tuple(pairs), tuple(candidates))
                                for pairs, candidates in components)
        total = 1
        for _, candidates in self.components:
            total *= len(candidates)
        self.total = total if self.components else 0

        groups: Counter = Counter()
        owners: Counter = Counter()
        for _, candidates in self.components:
            denominator = max(len(candidates), 1)
            for hypothesis in candidates:
                for group in hypothesis:
                    groups[group] += 1.0 / denominator
                    for pair in group.pairs():
                        owners[(pair[0], pair[1], group.owner)] += 1.0 / denominator
        self.group_frequency = dict(groups)
        self.owner_frequency = dict(owners)
        # OUT OF SCOPE IS UNSURE, NOT WRONG -- see `cb/factored_attribution.py`. Scope here is
        # the union of the components kept, so a component too dense to enumerate takes its
        # own pairs out of scope and leaves every other component fully attributable.
        self.scope = frozenset(scope or ())

        self.n_boot = 0
        self.ci_tests = 0
        self.truncated_fraction = 0.0
        self.replicates = None

    @property
    def space(self):
        """No structure enumeration exists here, so there is no candidate list to expose."""
        return ()

    def edge_marginals(self) -> np.ndarray:
        return self.directed

    def confounded_pairs(self, threshold: float = 0.5) -> tuple:
        return tuple((u, v) for u, v in combinations(range(self.k), 2)
                     if self.bidirected[u, v] >= threshold)

    def owner_channel(self, n_agents: int) -> np.ndarray:
        """[pairs, n_agents] -- how much of the belief blames each agent for each pair."""
        pair_list = list(combinations(range(self.k), 2))
        out = np.zeros((len(pair_list), n_agents), dtype=float)
        for index, (u, v) in enumerate(pair_list):
            for owner in range(n_agents):
                out[index, owner] = self.owner_frequency.get((u, v, owner), 0.0)
        return out


class ComponentAttributedBackend(FactoredAttributedBackend):
    """Factored structure, component-factored ownership. Call-compatible with the others.

    Inherits the lifecycle, the two evidence channels and the replay log from
    `FactoredAttributedBackend` and replaces only how the candidate set is held. The parent
    stays as the crosscheck reference; this is not a drop-in replacement for it in tests.
    """

    can_handle_multi_hidden = True

    def __init__(self, k: int, shared_positions: Sequence[int] = (), n_agents: int = 2,
                 agent: int = 0, evidence: str = "oracle", evidence_alpha: float = 0.001,
                 max_component_pairs: int = 8, max_component_candidates: int = 50_000,
                 local_disturbance: bool = True, **_ignored):
        super().__init__(k, shared_positions=shared_positions, n_agents=n_agents, agent=agent,
                         evidence=evidence, evidence_alpha=evidence_alpha,
                         # The parent's global pair cap is exactly what this class removes.
                         max_attribution_pairs=10 ** 9,
                         local_disturbance=local_disturbance)
        self.max_component_pairs = int(max_component_pairs)
        self.max_component_candidates = int(max_component_candidates)
        # Messages skipped because the pairs that could have supported rule 1 sit in a
        # component this belief dropped. NOT a contradiction -- the belief simply was not
        # asked -- and kept apart from `contradictions` so the two cannot be confused.
        self.out_of_scope = 0
        self._components: Tuple[Tuple[Tuple[Pair, ...], Tuple[tuple, ...]], ...] = ()
        self._masks: Dict[tuple, Tuple[set, set]] = {}

    # -- the candidate set ---------------------------------------------------------------

    def reset(self, true_mag: np.ndarray, adjacency=None, topology=None) -> None:
        self._masks = {}
        self._components = ()
        super().reset(true_mag, adjacency=adjacency, topology=topology)

    def _rebuild(self, force: bool = False) -> None:
        settled = self.settled_bidirected()
        if settled == self._settled and not force:
            # The settled set is unchanged, so the candidates are too -- but the STRUCTURE
            # frequencies move on every own intervention, so the belief object is rebuilt.
            self.last = ComponentAttributedBelief(
                self.structure.last or self._empty_structure(), self._components, self.k,
                scope=self._scope_of(self._components))
            return

        blocks: List[Tuple[Tuple[Pair, ...], tuple]] = []
        self.truncated = False
        for pairs in connected_components(settled):
            # A component too dense to enumerate is TRUNCATED, not dropped. Dropping it takes
            # every one of its pairs out of scope; truncating keeps a prefix, and a prefix is
            # exactly the belief this agent would hold if only those pairs had settled --
            # sound by the same argument as the global cap it replaces. A true group naming a
            # pair that was cut is out of scope and scores UNSURE, never wrong. Measured
            # before this: dropping put component scope BELOW the global cap it was meant to
            # beat, 0.66 against 0.80 at k=12.
            budget = self._pair_budget()
            if len(pairs) > budget:
                self.truncated = True
                pairs = pairs[:budget]
            blocks.append((pairs, attributions_for(pairs, self.owners)))

        # COUNTED PER REPLAY, NOT ACCUMULATED ACROSS REPLAYS. The log is replayed in full
        # every time the settled set changes, so incrementing a running total would count the
        # same contradicting message once per rebuild and report an episode as far more
        # broken than it is. These are counts of MESSAGES.
        self.contradictions = 0
        self.out_of_scope = 0
        live = self._prune(blocks)

        self._components = tuple(
            (pairs, tuple(candidates[i] for i in sorted(alive)))
            for (pairs, candidates), alive in zip(blocks, live))
        self._settled = settled
        self.last = ComponentAttributedBelief(
            self.structure.last or self._empty_structure(), self._components, self.k,
            scope=self._scope_of(self._components))

    def _pair_budget(self) -> int:
        """How many pairs one component may hold, from the enumeration cost it implies.

        The COST of enumerating a component is (2^owners - 1)^pairs -- the owner-set
        assignments tried, before canonical dedup collapses them, which is not the number
        that survives. So the budget is derived from the estimate BEFORE any enumeration
        happens: a budget checked on the result is a budget checked after the work it was
        meant to prevent, and at three partners a seven-pair component is 800,000
        assignments.
        """
        span = 2 ** len(self.owners) - 1
        if span <= 1:
            return self.max_component_pairs
        allowed, size = 0, 1
        while (allowed < self.max_component_pairs
               and size * span <= self.max_component_candidates):
            size *= span
            allowed += 1
        return max(1, allowed)

    @staticmethod
    def _scope_of(components) -> frozenset:
        return frozenset(pair for pairs, _ in components for pair in pairs)

    def _masks_for(self, pairs, candidates, owner, moved):
        """(atomicity survivors, atomicity-and-rule-1 survivors) as INDEX SETS, memoised.

        WHY THIS IS THE OBJECT TO CACHE AND THE PRUNED LIST IS NOT. Whether a message may be
        applied depends on the whole scope -- a clause that is unit today can stop being unit
        when a pair settles somewhere else -- so carrying a component's PRUNED list across
        rebuilds is unsound: it would keep a rule-1 prune that the enlarged scope no longer
        licenses. Caught before it shipped, and it is exactly the failure the replay log was
        introduced to prevent.

        These two sets have no such dependence. Both are a function of (this component's
        candidate list, this message) alone, and the candidate list is a function of the
        component's pairs alone. So they survive any change of scope, and the fixpoint below
        becomes intersections of small integer sets instead of rescans of every candidate.
        """
        key = (pairs, owner, moved)
        hit = self._masks.get(key)
        if hit is None:
            atomic, satisfying = set(), set()
            for index, candidate in enumerate(candidates):
                if not consistent_with_partner(candidate, owner, moved,
                                               local_disturbance=False):
                    continue
                atomic.add(index)
                if consistent_with_partner(candidate, owner, moved, local_disturbance=True):
                    satisfying.add(index)
            hit = (atomic, satisfying)
            self._masks[key] = hit
        return hit

    def _prune(self, blocks) -> List[set]:
        """Replay the whole message log to a FIXPOINT. Returns one live index set per block.

        A fixpoint rather than one pass in log order, because a clause that spans components
        can become unit only after some other message has pruned one of them, and a single
        pass would leave that inference on the table.

        NOT order-independent, and the enumerated backend is not either: a message is DROPPED
        when nothing can satisfy it, and whether it reaches that state can depend on what has
        already been applied. What the replay log does guarantee is independence from the
        order messages ARRIVED in relative to structure updates, which is the property that
        was actually at risk. The pass order here is log order, so it is deterministic.
        """
        live = [set(range(len(candidates))) for _, candidates in blocks]
        scope = frozenset(pair for pairs, _ in blocks for pair in pairs)
        dropped: set = set()
        # A message need only be re-examined when some block has CHANGED since it was last
        # examined; otherwise it re-derives the filtering it already performed. Without this
        # the fixpoint costs (passes x messages x blocks) set operations on every rebuild.
        version = [0] * len(blocks)
        seen: Dict[int, tuple] = {}
        changed = True
        while changed:
            changed = False
            for index, (owner, moved) in enumerate(self._log):
                if index in dropped or seen.get(index) == tuple(version):
                    continue
                seen[index] = tuple(version)
                masks = [self._masks_for(pairs, candidates, owner, moved)
                         for pairs, candidates in blocks]
                atomic = [live[i] & masks[i][0] for i in range(len(blocks))]
                if any(not survivors for survivors in atomic):
                    # Some block has no candidate left at all, so no global assignment
                    # survives. Refuse the MESSAGE rather than the belief, exactly as the
                    # enumerated backend does -- the truth is in the set, so the evidence is
                    # what is at fault.
                    dropped.add(index)
                    self.contradictions += 1
                    continue
                if self.local_disturbance:
                    support = [i for i in range(len(blocks))
                               if atomic[i] & masks[i][1]]
                    if not support:
                        dropped.add(index)
                        if set(moved) <= scope:
                            self.contradictions += 1
                        else:
                            self.out_of_scope += 1
                        continue
                    if len(support) == 1:
                        only = support[0]
                        atomic[only] = atomic[only] & masks[only][1]
                for i in range(len(blocks)):
                    if atomic[i] != live[i]:
                        live[i] = atomic[i]
                        version[i] += 1
                        changed = True
        return live

    # -- reporting -----------------------------------------------------------------------

    @property
    def n_candidates(self) -> int:
        """The size of the product, which is never built. Big integers, deliberately."""
        total = 1
        for _, candidates in self._components:
            total *= len(candidates)
        return total if self._components else 0

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def largest_component(self) -> int:
        return max((len(pairs) for pairs, _ in self._components), default=0)
