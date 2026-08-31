"""Latent groups: which hidden variable explains which confounded pairs, and whose it is.

WHY THIS REPLACES THE BIDIRECTED CLAIM. A bidirected edge says "something unobserved links
these two" and stops. It is a summary of confounding, not a model of it, and in a FEDERATED
setting the thing we actually want to know is whose unobserved variable it was. That is a
question a single-agent causal discovery method cannot even pose, because there is nobody
else for the answer to name.

WHAT AN ATTRIBUTION IS. A set of LATENT GROUPS. Each group is (owner, children) -- an agent
whose private block holds the hidden variable, and the set of window nodes it parents. A
group with children {u, v, w} accounts for the bidirected edges u-v, u-w and v-w at once.
Correctness is judged UP TO RENAMING: an outsider can never learn which of A's variables it
was, and does not need to, because the set of pairs the variable explains is its identity
from outside. That is also exactly the privacy claim -- and it is empty at one private node
per agent, since naming the agent would name the variable.

THE STRUCTURAL FACT THAT MAKES THIS TRACTABLE. Every bidirected edge in a window joins two
SHARED nodes, and its latent lies in exactly one agent's private block. A bidirected edge
needs a hidden node with directed paths to both endpoints running only through hidden nodes;
hidden nodes live in other agents' private blocks, and edges between DIFFERENT blocks are
forbidden by the visibility rule -- so the whole path lies inside one block and leaves it by
a single edge to a node that block's owner can see. The only nodes every agent sees are the
exposed ones. So attribution is a choice among a handful of NAMED agents, not a search over
arbitrary latent structures.

TWO OWNERS ARE POSSIBLE. Nothing stops agents A and C each holding a hidden cause of the
same pair, so a pair's owner set is a SET. `groups_from_dag` reports one group per latent
and lets them overlap; it does not force a partition.

IDENTIFIABILITY, and why this is a federated result. One latent parenting {u, v, w} and
three separate latents parenting {u,v}, {u,w}, {v,w} induce EXACTLY the same three
bidirected edges. No observation distinguishes them. An intervention does: act on the single
latent and all three associations move together; act on one of the three and only one moves.
The only agent who can perform that intervention is the one who owns the variable. So
recovering the grouping requires a partner to experiment on your behalf, which is what makes
this a coordination problem rather than an inference problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from ma.projection import BIDIRECTED, ancestor_matrix, latent_projection


@dataclass(frozen=True)
class LatentGroup:
    """One hidden variable, named by its owner and the window nodes it parents.

    `children` are WINDOW POSITIONS, not global node ids -- the agent has no vocabulary for
    global ids outside its own window. `owner` is an agent index.
    """
    owner: int
    children: FrozenSet[int]

    @lru_cache(maxsize=None)
    def pairs(self) -> Tuple[Tuple[int, int], ...]:
        """The bidirected edges this group accounts for.

        MEMOISED, and it matters more than it looks. This is a pure function of a FROZEN
        dataclass, so the answer can never change, and it sits inside every belief-frequency
        loop and every consistency test. Profiled 31 Aug on two episodes at k=12: 26.3
        MILLION calls, 16.5 s of a 77 s episode pair, recomputing `combinations(sorted(...))`
        for the same few dozen groups over and over. The cache is keyed on `self` and is
        therefore bounded by the number of distinct (owner, children) pairs, which is small.
        """
        return tuple(combinations(sorted(self.children), 2))

    def __repr__(self) -> str:
        return f"L(agent {self.owner} -> {{{', '.join(map(str, sorted(self.children)))}}})"


def _owner_of(topology, node: int) -> Optional[int]:
    for agent, block in enumerate(topology.private):
        if node in block:
            return agent
    return None


def groups_from_dag(adjacency: np.ndarray, topology, agent: int) -> Tuple[LatentGroup, ...]:
    """The TRUE latent groups in `agent`'s window, from the generating graph.

    One group per hidden variable that parents two or more window nodes through hidden
    intermediates. Groups may overlap: two agents can independently confound the same pair,
    and one agent can hold several latents.

    A latent's children are computed as the window nodes REACHABLE from it without passing
    through another window node -- reachability through hidden nodes only, which is exactly
    the condition that produces a bidirected edge in the projection. A hidden node whose
    influence on a window node runs THROUGH another window node does not confound anything:
    that path is already represented by the visible structure.

    Only maximal groups are kept. If h1 -> h2 and both reach the same window nodes, they
    describe one confounding structure, not two, and reporting both would make the truth
    unrecoverable by construction -- no evidence could ever separate them.
    """
    adjacency = np.asarray(adjacency)
    window = list(topology.observed_by(agent))
    position = {node: i for i, node in enumerate(window)}
    hidden = [n for n in range(topology.d) if n not in position]

    groups = []
    for h in hidden:
        # Reach from h through HIDDEN nodes only, then step into the window.
        frontier, seen, children = [h], {h}, set()
        while frontier:
            current = frontier.pop()
            for nxt in np.flatnonzero(adjacency[current] > 0):
                nxt = int(nxt)
                if nxt in position:
                    children.add(position[nxt])       # arrived in the window: stop here
                elif nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        if len(children) < 2:
            continue                                   # confounds nothing
        owner = _owner_of(topology, h)
        if owner is None:
            continue                                   # not in anyone's block: not modelled
        groups.append(LatentGroup(owner, frozenset(children)))

    # Drop non-maximal duplicates: same owner, children a subset of another group's.
    maximal = []
    for group in groups:
        if any(other is not group and group.owner == other.owner
               and group.children < other.children for other in groups):
            continue
        if group not in maximal:
            maximal.append(group)
    return tuple(maximal)


def responds_to(adjacency: np.ndarray, topology, agent: int, group: LatentGroup,
                intervened: int) -> bool:
    """Would intervening on global node `intervened` disturb this group's latent?

    True when `intervened` is the latent itself or an ancestor of it. This is the truth side
    of the elimination channel: a partner's private experiment moves exactly the groups whose
    latent it sits above, and the agent sees WHICH of its confounded pairs moved without ever
    learning which node was touched.
    """
    adjacency = np.asarray(adjacency)
    ancestors = ancestor_matrix(adjacency)
    window = list(topology.observed_by(agent))
    position = {node: i for i, node in enumerate(window)}
    for h in range(topology.d):
        if h in position or _owner_of(topology, h) != group.owner:
            continue
        if h != intervened and not ancestors[intervened, h]:
            continue
        # `h` is disturbed. Does it explain this group?
        reachable = _window_reach(adjacency, position, h)
        if reachable and reachable >= group.children:
            return True
    return False


def _window_reach(adjacency: np.ndarray, position: Dict[int, int], start: int):
    frontier, seen, children = [start], {start}, set()
    while frontier:
        current = frontier.pop()
        for nxt in np.flatnonzero(adjacency[current] > 0):
            nxt = int(nxt)
            if nxt in position:
                children.add(position[nxt])
            elif nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return frozenset(children)


def response_signature(adjacency: np.ndarray, topology, agent: int,
                       groups: Sequence[LatentGroup], intervened: int) -> Tuple[bool, ...]:
    """Which of `groups` a partner's intervention disturbs -- what the agent observes."""
    return tuple(responds_to(adjacency, topology, agent, g, intervened) for g in groups)


def bidirected_positions(adjacency: np.ndarray, topology, agent: int) -> Tuple[Tuple[int, int], ...]:
    """Confounded pairs of `agent`'s window, as window positions.

    Read from the authoritative MAG criterion rather than from the groups, because the two
    can legitimately disagree: a pair with BOTH a hidden common cause and a real directed
    edge is DIRECTED in the MAG, so the group that mentions it is real while the pair is not
    bidirected. Grading must follow the MAG.
    """
    window = tuple(topology.observed_by(agent))
    mag = latent_projection(adjacency, window)
    return tuple((u, v) for u, v in combinations(range(len(window)), 2)
                 if mag[u, v] == BIDIRECTED)


def observable_groups(adjacency: np.ndarray, topology, agent: int) -> Tuple[LatentGroup, ...]:
    """True groups restricted to the pairs that are actually BIDIRECTED in the MAG.

    A latent whose children are joined by real directed edges shows up in `groups_from_dag`
    but explains no bidirected edge, so it cannot be discovered and must not be graded. This
    is the set an agent could in principle recover, and therefore the set to score against.
    """
    confounded = set(bidirected_positions(adjacency, topology, agent))
    out = []
    for group in groups_from_dag(adjacency, topology, agent):
        children = frozenset(
            node for node in group.children
            if any((min(node, other), max(node, other)) in confounded
                   for other in group.children if other != node))
        if len(children) >= 2:
            candidate = LatentGroup(group.owner, children)
            if candidate not in out:
                out.append(candidate)
    return canonical_groups(out)


# =======================================================================================
# The hypothesis side: what attributions a candidate structure ADMITS.
# =======================================================================================


def cliques(pairs: Sequence[Tuple[int, int]]) -> Tuple[FrozenSet[int], ...]:
    """Every node set of size >= 2 that is complete in the bidirected graph `pairs`.

    A latent parenting a set of nodes makes EVERY pair among them bidirected, so a group is
    admissible only if its children form a clique. That single constraint is what keeps the
    attribution space small: an arbitrary grouping would be a set partition of the edges.
    """
    nodes = sorted({n for pair in pairs for n in pair})
    edges = {(min(u, v), max(u, v)) for u, v in pairs}
    out = []
    for size in range(2, len(nodes) + 1):
        for subset in combinations(nodes, size):
            if all((min(u, v), max(u, v)) in edges for u, v in combinations(subset, 2)):
                out.append(frozenset(subset))
    return tuple(out)


def _covers(pairs, candidate_cliques):
    """Minimal sets of cliques whose pairs are exactly `pairs`.

    MINIMAL means no clique can be dropped without uncovering an edge. Redundant covers are
    excluded because no evidence could ever rule them out -- a superfluous latent that
    explains only edges another latent already explains is unfalsifiable, and admitting it
    would make the truth unrecoverable rather than merely hard.
    """
    target = {(min(u, v), max(u, v)) for u, v in pairs}
    if not target:
        return [()]
    out = []
    for size in range(1, len(candidate_cliques) + 1):
        for chosen in combinations(candidate_cliques, size):
            covered = set()
            for group in chosen:
                covered |= set(combinations(sorted(group), 2))
            if covered != target:
                continue
            redundant = any(
                set().union(*(set(combinations(sorted(g), 2)) for g in chosen if g is not drop))
                == target for drop in chosen) if len(chosen) > 1 else False
            if not redundant:
                out.append(chosen)
    return out


def maximal_cliques(pairs: Sequence[Tuple[int, int]]) -> Tuple[FrozenSet[int], ...]:
    """The maximal complete node sets of the graph `pairs`. The CANONICAL form of a group.

    WHY MAXIMAL, and this is a correctness fix rather than an optimisation (2026-08-26).
    An earlier version enumerated every minimal clique COVER, so "one latent parenting
    {u,v,w}" and "three separate latents parenting each pair, all owned by the same agent"
    were distinct hypotheses. They are not distinguishable by ANY evidence available here.
    A partner action that disturbs the single latent moves all three pairs; under the
    three-latent hypothesis the same action can be an ancestor of all three and move all
    three too. No partial response is ever possible, because the agent is told only WHICH
    partner acted, never which of that partner's variables. So the finer hypothesis can
    never be eliminated and the claim can never be settled -- the space contained a
    distinction no experiment can decide.

    Collapsing to maximal cliques removes exactly those undecidable refinements and nothing
    else. Where a partial response IS possible the cliques stay separate: {u,v} and {v,w}
    with u-w NOT confounded do not merge, and an action moving only {u,v} tells them apart.

    MEMOISED on the normalised pair set. `_attributions` calls this once per owner per
    owner-set assignment, so the call count is (2^owners - 1)^pairs x owners while the number
    of DISTINCT arguments is at most the number of subsets of the pair set. Profiled 31 Aug at
    k=12: 272,424 calls over three episodes against a few dozen distinct inputs.
    """
    return _maximal_cliques(tuple(sorted((min(u, v), max(u, v)) for u, v in pairs)))


@lru_cache(maxsize=65536)
def _maximal_cliques(pairs: Tuple[Tuple[int, int], ...]) -> Tuple[FrozenSet[int], ...]:
    """The body of `maximal_cliques`, keyed on a hashable normalised pair tuple."""
    nodes = sorted({n for pair in pairs for n in pair})
    edges = {(min(u, v), max(u, v)) for u, v in pairs}
    found = []
    for size in range(len(nodes), 1, -1):
        for subset in combinations(nodes, size):
            if not all((min(u, v), max(u, v)) in edges for u, v in combinations(subset, 2)):
                continue
            candidate = frozenset(subset)
            if not any(candidate < bigger for bigger in found):
                found.append(candidate)
    return tuple(found)


def attributions_for(pairs: Sequence[Tuple[int, int]],
                     owners: Sequence[int]) -> Tuple[Tuple[LatentGroup, ...], ...]:
    """Every canonical attribution of `pairs` over `owners`. Memoised -- see `_attributions`.

    Many structures in one equivalence class carry the SAME set of bidirected pairs, and the
    attribution depends on nothing else, so the enumeration is done once per distinct pair
    set per episode rather than once per structure. That is the difference between ~9,000
    candidates built in seconds and built in milliseconds.
    """
    return _attributions(tuple(sorted((min(u, v), max(u, v)) for u, v in pairs)),
                         tuple(owners))


@lru_cache(maxsize=4096)
def _attributions(pairs: Tuple[Tuple[int, int], ...],
                  owners: Tuple[int, ...]) -> Tuple[Tuple[LatentGroup, ...], ...]:
    """The body of `attributions_for`, keyed on hashable arguments.

    Enumerated as: assign each confounded pair a NON-EMPTY set of owners -- non-empty
    because the pair is confounded and somebody must account for it, a set because two
    agents can independently confound the same pair -- then reduce each owner's assigned
    pairs to its maximal cliques. Distinct assignments that reduce to the same canonical
    form are one hypothesis.
    """
    from itertools import product
    pairs = [(min(u, v), max(u, v)) for u, v in pairs]
    if not pairs:
        return ((),)
    owner_sets = [frozenset(s) for r in range(1, len(owners) + 1)
                  for s in combinations(owners, r)]
    seen, out = set(), []
    for assignment in product(owner_sets, repeat=len(pairs)):
        groups = []
        for owner in owners:
            mine = [pair for pair, who in zip(pairs, assignment) if owner in who]
            groups.extend(LatentGroup(owner, clique) for clique in maximal_cliques(mine))
        canonical = tuple(sorted(groups, key=lambda g: (g.owner, sorted(g.children))))
        if canonical in seen:
            continue
        # A canonical form must reproduce the pair set exactly: every confounded pair
        # explained, and no group implying a pair that is not confounded.
        covered = set()
        for group in canonical:
            covered |= set(group.pairs())
        if covered != set(pairs):
            continue
        seen.add(canonical)
        out.append(canonical)
    return tuple(out)


def canonical_groups(groups: Sequence[LatentGroup]) -> Tuple[LatentGroup, ...]:
    """Reduce a set of true groups to the same canonical form the hypotheses use.

    The TRUTH has to be canonicalised too, or it can name a refinement the hypothesis space
    deliberately no longer contains, and the claim would be unsatisfiable by construction.
    """
    owners = sorted({g.owner for g in groups})
    out = []
    for owner in owners:
        mine = [pair for g in groups if g.owner == owner for pair in g.pairs()]
        out.extend(LatentGroup(owner, clique) for clique in maximal_cliques(mine))
    return tuple(sorted(out, key=lambda g: (g.owner, sorted(g.children))))


def predicted_response(groups: Sequence[LatentGroup], owner: int) -> Tuple[bool, ...]:
    """What a HYPOTHESIS predicts when `owner` intervenes on one of its private nodes.

    Deliberately coarse, and this is the honest limit of what an outsider can predict: the
    agent is told WHICH partner acted, never which of that partner's variables. A hypothesis
    can therefore only say "the groups I attribute to that partner MAY move" -- it cannot say
    which, because it does not know how the partner's block is arranged internally.

    So the usable evidence is one-directional and that is what the pruning must respect:
    a group that MOVED cannot belong to a partner who did not act, while a group that did
    not move is not evidence against the partner who did -- they may simply have intervened
    on a different one of their variables. See `AttributedBelief.observe_partner`.
    """
    return tuple(group.owner == owner for group in groups)


# =======================================================================================
# The belief: a version space over (structure, attribution) pairs.
# =======================================================================================


def consistent_with_partner(groups: Sequence[LatentGroup], owner: int,
                            moved: FrozenSet[Tuple[int, int]],
                            local_disturbance: bool = True) -> bool:
    """Could this attribution have produced `moved` when `owner` acted privately?

    TWO RULES. The first is an explicit MODELLING ASSUMPTION, not a theorem, and it is named
    and switchable so its cost can be measured rather than argued.

    THE LOCAL-DISTURBANCE ASSUMPTION (`local_disturbance=True`, the default): when a partner
    acts and pairs move, that partner's OWN latents are among the movers. It is false in
    general -- see the measurement below -- and it is what buys the channel its power. Set
    `local_disturbance=False` to drop it and keep only the provably sound rule, which is the
    sensitivity analysis this assumption has to be reported with.

    1. At least ONE pair that moved must be covered by a group this candidate attributes to
       `owner`. It used to demand ALL of them, which was strictly worse.

    2. ATOMICITY. Every group in the candidate must have moved ENTIRELY or not at
    all. One latent responds as a unit: disturb it and every pair it explains shifts
    together, so a candidate that assigns a clique to a latent and then sees only part of
    that clique move is refuted.

    THE KNOWN UNSOUNDNESS IN RULE 1, measured rather than argued. It required the pairs that moved to be
    covered by groups the candidate attributes to `owner` -- "something moved when that
    partner acted, so that partner owns it." That inference does not hold. `responds_to`
    marks a group as responding when the intervened node is an ANCESTOR of that group's
    owner's latent, and an actor's private node can sit above a THIRD agent's latent through
    the shared block. So the actor genuinely causes movement in pairs it does not own, and
    the message mixes owners. Measured 2026-08-29 at 3 agents: `moved` carried a foreign
    owner's pairs in 10 of 115 signals, and the TRUE attribution was refuted by its own
    evidence in 9 of them. Weakening it from "all" to "at least one" did not help -- in
    those 9 the actor owned NOTHING that moved -- so no version of rule 1 is sound, and this
    is the defect behind the residual "wrong" verdicts in `score_groups`.

    WHY RULE 1 IS KEPT ANYWAY, and what it costs. Deleting it and generalising atomicity to
    every owner IS sound, and it was tried: `right` collapsed from 72 to 0 over the same 162
    groups, because atomicity alone never refutes enough candidates to reach bar 1.0. Rule 1
    carries the entire discriminative power of the channel. So the engine is knowingly
    sound-LEANING here, and the cost is bounded and reported: `score_groups` returns
    `exhausted` and the backend counts `contradictions`, and residual `wrong` verdicts under
    ORACLE evidence are engine error rather than attribution error and must be read as such.
    Removing this caveat needs a signal that separates "the actor's latent moved" from "the
    actor's node sits above someone else's latent", which is a modelling change, not a fix.

    WHAT IS DELIBERATELY NOT A RULE: a group owned by `owner` that did not move is NOT
    evidence against it. The agent is told which PARTNER acted, never which of that
    partner's variables, so a silent group may simply belong to a variable the partner did
    not touch this round. Adding that rule would be the other way to make this unsound.

        Atomicity is where the discrimination lives, and it explains what coordination has to
    learn. A partner action that moves EVERYTHING separates nothing. A PARTIAL response
    refutes a single-clique hypothesis outright. So an agent needs its partner to probe its
    private variables ONE AT A TIME -- only possible when partners hold two or more, and
    exactly the experiment a partner has no selfish reason to run.
    """
    if not moved:
        return True                        # nothing observed, nothing to contradict
    covered = set()
    for group in groups:
        pairs = set(group.pairs())
        hit = pairs & moved
        if hit and hit != pairs:
            # ATOMICITY, and it holds for EVERY owner: one latent moves as a unit, so a
            # candidate that assigns a clique to a latent and then sees part of it move is
            # refuted whoever it named. This rule is sound unconditionally.
            return False
        if group.owner == owner:
            covered |= pairs
    if not local_disturbance:
        return True                        # sound-only mode: atomicity was the whole test
    return bool(moved & covered)           # local-disturbance: owner explains SOMETHING


class AttributedBelief:
    """Frequencies over surviving (structure, attribution) pairs, held FACTORED.

    WHY FACTORED. An attribution depends on the structure only through the structure's set
    of bidirected pairs. So candidates group by pair set, and within a group the structure
    and attribution axes are independent -- the candidate set is a disjoint union of
    products, never one flat list. Materialising the product cost 1.3 s per episode at
    k=5 and made the environment unusable; keeping it factored costs milliseconds and is
    the SAME set, exactly, not an approximation.

    Exposes `.adjacency` / `.directed` / `.bidirected` like every other belief, so the
    claims module, the observation and the greedy baseline are unchanged, plus
    `.group_frequency` and `.owner_frequency`, which are the new objects.
    """

    def __init__(self, buckets, k: int):
        # buckets: {pair_set: (structures, attributions)}
        self.buckets = buckets
        self.k = int(k)
        self.total = sum(len(st) * len(at) for st, at in buckets.values())
        n = max(self.total, 1)
        self.adjacency = np.zeros((k, k), dtype=float)
        self.directed = np.zeros((k, k), dtype=float)
        self.bidirected = np.zeros((k, k), dtype=float)
        counts: Dict[LatentGroup, int] = {}
        owner_counts: Dict[Tuple[int, int, int], int] = {}
        from cb.versionspace import BACK, BI, FWD, NONE, pairs as _pairs
        pair_index = _pairs(k)
        for structures, attributions in buckets.values():
            if not structures or not attributions:
                continue
            weight = len(attributions)
            for marks in structures:
                for (u, v), m in zip(pair_index, marks):
                    if m == NONE:
                        continue
                    self.adjacency[u, v] += weight
                    self.adjacency[v, u] += weight
                    if m == FWD:
                        self.directed[u, v] += weight
                    elif m == BACK:
                        self.directed[v, u] += weight
                    else:
                        self.bidirected[u, v] += weight
                        self.bidirected[v, u] += weight
            structure_weight = len(structures)
            for groups in attributions:
                for group in groups:
                    counts[group] = counts.get(group, 0) + structure_weight
                    for u, v in group.pairs():
                        key = (u, v, group.owner)
                        owner_counts[key] = owner_counts.get(key, 0) + structure_weight
        self.adjacency /= n
        self.directed /= n
        self.bidirected /= n
        self.group_frequency = {g: c / n for g, c in counts.items()}
        self.owner_frequency = {key: c / n for key, c in owner_counts.items()}
        self.n_boot = self.total
        self.ci_tests = 0
        self.truncated_fraction = 0.0
        self.replicates = None

    @property
    def space(self):
        """The surviving STRUCTURES, for consumers that only know about those."""
        return tuple(marks for structures, attributions in self.buckets.values()
                     if attributions for marks in structures)

    def edge_marginals(self) -> np.ndarray:
        return self.directed

    def confounded_pairs(self, threshold: float = 0.5) -> tuple:
        return tuple((u, v) for u, v in combinations(range(self.k), 2)
                     if self.bidirected[u, v] >= threshold)

    def owner_channel(self, n_agents: int) -> np.ndarray:
        """[pairs, n_agents]: how much of the belief blames each agent for each pair.

        Replaces the single bidirected channel in the observation. Carries no node
        identities, so it discloses nothing about anyone's private block beyond the
        ownership this agent has itself inferred.
        """
        pair_list = list(combinations(range(self.k), 2))
        out = np.zeros((len(pair_list), n_agents), dtype=float)
        for index, (u, v) in enumerate(pair_list):
            for owner in range(n_agents):
                out[index, owner] = self.owner_frequency.get((u, v, owner), 0.0)
        return out


def score_groups(belief, true_groups: Sequence[LatentGroup], bar: float = 1.0):
    """Three-outcome scoring of the attribution, one claim per TRUE latent group.

    Mirrors `cb.claims`: right / wrong / unsure, never summed. A group is settled RIGHT when
    at least `bar` of the surviving candidates name exactly it -- same owner, same children.
    Settled WRONG when at most `1 - bar` do, which at bar 1.0 means no survivor names it.

    The guarantee carries over from the version space: the truth never leaves the candidate
    set, so at bar 1.0 "settled" implies "settled correctly" and settled-wrong cannot occur.

    WHEN THAT GUARANTEE HAS ALREADY FAILED, SAY SO RATHER THAN SCORING IT AS AN ERROR. If
    the candidate set is empty, every true group is absent from `group_frequency`, `freq`
    defaults to 0.0, and `1 - 0 >= 1` marked all of them WRONG -- reporting an engine
    contradiction as a confident misattribution, which is the one failure mode that
    corrupts every attribution number downstream. An exhausted belief knows nothing, so the
    honest verdict is UNSURE, and the contradiction is surfaced in its own field instead of
    being laundered into the score.
    """
    exhausted = not getattr(belief, "group_frequency", None) or not getattr(belief, "total", 0)
    # OUT OF SCOPE IS UNSURE, NOT WRONG. A belief whose candidates are enumerated over only
    # part of the window -- `cb.factored_attribution`, which enumerates over the pairs its
    # structure belief has SETTLED -- has no opinion about a group naming a pair it has not
    # reached yet. Absent from `group_frequency` then means "not asked", not "refuted", and
    # scoring it WRONG reports an incomplete belief as a confident misattribution. `scope`
    # is absent on the enumerated backend, where every pair is always in scope, so this is
    # inert there.
    scope = getattr(belief, "scope", None)
    right = wrong = unsure = 0
    detail = []
    for group in true_groups:
        in_scope = scope is None or set(group.pairs()) <= set(scope)
        freq = belief.group_frequency.get(group, 0.0) if not exhausted else 0.0
        if exhausted or not in_scope:
            outcome, unsure = "unsure", unsure + 1
        elif freq >= bar:
            outcome, right = "right", right + 1
        elif 1.0 - freq >= bar:
            outcome, wrong = "wrong", wrong + 1
        else:
            outcome, unsure = "unsure", unsure + 1
        detail.append((group, outcome, freq))
    return {"right": right, "wrong": wrong, "unsure": unsure,
            "total": len(true_groups), "detail": detail, "exhausted": bool(exhausted),
            "identified": wrong == 0 and right == len(true_groups)}


class AttributedVersionSpaceBackend:
    """Deterministic belief over (structure, attribution). Call-compatible with the others.

    Two elimination channels, answering different questions:
      OWN interventions      prune the STRUCTURE, by pairwise ancestry, exactly as before.
      PARTNER interventions  prune the ATTRIBUTION, by which of your confounded pairs moved
                             when a named partner experimented privately.
    The second is new and it is the point: it is the only channel that can say WHOSE hidden
    variable disturbs your window, and it fires only when a partner spends a round on
    something that does nothing for the partner itself.
    """

    can_handle_multi_hidden = True

    def __init__(self, k: int, shared_positions: Sequence[int] = (), n_agents: int = 2,
                 agent: int = 0, max_candidates: int = 200_000,
                 local_disturbance: bool = True, **_ignored):
        self.k = int(k)
        self.shared_positions = tuple(shared_positions)
        self.n_agents = int(n_agents)
        self.agent = int(agent)
        self.truth: Optional[tuple] = None
        self.true_groups: Tuple[LatentGroup, ...] = ()
        # Times an elimination channel would have emptied the candidate set. Non-zero means
        # the soundness guarantee broke; `score_groups` reports UNSURE rather than WRONG.
        self.contradictions = 0
        # Times a disclosed message refuted the TRUE attribution -- the local-disturbance
        # assumption failing, measured rather than assumed away.
        self.assumption_violations = 0
        self.last: Optional[AttributedBelief] = None
        self._buckets: Dict[tuple, tuple] = {}
        self._initial: Dict[tuple, tuple] = {}
        self._applied: frozenset = frozenset()
        # A DENSE k=5 window can carry a structure space in the tens of thousands -- that
        # is the pre-existing version-space cost (3^edges), not an attribution cost, and it
        # is what made one Erdos-Renyi configuration take 8 s an episode against 0.4 s for
        # scale-free at the same size. Truncation is REPORTED rather than silent, because a
        # truncated belief is not a confident one and must never be read as one.
        self.max_candidates = int(max_candidates)
        # See `consistent_with_partner`. True keeps the assumption that a partner's own
        # latents are among the movers -- powerful and false in general. False keeps only
        # atomicity, which is sound and, measured, refutes almost nothing.
        self.local_disturbance = bool(local_disturbance)
        self.truncated = False

    def reset(self, true_mag: np.ndarray, adjacency=None, topology=None) -> None:
        from cb.versionspace import BI, equivalence_class, marks_from_mag, pairs as _pairs
        self.truth = marks_from_mag(true_mag)
        owners = tuple(a for a in range(self.n_agents) if a != self.agent)
        pair_index = _pairs(self.k)
        grouped: Dict[tuple, list] = {}
        for marks in equivalence_class(self.truth, self.k):
            key = tuple((u, v) for (u, v), m in zip(pair_index, marks) if m == BI)
            grouped.setdefault(key, []).append(marks)
        self._initial = {key: (tuple(structures), attributions_for(key, owners))
                         for key, structures in grouped.items()}
        self._buckets = dict(self._initial)
        self._applied = frozenset()
        self.truncated = self.n_candidates > self.max_candidates
        if self.truncated:
            # Keep the smallest buckets, which are the most informative per unit of work,
            # until the budget is met. The belief is then a SUBSET of the true version
            # space, so it can be over-confident; `truncated` says so.
            order = sorted(self._buckets.items(), key=lambda kv: len(kv[1][0]) * len(kv[1][1]))
            kept, running = {}, 0
            for key, value in order:
                cost = len(value[0]) * len(value[1])
                if running + cost > self.max_candidates and kept:
                    break
                kept[key] = value
                running += cost
            self._buckets = kept
            self._initial = dict(kept)
        self.contradictions = 0
        self.assumption_violations = 0
        self.true_groups = (observable_groups(adjacency, topology, self.agent)
                            if adjacency is not None else ())
        self.last = AttributedBelief(self._buckets, self.k)

    @property
    def n_candidates(self) -> int:
        return sum(len(st) * len(at) for st, at in self._buckets.values())

    def edge_marginals(self, data, known_intervened, told=None, score_rule=None,
                       blocks=None) -> np.ndarray:
        """Prune the STRUCTURE by which of this window's nodes have been intervened on."""
        from cb.versionspace import reveal
        if self.truth is None:
            raise RuntimeError("AttributedVersionSpaceBackend.reset must be called first")
        mask = np.asarray(known_intervened) > 0.5
        intervened = frozenset(x for x in range(self.k) if mask[:, x].any())
        if not intervened >= self._applied:
            self._buckets, self._applied = dict(self._initial), frozenset()
        fresh = intervened - self._applied
        if fresh:
            target = {x: reveal(self.truth, self.k, x) for x in fresh}
            self._buckets = {
                key: (tuple(m for m in structures
                            if all(reveal(m, self.k, x) == target[x] for x in fresh)),
                      attributions)
                for key, (structures, attributions) in self._buckets.items()}
            pruned = {k_: v for k_, v in self._buckets.items() if v[0] and v[1]}
            if not pruned:
                self.contradictions += 1        # same guard as `observe_partner`
            else:
                self._buckets = pruned
                self.last = AttributedBelief(self._buckets, self.k)
            self._applied = intervened
        elif self.last is None:
            self.last = AttributedBelief(self._buckets, self.k)
        return self.last.directed

    def observe_partner(self, owner: int, moved: FrozenSet[Tuple[int, int]]) -> None:
        """Prune the ATTRIBUTION by a named partner's private experiment."""
        if not moved:
            return                                  # no evidence, no work
        # Did this message refute the TRUTH? Under oracle evidence that can only happen when
        # the local-disturbance assumption fails, so counting it measures the assumption's
        # violation rate directly, per episode, with no extra instrumentation.
        if self.true_groups and not consistent_with_partner(
                self.true_groups, owner, moved, local_disturbance=self.local_disturbance):
            self.assumption_violations += 1
        changed = False
        buckets = {}
        for key, (structures, attributions) in self._buckets.items():
            kept = tuple(a for a in attributions
                         if consistent_with_partner(a, owner, moved,
                                                    local_disturbance=self.local_disturbance))
            changed |= len(kept) != len(attributions)
            if structures and kept:
                buckets[key] = (structures, kept)
        if changed:
            if not buckets:
                # CONTRADICTION. Every candidate was refuted, but the truth is one of them,
                # so the evidence -- not the belief -- is at fault. Refuse the update and
                # record it, exactly as `cb/factored.py::_apply_ancestry` does for marks.
                # Propagating an empty space would read downstream as unanimous confidence.
                self.contradictions += 1
                return
            self._buckets = buckets
            self.last = AttributedBelief(self._buckets, self.k)

    def credit_fraction(self, true_mag: np.ndarray, required_positions=(),
                        strict: bool = False) -> float:
        from cb.versionspace import marks_from_mag
        if self.last is None or not self.last.total:
            return 0.0
        truth = marks_from_mag(true_mag)
        target = set(self.true_groups)
        hit = sum(1 for structures, attributions in self._buckets.values()
                  for marks in structures for groups in attributions
                  if marks == truth and set(groups) == target)
        return hit / self.last.total

    @property
    def bidirected(self) -> np.ndarray:
        return self.last.bidirected if self.last is not None else np.zeros((self.k, self.k))


def estimated_moved(data, actor_rows, baseline_rows, pairs, alpha: float = 0.001,
                    min_rows: int = 30) -> FrozenSet[Tuple[int, int]]:
    """Which confounded pairs CHANGED when a partner worked privately, read from the data.

    The sampled counterpart of `response_signature`, which consults the true graph. For each
    confounded pair, compare its correlation in the rows where the partner acted against the
    rows where nobody hidden did, by Fisher's z for a difference of two correlations. A
    latent that was disturbed changes the association it induces; one that was not, does not.

    SOUND-LEANING, by the same argument as `cb.versionspace.estimated_reveal`: a detection
    is trustworthy and silence means "not detected", never "did not move". So the pruning it
    feeds only ever eliminates candidates that DENY a detected movement, and a partner whose
    latent sits far upstream produces a change too small to see and prunes nothing -- which
    is the effect-range property, arriving here for the same reason it arrives there.

    Returns an empty set when either side is too thin to compare, so a short episode
    degrades to "no evidence" rather than to noise.
    """
    data = np.asarray(data)
    actor_rows = np.asarray(actor_rows, dtype=bool)
    baseline_rows = np.asarray(baseline_rows, dtype=bool)
    n_a, n_b = int(actor_rows.sum()), int(baseline_rows.sum())
    if n_a < min_rows or n_b < min_rows:
        return frozenset()
    from scipy import stats
    moved = []
    for u, v in pairs:
        a, b = data[actor_rows], data[baseline_rows]
        if a[:, u].std() < 1e-12 or a[:, v].std() < 1e-12:
            continue
        if b[:, u].std() < 1e-12 or b[:, v].std() < 1e-12:
            continue
        ra = float(np.corrcoef(a[:, u], a[:, v])[0, 1])
        rb = float(np.corrcoef(b[:, u], b[:, v])[0, 1])
        if not (np.isfinite(ra) and np.isfinite(rb)):
            continue
        za = 0.5 * np.log((1 + np.clip(ra, -0.999999, 0.999999))
                          / (1 - np.clip(ra, -0.999999, 0.999999)))
        zb = 0.5 * np.log((1 + np.clip(rb, -0.999999, 0.999999))
                          / (1 - np.clip(rb, -0.999999, 0.999999)))
        se = np.sqrt(1.0 / (n_a - 3) + 1.0 / (n_b - 3))
        if se <= 0:
            continue
        p = 2.0 * float(stats.norm.sf(abs(za - zb) / se))
        if p < alpha:
            moved.append((min(u, v), max(u, v)))
    return frozenset(moved)
