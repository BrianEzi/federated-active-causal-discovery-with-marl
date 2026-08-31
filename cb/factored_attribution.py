"""Attribution at scale: factored structure, enumerated ownership.

WHY THIS EXISTS, AND WHY IT IS SMALLER THAN IT LOOKS. `AttributedVersionSpaceBackend` was
believed to cap at k~5, and attribution was cut from the thesis on that basis. Measured
31 Aug 2026, the cap is not attribution's:

    k    edges/window   structure space 3^E   attribution hypotheses   binding
    4         3.9                       69                        5    structure
    6         7.9                    5,700                        9    structure
    8        12.7                1.18e+06                       35    structure
   12        22.4                5.01e+10                      482    structure

The two enumerations are INDEPENDENT and only one of them is expensive. An attribution
depends on the structure only through the structure's set of BIDIRECTED pairs -- which
`AttributedBelief` already exploits by bucketing -- so the structure belief can be swapped
for one that does not enumerate at all, and the attribution belief carries on unchanged at
482 hypotheses. That is what this class does:

    STRUCTURE     `cb.factored.FactoredBackend` -- per-pair surviving marks, O(k^2), the
                  backend the rest of the thesis already runs on. Scales to k=30.
    ATTRIBUTION   the existing enumerated hypotheses over owner assignments, pruned by
                  `consistent_with_partner`, exactly as before. Exact, not approximated.

WHICH PAIRS ATTRIBUTION IS ABOUT. Only pairs the structure belief has SETTLED as bidirected
(surviving marks == {BI}). Not merely-possible pairs: at reset every adjacent pair still
admits BI, so keying on possibility would enumerate over every pair in the window and
reproduce the blow-up this class exists to avoid. Settled pairs appear as the agent covers
its window, so the attribution set grows during an episode rather than starting complete.

WHY A REPLAY LOG. A partner's message can arrive before the pair it speaks about has been
settled. Rather than drop that evidence or apply it to a candidate set that does not yet
exist, every message is kept and the whole log is replayed whenever the settled set changes.
Replaying is cheap (the log is at most one entry per round) and it makes the result
INDEPENDENT OF ARRIVAL ORDER, which a prune-on-receipt design would not be.

WHAT IS LOST relative to the enumerated backend. The structure side inherits
`FactoredBelief`'s conservatism: it stays unsure where the enumeration would have settled a
mark by joint reasoning, so pairs enter the attribution set later. Attribution itself is
unchanged -- the same hypotheses, the same pruning rule, the same soundness argument.
`tests/crosscheck/` pins agreement with the enumerated backend where both can run.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from cb.attribution import (LatentGroup, attributions_for, consistent_with_partner,
                            observable_groups)
from cb.factored import FactoredBackend
from cb.versionspace import BI


class FactoredAttributedBelief:
    """Structure frequencies from the factored belief, ownership from the hypotheses.

    Exposes the same surface as `AttributedBelief` -- `.adjacency` / `.directed` /
    `.bidirected` for the claims module and the observation, plus `.group_frequency`,
    `.owner_frequency` and `.owner_channel` for the attribution consumers -- so nothing
    downstream needs to know which backend produced it.
    """

    def __init__(self, structure, attributions, k: int, scope=()):
        self.k = int(k)
        self.adjacency = structure.adjacency
        self.directed = structure.directed
        self.bidirected = structure.bidirected
        self.attributions = tuple(attributions)
        self.total = len(self.attributions)

        groups: Counter = Counter()
        owners: Counter = Counter()
        for hypothesis in self.attributions:
            for group in hypothesis:
                groups[group] += 1
                for pair in group.pairs():
                    owners[(pair[0], pair[1], group.owner)] += 1
        denominator = max(self.total, 1)
        self.group_frequency = {g: c / denominator for g, c in groups.items()}
        self.owner_frequency = {key: c / denominator for key, c in owners.items()}
        # THE PAIRS THIS BELIEF CAN SPEAK ABOUT AT ALL. Candidates are enumerated over the
        # pairs the structure belief has SETTLED as bidirected, so a true group naming a
        # pair that is not settled yet appears in no hypothesis and has frequency zero --
        # which `score_groups` would otherwise read as "confidently wrong" rather than "no
        # opinion yet". Measured before this field existed: 16 of 76 true groups scored
        # WRONG at k=6 with no partner evidence at all, which is the one failure mode that
        # corrupts every attribution number downstream. Out of scope means UNSURE.
        self.scope = frozenset(scope or ())

        # Reporting fields the other beliefs carry, so nothing downstream special-cases this.
        self.n_boot = 0
        self.ci_tests = 0
        self.truncated_fraction = 0.0
        self.replicates = None

    @property
    def space(self):
        """The factored belief does not enumerate structures, so there is no candidate list
        to expose. Empty rather than absent, so consumers that iterate it simply find
        nothing instead of raising."""
        return ()

    def edge_marginals(self) -> np.ndarray:
        return self.directed

    def confounded_pairs(self, threshold: float = 0.5) -> tuple:
        return tuple((u, v) for u, v in combinations(range(self.k), 2)
                     if self.bidirected[u, v] >= threshold)

    def owner_channel(self, n_agents: int) -> np.ndarray:
        """[pairs, n_agents] -- how much of the belief blames each agent for each pair.

        Carries no node identities, so it discloses nothing about anyone's private block
        beyond the ownership this agent has itself inferred.
        """
        pair_list = list(combinations(range(self.k), 2))
        out = np.zeros((len(pair_list), n_agents), dtype=float)
        for index, (u, v) in enumerate(pair_list):
            for owner in range(n_agents):
                out[index, owner] = self.owner_frequency.get((u, v, owner), 0.0)
        return out


class FactoredAttributedBackend:
    """Call-compatible with the other backends; attribution that reaches k=12 and beyond."""

    can_handle_multi_hidden = True

    def __init__(self, k: int, shared_positions: Sequence[int] = (), n_agents: int = 2,
                 agent: int = 0, evidence: str = "oracle", evidence_alpha: float = 0.001,
                 max_attribution_pairs: int = 8, local_disturbance: bool = True,
                 **_ignored):
        self.k = int(k)
        self.n_agents = int(n_agents)
        self.agent = int(agent)
        self.structure = FactoredBackend(k, shared_positions=shared_positions,
                                         evidence=evidence, evidence_alpha=evidence_alpha)
        self.owners = tuple(a for a in range(self.n_agents) if a != self.agent)
        # The enumeration is (2^(n-1) - 1)^P in the settled-pair count P, so it is bounded
        # by capping P rather than by a candidate budget: 482 hypotheses at the measured
        # k=12 mean of 3.2 pairs, and 5.7 million at 8. Beyond the cap the newest settled
        # pairs are held back and `truncated` says so, because a belief that quietly
        # forgot half its evidence must never be read as a confident one.
        self.max_attribution_pairs = int(max_attribution_pairs)
        self.local_disturbance = bool(local_disturbance)

        self.truth = None
        self.true_groups: Tuple[LatentGroup, ...] = ()
        self.contradictions = 0
        self.assumption_violations = 0
        self.truncated = False
        self._log: list = []
        self._settled: Optional[tuple] = None
        self._attributions: tuple = ()
        self.last: Optional[FactoredAttributedBelief] = None

    # -- lifecycle -----------------------------------------------------------------------

    def reset(self, true_mag: np.ndarray, adjacency=None, topology=None) -> None:
        self.structure.reset(true_mag)
        self.truth = self.structure.truth
        self._log = []
        self._settled = None
        self._attributions = ()
        self.contradictions = 0
        self.assumption_violations = 0
        self.truncated = False
        self.true_groups = (observable_groups(adjacency, topology, self.agent)
                            if adjacency is not None and topology is not None else ())
        self._rebuild()

    # -- the two evidence channels -------------------------------------------------------

    def edge_marginals(self, data, known_intervened, told=None, score_rule=None,
                       blocks=None) -> np.ndarray:
        """Own interventions prune the STRUCTURE; the settled set may then grow."""
        directed = self.structure.edge_marginals(data, known_intervened, told=told,
                                                 score_rule=score_rule, blocks=blocks)
        self._rebuild()
        return directed

    def observe_partner(self, owner: int, moved: FrozenSet[Tuple[int, int]]) -> None:
        """A named partner's private experiment prunes the ATTRIBUTION."""
        if not moved:
            return                                   # no evidence, no work
        # Did this message refute the TRUTH? Under oracle evidence that can only happen when
        # the local-disturbance assumption fails, so counting it measures the assumption's
        # violation rate directly, per episode, with no extra instrumentation.
        if self.true_groups and not consistent_with_partner(
                self.true_groups, owner, moved, local_disturbance=self.local_disturbance):
            self.assumption_violations += 1
        self._log.append((int(owner), frozenset(moved)))
        self._rebuild(force=True)

    # -- the attribution belief ----------------------------------------------------------

    def settled_bidirected(self) -> tuple:
        """Pairs the structure belief has settled as BIDIRECTED -- marks == {BI} exactly."""
        possible = self.structure._possible
        return tuple(pair for pair, marks in sorted(possible.items())
                     if marks == frozenset({BI}))

    def _rebuild(self, force: bool = False) -> None:
        settled = self.settled_bidirected()
        if settled == self._settled and not force:
            self.last = FactoredAttributedBelief(
                self.structure.last or self._empty_structure(),
                self._attributions, self.k, scope=self._settled or ())
            return
        kept = settled
        self.truncated = len(settled) > self.max_attribution_pairs
        if self.truncated:
            kept = settled[:self.max_attribution_pairs]
        candidates = attributions_for(kept, self.owners) if kept else ((),)
        # REPLAYED IN FULL, not pruned incrementally: a message may name a pair that was not
        # settled when it arrived, and replaying makes the result independent of the order
        # messages happened to come in.
        for owner, moved in self._log:
            survivors = tuple(a for a in candidates
                              if consistent_with_partner(
                                  a, owner, moved,
                                  local_disturbance=self.local_disturbance))
            if not survivors:
                # SKIP THE MESSAGE, DO NOT STOP. Refusing to empty the set is the same guard
                # the factored structure belief uses. Breaking out here instead discarded
                # every LATER message too, which both wasted sound evidence and left a
                # candidate set pruned by a prefix that could already exclude the truth --
                # measured at one confidently-wrong attribution in 76 before this changed.
                # A contradicting message is dropped and counted; the rest still apply.
                self.contradictions += 1
                continue
            candidates = survivors
        self._settled = settled
        self._attributions = candidates
        self.last = FactoredAttributedBelief(self.structure.last or self._empty_structure(),
                                             candidates, self.k, scope=kept)

    def _empty_structure(self):
        from cb.factored import FactoredBelief
        return FactoredBelief(self.structure._possible, self.k)

    # -- reporting -----------------------------------------------------------------------

    @property
    def n_candidates(self) -> int:
        return len(self._attributions)

    def credit_fraction(self, true_mag: np.ndarray, required_positions=(), **kwargs) -> float:
        """Delegated to the structure half, keyword arguments included.

        `**kwargs` rather than a fixed signature because the callers pass different things --
        `ma/evaluate.py` uses `strict=True` — and a backend that is call-compatible with the
        others has to accept whatever they accept. Dropping the argument silently would be
        worse than forwarding it, so it is forwarded.
        """
        return self.structure.credit_fraction(true_mag, required_positions, **kwargs)
