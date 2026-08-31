"""A belief that never materialises candidates: one small version space PER PAIR.

WHY THIS EXISTS. `cb/versionspace.py` holds the belief as an explicit set of whole-window
structures and enumerates 3^(edges). That is exact, it supports an exact ceiling and an
exact optimum, and it dies around k=6: a 30-node window with 40 edges would carry 3^40
candidates, and the m-separation check inside it enumerates PATHS, which is exponential too.
No constant factor reaches k=30.

THE FACT THAT MAKES FACTORING WORK, and it is the same one the whole interventional design
rests on: **the evidence is already pairwise**. For an ADJACENT pair,

    x is an ancestor of y   =>  the edge is x -> y
        (not y -> x, that would be a cycle; not x <-> y, because a bidirected edge forbids
         either endpoint being an ancestor of the other -- the almost-directed cycle rule)
    neither is an ancestor  =>  the edge is x <-> y

So intervening on both endpoints of a pair determines its mark with NO joint reasoning. The
candidate set was never needed to absorb this evidence; it was a representation choice.

WHAT IS REPRESENTED. Per pair, the SET of marks still possible -- a version space of size at
most four, held independently for each pair. 4 * C(k,2) numbers: 1,740 at k=30 against 10^19
candidates. Frequencies are uniform over each pair's surviving set, so every existing
consumer (`cb.claims`, the observation vector, the greedy baseline) reads it unchanged, and
"settled at bar 1.0" still means "one mark left", which still means "settled correctly"
provided each update is sound.

WHAT IS LOST, stated plainly. The enumerated belief also carries JOINT constraints --
ancestrality, maximality, and compatibility with the observed independencies -- which couple
edges and let it settle a mark by elimination ("given these three, that fourth combination
is impossible"). A factored belief cannot represent those. The consequence is CONSERVATIVE:
it stays unsure where the enumeration would have settled, and never settles wrongly, because
each update is individually sound. Slower convergence bought scale.

WHAT IS ALSO LOST, and this one costs a headline. The exact ceiling and the exact
optimal-rounds figures are computed by enumerating reachable intervention sets and pruning
(`scripts/vs_evaluate.py`). Without an enumerable belief they become bounds rather than
exact values, so "closed X% of the achievable headroom" is available at small k and not at
large k. That is a real limitation of the scale results and should be written as one.

NOT DONE HERE, and the obvious next step: constraint propagation. The standard FCI
orientation rules are local and polynomial, and would recover a good part of the joint
reasoning without materialising anything. `cb/orient.py` already implements them for the
statistical engine. Left out deliberately so that this backend's behaviour is exactly "what
pairwise interventional evidence alone can prove", which is the claim the scale results
should make.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from cb.versionspace import BACK, BI, FWD, NONE, marks_from_mag, pairs


class FactoredBelief:
    """Per-pair surviving marks, exposed as the frequency matrices every consumer reads."""

    def __init__(self, possible, k: int):
        # possible[(u, v)] -> frozenset of marks still admissible for that pair
        self.possible = possible
        self.k = int(k)
        self.adjacency = np.zeros((k, k), dtype=float)
        self.directed = np.zeros((k, k), dtype=float)
        self.bidirected = np.zeros((k, k), dtype=float)
        for (u, v), marks in possible.items():
            if not marks:
                continue
            weight = 1.0 / len(marks)
            for mark in marks:
                if mark == NONE:
                    continue
                self.adjacency[u, v] += weight
                self.adjacency[v, u] += weight
                if mark == FWD:
                    self.directed[u, v] += weight
                elif mark == BACK:
                    self.directed[v, u] += weight
                else:
                    self.bidirected[u, v] += weight
                    self.bidirected[v, u] += weight
        # Reporting fields the other beliefs carry, so nothing downstream special-cases this.
        self.n_boot = 0
        self.ci_tests = 0
        self.truncated_fraction = 0.0
        self.replicates = None

    @property
    def space(self):
        """The single settled structure, when every pair has one mark left; else empty.

        Deliberately NOT the product of the per-pair sets -- that product is exactly the
        object this class exists to avoid building, and it would also be wrong, since it
        contains combinations the joint constraints forbid.
        """
        if all(len(m) == 1 for m in self.possible.values()):
            return (tuple(next(iter(self.possible[p])) for p in pairs(self.k)),)
        return ()

    def edge_marginals(self) -> np.ndarray:
        return self.directed

    def confounded_pairs(self, threshold: float = 0.5) -> tuple:
        return tuple((u, v) for u, v in combinations(range(self.k), 2)
                     if self.bidirected[u, v] >= threshold)

    @property
    def settled(self) -> int:
        return sum(1 for m in self.possible.values() if len(m) == 1)


def credit_for_set(true_mag, k: int, positions) -> float:
    """Window credit had EXACTLY `positions` been intervened on. Oracle evidence only.

    This is what makes a difference reward computable rather than estimable here: under
    oracle evidence `edge_marginals` prunes from the SET of intervened nodes and nothing
    else, so "what would my window look like if I had never acted" is a replay, not a
    counterfactual guess. Rebuilt from scratch rather than rolled back, because
    `_apply_ancestry` only ever narrows a pair and undoing it is not defined.

    Denominator matches `FactoredBackend.credit_fraction` -- ALL pairs, not just adjacent
    ones -- so the two numbers can be subtracted from each other without a scale error.
    """
    from cb.versionspace import reveal

    backend = FactoredBackend(k)
    backend.reset(true_mag)
    backend.reset_marks()
    for x in sorted(positions):
        backend._apply_ancestry(x, reveal(backend.truth, k, x))
    truth = marks_from_mag(true_mag)
    hits = sum(1 for index, key in enumerate(pairs(k))
               if backend._possible[key] == frozenset({truth[index]}))
    return hits / max(len(backend._possible), 1)


class FactoredBackend:
    """Pairwise belief, updated by ancestry. O(k^2) state, O(k^2) per update.

    Call-compatible with the other backends. `evidence="oracle"` prunes by the true ancestry;
    `evidence="sampled"` prunes by what the data shows, with the same one-sided treatment of
    an under-powered test as `cb.versionspace` -- silence is not refutation.
    """

    can_handle_multi_hidden = True

    def __init__(self, k: int, shared_positions: Sequence[int] = (),
                 evidence: str = "oracle", evidence_alpha: float = 0.001,
                 assume_skeleton: bool = True, evidence_power: float = 1.0,
                 power_seed: int = 0, **_ignored):
        self.k = int(k)
        self.shared_positions = tuple(shared_positions)
        if evidence not in ("oracle", "sampled"):
            raise ValueError(f"evidence must be 'oracle' or 'sampled', got {evidence!r}")
        self.evidence = evidence
        self.evidence_alpha = float(evidence_alpha)
        # The ORACLE OBSERVATIONAL SKELETON, which is an accepted arm elsewhere in this
        # project (`oracle_obs_structure`). Adjacency is fixed and correct from the start;
        # only the marks are open. Without it the backend would need FCI's skeleton search,
        # which is where the polynomial cost actually lives -- and the thesis question is
        # about choosing EXPERIMENTS, not about estimating skeletons.
        self.assume_skeleton = bool(assume_skeleton)
        self.truth: Optional[tuple] = None
        self.last: Optional[FactoredBelief] = None
        self._possible: dict = {}
        self._applied: frozenset = frozenset()
        self._detected: dict = {}
        # Which pairs the SKELETON declared absent. Kept separately from `truth` so that
        # `reset_marks` can re-open the belief without silently re-deriving adjacency from
        # the true MAG, which under an estimated skeleton would hand back the very
        # assumption the ablation exists to remove.
        self._seeded_absent: dict = {}
        # POWER-LIMITED ORACLE EVIDENCE. `evidence_power` is the probability that a given
        # ancestry question yields a usable answer at all; the rest of the time the pair is
        # left untouched, exactly as if the test had run and been under-powered.
        #
        # WHY THIS SHAPE AND NOT NOISE ON THE ANSWER. Sampled evidence is SOUND BUT NOT
        # COMPLETE -- it never asserts a false ancestry, it only fails to detect weak and
        # distant ones, which is why the belief carries intermediate frequencies instead of
        # all-or-nothing marks. Corrupting the answer would break soundness and produce a
        # belief the truth can leave; declining to answer reproduces the real failure mode
        # and keeps the version-space guarantee intact. `evidence_power=1.0` is the
        # untouched oracle, so this is inert unless asked for.
        #
        # WHAT IT IS FOR. Measured 31 Aug: oracle training costs 0.085 s/episode and sampled
        # training 6.3-9.4, a factor of 74-110, which is why the sampled sweep needs a
        # cluster. Policies trained under oracle evidence do NOT transfer to sampled
        # (`FINDINGS_2026_08_27` section 3: 0.171 against random's 0.208) because they have
        # never seen a half-settled belief. This gives that input distribution at oracle
        # speed, so "train with the noise you will be tested under" becomes affordable.
        self.evidence_power = float(evidence_power)
        if not 0.0 < self.evidence_power <= 1.0:
            raise ValueError(f"evidence_power must be in (0, 1], got {evidence_power!r}")
        # NOT reseeded per episode, deliberately: the point is that the policy meets a
        # DIFFERENT pattern of missing evidence every episode, which is what domain
        # randomisation means here. Deterministic given `power_seed` for the run as a whole.
        self._power_rng = np.random.default_rng(int(power_seed))
        # Rows seen per node at the last update. A rising count means another experiment on
        # that node, which earns another draw against `evidence_power`.
        self._attempts: dict = {}

    def reset(self, true_mag: np.ndarray, adjacency=None, topology=None,
              skeleton: Optional[np.ndarray] = None) -> None:
        """`skeleton` overrides which pairs are treated as adjacent.

        WHY THE OVERRIDE EXISTS. The default seeds absence from `self.truth`, which reads as
        oracle knowledge and is not: a MAG's adjacencies are exactly the pairs no OBSERVED
        conditioning set can separate, so the skeleton is recoverable from observational data
        alone -- measured at 100% agreement over 4,710 pairs (see
        docs/FINDINGS_SKELETON_2026_08_31.md). What the default supplies is the INFINITE-DATA
        answer. Passing an estimated skeleton here is how the finite-sample cost of that
        supply gets measured instead of assumed away.

        A [k, k] boolean, True where the pair is taken to be adjacent. A pair the skeleton
        calls absent is closed to NONE; a pair it calls present is opened to
        {FWD, BACK, BI} -- INCLUDING a pair that is truly absent, which is exactly the
        damage a spurious adjacency does: it can never be settled, so it caps identification.
        """
        self.truth = marks_from_mag(true_mag)
        self._possible = {}
        self._seeded_absent = {}
        if skeleton is not None:
            skeleton = np.asarray(skeleton, dtype=bool)
            for index, (u, v) in enumerate(pairs(self.k)):
                absent = not bool(skeleton[u, v])
                self._seeded_absent[(u, v)] = absent
                self._possible[(u, v)] = (frozenset({NONE}) if absent
                                          else frozenset({FWD, BACK, BI}))
            self._applied = frozenset()
            self._attempts = {}
            self._detected = {}
            self.last = FactoredBelief(self._possible, self.k)
            return
        for index, (u, v) in enumerate(pairs(self.k)):
            self._seeded_absent[(u, v)] = self.truth[index] == NONE
            if self.truth[index] == NONE:
                # Correctly absent, and known to be: this is the observational half.
                self._possible[(u, v)] = frozenset({NONE})
            else:
                # Adjacent, orientation wide open. Deliberately NOT narrowed by observational
                # orientation rules -- see the module docstring; this backend reports what
                # pairwise interventional evidence alone can prove.
                self._possible[(u, v)] = frozenset({FWD, BACK, BI})
        self._applied = frozenset()
        self._attempts = {}
        self._detected = {}
        self.last = FactoredBelief(self._possible, self.k)

    # -- the update ----------------------------------------------------------------------

    def _apply_ancestry(self, x: int, ancestry, powered=None, blind=None) -> None:
        """`ancestry[i]` -- is x an ancestor of the i-th other node? Prune each pair on x.

        EXACT AND LOCAL, for the reason in the module docstring: on an adjacent pair,
        ancestry from x to y admits only x -> y, and its absence admits only {y -> x, x <-> y}.
        Nothing about any other pair is consulted, which is the whole point.

        `powered` gates the NEGATIVE direction under sampled evidence. An undetected effect
        refutes `x -> y` only where the test had the power to have seen it; otherwise silence
        carries no information and the pair is left alone. That is what stops a distant,
        attenuated effect from being read as absent.

        `blind` gates BOTH directions and is what `evidence_power` uses: the question was
        asked and the test returned nothing usable, so neither the positive nor the negative
        conclusion is available. See `FactoredBackend.__init__` for why that is the right
        shape for simulating a weak test rather than adding noise to the answer.
        """
        others = [y for y in range(self.k) if y != x]
        for position, y in enumerate(others):
            if blind is not None and blind[position]:
                continue                      # no power here: the pair learns nothing
            key = (x, y) if x < y else (y, x)
            marks = self._possible[key]
            if marks == frozenset({NONE}) or len(marks) == 1:
                continue
            # The mark meaning "x -> y" depends on which way round the pair is stored.
            forward = FWD if x < y else BACK
            reverse = BACK if x < y else FWD
            if ancestry[position]:
                self._possible[key] = marks & frozenset({forward})
            elif powered is None or powered[position]:
                self._possible[key] = marks - frozenset({forward})
            # An empty set is necessarily wrong -- the truth is one of the marks -- so a
            # contradiction is refused rather than propagated as unanimous confidence.
            if not self._possible[key]:
                self._possible[key] = marks
                # Contradiction: keep the pair as it was, and record it. Both directions
                # were refuted, which can only happen if a test fired falsely.
                self._contradictions = getattr(self, "_contradictions", 0) + 1

    def edge_marginals(self, data, known_intervened, told=None, score_rule=None,
                       blocks=None) -> np.ndarray:
        if self.truth is None:
            raise RuntimeError("FactoredBackend.reset(true_mag) must be called first")
        mask = np.asarray(known_intervened) > 0.5
        intervened = frozenset(x for x in range(self.k) if mask[:, x].any())

        if self.evidence == "oracle":
            fresh = intervened - self._applied
            # Under power limiting a repeat is informative, so "nothing new was intervened
            # on" is no longer a reason to skip the update -- the row count may still have
            # risen. See the block below.
            if not fresh and (self.evidence_power >= 1.0
                              or all(int(mask[:, x].sum()) == self._attempts.get(x, 0)
                                     for x in range(self.k))):
                if self.last is None:
                    self.last = FactoredBelief(self._possible, self.k)
                return self.last.directed
            from cb.versionspace import reveal
            if self.evidence_power < 1.0:
                # A REPEAT MUST BUY ANOTHER DRAW, or this reproduces the wrong thing. Under
                # plain oracle evidence a second intervention on the same node reveals
                # nothing -- ancestry is already known -- so `fresh` correctly skips it, and
                # the learner correctly learns never to repeat. Under SAMPLED evidence a
                # repeat is exactly how you buy statistical power, and that inverted rule is
                # the mechanism the transfer failure was traced to (repeat rate: greedy
                # 0.247/0.331 against the learner's 0.110/0.138,
                # HANDOVER_CLUSTER_SAMPLED_2026_08_29 section 1).
                #
                # So withheld questions have to become ASKABLE AGAIN when the node is
                # intervened on again. Rows are the currency: `known_intervened` accumulates,
                # so a rising row count for x means another experiment on x, and each one
                # gets a fresh draw against `evidence_power`. Without this the prototype
                # makes evidence scarcer without making repetition worth anything -- it
                # would teach a policy the same "never repeat" rule, and fail transfer for
                # the same reason.
                attempts = {x: int(mask[:, x].sum()) for x in range(self.k)}
                for x in range(self.k):
                    if attempts[x] == 0 or attempts[x] == self._attempts.get(x, 0):
                        continue
                    blind = self._power_rng.random(self.k - 1) >= self.evidence_power
                    self._apply_ancestry(x, reveal(self.truth, self.k, x), blind=blind)
                self._attempts = attempts
            else:
                for x in fresh:
                    self._apply_ancestry(x, reveal(self.truth, self.k, x))
            self._applied = intervened
        else:
            if not intervened:
                if self.last is None:
                    self.last = FactoredBelief(self._possible, self.k)
                return self.last.directed
            from cb.versionspace import estimated_reveal_all
            detected = estimated_reveal_all(data, known_intervened, tuple(intervened),
                                            self.k, alpha=self.evidence_alpha, foreign=told)
            if detected == self._detected and self.last is not None:
                return self.last.directed
            self._detected = detected
            # Rebuild from the start: evidence accumulates with every round, so a pair
            # refuted on thin data must get its chance back as rows arrive. Cheap here in a
            # way it is not for the enumerated belief -- the state is O(k^2).
            self.reset_marks()
            for x, (ancestry, powered) in detected.items():
                self._apply_ancestry(x, ancestry, powered)
            self._applied = intervened
        self.last = FactoredBelief(self._possible, self.k)
        return self.last.directed

    def reset_marks(self) -> None:
        """Re-open every adjacent pair, keeping whatever skeleton `reset` established.

        The skeleton is NOT re-derived from truth here: under an estimated skeleton the
        sampled path rebuilds marks from scratch every round, and re-deriving would silently
        restore the true adjacencies partway through the episode -- handing back exactly the
        assumption the ablation exists to remove.
        """
        for index, (u, v) in enumerate(pairs(self.k)):
            absent = self._seeded_absent.get((u, v), self.truth[index] == NONE)
            self._possible[(u, v)] = (frozenset({NONE}) if absent
                                      else frozenset({FWD, BACK, BI}))

    # -- reporting -----------------------------------------------------------------------

    def credit_fraction(self, true_mag: np.ndarray, required_positions=(),
                        strict: bool = False) -> float:
        """Fraction of pairs settled to the TRUE mark. Not a posterior mass -- there is no
        joint here to take a mass of -- so it is reported as what it is."""
        if self.last is None:
            return 0.0
        truth = marks_from_mag(true_mag)
        hits = sum(1 for index, key in enumerate(pairs(self.k))
                   if self._possible[key] == frozenset({truth[index]}))
        return hits / max(len(self._possible), 1)

    @property
    def bidirected(self) -> np.ndarray:
        return self.last.bidirected if self.last is not None else np.zeros((self.k, self.k))
