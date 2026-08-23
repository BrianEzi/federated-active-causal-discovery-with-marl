"""Orientation: turn an undirected skeleton into a partial ancestral graph.

EDGE ENDS, NOT EDGE DIRECTIONS. Each end of each edge carries its own mark:

    CIRCLE   undetermined
    ARROW    "not an ancestor of the other endpoint"
    TAIL     "is an ancestor of the other endpoint"

so an edge is read from its two ends:

    u --TAIL...ARROW--> v     u -> v      direct cause
    u <--ARROW...ARROW--> v   u <-> v     CONFOUNDED: neither causes the other
    u --CIRCLE..CIRCLE-- v    u o-o v     adjacent, nothing determined

THIS REPRESENTATION IS THE WHOLE POINT AND THE FIRST VERSION GOT IT WRONG. That version
tracked only directions and marked every edge it failed to orient as bidirected, which
reported confounding on a plain three-node chain with no latent anywhere -- because a
Markov equivalence class leaves edges unoriented for want of INFORMATION, which is not the
same as evidence that neither node causes the other. "Undetermined" and "confounded" are
different claims and they need different symbols. Caught by the first smoke test.

SOUND, NOT COMPLETE. Complete FCI (Zhang 2008) carries a large rule set; implemented here
are collider detection, Meek's R1/R2, and interventional orientation. Every mark placed is
correct; some marks a complete implementation would place are left as circles.
`Orientation.circle_count` reports how many, so a run can be judged rather than assumed.
The asymmetry is deliberate: a wrong arrowhead propagates through the rules and corrupts
the graph, a missing one costs only information.

THREE SOURCES OF MARKS, in the order applied:

  1. INTERVENTIONS -- clamping x changed y, so x IS an ancestor of y. Certain, and
     unavailable to any observational method. Applied first so later rules build on facts.
  2. COLLIDERS -- for u *-* w *-* v with u, v NON-adjacent and w absent from their
     separating set, w cannot explain their dependence, so both edges get an ARROW at w.
     This is the one place observational data yields a mark, and it is why the skeleton
     search records separating sets.
  3. MEEK PROPAGATION -- what follows from creating no new collider and no cycle.

Confounding then falls out with no enumeration at all: an edge with arrowheads at BOTH ends
says neither endpoint is an ancestor of the other, yet they are dependent given every
observed subset -- which is exactly a latent common cause. Contrast the exact engine, which
had to enumerate DAGs-over-the-shared-set confounding patterns and died at |X| = 5.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np

# Endpoint marks. `marks[u, v]` is the mark at the **v end** of edge u-v.
NO_EDGE = 0
CIRCLE = 1
ARROW = 2
TAIL = 3

# Edge classifications returned by `Orientation.codes`.
CODE_NONE = 0
CODE_DIRECTED = 1      # codes[u, v] == CODE_DIRECTED means u -> v
CODE_BIDIRECTED = 2    # symmetric
CODE_UNDETERMINED = 3  # symmetric


class Orientation:
    """Endpoint marks, plus the edge classification derived from them."""

    def __init__(self, marks: np.ndarray):
        self.marks = marks
        self.codes = self._classify()

    @property
    def k(self) -> int:
        return int(self.marks.shape[0])

    def _classify(self) -> np.ndarray:
        k = self.marks.shape[0]
        codes = np.zeros((k, k), dtype=np.int8)
        for u, v in combinations(range(k), 2):
            if self.marks[u, v] == NO_EDGE:
                continue
            at_v, at_u = self.marks[u, v], self.marks[v, u]
            if at_v == ARROW and at_u == ARROW:
                codes[u, v] = codes[v, u] = CODE_BIDIRECTED
            elif at_v == ARROW and at_u in (TAIL, CIRCLE):
                # ARROW opposite CIRCLE is read as DIRECTED. In a PAG `u o-> v` means
                # "u -> v OR u <-> v" and is genuinely ambiguous; we resolve it towards the
                # causal reading and require positive interventional evidence to call it
                # confounded (step 4 of `orient`). Documented assumption, not an oversight:
                # defaulting the other way would report confounding on every collider.
                codes[u, v] = CODE_DIRECTED
            elif at_u == ARROW and at_v in (TAIL, CIRCLE):
                codes[v, u] = CODE_DIRECTED
            else:
                codes[u, v] = codes[v, u] = CODE_UNDETERMINED
        return codes

    @property
    def circle_count(self) -> int:
        """Edges left undetermined -- the honest measure of incompleteness."""
        return int((np.triu(self.codes) == CODE_UNDETERMINED).sum())

    def bidirected_pairs(self) -> tuple:
        return tuple((u, v) for u, v in combinations(range(self.k), 2)
                     if self.codes[u, v] == CODE_BIDIRECTED)

    def directed_matrix(self) -> np.ndarray:
        return self.codes == CODE_DIRECTED


def _ancestors(directed: np.ndarray) -> np.ndarray:
    reach = directed.copy()
    for m in range(reach.shape[0]):
        reach |= np.outer(reach[:, m], reach[m, :])
    return reach


def orient(skeleton, ancestral: Optional[np.ndarray] = None,
           clamped: Optional[np.ndarray] = None,
           require_power: bool = True,
           powered: Optional[np.ndarray] = None) -> Orientation:
    """Orient `skeleton`.

    `ancestral[x, y]` -- clamping x demonstrably changed y.
    `clamped[x]`      -- x was clamped often enough for its row to be informative. Without
                         it, an absent effect cannot be told from an absent experiment.
    `require_power`   -- before calling a pair confounded, demand the experiment had the
                         power to have detected causation on THAT PAIR, had it existed.
    `powered[x, y]`   -- the per-pair power verdict, from `FisherZ.pair_power`: the clamp on
                         x could have detected an effect on y of the size the pair's own
                         observed dependence implies.

    TENSION RESOLVED 2026-08-24 by the per-pair check. The previous check was global --
    "did the clamp on x move ANYTHING" -- which proved the experiment ran, not that it
    could have seen an effect on the specific partner. That was wrong in both directions
    at once: a clamp validated by a strong edge licensed a confounding verdict on a weak
    one (false positive, the chain-and-branch xfail), and a true latent pair with no
    observed descendant could never demonstrate power at all (false negative, the two-node
    case). `powered` answers the question per pair, so both cases now come out right.
    `require_power=True` with `powered=None` falls back to the old global test and exists
    only for callers without a `FisherZ` at hand; `cb/bootstrap.py` always passes `powered`.
    """
    k = skeleton.k
    adjacency = skeleton.adjacency
    marks = np.where(adjacency, CIRCLE, NO_EDGE).astype(np.int8)

    def put(at_node: int, other: int, mark: int) -> bool:
        """Place `mark` at `at_node`'s end of edge (other, at_node). True if it changed."""
        if marks[other, at_node] == CIRCLE:
            marks[other, at_node] = mark
            return True
        return False

    # --- 1. interventional marks: certain, so they go first -----------------------------
    if ancestral is not None:
        for u, v in combinations(range(k), 2):
            if marks[u, v] == NO_EDGE:
                continue
            if ancestral[u, v] and not ancestral[v, u]:
                put(v, u, ARROW); put(u, v, TAIL)      # u -> v
            elif ancestral[v, u] and not ancestral[u, v]:
                put(u, v, ARROW); put(v, u, TAIL)      # v -> u

    # --- 2. colliders --------------------------------------------------------------------
    for w in range(k):
        neighbours = [x for x in range(k) if adjacency[w, x]]
        for u, v in combinations(neighbours, 2):
            if adjacency[u, v]:
                continue                               # adjacent -> not a v-structure
            sep = skeleton.sepset(u, v)
            if sep is None or w in sep:
                continue                               # w explains them -> not a collider
            put(w, u, ARROW)
            put(w, v, ARROW)

    # --- 3. Meek propagation --------------------------------------------------------------
    changed = True
    while changed:
        changed = False
        directed = np.zeros((k, k), dtype=bool)
        for a, b in combinations(range(k), 2):
            if marks[a, b] == ARROW and marks[b, a] == TAIL:
                directed[a, b] = True
            elif marks[b, a] == ARROW and marks[a, b] == TAIL:
                directed[b, a] = True

        for a, b in combinations(range(k), 2):
            for x, y in ((a, b), (b, a)):
                if marks[x, y] != CIRCLE or marks[y, x] != CIRCLE:
                    continue
                # R1: c -> x o-o y, with c and y non-adjacent  =>  x -> y
                #     (any other choice would make x a collider that the skeleton denies)
                if any(directed[c, x] and not adjacency[c, y]
                       for c in range(k) if c not in (x, y)):
                    put(y, x, ARROW); put(x, y, TAIL)
                    changed = True
                    continue
                # R2: x -> c -> y  =>  x -> y (anything else is a cycle)
                if any(directed[x, c] and directed[c, y]
                       for c in range(k) if c not in (x, y)):
                    put(y, x, ARROW); put(x, y, TAIL)
                    changed = True

    # --- 4. confounding requires POSITIVE evidence, and interventions supply it ----------
    #
    # THE SECOND BUG THIS FILE HAS HAD. The first version marked any edge it could not
    # orient as bidirected. The second inferred an arrowhead wherever the opposite end was
    # an arrowhead and no DIRECTED path made the near end an ancestor -- which reported a
    # textbook collider 0 -> 2 <- 1 as two confounded pairs, because at that point in the
    # algorithm nothing is directed yet, so "not an ancestor" is vacuously true.
    #
    # Absence of orientation is not evidence of confounding. The claim needs its own
    # evidence, and our agents generate exactly the right kind:
    #
    #     u and v are adjacent          -- dependent given EVERY observed subset
    #     clamping u does not move v    -- u is not a cause of v
    #     clamping v does not move u    -- v is not a cause of u
    #     => something unobserved drives both
    #
    # This is the two-intervention argument, and it is only available because the agents
    # act. A purely observational engine would need FCI's discriminating-path machinery to
    # get here, and would still fail on the ~98% of confounded windows the structural
    # ceiling measured as unreachable from observation alone.
    #
    # `clamped` guards it: a pair where we never clamped either node has NO evidence
    # either way and must stay undetermined rather than default to confounded.
    if ancestral is not None and clamped is not None:
        for u, v in combinations(range(k), 2):
            if marks[u, v] == NO_EDGE:
                continue
            if not (clamped[u] and clamped[v]):
                continue                          # no interventional evidence for this pair
            # POWER CHECK, and it is what keeps this sound. Step 4 turns "no effect
            # detected" into a POSITIVE claim of confounding, so a MISSED detection becomes
            # a false confounder -- the exact inversion this file has already made twice.
            # Per PAIR: the clamp on u must have had the power to detect an effect on v of
            # the size their own measured dependence implies, and vice versa. Only then is
            # "nothing moved" evidence rather than noise.
            if require_power:
                if powered is not None:
                    if not (powered[u, v] and powered[v, u]):
                        continue
                elif not (ancestral[u].any() and ancestral[v].any()):
                    continue          # legacy global fallback; see docstring
            if not ancestral[u, v] and not ancestral[v, u]:
                marks[u, v] = marks[v, u] = ARROW

    return Orientation(marks)
