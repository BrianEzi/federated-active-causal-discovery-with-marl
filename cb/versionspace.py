"""The deterministic belief: a version space over MAGs, and no statistics at all.

WHY THIS EXISTS. The statistical engine's failures (missed edges, confounding misreads,
under-powered tests -- measured 2026-08-25) sit between the policy and the thing the thesis
is about, which is WHICH EXPERIMENT TO RUN. This backend removes them by construction and
answers the infinite-data question instead: given perfect measurement, can agents learn to
divide experiments between them?

WHAT A BELIEF IS HERE. The set of MAGs still consistent with what has been established --
the version space. It starts as the observational equivalence class of the window's true
MAG (what infinite observational data gives you, i.e. the PAG) and an intervention on X
prunes it to the members agreeing with the truth about X's ancestry. That reveal channel is
the infinite-data limit of the engine's OWN ancestral evidence, so this is the idealisation
of the actual method rather than a different method that happens to be easier.

THE GUARANTEE, and the reason the claim bar must be 1.0. The true MAG never leaves the
space, so a claim every survivor agrees on is agreed CORRECTLY. At a bar below 1.0 a
majority of survivors could carry a wrong answer over the line and "settled-wrong" would
reappear -- the very thing this environment exists to eliminate. `ma/env.py` therefore
refuses `claim_bar < 1.0` on this backend.

SCALING. Markov-equivalent MAGs share adjacencies, so the class varies only in the marks on
the true skeleton: 3^(edges) candidates, not 4^(pairs). Cost is governed by WINDOW DENSITY,
not by the number of variables -- global graphs of 20+ nodes are cheap, while a dense k=7
window is not (measured: 94% of such windows exceeded a 10-edge guard). k <= 6 is the
usable range.

Because adjacency is shared across the whole class, every ADJACENCY claim is resolved from
the start and only the type claims (direction, confounding) are open. That is the honest
PAG semantics: observation fixes the skeleton, interventions do the rest.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Optional, Sequence

import numpy as np

from ma.projection import BIDIRECTED as MAG_BIDIRECTED
from ma.projection import DIRECTED as MAG_DIRECTED

NONE, FWD, BACK, BI = 0, 1, 2, 3


def pairs(k: int):
    return list(combinations(range(k), 2))


def marks_from_mag(mag: np.ndarray) -> tuple:
    """A true MAG (`ma.projection.latent_projection` output) as a mark tuple."""
    mag = np.asarray(mag)
    k = mag.shape[0]
    out = []
    for (u, v) in pairs(k):
        if mag[u, v] == MAG_BIDIRECTED:
            out.append(BI)
        elif mag[u, v] == MAG_DIRECTED:
            out.append(FWD)
        elif mag[v, u] == MAG_DIRECTED:
            out.append(BACK)
        else:
            out.append(NONE)
    return tuple(out)


def _edges(marks, k):
    directed, bidirected = set(), set()
    for (u, v), m in zip(pairs(k), marks):
        if m == FWD:
            directed.add((u, v))
        elif m == BACK:
            directed.add((v, u))
        elif m == BI:
            bidirected.add((u, v))
    return directed, bidirected


def _ancestors(directed, k):
    reach = [[False] * k for _ in range(k)]
    for (u, v) in directed:
        reach[u][v] = True
    for mid in range(k):
        for u in range(k):
            if reach[u][mid]:
                for v in range(k):
                    if reach[mid][v]:
                        reach[u][v] = True
    return reach


def valid_mag(marks, k):
    """Ancestral: no directed cycle, and no almost-directed cycle (u <-> v forbids either
    from being an ancestor of the other). Returns (directed, bidirected, ancestors) or
    None."""
    directed, bidirected = _edges(marks, k)
    anc = _ancestors(directed, k)
    if any(anc[u][u] for u in range(k)):
        return None
    for (u, v) in bidirected:
        if anc[u][v] or anc[v][u]:
            return None
    return directed, bidirected, anc


def m_separated(marks, k, x, y, cond) -> bool:
    """m-separation: every path between x and y is blocked given `cond`.

    A path is m-connecting iff every intermediate node is either a COLLIDER that is in
    `cond` or has a descendant there, or a NON-COLLIDER outside `cond`.
    """
    directed, bidirected, anc = valid_mag(marks, k)
    adjacency = [[False] * k for _ in range(k)]
    for (u, v), m in zip(pairs(k), marks):
        if m != NONE:
            adjacency[u][v] = adjacency[v][u] = True

    def arrow_into(a, b):
        return (a, b) in directed or (a, b) in bidirected or (b, a) in bidirected

    def opens_collider(node):
        return node in cond or any(anc[node][z] for z in cond)

    stack = [(x, (x,))]
    while stack:
        node, path = stack.pop()
        for nxt in range(k):
            if not adjacency[node][nxt] or nxt in path:
                continue
            extended = path + (nxt,)
            if nxt == y:
                if all(opens_collider(mid)
                       if (arrow_into(extended[i], mid) and arrow_into(extended[i + 2], mid))
                       else mid not in cond
                       for i, mid in enumerate(extended[1:-1])):
                    return False
                continue
            stack.append((nxt, extended))
    return True


def separation_queries(k):
    """Every (pair, conditioning set) question, smallest sets first so a mismatched
    candidate is rejected early."""
    queries = []
    for (x, y) in pairs(k):
        others = [c for c in range(k) if c not in (x, y)]
        for r in range(len(others) + 1):
            for cond in combinations(others, r):
                queries.append((x, y, frozenset(cond)))
    queries.sort(key=lambda q: len(q[2]))
    return queries


def equivalence_class(true_marks, k):
    """Every MAG observationally indistinguishable from the truth -- the starting PAG.

    Searches orientations of the TRUE SKELETON only, which is exact because Markov
    equivalent MAGs share adjacencies, and which is what makes larger windows tractable.
    """
    queries = separation_queries(k)
    target = tuple(m_separated(true_marks, k, x, y, cond) for x, y, cond in queries)
    slots = [i for i, m in enumerate(true_marks) if m != NONE]
    members = []
    for assignment in product((FWD, BACK, BI), repeat=len(slots)):
        marks = [NONE] * len(true_marks)
        for slot, mark in zip(slots, assignment):
            marks[slot] = mark
        marks = tuple(marks)
        if valid_mag(marks, k) is None:
            continue
        if all(m_separated(marks, k, x, y, cond) == value
               for value, (x, y, cond) in zip(target, queries)):
            members.append(marks)
    return tuple(members)


def reveal(marks, k, x) -> tuple:
    """What do(x) shows with infinite data: whether x is an ancestor of each other node.

    Deliberately the same channel `cb.citest.FisherZ.ancestral_evidence` estimates, so this
    is that engine's infinite-data limit.
    """
    _, _, anc = valid_mag(marks, k)
    return tuple(anc[x][y] for y in range(k) if y != x)


class VersionSpaceBelief:
    """Frequencies over the surviving candidates, shaped exactly like `BootstrapBelief`.

    Every consumer -- `cb.claims`, the observation vector, the greedy baseline -- reads
    `.adjacency`, `.directed` and `.bidirected`, so nothing downstream knows the difference.
    A frequency here is the fraction of SURVIVORS asserting the feature, which is 0 or 1
    once a claim is resolved.
    """

    def __init__(self, space, k: int):
        self.space = tuple(space)
        self.k = int(k)
        n = max(len(self.space), 1)
        self.adjacency = np.zeros((k, k), dtype=float)
        self.directed = np.zeros((k, k), dtype=float)
        self.bidirected = np.zeros((k, k), dtype=float)
        for marks in self.space:
            for (u, v), m in zip(pairs(k), marks):
                if m == NONE:
                    continue
                self.adjacency[u, v] += 1.0
                self.adjacency[v, u] += 1.0
                if m == FWD:
                    self.directed[u, v] += 1.0
                elif m == BACK:
                    self.directed[v, u] += 1.0
                else:
                    self.bidirected[u, v] += 1.0
                    self.bidirected[v, u] += 1.0
        self.adjacency /= n
        self.directed /= n
        self.bidirected /= n
        # Compatibility with the bootstrap belief's reporting fields.
        self.n_boot = len(self.space)
        self.ci_tests = 0
        self.truncated_fraction = 0.0
        self.replicates = None

    def edge_marginals(self) -> np.ndarray:
        return self.directed

    def confounded_pairs(self, threshold: float = 0.5) -> tuple:
        return tuple((u, v) for u, v in pairs(self.k)
                     if self.bidirected[u, v] >= threshold)


class VersionSpaceBackend:
    """Deterministic belief for one window. Call-compatible with `ConstraintBackend`.

    `reset(true_mag)` must be called at the start of every episode -- the environment does
    this -- because the version space is defined relative to the episode's truth. Truth is
    used ONLY to prune (oracle-side, exactly as the reward is); nothing about it reaches the
    observation vector.
    """

    can_handle_multi_hidden = True

    def __init__(self, k: int, shared_positions: Sequence[int] = (), **_ignored):
        self.k = int(k)
        self.shared_positions = tuple(shared_positions)
        self.truth: Optional[tuple] = None
        self.last: Optional[VersionSpaceBelief] = None
        self._space: tuple = ()

    def reset(self, true_mag: np.ndarray) -> None:
        self.truth = marks_from_mag(true_mag)
        self._space = equivalence_class(self.truth, self.k)
        self.last = VersionSpaceBelief(self._space, self.k)

    def edge_marginals(self, data, known_intervened, told=None, score_rule=None,
                       blocks=None) -> np.ndarray:
        """Prune by which nodes have been intervened on, and report the frequencies.

        `data` is ignored -- that is the whole point. Only the intervention MASK matters,
        because with infinite data the values add nothing the mask does not already imply.
        """
        if self.truth is None:
            raise RuntimeError("VersionSpaceBackend.reset(true_mag) must be called first")
        mask = np.asarray(known_intervened) > 0.5
        intervened = [x for x in range(self.k) if mask[:, x].any()]
        space = self._space
        for x in intervened:
            truth_reveal = reveal(self.truth, self.k, x)
            space = tuple(m for m in space if reveal(m, self.k, x) == truth_reveal)
        self.last = VersionSpaceBelief(space, self.k)
        return self.last.directed

    def credit_fraction(self, true_mag: np.ndarray, required_positions: Sequence[int] = (),
                        strict: bool = False) -> float:
        """Fraction of survivors that ARE the truth -- the analogue of posterior mass.

        Exact here rather than approximate: with the truth guaranteed present, this is
        1/|space| when nothing is resolved and 1.0 when the space collapses to the truth.
        """
        if self.last is None or not self.last.space:
            return 0.0
        truth = marks_from_mag(true_mag)
        return sum(m == truth for m in self.last.space) / len(self.last.space)

    @property
    def bidirected(self) -> np.ndarray:
        return self.last.bidirected if self.last is not None else np.zeros((self.k, self.k))
