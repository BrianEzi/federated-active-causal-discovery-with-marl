"""Enumeration of DAGs and their Markov equivalence classes.

This is the foundation the rest of `sa/` rests on, and it exists as its own module
because the equivalence-class structure is not a detail -- it is what defines how hard
the task is, and it gives us a *predicted number* to check the environment against.

Why that matters here (see docs/THEORY_NOTES.md #1/#2): observational data can never
distinguish two DAGs in the same Markov equivalence class. So the fraction of DAGs
sitting alone in their own class is exactly the fraction of problems solvable without
intervening. Everything else *requires* an intervention. In the previous two-agent
codebase, theory said that fraction was 0% and measurement said 50% -- the gap was a
leak (equal error variances making the graph observationally identifiable), and it went
unnoticed for weeks because nobody had computed the theoretical number to compare
against. Computing it is cheap; not computing it cost this project a lot.

Two DAGs are Markov equivalent iff they share a skeleton and a set of v-structures
(Verma & Pearl 1990). Both are computed directly here rather than via a CPDAG/Meek-rules
construction, because at d <= 5 exhaustive enumeration is trivially affordable and a
direct implementation is far easier to verify against known counts.

Known counts, used as tests: the number of labelled DAGs on d nodes is
1, 3, 25, 543, 29281 for d = 1..5, and the number of Markov equivalence classes is
1, 2, 11, 185, 8782.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

# Known values, kept here as the single source of truth for the tests that pin
# enumeration correctness. Sequences: OEIS A003024 (labelled DAGs) and A007984
# (Markov equivalence classes / CPDAGs).
N_DAGS = {1: 1, 2: 3, 3: 25, 4: 543, 5: 29281, 6: 3781503}
N_MECS = {1: 1, 2: 2, 3: 11, 4: 185, 5: 8782, 6: 1067825}


def is_acyclic(adjacency: np.ndarray) -> bool:
    """True if `adjacency` (adjacency[i, j] == 1 meaning i -> j) has no directed cycle.

    Uses repeated removal of sink nodes (Kahn's algorithm on the reversed graph), which
    is exact and avoids the numerical fragility of matrix-power trace tests.
    """
    a = np.asarray(adjacency) > 0.5
    d = a.shape[0]
    remaining = list(range(d))
    while remaining:
        # A sink under `a` has no outgoing edges to any remaining node.
        sinks = [i for i in remaining if not a[i, remaining].any()]
        if not sinks:
            return False
        for s in sinks:
            remaining.remove(s)
    return True


def enumerate_dags(d: int) -> np.ndarray:
    """All labelled DAGs on `d` nodes as an [N, d, d] array, adjacency[i, j] = i -> j.

    Brute force over the 3^(d choose 2) ways to leave each unordered pair absent,
    forward or reversed -- which is far smaller than 2^(d(d-1)) over directed edges and
    rules out 2-cycles by construction. Affordable through d = 5 (29281 DAGs).
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    pairs = list(itertools.combinations(range(d), 2))
    dags: List[np.ndarray] = []
    # 0 = no edge, 1 = i -> j, 2 = j -> i
    for choice in itertools.product((0, 1, 2), repeat=len(pairs)):
        a = np.zeros((d, d), dtype=np.int8)
        for (i, j), c in zip(pairs, choice):
            if c == 1:
                a[i, j] = 1
            elif c == 2:
                a[j, i] = 1
        if is_acyclic(a):
            dags.append(a)
    return np.array(dags, dtype=np.int8)


def _pair_list(d: int) -> List[Tuple[int, int]]:
    return list(itertools.combinations(range(d), 2))


def enumerate_dags_fast(d: int, chunk: int = 1 << 20) -> np.ndarray:
    """`enumerate_dags`, vectorised over candidates. Exactly the same output.

    The pure-Python version tests 3^C(d,2) candidates one at a time, which is fine through
    d=5 (59049 candidates) and hopeless at d=6 (14.3 million -- roughly 28 minutes, paid
    again by every job that builds the space). This tests a whole chunk of candidates at
    once with array operations, which brings d=6 down to well under a minute.

    Acyclicity is still Kahn's algorithm, just run on every candidate simultaneously:
    repeatedly delete all nodes that have no outgoing edge to a surviving node. A graph is
    acyclic iff nothing survives after `d` rounds, since each round removes at least one
    node from every acyclic graph.
    """
    pairs = _pair_list(d)
    n_pairs = len(pairs)
    total = 3 ** n_pairs
    # Descending weights so candidate `idx` is the same graph `itertools.product` yields
    # at position `idx` -- its LAST position varies fastest. Matching that exactly matters:
    # a DAG's index is its identity everywhere else in the codebase, so a different
    # enumeration order would silently renumber every graph.
    powers = 3 ** np.arange(n_pairs - 1, -1, -1)

    keep: List[np.ndarray] = []
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        digits = (idx[:, None] // powers[None, :]) % 3      # [c, n_pairs] in {0, 1, 2}

        a = np.zeros((len(idx), d, d), dtype=bool)
        for k, (i, j) in enumerate(pairs):
            a[:, i, j] = digits[:, k] == 1
            a[:, j, i] = digits[:, k] == 2

        alive = np.ones((len(idx), d), dtype=bool)
        for _ in range(d):
            # has_out[c, i]: node i still points at some surviving node.
            has_out = (a & alive[:, None, :]).any(axis=2)
            alive &= has_out          # sinks (no outgoing edge) are deleted
        acyclic = ~alive.any(axis=1)

        if acyclic.any():
            keep.append(a[acyclic].astype(np.int8))

    return np.concatenate(keep) if keep else np.zeros((0, d, d), dtype=np.int8)


def mec_ids_fast(dags: np.ndarray) -> np.ndarray:
    """[N] Markov equivalence class id for every DAG, computed with array operations.

    Same criterion as `mec_signature` (Verma & Pearl 1990: equal skeleton and equal
    v-structures), but the signature is packed into integer bit-codes rather than nested
    frozensets, so millions of graphs can be grouped with one `np.unique`. Building 3.78
    million frozenset pairs in Python is both slow and memory-hungry.

    Bit layout: one bit per unordered pair for the skeleton, and one bit per candidate
    triple (i -> k <- j, i < j) for the v-structures, spread across as many uint64 words
    as needed.
    """
    a = np.asarray(dags) > 0
    n, d = a.shape[0], a.shape[1]

    bits: List[np.ndarray] = []
    for i, j in _pair_list(d):
        bits.append(a[:, i, j] | a[:, j, i])                       # skeleton
    for k in range(d):
        others = [x for x in range(d) if x != k]
        for i, j in itertools.combinations(others, 2):
            # Unshielded collider only: if i and j are adjacent the pattern carries no
            # orientation information (see `v_structures`).
            bits.append(a[:, i, k] & a[:, j, k] & ~a[:, i, j] & ~a[:, j, i])

    n_words = (len(bits) + 63) // 64
    codes = np.zeros((n, n_words), dtype=np.uint64)
    for b, bit in enumerate(bits):
        codes[:, b // 64] |= bit.astype(np.uint64) << np.uint64(b % 64)

    _, inverse = np.unique(codes, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int32)


def skeleton(adjacency: np.ndarray) -> FrozenSet[FrozenSet[int]]:
    """The undirected edge set: which pairs are adjacent, ignoring direction."""
    a = np.asarray(adjacency) > 0.5
    d = a.shape[0]
    return frozenset(
        frozenset((i, j)) for i in range(d) for j in range(d) if a[i, j]
    )


def v_structures(adjacency: np.ndarray) -> FrozenSet[Tuple[int, int, int]]:
    """Colliders `i -> k <- j` where i and j are NOT adjacent, as (min(i,j), k, max(i,j)).

    The non-adjacency requirement is the whole point: if i and j were adjacent the
    pattern is a "shielded collider" and carries no orientation information, because it
    is indistinguishable from other orientations of the same triangle.
    """
    a = np.asarray(adjacency) > 0.5
    d = a.shape[0]
    out = set()
    for k in range(d):
        parents = [i for i in range(d) if a[i, k]]
        for i, j in itertools.combinations(parents, 2):
            if not a[i, j] and not a[j, i]:
                out.add((min(i, j), k, max(i, j)))
    return frozenset(out)


def mec_signature(adjacency: np.ndarray):
    """The pair that determines Markov equivalence (Verma & Pearl 1990).

    Two DAGs are Markov equivalent iff this value is equal for both.
    """
    return (skeleton(adjacency), v_structures(adjacency))


@dataclass(frozen=True)
class GraphSpace:
    """All DAGs on `d` nodes, together with their equivalence-class structure.

    Attributes:
        d: number of nodes.
        dags: [N, d, d] int8 adjacency matrices, adjacency[i, j] = 1 meaning i -> j.
        mec_id: [N] which equivalence class each DAG belongs to.
        mec_sizes: [n_mecs] how many DAGs are in each class.
    """

    d: int
    dags: np.ndarray
    mec_id: np.ndarray
    mec_sizes: np.ndarray

    @property
    def n_dags(self) -> int:
        return int(self.dags.shape[0])

    @property
    def n_mecs(self) -> int:
        return int(self.mec_sizes.shape[0])

    @property
    def is_singleton(self) -> np.ndarray:
        """[N] bool: is this DAG alone in its equivalence class?

        A singleton DAG is fully determined by observational data -- no intervention is
        needed to identify it. Everything else needs at least one.
        """
        return self.mec_sizes[self.mec_id] == 1

    @property
    def singleton_fraction(self) -> float:
        """The fraction of DAGs identifiable without intervening.

        This is GATE 1's target: an environment sampling uniformly from this space must
        show an observational-only identification rate equal to this number. Higher
        means information is leaking; lower means the estimator is underperforming.
        """
        return float(np.mean(self.is_singleton))

    def mec_members(self, dag_index: int) -> np.ndarray:
        """Indices of every DAG Markov-equivalent to `dag_index` (including itself)."""
        return np.flatnonzero(self.mec_id == self.mec_id[dag_index])


def build_graph_space(d: int, fast: bool = True) -> GraphSpace:
    """Enumerate all DAGs on `d` nodes and group them into equivalence classes.

    `fast=False` selects the straightforward per-graph implementation. It is kept as the
    reference the vectorised path is tested against, not as a fallback: the two must agree
    exactly, and `tests/sa/test_graphs.py` asserts that they do.
    """
    if fast:
        dags = enumerate_dags_fast(d)
        mec_id = mec_ids_fast(dags)
    else:
        dags = enumerate_dags(d)
        signatures: Dict[object, int] = {}
        mec_id = np.empty(len(dags), dtype=np.int32)
        for idx, a in enumerate(dags):
            sig = mec_signature(a)
            if sig not in signatures:
                signatures[sig] = len(signatures)
            mec_id[idx] = signatures[sig]
    mec_sizes = np.bincount(mec_id)
    return GraphSpace(d=d, dags=dags, mec_id=mec_id, mec_sizes=mec_sizes)


def descendants(adjacency: np.ndarray) -> np.ndarray:
    """[d, d] bool: reach[i, j] is True iff j is a strict descendant of i.

    Used by the oracle: a hard intervention on i shifts the distribution of exactly its
    descendants, so two hypotheses are distinguishable by do(X_i) precisely when their
    descendant sets from i differ.
    """
    a = np.asarray(adjacency) > 0.5
    d = a.shape[0]
    reach = a.copy()
    for k in range(d):  # Floyd-Warshall transitive closure; d is tiny.
        reach |= reach[:, k][:, None] & reach[k, :][None, :]
    return reach
