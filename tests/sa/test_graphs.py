"""Tests for DAG enumeration and Markov equivalence classes.

These pin the numbers everything downstream is checked against. In particular
`singleton_fraction` is GATE 1's predicted target -- if enumeration is wrong, the gate
silently compares against the wrong number and the leak that motivated this rebuild
could reappear unnoticed.
"""
import itertools

import numpy as np
import pytest

from sa.graphs import (
    N_DAGS,
    N_MECS,
    build_graph_space,
    descendants,
    enumerate_dags,
    is_acyclic,
    mec_signature,
    skeleton,
    v_structures,
)


# --- acyclicity ------------------------------------------------------------------

def test_is_acyclic_accepts_empty_and_chain():
    assert is_acyclic(np.zeros((3, 3)))
    chain = np.zeros((3, 3)); chain[0, 1] = 1; chain[1, 2] = 1
    assert is_acyclic(chain)


def test_is_acyclic_rejects_cycles():
    two = np.zeros((2, 2)); two[0, 1] = 1; two[1, 0] = 1
    assert not is_acyclic(two)
    three = np.zeros((3, 3)); three[0, 1] = 1; three[1, 2] = 1; three[2, 0] = 1
    assert not is_acyclic(three)


# --- enumeration counts ----------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_dag_counts_match_known_values(d):
    """Labelled DAG counts: 1, 3, 25, 543 for d = 1..4 (OEIS A003024)."""
    assert len(enumerate_dags(d)) == N_DAGS[d]


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_mec_counts_match_known_values(d):
    """Markov equivalence class counts: 1, 2, 11, 185 for d = 1..4 (OEIS A007984)."""
    assert build_graph_space(d).n_mecs == N_MECS[d]


def test_all_enumerated_graphs_are_acyclic_and_distinct():
    dags = enumerate_dags(4)
    assert all(is_acyclic(a) for a in dags)
    assert len({a.tobytes() for a in dags}) == len(dags)


# --- equivalence semantics -------------------------------------------------------

def test_three_node_chain_orientations_are_equivalent_but_collider_is_not():
    """The textbook case, and the reason interventions are needed at all.

    A->B->C, A<-B<-C and A<-B->C share a skeleton and have no v-structure, so no amount
    of observational data separates them. A->B<-C has a v-structure and is alone in its
    class -- identifiable without intervening.
    """
    def g(*edges):
        a = np.zeros((3, 3), dtype=np.int8)
        for i, j in edges:
            a[i, j] = 1
        return a

    chain_fwd = g((0, 1), (1, 2))
    chain_bwd = g((2, 1), (1, 0))
    fork = g((1, 0), (1, 2))
    collider = g((0, 1), (2, 1))

    assert mec_signature(chain_fwd) == mec_signature(chain_bwd) == mec_signature(fork)
    assert mec_signature(collider) != mec_signature(chain_fwd)
    assert v_structures(collider) == frozenset({(0, 1, 2)})
    assert v_structures(chain_fwd) == frozenset()


def test_shielded_collider_is_not_a_v_structure():
    """i -> k <- j only carries orientation information when i and j are non-adjacent."""
    a = np.zeros((3, 3), dtype=np.int8)
    a[0, 1] = 1; a[2, 1] = 1; a[0, 2] = 1  # 0 and 2 adjacent, so the collider is shielded
    assert v_structures(a) == frozenset()


def test_equivalence_classes_partition_the_space_consistently():
    """Every pair in a class shares a signature; no pair across classes does."""
    space = build_graph_space(3)
    for i, j in itertools.combinations(range(space.n_dags), 2):
        same_class = space.mec_id[i] == space.mec_id[j]
        same_sig = mec_signature(space.dags[i]) == mec_signature(space.dags[j])
        assert same_class == same_sig


def test_mec_members_are_symmetric_and_include_self():
    space = build_graph_space(3)
    for i in range(space.n_dags):
        members = space.mec_members(i)
        assert i in members
        for m in members:
            assert i in space.mec_members(int(m))


# --- the GATE 1 target -----------------------------------------------------------

def test_singleton_fraction_is_wellformed_and_nonzero():
    """GATE 1 compares the environment's observational-only solve rate against this.

    It must be strictly between 0 and 1: some graphs are identifiable observationally
    (so the gate isn't trivially 'never solve anything'), and some are not (so
    interventions are actually required and there is a task).
    """
    for d in (3, 4):
        space = build_graph_space(d)
        frac = space.singleton_fraction
        assert 0.0 < frac < 1.0, f"d={d}: singleton fraction {frac} leaves no task"
        assert space.is_singleton.sum() == (space.mec_sizes[space.mec_id] == 1).sum()


def test_singleton_dags_are_exactly_those_alone_in_their_class():
    space = build_graph_space(3)
    for i in range(space.n_dags):
        assert space.is_singleton[i] == (len(space.mec_members(i)) == 1)


def test_empty_graph_is_never_a_singleton_for_d_at_least_two():
    """Sanity anchor: with no edges there is nothing to orient, and the empty graph is
    Markov equivalent only to itself -- so it IS a singleton. This test records that
    deliberately, since it is a common source of confusion."""
    space = build_graph_space(3)
    empty_idx = next(i for i in range(space.n_dags) if space.dags[i].sum() == 0)
    assert space.is_singleton[empty_idx]


# --- descendants -----------------------------------------------------------------

def test_descendants_is_transitive_and_strict():
    a = np.zeros((4, 4), dtype=np.int8)
    a[0, 1] = 1; a[1, 2] = 1; a[2, 3] = 1
    reach = descendants(a)
    assert reach[0, 1] and reach[0, 2] and reach[0, 3]  # transitive
    assert not reach[0, 0]                              # strict: not its own descendant
    assert not reach[3, 0]                              # direction respected


def test_descendants_of_empty_graph_is_empty():
    assert not descendants(np.zeros((3, 3))).any()
