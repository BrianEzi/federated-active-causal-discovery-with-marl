"""`score_groups` must never turn an ENGINE failure into an attribution error.

Its docstring promises settled-wrong cannot occur at bar 1.0, because the truth never leaves
the candidate set. Measured 2026-08-29 it reported wrong at 0.075-0.113 under ORACLE
evidence. Two defects, both pinned here.
"""
import numpy as np
import pytest

from cb.attribution import LatentGroup, consistent_with_partner, score_groups


class _Belief:
    def __init__(self, freq, total):
        self.group_frequency = dict(freq)
        self.total = total


def test_exhausted_belief_scores_unsure_not_wrong():
    """An emptied version space knows NOTHING. Scoring it as confidently wrong is the bug
    that corrupted every attribution number downstream."""
    truth = (LatentGroup(1, frozenset({2, 3})),)
    out = score_groups(_Belief({}, 0), truth, bar=1.0)
    assert out["exhausted"] is True
    assert out["wrong"] == 0
    assert out["unsure"] == 1


def test_a_live_belief_that_never_names_the_group_still_scores_wrong():
    """The guard must not swallow genuine misattribution: a populated belief that gives the
    true group zero mass is a real error and must keep reading as one."""
    truth = (LatentGroup(1, frozenset({2, 3})),)
    belief = _Belief({LatentGroup(0, frozenset({2, 3})): 1.0}, total=4)
    out = score_groups(belief, truth, bar=1.0)
    assert out["exhausted"] is False
    assert out["wrong"] == 1 and out["unsure"] == 0


def test_full_mass_on_the_true_group_scores_right():
    truth = (LatentGroup(1, frozenset({2, 3})),)
    out = score_groups(_Belief({truth[0]: 1.0}, total=4), truth, bar=1.0)
    assert out["right"] == 1 and out["identified"] is True


# -- the elimination rule ---------------------------------------------------------------

def test_atomicity_refutes_a_clique_that_moved_only_partly():
    """One latent responds as a unit, so a partial response refutes the candidate. This is
    where the channel's discrimination lives."""
    cand = (LatentGroup(1, frozenset({0, 1, 2})),)
    partial = frozenset({(0, 1)})                      # (0,2) and (1,2) did not move
    assert not consistent_with_partner(cand, owner=1, moved=partial)


def test_atomicity_accepts_a_clique_that_moved_entirely():
    cand = (LatentGroup(1, frozenset({0, 1})),)
    assert consistent_with_partner(cand, owner=1, moved=frozenset({(0, 1)}))


def test_owner_need_only_explain_SOMETHING_that_moved():
    """Rule 1 used to demand the owner cover EVERY moved pair. `moved` mixes owners -- an
    actor's private node can sit above a third agent's latent -- so demanding all of it
    refuted the truth. Owning one of the moved pairs must suffice."""
    cand = (LatentGroup(1, frozenset({0, 1})), LatentGroup(2, frozenset({3, 4})))
    mixed = frozenset({(0, 1), (3, 4)})                # actor 1 owns only the first
    assert consistent_with_partner(cand, owner=1, moved=mixed)


def test_an_owner_who_explains_nothing_that_moved_is_still_refuted():
    cand = (LatentGroup(2, frozenset({3, 4})),)
    assert not consistent_with_partner(cand, owner=1, moved=frozenset({(3, 4)}))


def test_no_evidence_never_refutes():
    cand = (LatentGroup(1, frozenset({0, 1})),)
    assert consistent_with_partner(cand, owner=1, moved=frozenset())
