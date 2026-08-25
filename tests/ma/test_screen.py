"""The confounding-assignment screen: what it must preserve.

The screen replaces `3^pairs` exact partition calls with `1 + 2 * n_pairs` surrogate calls
plus `screen_keep` exact ones. That is the change that makes |X| >= 5 computable at all --
543 assignments at |X| = 4, 29281 at |X| = 5, and `product(*options)` could not even be
BUILT at |X| = 10. It is also an approximation, so what it may not break is written down
here rather than assumed.

The load-bearing test is `test_the_true_assignment_survives_the_screen`. `joint_conf_dag_
probability` credits an agent only for assignments naming the true confounded pairs, so a
screen that drops the true assignment reports 0.000 for a reason that has nothing to do
with the agent -- unearnable, and indistinguishable from "has not learned yet". A
structurally unearnable metric passed 529 tests once already; this is the standing rule
that came out of it.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from ma.belief_dp import WindowBeliefDP
from sa.scm import sample, sample_multi, sample_scm_params


def _confounded_episode(seed: int = 0):
    """Hidden node 0 parents shared nodes 1 and 2; the agent observes only (1, 2, 3).

    The true confounded pair is therefore (0, 1) in WINDOW coordinates. `3` is an observed
    descendant of `2`, which is what gives a clamp demonstrable power.
    """
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[0, 1] = adjacency[0, 2] = adjacency[2, 3] = 1
    params = sample_scm_params(adjacency, rng)
    obs, obs_known = sample(params, 600, rng)
    blocks, knowns = [obs], [obs_known]
    for node in (1, 2, 3):
        d, m = sample_multi(params, 80, rng, intervene_nodes={node: 0.0})
        blocks.append(d); knowns.append(m)
    observed = [1, 2, 3]
    samples = np.vstack(blocks)[:, observed]
    known = np.vstack(knowns)[:, observed]
    return samples, known, np.zeros(len(samples))


def test_the_true_assignment_survives_the_screen():
    """METRIC REACHABILITY. The pair the truth confounds must be on the shortlist.

    Scored per regime rather than in aggregate: an assignment that names the true pair in
    EITHER orientation counts, because "u and v share a hidden cause" is one claim and the
    two orientations of the modelling edge express it equally -- the same equivalence
    `joint_conf_dag_probability` already applies.
    """
    samples, known, clean = _confounded_episode()
    k = samples.shape[1]
    shared = [0, 1, 2]
    screened = WindowBeliefDP(k, shared, screen_keep=8, max_eager=0)
    kept = [a for a, _, _ in screened.prepared_assignments(samples, known, clean)]

    named = [{frozenset(e) for e in a if e is not None} for a in kept]
    assert {frozenset((0, 1))} in named, (
        "the true confounded pair is not on the shortlist, so the identification metric "
        f"is unearnable at this shape. kept: {named}")


def test_screening_and_full_enumeration_agree_on_the_marginals():
    """The shortlist is an approximation; this pins how large it is allowed to be.

    Measured across five seeds by `scripts/bayes_screen_error.py`: at `keep = 64` the kept
    assignments hold >= 99.6% of the posterior mass and no edge marginal moves by more than
    3.3e-3. The bound here is deliberately looser than the measurement, so this fails on a
    regression rather than on sampling noise.
    """
    samples, known, clean = _confounded_episode(seed=1)
    k = samples.shape[1]
    shared = [0, 1, 2]
    exact = WindowBeliefDP(k, shared).joint_conf_marginals(samples, known, clean)
    screened = WindowBeliefDP(k, shared, screen_keep=16, max_eager=0
                              ).joint_conf_marginals(samples, known, clean)
    assert np.max(np.abs(exact - screened)) < 0.05


def test_below_the_cap_nothing_is_screened_at_all():
    """The exact path must stay exact, or every number banked before 2026-08-25 moves."""
    samples, known, clean = _confounded_episode(seed=2)
    k = samples.shape[1]
    belief = WindowBeliefDP(k, [0, 1, 2])
    assert belief._eager and belief.assignments is not None
    assert len(belief.assignments) == 25          # the |X| = 3 count, cycles removed
    prepared = belief.prepared_assignments(samples, known, clean)
    assert len(prepared) == 25


def test_a_wide_shared_set_constructs_without_enumerating():
    """|X| = 10 is 3^45 raw assignments. The old constructor could not BUILD that list.

    This is the wall the lazy path removes, and it is a construction-time wall rather than
    a scoring-time one -- which is why it is timed rather than merely called.
    """
    started = time.perf_counter()
    belief = WindowBeliefDP(12, list(range(10)))
    elapsed = time.perf_counter() - started
    assert belief.assignments is None, "should not have materialised"
    assert not belief._eager
    assert elapsed < 5.0, f"construction took {elapsed:.1f}s"
    assert belief.n_assignments == belief.screen_keep


def test_the_screen_ranks_on_the_prior_too():
    """TRAP 2, at screen scale.

    `joint_conf_marginals` already prunes on the likelihood, and DISCLOSURE_SPEC.md section 5
    warns that an assignment the likelihood alone discards may be exactly the one a
    disclosed prior rescues. The screen is a far more aggressive prune, so it inherits the
    trap in a worse form: a shortlist chosen on the likelihood alone can never be rescued
    afterwards, because the rescued assignment was never scored.

    Here the prior is pushed hard onto the LAST pair, which the likelihood has no reason to
    favour, and that pair must then appear among the kept assignments.
    """
    samples, known, clean = _confounded_episode(seed=3)
    k = samples.shape[1]
    shared = [0, 1, 2]
    belief = WindowBeliefDP(k, shared, screen_keep=4, max_eager=0)
    target = belief.pairs.index((1, 2))

    log_prior = np.zeros((len(belief.pairs), 3))
    log_prior[target, 1] = 50.0            # overwhelming, so the test cannot be marginal
    kept = [a for a, _, _ in
            belief.prepared_assignments(samples, known, clean, log_prior=log_prior)]
    named = [{frozenset(e) for e in a if e is not None} for a in kept]
    assert any(frozenset((1, 2)) in n for n in named), (
        f"a prior of e^50 on (1,2) did not put it on the shortlist: {named}")
