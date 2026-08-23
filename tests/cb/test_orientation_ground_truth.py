"""The constraint engine against graphs whose answer is known by hand.

VALIDATED DIRECTLY, NOT THROUGH A CONSUMER. Every case here checks the engine's own output
against a graph written down in the test. Checking it through an identification metric would
hide exactly the failures this file caught -- and it caught three, none of which any
downstream metric would have distinguished from "the agent has not learned yet":

  1. the first orientation pass marked every unoriented edge as bidirected, so a plain chain
     reported confounding everywhere. Undetermined and confounded are different claims.
  2. the second inferred an arrowhead from absence of a directed path, so a textbook collider
     reported two confounded pairs -- at that point nothing is directed, so "not an ancestor"
     is vacuously true of everything.
  3. `ancestral_evidence` tested MEANS only. A clamp sets a node to 0.0 and E[x] is already
     0, so a child sees no mean shift at all -- only a variance drop. The test detected
     nothing on precisely the experiment the agents perform, and since step 4 reads "no
     effect either way" as confounding, every clamped pair came back confounded.
  4. power was checked per NODE ("did this clamp move anything anywhere"), not per PAIR, so
     a clamp validated by a strong edge licensed a confounding verdict on a weak one --
     the (0,3) false positive this file pinned as an xfail on 2026-08-23. Fixed by
     `FisherZ.pair_power`: the pair's own measured dependence sets the effect size the
     clamp must have been able to detect.
  5. `ancestral_evidence`'s comparison group mixed regimes: "x free, y free" includes rows
     from OTHER nodes' clamp blocks, where y's distribution genuinely differs -- so z's
     effect was attributed to x, and clamping a childless sink was reported as moving a
     node two edges away. That false entry is what satisfied the old global power check,
     so the latent-detection test below passed BECAUSE of this bug.

All five passed every existing test in the suite when introduced.
"""
from __future__ import annotations

import numpy as np
import pytest

from cb.bootstrap import bootstrap_belief
from cb.citest import FisherZ
from ma.scm import SCMParams, sample, sample_multi, sample_scm_params

N_ROWS = 1200
N_BOOT = 12


def _params(adjacency, weights=None, seed=0):
    """Hand-set weights where the test's argument depends on effect size."""
    adjacency = np.asarray(adjacency)
    if weights is None:
        return sample_scm_params(adjacency, np.random.default_rng(seed))
    return SCMParams(adjacency=adjacency.astype(np.int8),
                     weights=np.asarray(weights, dtype=float),
                     noise_scales=np.linspace(0.8, 1.2, adjacency.shape[0]))


def _blocks(params, clamps, seed=0):
    rng = np.random.default_rng(seed)
    data, mask = sample(params, N_ROWS, rng)
    blocks, masks = [data], [mask]
    for node in clamps:
        d, m = sample_multi(params, N_ROWS, rng, intervene_nodes={node: 0.0})
        blocks.append(d); masks.append(m)
    return np.vstack(blocks), np.vstack(masks)


def _episode(adjacency, clamps=(), observed=None, seed=0, weights=None):
    """Observational batch plus one clamped batch per node in `clamps`."""
    params = _params(adjacency, weights, seed)
    data, mask = _blocks(params, clamps, seed)
    if observed is not None:
        data, mask = data[:, observed], mask[:, observed]
    return bootstrap_belief(data, mask, n_boot=N_BOOT, seed=seed)


def test_chain_and_branch_reports_no_confounding():
    """0->1->2 with 0->3. No latent anywhere, so no bidirected edge may appear.

    Regression for bug 1 (unoriented read as confounded) AND bug 4: this was the xfail of
    2026-08-23. Clamping 0 was detected on 1, so the old global power check passed, while
    the weak effect on 3 fell under the variance threshold -- and (0,3) came back
    confounded on a graph with no latent anywhere. With per-pair power, the clamp on 0 is
    not credited with the ability to detect an effect of (0,3)'s implied size, so the pair
    stays undetermined instead of turning into a false confounder.
    """
    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[0, 1] = adjacency[1, 2] = adjacency[0, 3] = 1
    belief = _episode(adjacency, clamps=(0, 1, 2, 3))
    assert belief.confounded_pairs() == (), belief.bidirected


def test_collider_is_oriented_and_not_called_confounded():
    """0->2<-1: a genuine v-structure. Must orient, must not report confounding.

    Regression for bug 2.
    """
    adjacency = np.zeros((3, 3), dtype=int)
    adjacency[0, 2] = adjacency[1, 2] = 1
    belief = _episode(adjacency, clamps=(0, 1, 2))
    assert belief.confounded_pairs() == ()
    marginals = belief.edge_marginals()
    assert marginals[1, 2] > 0.5, "1 -> 2 should be recovered"
    assert marginals[0, 2] > 0.5, "0 -> 2 should be recovered"


# A hidden confounder strong enough that its implied variance drop is comfortably within
# the power of the variance test at N_ROWS: r(1,2) ~ 0.66, so 1-r^2 ~ 0.56 against a
# detectability floor of ~0.85 at these sample sizes.
_STRONG_LATENT = np.array([
    [0.0,  1.4, -1.4, 0.0],
    [0.0,  0.0,  0.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
    [0.0,  0.0,  0.0, 0.0],
])


def test_a_true_latent_is_detected_when_the_clamps_have_power():
    """Hidden 0 causes 1 and 2 STRONGLY; 2->3; the agent observes only (1, 2, 3).

    1 and 2 are dependent given every observed subset, and clamping either moves the other
    not at all -- the two-intervention signature of a common cause. The confounding is
    strong enough that a causal explanation of it would have been detected, which is what
    per-pair power demands before 'nothing moved' counts as evidence.

    Weights are hand-set: the earlier version of this test used seeded random weights and
    passed only because contaminated ancestral evidence (bug 5) satisfied the old global
    power check -- the confounder it drew (r ~ -0.28) is genuinely undetectable at these
    sample sizes.
    """
    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[0, 1] = adjacency[0, 2] = adjacency[2, 3] = 1
    belief = _episode(adjacency, clamps=(1, 2, 3), observed=[1, 2, 3],
                      weights=_STRONG_LATENT)
    # observed indices 0,1,2 correspond to true nodes 1,2,3
    assert (0, 1) in belief.confounded_pairs(), belief.bidirected


def test_a_true_latent_needs_no_observed_descendant():
    """Hidden 0 causes 1 and 2; the agent observes ONLY those two. Formerly the false
    NEGATIVE of the global power check: with no observed descendant, clamping either node
    moves nothing observable, so the clamps could never demonstrate power and a genuinely
    confounded pair stayed silent. Per-pair power asks only about the partner -- was the
    clamp on 1 able to detect an effect on 2 of the size their dependence implies -- so
    the descendant is no longer needed.
    """
    adjacency = np.zeros((3, 3), dtype=int)
    adjacency[0, 1] = adjacency[0, 2] = 1
    belief = _episode(adjacency, clamps=(1, 2), observed=[1, 2],
                      weights=_STRONG_LATENT[:3, :3])
    assert belief.confounded_pairs() == ((0, 1),), belief.bidirected


def test_an_underpowered_pair_stays_undetermined_not_confounded():
    """A WEAK hidden confounder (r ~ -0.28, implied variance drop ~8%) is below what the
    variance test can detect at these sample sizes. The sound answer is 'undetermined',
    never 'confounded': no power calculation conjures detection out of a sample the effect
    is invisible in, and over-reporting confounding is the failure mode this project can
    least afford. The edge must still be THERE -- the skeleton sees the dependence.
    """
    weak = np.array([
        [0.0, 0.905, -0.561],
        [0.0, 0.0,    0.0],
        [0.0, 0.0,    0.0],
    ])
    adjacency = np.zeros((3, 3), dtype=int)
    adjacency[0, 1] = adjacency[0, 2] = 1
    belief = _episode(adjacency, clamps=(1, 2), observed=[1, 2], weights=weak)
    assert belief.adjacency[0, 1] > 0.5, "the dependence itself must be seen"
    assert belief.confounded_pairs() == (), belief.bidirected


def test_a_sinks_clamp_yields_no_ancestral_evidence():
    """Regression for bug 5, checked at the `FisherZ` layer where it lives.

    True node 1 is a childless sink: clamping it moves NOTHING. The contaminated
    comparison group ("x free, y free", which includes other nodes' clamp blocks)
    attributed node 2's effect on 3 to node 1, reporting a sink as an ancestor of a node
    two edges away. The clean two-regime contrast must report an all-False row.
    """
    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[0, 1] = adjacency[0, 2] = adjacency[2, 3] = 1
    params = _params(adjacency, _STRONG_LATENT)
    data, mask = _blocks(params, clamps=(1, 2, 3))
    test = FisherZ(data[:, [1, 2, 3]], mask[:, [1, 2, 3]])
    ancestral = test.ancestral_evidence()
    assert not ancestral[0].any(), ancestral.astype(int)   # obs 0 == true node 1, the sink
    assert ancestral[1, 2], "the real ancestry 2->3 must still be seen"


def test_skeleton_recovers_the_true_adjacency():
    """Before any orientation question, the adjacency itself must be right."""
    adjacency = np.zeros((4, 4), dtype=int)
    adjacency[0, 1] = adjacency[1, 2] = adjacency[0, 3] = 1
    belief = _episode(adjacency, clamps=(0, 1, 2, 3))
    truth = (adjacency + adjacency.T) > 0
    recovered = belief.adjacency > 0.5
    assert np.array_equal(recovered, truth), f"\n{recovered.astype(int)}\nvs\n{truth.astype(int)}"
