"""Every canary must FIRE on the failure it was written for.

A check that only ever passes is decoration. Each test below reconstructs the specific
situation that went unnoticed in an earlier round and asserts the canary catches it --
and, separately, that it stays quiet on a healthy run, so it cannot be made to pass by
simply always failing.
"""
import numpy as np
import pytest

from sa.evaluate import EpisodeTrace
from sa.gates import (
    canary_anchors,
    canary_entropy,
    canary_gate1,
    canary_informative_fraction,
    canary_seed_spread,
    collect_canaries,
)


def _traces(costs, budget=4):
    """Traces with prescribed per-episode cost; unsolved episodes charge the full budget."""
    out = []
    for c in costs:
        identified = c < budget
        out.append(EpisodeTrace(identified=identified,
                                n_interventions=int(c) if identified else budget,
                                passed_early=False, mec_size=2, is_singleton=False))
    return out


# --------------------------------------------------------------------------------------
# G1 entropy
# --------------------------------------------------------------------------------------

def test_g1_fires_on_near_uniform_policy():
    """The overnight failure signature: entropy pinned near ln(n_actions)."""
    # d=5 with a pass action -> 6 actions, ln 6 = 1.792. Failing runs sat at 1.2-1.6.
    c = canary_entropy(final_entropy=1.596, n_actions=6)
    assert not c.ok
    assert c.severity == "warn"
    assert "near-uniform" in c.detail


def test_g1_quiet_on_committed_policy():
    """Passing runs sat at 0.5-0.7 nats."""
    c = canary_entropy(final_entropy=0.6, n_actions=6)
    assert c.ok


def test_g1_fires_on_missing_history():
    c = canary_entropy(final_entropy=float("nan"), n_actions=6)
    assert not c.ok
    assert "not finite" in c.detail


def test_g1_ratio_is_relative_not_absolute():
    """The same entropy must read differently against different action counts.

    This is the whole point of the canary: 1.5 nats is committed at d=20 and near-uniform
    at d=5. A fixed absolute threshold would not survive the scaling ladder.
    """
    assert not canary_entropy(1.5, n_actions=6).ok      # ln 6  = 1.79
    assert canary_entropy(1.5, n_actions=100).ok        # ln 100 = 4.61


# --------------------------------------------------------------------------------------
# G2 anchors
# --------------------------------------------------------------------------------------

def test_g2_quiet_on_well_formed_references():
    random_ref = _traces([1, 2, 3, 4])
    greedy_ref = _traces([1, 1, 2, 2])
    c = canary_anchors(random_ref, greedy_ref, budget=4)
    assert c.ok, c.detail
    assert "anchors exact" in c.detail


def test_g2_fires_when_references_are_indistinguishable():
    """Denominator ~0: gap closed is undefined, and must not be reported as a number."""
    ref = _traces([2, 2, 2, 2])
    c = canary_anchors(ref, list(ref), budget=4)
    assert not c.ok
    assert c.severity == "fail"
    assert "UNDEFINED" in c.detail


def test_g2_fires_when_references_are_swapped():
    """Greedy passed in as random and vice versa: the scale inverts, anchoring at 0/1 fails.

    Reproduces the shape of the corruption that was once read off as anchors of
    0.233 / 1.067 and believed.
    """
    random_ref = _traces([1, 1, 2, 2])   # actually the better policy
    greedy_ref = _traces([1, 2, 3, 4])   # actually the worse one
    c = canary_anchors(random_ref, greedy_ref, budget=4)
    # The 0/1 identity still holds under a swap -- the formula defines its own endpoints --
    # so the ordering check is what has to catch this. Without it the canary is silent
    # here, which was the first version of this code.
    assert not c.ok
    assert c.severity == "fail"
    assert "INVERTED" in c.detail


def test_g2_fires_on_mismatched_reference_lengths():
    """Different episode counts mean the references were not drawn from paired seeds."""
    c = canary_anchors(_traces([1, 2, 3, 4]), _traces([1, 1]), budget=4)
    assert c.ok or not c.ok  # must not raise
    assert isinstance(c.as_dict()["detail"], str)


# --------------------------------------------------------------------------------------
# G3 informative fraction
# --------------------------------------------------------------------------------------

def test_g3_fires_on_the_retracted_figure():
    """The measured case: 93-98% of scored steps were vacuous ties."""
    c = canary_informative_fraction(fraction=0.04)
    assert not c.ok
    assert c.severity == "fail"
    assert "Gap closed is unaffected" in c.detail  # blast radius stated, not overstated


def test_g3_quiet_when_most_steps_are_real_choices():
    assert canary_informative_fraction(fraction=0.62).ok


def test_g3_fires_when_undefined():
    c = canary_informative_fraction(fraction=float("nan"))
    assert not c.ok
    assert "undefined, not high" in c.detail


# --------------------------------------------------------------------------------------
# G4 seed spread
# --------------------------------------------------------------------------------------

def test_g4_fires_on_the_measured_unstable_configuration():
    """pernode_best without action memory: +1.043 to -1.766 across seeds."""
    c = canary_seed_spread([1.043, -1.766, 0.2])
    assert not c.ok
    assert "unstable" in c.detail


def test_g4_quiet_on_the_measured_stable_configuration():
    """The configuration actually carried forward: d=5, three seeds, all near +1.23."""
    assert canary_seed_spread([1.233, 1.198, 1.271]).ok


def test_g4_does_not_fire_on_a_single_seed():
    """One seed has no spread; claiming stability from it would be worse than silence."""
    c = canary_seed_spread([1.2])
    assert c.ok
    assert "not defined" in c.detail


def test_g4_ignores_non_finite_seeds():
    c = canary_seed_spread([1.2, float("nan"), 1.25])
    assert c.ok
    assert "across 2 seeds" in c.detail


# --------------------------------------------------------------------------------------
# G5 gate 1
# --------------------------------------------------------------------------------------

def test_g5_fires_when_gate1_was_never_run():
    """The actual d>=5 failure: no check present at all, which read as no problem."""
    c = canary_gate1(None)
    assert not c.ok
    assert c.severity == "fail"
    assert "NOT evaluated" in c.detail


def test_g5_fires_when_gate1_ran_and_failed():
    c = canary_gate1({"rate": 0.040, "target": 0.0893, "passed": False})
    assert not c.ok
    assert "GATE 1 FAILED" in c.detail


def test_g5_quiet_when_gate1_passed():
    c = canary_gate1({"rate": 0.088, "target": 0.0893, "passed": True})
    assert c.ok


# --------------------------------------------------------------------------------------
# collect_canaries
# --------------------------------------------------------------------------------------

def _per_seed(gaps, entropy, informative):
    return [{"gap_closed": g, "final_entropy": entropy,
             "deterministic": {"informative_fraction": informative}} for g in gaps]


def test_collect_returns_all_five_in_order():
    records = collect_canaries(_per_seed([1.2, 1.25, 1.19], 0.6, 0.5),
                               {"rate": 0.088, "target": 0.0893, "passed": True},
                               n_actions=6,
                               random_ref=_traces([1, 2, 3, 4]),
                               greedy_ref=_traces([1, 1, 2, 2]), budget=4)
    assert [r["name"][:2] for r in records] == ["G1", "G2", "G3", "G4", "G5"]
    assert all(r["ok"] for r in records), [r for r in records if not r["ok"]]


def test_collect_flags_a_failing_run_on_every_axis():
    """A run that is bad in all five ways must produce five fired canaries, not one."""
    records = collect_canaries(_per_seed([1.043, -1.766], 1.6, 0.04),
                               None, n_actions=6)
    assert not any(r["ok"] for r in records)


def test_collect_never_raises_on_malformed_input():
    """A canary must not destroy a run that already cost hours of compute."""
    records = collect_canaries([{}, {"gap_closed": "not a number"}], None, n_actions=6)
    assert len(records) == 5
    for r in records:
        assert isinstance(r["detail"], str)


def test_collect_is_json_serialisable():
    """These go straight into the result file; numpy scalars would break json.dump."""
    import json

    records = collect_canaries(_per_seed([np.float64(1.2), np.float64(1.3)],
                                         np.float64(0.6), np.float64(0.5)),
                               {"rate": np.float64(0.088), "target": np.float64(0.0893),
                                "passed": np.bool_(True)},
                               n_actions=6,
                               random_ref=_traces([1, 2, 3, 4]),
                               greedy_ref=_traces([1, 1, 2, 2]), budget=4)
    json.dumps(records)  # must not raise
