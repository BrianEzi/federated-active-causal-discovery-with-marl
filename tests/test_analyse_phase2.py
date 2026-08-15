"""The paired analysis must be correct before the data arrives, not after.

Built against synthetic results with known answers, so the classification logic is fixed
while nobody can see what it would say about the real sweep.
"""
import json

import pytest

from scripts.analyse_phase2 import (
    EFFECT_THRESHOLD,
    classify,
    load_canaries,
    summarise_by_tag,
    tag_to_arm,
)
from scripts.analyse_sweep import load_rows
from scripts.sweep_phase2 import build_matrix


def _payload(tag, arch, gaps, canaries=None):
    return {
        "tag": tag,
        "args": {"d": 5, "observation": "edge_marginals", "train_episodes": 6000,
                 "arch": arch, "n_obs": 5000, "budget": 20},
        "space": {"n_dags": 29281, "n_mecs": 9484, "singleton_fraction": 0.0893},
        "provenance": {"git_commit": "abc12345", "torch": "2.6.0+cpu"},
        "references": {"random": {"mean_cost": 3.2},
                       "greedy_oracle": {"mean_cost": 1.8}},
        "canaries": canaries or [],
        "per_seed": [
            {"seed": i, "final_entropy": 0.6, "passed": True, "train_seconds": 100,
             "deterministic": {"gap_closed": g, "solve_rate": 0.95,
                               "greedy_solve_rate": 0.98, "mean_cost": 2.0,
                               "cost_ci": [1.8, 2.2], "under_acting_rate": 0.0,
                               "optimal_rate": 0.6, "informative_fraction": 0.7,
                               "mean_regret": 0.1, "repeat_rate": 0.1,
                               "distinct_targets": 3.0},
             "sampled": {"gap_closed": g - 0.1}}
            for i, g in enumerate(gaps)
        ],
    }


@pytest.fixture
def results(tmp_path):
    def write(tag, arch, gaps, canaries=None):
        (tmp_path / f"{tag}.json").write_text(
            json.dumps(_payload(tag, arch, gaps, canaries)), encoding="utf-8")
    return tmp_path, write


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------

def test_lever_moving_both_architectures_is_a_task_property():
    assert classify(-1.2, -1.4) == "task"


def test_lever_moving_only_flat_is_an_artefact():
    """The case the whole comparison exists for: the overnight conclusion does not carry."""
    assert classify(0.05, -1.5) == "artefact"


def test_lever_moving_only_pernode_is_unlocked():
    assert classify(-1.5, 0.02) == "unlocked"


def test_lever_moving_neither_is_dead():
    assert classify(0.1, -0.2) == "dead"


def test_threshold_is_symmetric_in_sign():
    """A lever that HELPS must be detected as readily as one that hurts."""
    assert classify(EFFECT_THRESHOLD + 0.01, 0.0) == "unlocked"
    assert classify(-(EFFECT_THRESHOLD + 0.01), 0.0) == "unlocked"


def test_threshold_is_exclusive():
    assert classify(EFFECT_THRESHOLD, EFFECT_THRESHOLD) == "dead"


# --------------------------------------------------------------------------------------
# Arm recovery -- the bug that motivated a separate script
# --------------------------------------------------------------------------------------

def test_arms_come_from_the_matrix_not_from_parsing_the_tag():
    """The old tag parser returns "pernode" for every Phase 2 configuration.

    Tags are now "<arch>_<lever>_<value>", so splitting on the last underscore recovers the
    architecture rather than the lever, and every arm would collapse into one.
    """
    arms = tag_to_arm()
    assert arms["pernode_lr_0.0001"] == "lr"
    assert arms["flat_entropy_coef_0.03"] == "entropy_coef"
    assert arms["pernode_baseline"] == "baseline"

    from scripts.analyse_sweep import _arm_of
    assert _arm_of("pernode_lr_0.0001") != "lr"   # the reason this module exists


def test_every_matrix_tag_has_an_arm():
    arms = tag_to_arm()
    for config in build_matrix():
        assert config["tag"] in arms


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------

def test_summary_uses_median_and_reports_spread(results):
    tmp_path, write = results
    write("pernode_baseline", "pernode", [1.0, 1.2, 1.9])

    summary = summarise_by_tag(load_rows(str(tmp_path)), tag_to_arm())
    entry = summary["pernode_baseline"]
    assert entry["median_gap"] == pytest.approx(1.2)
    assert entry["min_gap"] == pytest.approx(1.0)
    assert entry["spread"] == pytest.approx(0.9)


def test_pairing_recovers_deltas_against_each_architectures_own_baseline(results):
    """Deltas, not raw values: the two architectures sit at different absolute levels, so
    raw numbers would only re-measure the architecture gap that is already known."""
    tmp_path, write = results
    write("pernode_baseline", "pernode", [1.2, 1.2, 1.2])
    write("flat_baseline", "flat", [-1.8, -1.8, -1.8])
    write("pernode_lr_0.0001", "pernode", [1.15, 1.15, 1.15])
    write("flat_lr_0.0001", "flat", [-0.2, -0.2, -0.2])

    summary = summarise_by_tag(load_rows(str(tmp_path)), tag_to_arm())
    delta_pernode = summary["pernode_lr_0.0001"]["median_gap"] \
        - summary["pernode_baseline"]["median_gap"]
    delta_flat = summary["flat_lr_0.0001"]["median_gap"] \
        - summary["flat_baseline"]["median_gap"]

    assert delta_pernode == pytest.approx(-0.05)
    assert delta_flat == pytest.approx(1.6)
    # Big under flat, nothing under per-node: the lever was compensating.
    assert classify(delta_pernode, delta_flat) == "artefact"


def test_canaries_are_loaded_alongside_the_numbers(results):
    tmp_path, write = results
    write("pernode_NEGCONTROL_n_obs_1000_gate1_fails", "pernode", [0.1],
          canaries=[{"name": "G5 gate 1 recorded", "ok": False, "severity": "fail",
                     "observed": 0.0267, "threshold": 0.0893, "detail": "GATE 1 FAILED"}])

    loaded = load_canaries(tmp_path)
    fired = [c for recs in loaded.values() for c in recs if not c["ok"]]
    assert len(fired) == 1
    assert fired[0]["name"].startswith("G5")


def test_incomplete_pairs_are_reported_not_silently_dropped(results, capsys):
    """A task that died leaves one architecture only. Dropping it would quietly turn a
    missing run into an absent lever."""
    from scripts.analyse_phase2 import main
    import sys

    tmp_path, write = results
    write("pernode_baseline", "pernode", [1.2])
    write("flat_baseline", "flat", [-1.8])
    write("pernode_gamma_1.0", "pernode", [1.1])   # flat counterpart missing

    sys.argv = ["analyse_phase2", "--results", str(tmp_path)]
    main()
    assert "INCOMPLETE" in capsys.readouterr().out
