"""The Phase 2 matrix must mean what it says.

A sweep definition is executed 66 times without anyone reading the resulting command
lines, so a rendering bug here does not cause a crash -- it produces 198 runs that are
quietly not the experiment that was designed. These tests read the generated CLI back
through the real argument parser.
"""
import pytest

from scripts.run_experiment import build_parser
from scripts.sweep_phase2 import ARCHES, BASELINE, build_matrix, to_cli


@pytest.fixture(scope="module")
def matrix():
    return build_matrix()


def _parse(config):
    return build_parser().parse_args(to_cli(config).split())


def test_both_architectures_have_identical_configuration_sets(matrix):
    """E1 vs E2 is only interpretable if arch is the ONLY difference."""
    by_arch = {arch: [] for arch in ARCHES}
    for c in matrix:
        by_arch[c["arch"]].append({k: v for k, v in c.items()
                                   if k not in ("arch", "tag")})
    first, second = (by_arch[a] for a in ARCHES)
    assert first == second


def test_every_tag_is_unique(matrix):
    """Tags name the output file; a collision silently overwrites a result."""
    tags = [c["tag"] for c in matrix]
    assert len(tags) == len(set(tags))


def test_include_counts_false_does_not_emit_the_flag(matrix):
    """`--include_counts False` would be parsed as the flag being SET.

    The opposite of the intent, and invisible: the run would succeed, the ablation would
    silently not happen, and the arm would agree with the baseline for a reason nobody
    could see.
    """
    ablation = next(c for c in matrix
                    if c["arch"] == "pernode" and c["arm"] == "include_counts")
    assert ablation["include_counts"] is False
    assert "--include_counts" not in to_cli(ablation)
    assert _parse(ablation).include_counts is False


def test_baseline_keeps_action_memory_on(matrix):
    baseline = next(c for c in matrix
                    if c["arch"] == "pernode" and c["arm"] == "baseline")
    assert _parse(baseline).include_counts is True


def test_no_pass_arm_emits_the_flag(matrix):
    arm = next(c for c in matrix if c["arch"] == "pernode" and c["arm"] == "no_pass")
    assert _parse(arm).no_pass is True


def test_every_config_parses(matrix):
    """The whole matrix must survive the real parser, not a mock of it."""
    for config in matrix:
        parsed = _parse(config)
        assert parsed.d == config["d"]
        assert parsed.arch == config["arch"]
        assert parsed.seeds == config["seeds"]


def test_gate1_is_recorded_for_every_run(matrix):
    """G5 exists because a run without its validity check reads as a run without a problem."""
    for config in matrix:
        assert _parse(config).gate1_episodes > 0


def test_only_the_negative_control_uses_a_gate_failing_n_obs(matrix):
    """GATE 1 does not pass at d=5 below n_obs=5000. Any other arm sitting there would be
    an invalid environment presented as a normal result."""
    for config in matrix:
        if config["n_obs"] < 5000:
            assert "NEGCONTROL" in config["tag"], config["tag"]
        if "NEGCONTROL" in config["tag"]:
            assert config["n_obs"] == 1000


def test_baseline_is_the_configuration_that_won(matrix):
    """The levers must be characterised around the network actually being used, not the
    one whose results were invalidated."""
    assert BASELINE["lr"] == 1e-3
    assert BASELINE["hidden"] == 256
    assert BASELINE["episodes_per_update"] == 16
    assert BASELINE["include_counts"] is True
    assert BASELINE["n_obs"] == 5000
    assert BASELINE["layers"] == 1


def test_no_arm_silently_repeats_the_baseline(matrix):
    """A lever value equal to the baseline wastes a task and invites two different numbers
    for the same setting."""
    for config in matrix:
        if config["arm"] in ("baseline", "n_obs", "shaping_coef"):
            continue
        lever = config["arm"]
        if lever in BASELINE:
            assert config[lever] != BASELINE[lever], config["tag"]


def test_output_paths_are_distinct(matrix):
    paths = [to_cli(c).split("--out ")[1].split()[0] for c in matrix]
    assert len(paths) == len(set(paths))
