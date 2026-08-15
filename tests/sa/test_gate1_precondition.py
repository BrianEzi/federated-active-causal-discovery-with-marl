"""The GATE 1 precondition that runs on every training run.

This exists because the gate was checked once, at d=3, passed, and was then assumed. It
silently stopped holding at d>=5 with the default n_obs, and a whole night of experiments
ran in an environment that did not match its specification before anyone noticed. The same
"checked once, then assumed" shape had already cost this project its previous round.

The point of these tests is that the check must FAIL when it should. A precondition that
only ever passes is decoration.
"""
import numpy as np
import pytest

from sa.env import EnvConfig
from sa.graphs import build_graph_space
from scripts.run_experiment import _check_gate1


@pytest.fixture(scope="module")
def space4():
    return build_graph_space(4)


def test_gate1_passes_at_d4_with_the_default_samples(space4):
    """d=4 with n_obs=1000 is the setting the gate was originally verified at."""
    result = _check_gate1(EnvConfig(d=4, n_obs=1000), space4, n_episodes=150)
    assert result["passed"], (
        f"rate {result['rate']:.4f} CI {result['ci']} excludes target {result['target']:.4f}"
    )


def test_gate1_fails_when_there_is_too_little_observational_data(space4):
    """Starved of data the posterior cannot concentrate, so even the graphs that ARE
    observationally identifiable go unidentified -- and the check must say so."""
    result = _check_gate1(EnvConfig(d=4, n_obs=15), space4, n_episodes=150)
    assert not result["passed"], (
        f"rate {result['rate']:.4f} should fall below target {result['target']:.4f} at "
        f"n_obs=15, but the gate reported a pass"
    )
    assert result["rate"] < result["target"]


def test_gate1_target_is_the_singleton_fraction(space4):
    result = _check_gate1(EnvConfig(d=4), space4, n_episodes=20)
    assert result["target"] == pytest.approx(space4.singleton_fraction)


def test_gate1_can_be_skipped(space4):
    assert _check_gate1(EnvConfig(d=4), space4, n_episodes=0) is None


def test_gate1_reports_a_confidence_interval_containing_its_estimate(space4):
    result = _check_gate1(EnvConfig(d=4), space4, n_episodes=100)
    low, high = result["ci"]
    assert low <= result["rate"] <= high
    assert 0.0 <= low <= high <= 1.0
