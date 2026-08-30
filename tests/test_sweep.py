"""The sweep grid must be a reparameterisation, not a new set of assumptions.

`private` and `shared` both move `k`, so varying them independently confounds "how hard is
my problem" with "how much of it is contended". The grid is therefore parameterised on
(k, sigma, n, beta) and the block sizes are derived.
"""
import math

import pytest

from scripts.sweep import (AXES, BASELINE, Cell, build_cells, command,
                           required_cover_fraction)


def _cell(**kw):
    return Cell(axis="test", **{**BASELINE, **kw})


# -- the reparameterisation is exact ------------------------------------------------------

@pytest.mark.parametrize("k", [4, 5, 8, 12, 20, 30])
@pytest.mark.parametrize("sigma", [0.25, 0.5, 0.75])
def test_private_plus_shared_always_reconstructs_k(k, sigma):
    cell = _cell(k=k, sigma=sigma)
    assert cell.private + cell.shared == k


@pytest.mark.parametrize("sigma", [0.0, 0.1, 0.9, 1.0])
def test_every_agent_keeps_at_least_one_private_node(sigma):
    """A site with nothing of its own has nothing to experiment on, and the attribution
    question becomes vacuous -- so the extremes must clamp rather than produce zero."""
    cell = _cell(sigma=sigma)
    assert cell.private >= 1 and cell.shared >= 1


def test_sigma_one_half_splits_evenly():
    assert _cell(k=12, sigma=0.5).private == _cell(k=12, sigma=0.5).shared == 6


def test_sigma_is_the_contended_fraction_it_claims_to_be():
    cell = _cell(k=12, sigma=0.75)
    assert cell.shared / cell.k == pytest.approx(0.75, abs=0.05)


# -- the historical ladder, which is what motivated the axis ------------------------------

def test_the_window_ladder_never_varied_sigma():
    """w04 sits at 0.75 while every other rung sits at 0.50, so w04 is not on the same line
    -- which is a live candidate explanation for every w04 anomaly."""
    ladder = {"w04": (1, 3), "w08": (4, 4), "w12": (6, 6), "w20": (10, 10), "w30": (15, 15)}
    sigmas = {name: shared / (private + shared) for name, (private, shared) in ladder.items()}
    assert sigmas["w04"] == pytest.approx(0.75)
    assert all(sigmas[n] == pytest.approx(0.50) for n in ("w08", "w12", "w20", "w30"))


# -- beta, and why it is not a raw budget -------------------------------------------------

def test_required_cover_is_sublinear_which_is_why_beta_exists():
    """A fixed budget-per-node hands large windows a MORE generous allowance, so the
    resulting decline is a budget effect wearing a window-size costume."""
    assert required_cover_fraction(4) > required_cover_fraction(30)


def test_required_cover_is_clamped_outside_the_measured_range():
    """Extrapolating a sublinear fit past the range it was fitted on is how a normalisation
    quietly becomes a fudge."""
    assert required_cover_fraction(2) == required_cover_fraction(4)
    assert required_cover_fraction(60) == required_cover_fraction(30)


def test_budget_scales_with_beta_and_with_agents():
    base = _cell(beta=1.0).budget
    assert _cell(beta=2.0).budget >= 2 * base - 1
    assert _cell(n=8).budget > _cell(n=4).budget


def test_beta_one_is_never_rounded_below_the_cover():
    """Rounding down would silently make beta=1 mean beta<1, and beta=1 is the cell that
    says 'exactly enough budget to be possible'."""
    for k in (4, 8, 12, 20, 30):
        cell = _cell(k=k, beta=1.0)
        assert cell.budget >= math.ceil(required_cover_fraction(k) * k * cell.n) - 1


# -- the design ---------------------------------------------------------------------------

def test_cells_are_deduplicated_so_the_baseline_is_not_run_five_times():
    cells = build_cells()
    assert len({c.name for c in cells}) == len(cells)


def test_every_axis_value_appears_somewhere():
    cells = build_cells()
    for axis, values in AXES.items():
        present = {getattr(c, axis) for c in cells}
        assert set(values) <= present, axis


def test_interaction_block_can_be_switched_off():
    assert len(build_cells(interaction=False)) < len(build_cells(interaction=True))


def test_emitted_command_carries_the_derived_topology_not_the_axes():
    """ma_train takes block sizes, not (k, sigma). The translation must happen here, once,
    rather than being retyped per run -- twice on this project a comparison was built from
    a neighbouring run's flags."""
    cell = _cell(k=12, sigma=0.75, n=3)
    argv = command(cell, seed=1, out_dir="results/x")
    assert argv[argv.index("--private_size") + 1] == str(cell.private)
    assert argv[argv.index("--n_shared") + 1] == str(cell.shared)
    assert argv[argv.index("--n_agents") + 1] == "3"
    assert argv[argv.index("--budget") + 1] == str(cell.budget)
    assert "--normalise_returns" in argv          # the fix that unblocked the agent axis
