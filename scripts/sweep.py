"""The experiment grid, parameterised on (k, sigma, n, beta) rather than raw block sizes.

WHY REPARAMETERISE. Varying `private_size` and `n_shared` independently confounds two
different things, because both move the window size: a run at 6+6 and a run at 2+10 have
the same k and nothing else in common. The axes that mean something are

    k      window size = private + shared            how hard is MY problem
    sigma  shared / k, the CONTENDED FRACTION        how much of it is also someone else's
    n      agents                                    how many partners contend for it
    beta   budget / required cover                   effort, normalised so k cannot confound

`private = round(k * (1 - sigma))`, `shared = k - private`. Pure reparameterisation, no new
machinery -- but it makes the federated axis visible, and it immediately exposed something:

    rung   private  shared   sigma
    w04       1        3      0.75
    w08       4        4      0.50
    w12       6        6      0.50
    w20      10       10      0.50
    w30      15       15      0.50

**The window ladder never varied sigma.** It is fixed at 0.50 for four of five rungs while
w04 sits at 0.75 -- so w04 is not on the same line as the others, and every w04 anomaly
(argmax reversing, the learner winning SHD only there, the de-dup gap going non-significant
only there) is a live candidate for a sigma effect read as a k effect.

WHY beta AND NOT A RAW BUDGET. The required cover is closed-form under oracle evidence -- a
directed edge needs its TAIL, a confounded pair needs BOTH endpoints -- and it is sublinear
in k, measured at 0.757k for k=4 falling to 0.542k at k=30. So a fixed budget-per-node hands
the large windows a MORE generous allowance and the resulting decline is a budget effect
wearing a window-size costume. beta removes that.

    Under SAMPLED evidence the required cover is not defined at all: the belief is not a
    function of the intervened SET alone, so no set is sufficient with certainty and
    `scripts/required_cover.py` refuses rather than returning a meaningless number. beta is
    therefore computed from the ORACLE cover of the same topology, and that substitution is
    stated rather than assumed.

WHY FRACTIONAL AND NOT A FULL GRID. 5 x 5 x 4 x 5 is 500 cells. Oracle evaluation of that is
affordable; sampled evaluation, at the measured ~3 s per episode per arm, is not. So: one
factor at a time from a baseline, plus ONE interaction block where theory predicts an
interaction (sigma x n, because contention per shared node depends on both). About 25 cells,
and every marginal claim is still supported.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# The measured required cover, as a fraction of k. See `scripts/required_cover.py`.
# Interpolated between the two anchors the ladder measured; outside them it is clamped,
# because extrapolating a sublinear fit past the range it was fitted on is how a
# normalisation quietly becomes a fudge.
_COVER_ANCHORS = ((4, 0.757), (30, 0.542))


def required_cover_fraction(k: int) -> float:
    """Fraction of a k-window that must be intervened on to identify it, under oracle."""
    (k_lo, f_lo), (k_hi, f_hi) = _COVER_ANCHORS
    if k <= k_lo:
        return f_lo
    if k >= k_hi:
        return f_hi
    t = (k - k_lo) / (k_hi - k_lo)
    return f_lo + t * (f_hi - f_lo)


@dataclass(frozen=True)
class Cell:
    """One experiment. `k`, `sigma`, `n` and `beta` are the axes; the rest is derived."""
    k: int
    sigma: float
    n: int
    beta: float
    axis: str                       # which axis this cell varies, for grouping in reports

    @property
    def private(self) -> int:
        """At least one private node per agent -- a site with none has nothing of its own
        to experiment on, and the attribution question becomes vacuous."""
        return max(1, round(self.k * (1.0 - self.sigma)))

    @property
    def shared(self) -> int:
        return max(1, self.k - self.private)

    @property
    def budget(self) -> int:
        """beta multiples of the required cover, over the WHOLE system.

        The budget is a shared pool of ROUNDS (see docs/TURN_BUDGET_SPEC.md), so the cover
        every agent needs is multiplied by the agent count. Rounded up: a budget below the
        cover cannot identify anything, so rounding down would silently make beta=1 mean
        beta<1."""
        per_agent = required_cover_fraction(self.k) * self.k
        return max(1, math.ceil(self.beta * per_agent * self.n))

    @property
    def name(self) -> str:
        return (f"k{self.k:02d}s{int(round(self.sigma * 100)):02d}"
                f"n{self.n:02d}b{int(round(self.beta * 100)):03d}")

    def as_dict(self) -> dict:
        return {"name": self.name, "axis": self.axis, "k": self.k, "sigma": self.sigma,
                "n": self.n, "beta": self.beta, "private": self.private,
                "shared": self.shared, "budget": self.budget}


# The baseline every one-factor sweep departs from. Chosen mid-range on every axis so a
# move in either direction is measurable rather than clipped at an edge.
BASELINE = dict(k=12, sigma=0.5, n=4, beta=1.5)

AXES: Dict[str, Sequence] = {
    "k": (4, 8, 12, 20, 30),
    "sigma": (0.25, 0.5, 0.75),
    "n": (2, 3, 5, 8, 15),
    "beta": (1.0, 1.2, 1.5, 2.0, 5.0),
}


def build_cells(interaction: bool = True, axes: Optional[Dict[str, Sequence]] = None,
                baseline: Optional[dict] = None) -> List[Cell]:
    """One factor at a time from the baseline, plus the sigma x n interaction block."""
    axes = axes or AXES
    base = dict(baseline or BASELINE)
    seen: Dict[str, Cell] = {}

    def add(axis: str, **overrides):
        cell = Cell(axis=axis, **{**base, **overrides})
        seen.setdefault(cell.name, cell)

    add("baseline")
    for axis, values in axes.items():
        for value in values:
            add(axis, **{axis: value})
    if interaction:
        # Contention per shared node depends on BOTH sigma and n, so this is the one place
        # a main-effects design would mislead.
        for sigma, n in itertools.product(axes["sigma"], (2, 4, 8)):
            add("sigma_x_n", sigma=sigma, n=n)
    return sorted(seen.values(), key=lambda c: (c.axis, c.k, c.sigma, c.n, c.beta))


def command(cell: Cell, seed: int, out_dir: str, *, evidence: str = "oracle",
            arch: str = "gnn_portable", episodes: int = 4000,
            extra: Sequence[str] = ()) -> List[str]:
    """The exact training invocation for one cell. Emitted rather than hand-typed, because
    twice on this project a comparison was built from a neighbouring run's flags."""
    out = f"{out_dir.rstrip('/')}/{cell.name}_s{seed}.json"
    return [".venv/bin/python", "scripts/ma_train.py",
            "--arm", cell.name, "--seed", str(seed),
            "--n_agents", str(cell.n),
            "--private_size", str(cell.private), "--n_shared", str(cell.shared),
            "--budget", str(cell.budget),
            "--n_obs", "60", "--n_int", "20",
            "--turn_order", "round_robin", "--backend", "factored",
            "--policy_arch", arch, "--vary_only",
            "--graph_model", "sf", "--sf_m", "2",
            "--claim_bar", "1.0", "--reward_criterion", "claims",
            "--per_agent_reward", "--episode_mix", "confounded",
            "--normalise_returns",
            "--vs_evidence", evidence,
            "--train_episodes", str(episodes), "--eval_episodes", "200",
            "--no_wandb", "--force", "--out", out, *extra]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out_dir", default="results/sweep")
    ap.add_argument("--evidence", default="oracle", choices=["oracle", "sampled"])
    ap.add_argument("--arch", default="gnn_portable")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--no_interaction", action="store_true")
    ap.add_argument("--emit", choices=["table", "sh", "jobs", "json"], default="table")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calibration", default=None,
                    help="a scripts/calibrate_sweep.py manifest, used to order the job "
                         "list longest-first")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    cells = build_cells(interaction=not args.no_interaction)
    if args.emit == "table":
        print(f"{'cell':22s} {'axis':10s} {'k':>3s} {'sigma':>6s} {'n':>3s} {'beta':>5s} "
              f"{'priv':>5s} {'shared':>6s} {'budget':>7s}")
        for cell in cells:
            print(f"{cell.name:22s} {cell.axis:10s} {cell.k:3d} {cell.sigma:6.2f} "
                  f"{cell.n:3d} {cell.beta:5.2f} {cell.private:5d} {cell.shared:6d} "
                  f"{cell.budget:7d}")
        print(f"\n{len(cells)} cells x {args.seeds} seeds = {len(cells) * args.seeds} runs")
    elif args.emit in ("sh", "jobs"):
        jobs = []
        for cell in cells:
            for seed in range(args.seeds):
                argv_ = command(cell, seed, args.out_dir, evidence=args.evidence,
                                arch=args.arch, episodes=args.episodes)
                out = argv_[argv_.index("--out") + 1]
                jobs.append((cell, seed, out, argv_))

        # LONGEST FIRST. The runs differ by more than 4x in length, and a list schedule
        # that starts the long ones last leaves seven workers idle while one finishes --
        # the classic greedy-scheduling tail. Ordered by the CALIBRATION where one exists,
        # since that is measured, and by budget otherwise, which is what cost tracks.
        estimates = {}
        if args.calibration:
            try:
                measured = json.loads(pathlib.Path(args.calibration).read_text())
                estimates = {row["name"]: row["run_seconds"] for row in measured["cells"]}
            except (OSError, ValueError, KeyError):
                print("# warning: calibration unreadable; ordering by budget instead")
        jobs.sort(key=lambda job: estimates.get(job[0].name, float(job[0].budget)),
                  reverse=True)

        if args.emit == "sh":
            print("#!/usr/bin/env bash")
            print("set -u")
            print("export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
        for cell, seed, out, argv_ in jobs:
            # The skip is inside the line, so the list is restart-safe however it is run:
            # sequentially as a script, or fed to xargs -P for the parallel launch.
            print(f'[ -f "{out}" ] || {" ".join(argv_)}')

    else:
        print(json.dumps([c.as_dict() for c in cells], indent=1))

    if args.manifest:
        path = pathlib.Path(args.manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"baseline": BASELINE, "axes": {k: list(v) for k, v in AXES.items()},
             "seeds": args.seeds, "evidence": args.evidence,
             "cells": [c.as_dict() for c in cells]}, indent=1))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
