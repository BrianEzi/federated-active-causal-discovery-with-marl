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
import sys
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
    # INTERVENTIONAL ROWS PER ROUND. Inert under oracle evidence -- the belief prunes by true
    # ancestry from the intervened SET and never reads the data matrix, so 20/100/400 give an
    # identical 0.967 success and 0.0004 soft SHD. Under SAMPLED evidence it is the dominant
    # axis: greedy runs 0.000 / 0.167 / 0.400 and soft SHD 0.0574 / 0.0222 / 0.0147 across
    # the same three values. Twenty times the data is the difference between no signal and a
    # working regime, where sixteen times the OBSERVATIONAL data moved soft SHD by 8%.
    n_int: int = 20

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
        # The n_int suffix appears only when it is off the default, so every existing cell
        # name -- and every result file already on disk under it -- stays exactly as it was.
        base = (f"k{self.k:02d}s{int(round(self.sigma * 100)):02d}"
                f"n{self.n:02d}b{int(round(self.beta * 100)):03d}")
        return base if self.n_int == 20 else f"{base}i{self.n_int:04d}"

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
    # n=15 WAS HERE AND WAS REMOVED, 30 Aug 2026, on a measurement rather than a guess.
    # Calibrated at 29.9 h per run: 89.7 core-hours at three seeds, 47.6% of the entire
    # sweep for ONE point on this axis. Worse, a single 29.9 h job is a hard FLOOR on the
    # sweep's wall clock at any worker count, because no schedule finishes before its
    # longest job -- and it exceeds any cluster walltime even chunked across resumes.
    # The cost compounds three ways at once: budget 187, fifteen agents forwarding and
    # being scored every round, and d = 96 nodes making the projection and the SCM
    # sampling expensive too -- 55x the baseline per episode.
    # n=10 keeps a point above n=8 and the axis still spans 5x across five points.
    "n": (2, 3, 5, 8, 10),
    "beta": (1.0, 1.2, 1.5, 2.0, 5.0),
}

# SAMPLED ONLY, and the reason is the whole point of the axis. Under oracle evidence n_int is
# inert, so sweeping it there would spend three seeds per value reproducing identical numbers.
# Under sampling it decides whether the regime has any signal at all, and the CONVERGENCE it
# traces is the result: as n_int grows the ancestry test approaches always-correct and sampled
# approaches oracle. That is not a threat to the realism claim, it is the confirmation of it --
# the two regimes are the same problem at different data budgets. 1000 is included to find
# where they meet, and the interesting quantity is whether the LEARNED policy reaches oracle
# performance at a lower n_int than greedy does, which would be a data-efficiency result
# rather than merely a performance one.
SAMPLED_ONLY_AXES: Dict[str, Sequence] = {
    "n_int": (20, 100, 400, 1000),
}


def build_cells(interaction: bool = True, axes: Optional[Dict[str, Sequence]] = None,
                baseline: Optional[dict] = None, evidence: str = "oracle") -> List[Cell]:
    """One factor at a time from the baseline, plus the sigma x n interaction block.

    `evidence="sampled"` additionally sweeps `SAMPLED_ONLY_AXES`, which are the axes that are
    inert under oracle evidence and would otherwise buy identical numbers three seeds at a time.
    """
    axes = dict(axes or AXES)
    if evidence == "sampled":
        axes.update(SAMPLED_ONLY_AXES)
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
            "--n_obs", "60", "--n_int", str(cell.n_int),
            "--turn_order", "round_robin", "--backend", "factored",
            "--policy_arch", arch, "--vary_only",
            "--graph_model", "sf", "--sf_m", "2",
            "--claim_bar", "1.0", "--reward_criterion", "claims",
            "--per_agent_reward", "--episode_mix", "confounded",
            "--normalise_returns",
            "--vs_evidence", evidence,
            "--train_episodes", str(episodes), "--eval_episodes", "200",
            "--no_wandb", "--force", "--out", out, *extra]


# THE COST IS EXTREMELY CONCENTRATED, which is what makes a split across machines clean
# rather than arbitrary. Measured 30 Aug 2026: twelve of the twenty cells cost 14% of the
# whole sweep and none of them takes an hour, while two cells (k=30 and n=15) cost 52%.
# So the boundaries are not round numbers picked for tidiness -- they sit in the two large
# gaps in the measured distribution.
#
#   cheap   < 60 min/run    12 cells, ~16 core-h at 3 seeds. Runs beside interactive work.
#   medium  1-6 h/run        5 cells, ~37 core-h. Too slow to want on a working machine,
#                            too small to be worth a cluster queue.
#   heavy   > 6 h/run        2 cells, ~60 core-h. k=30 exceeds any single cluster job's
#                            walltime, so it must be chunked -- see --resume_from.
TIERS = {"cheap": (0.0, 60.0), "medium": (60.0, 360.0), "heavy": (360.0, float("inf"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed_list", default=None,
                    help="comma-separated seeds, overriding --seeds. The heavy cells are "
                         "split across machines by SEED (seed 0 on the fallback machine, "
                         "1 and 2 on the cluster), so a partition needs this rather than "
                         "a count.")
    ap.add_argument("--out_dir", default="results/sweep")
    ap.add_argument("--evidence", default="oracle", choices=["oracle", "sampled"])
    ap.add_argument("--arch", default="gnn_portable")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--extra", default="",
                    help="flags appended to every emitted command, space separated. The "
                         "31 Aug sweep uses '--turn_aware_credit --local_epochs 4': credit "
                         "because 75%% of rows were otherwise discarded actions, and "
                         "local_epochs because plain FedAvg matched pooled at k=12 (0.977 "
                         "vs 0.980) and is genuinely federated, where pooling concatenates "
                         "raw trajectories.")
    ap.add_argument("--no_interaction", action="store_true")
    ap.add_argument("--emit", choices=["table", "sh", "jobs", "json"], default="table")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calibration", default=None,
                    help="a scripts/calibrate_sweep.py manifest, used to order the job "
                         "list longest-first and to resolve --tier")
    ap.add_argument("--tier", default=None, choices=["cheap", "medium", "heavy"],
                    help="split the sweep by MEASURED cost, for running it across several "
                         "machines. Requires --calibration. See TIERS for the boundaries "
                         "and why they sit where they do.")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    cells = build_cells(interaction=not args.no_interaction, evidence=args.evidence)
    if args.tier:
        if not args.calibration:
            print("--tier needs --calibration: the tiers are defined on MEASURED cost, "
                  "and guessing which cell is expensive is what the calibration exists "
                  "to stop.", file=sys.stderr)
            return 2
        try:
            measured = json.loads(pathlib.Path(args.calibration).read_text())
        except (OSError, ValueError) as error:
            print(f"cannot read {args.calibration}: {error}", file=sys.stderr)
            return 2
        minutes = {row["name"]: row["run_seconds"] / 60.0 for row in measured["cells"]}
        missing = [c.name for c in cells if c.name not in minutes]
        if missing:
            print(f"# warning: {len(missing)} cell(s) absent from the calibration and so "
                  f"in no tier: {', '.join(missing)}", file=sys.stderr)
        low, high = TIERS[args.tier]
        cells = [c for c in cells if low <= minutes.get(c.name, -1.0) < high]

    seeds = ([int(x) for x in args.seed_list.split(",") if x.strip()]
             if args.seed_list else list(range(args.seeds)))
    if args.emit == "table":
        print(f"{'cell':22s} {'axis':10s} {'k':>3s} {'sigma':>6s} {'n':>3s} {'beta':>5s} "
              f"{'priv':>5s} {'shared':>6s} {'budget':>7s}")
        for cell in cells:
            print(f"{cell.name:22s} {cell.axis:10s} {cell.k:3d} {cell.sigma:6.2f} "
                  f"{cell.n:3d} {cell.beta:5.2f} {cell.private:5d} {cell.shared:6d} "
                  f"{cell.budget:7d}")
        print(f"\n{len(cells)} cells x {len(seeds)} seeds = "
              f"{len(cells) * len(seeds)} runs  (seeds {seeds})")
    elif args.emit in ("sh", "jobs"):
        jobs = []
        for cell in cells:
            for seed in seeds:
                argv_ = command(cell, seed, args.out_dir, evidence=args.evidence,
                                arch=args.arch, episodes=args.episodes,
                                extra=tuple(args.extra.split()) if args.extra else ())
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
             "seeds": seeds, "tier": args.tier, "evidence": args.evidence,
             "cells": [c.as_dict() for c in cells]}, indent=1))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
