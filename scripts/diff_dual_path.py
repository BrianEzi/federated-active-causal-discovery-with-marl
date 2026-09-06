"""Do the local and Myriad runs of the same rho12 cell agree?

WHY THIS EXISTS. The rho12 fleet is running on two machines at once -- a six-worker laptop and
a 21-task SGE array -- because neither path's completion time was certain. That hedge is only
safe if I check, rather than assume, that a cell computed in one place matches the same cell
computed in the other. Two things could differ and neither would announce itself:

  * the CODE. The cluster checkout was 410 commits behind as recently as this morning. It is
    now at parity, but "is at parity" is a claim about a moment, and a job that started before
    a pull runs the code it started with.
  * the ENVIRONMENT. Different BLAS builds, different torch minor versions, different CPU
    instruction sets. Training is stochastic and seeded; identical seeds do NOT guarantee
    identical floating-point trajectories across machines, and a policy that diverges early
    can end up somewhere else entirely.

So this does not assert the two must be bit-identical. It reports HOW FAR APART they are and
leaves the judgement explicit: a small difference in final training diagnostics is expected
across machines, and a large one means the two fleets are not interchangeable and only one of
them may be used.

    python scripts/diff_dual_path.py --local results/rho12 --other results/rho12_myriad
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np

FIELDS = ("budget", "train_episodes", "n_int", "n_obs", "private_size", "n_shared",
          "n_agents", "vs_evidence", "vs_evidence_power", "observe_belief_channels")


def load(path: pathlib.Path):
    d = json.loads(path.read_text())
    return d[0] if isinstance(d, list) else d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", default="results/rho12b")
    ap.add_argument("--other", default="results/rho12b_myriad")
    # A tolerance on the FINAL training diagnostics, not on a scored metric. Nothing here is a
    # thesis number; it is a check that two fleets converged to comparable places.
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args(argv)

    a_dir, b_dir = pathlib.Path(args.local), pathlib.Path(args.other)
    names = sorted({p.name for p in a_dir.glob("rho*_s?.json")}
                   & {p.name for p in b_dir.glob("rho*_s?.json")})
    only_a = sorted({p.name for p in a_dir.glob("rho*_s?.json")}
                    - {p.name for p in b_dir.glob("rho*_s?.json")})
    only_b = sorted({p.name for p in b_dir.glob("rho*_s?.json")}
                    - {p.name for p in a_dir.glob("rho*_s?.json")})

    print(f"in both: {len(names)}   only local: {len(only_a)}   only other: {len(only_b)}")
    if not names:
        print("no cell has completed on both paths yet -- nothing to compare")
        return 0

    print(f"\n{'cell':18s} {'wr local':>9s} {'wr other':>9s} {'|d|':>7s} "
          f"{'solve local':>12s} {'solve other':>12s} {'config':>8s}")
    bad = 0
    for name in names:
        A, B = load(a_dir / name), load(b_dir / name)
        ha, hb = (A.get("history") or [])[-10:], (B.get("history") or [])[-10:]
        wa = float(np.mean([h.get("window_rate", 0.0) for h in ha])) if ha else float("nan")
        wb = float(np.mean([h.get("window_rate", 0.0) for h in hb])) if hb else float("nan")
        sa = float(np.mean([h.get("solve_rate", 0.0) for h in ha])) if ha else float("nan")
        sb = float(np.mean([h.get("solve_rate", 0.0) for h in hb])) if hb else float("nan")
        ca, cb = A.get("config", {}), B.get("config", {})
        cfg_diff = [f for f in FIELDS if ca.get(f) != cb.get(f)]
        far = abs(wa - wb) > args.tol
        bad += bool(cfg_diff) or far
        print(f"{name[:-5]:18s} {wa:9.3f} {wb:9.3f} {abs(wa - wb):7.3f} "
              f"{sa:12.3f} {sb:12.3f} "
              + ("SAME" if not cfg_diff else "DIFFERS: " + ",".join(cfg_diff)))

    print()
    if bad:
        print(f"!! {bad} cell(s) differ by more than {args.tol} window rate, or have differing "
              f"configs. The two fleets are NOT interchangeable -- pick ONE path per cell, say "
              f"which in the write-up, and do not average them.")
        return 1
    print(f"all {len(names)} shared cell(s) agree within {args.tol} window rate on matching "
          f"configs. Either path may be used, but say which one was.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
