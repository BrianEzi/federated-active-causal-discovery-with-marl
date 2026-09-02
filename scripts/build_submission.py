"""Assemble `submission/` -- everything needed to verify or reproduce the thesis.

WHY. `results/` is 2.1 GB of working directory: superseded builds, abandoned experiments, and
4,541 checkpoints of which the thesis quotes a small fraction. Submitting it would be
unreadable, and submitting nothing would make every number unverifiable. This selects the runs
the thesis actually cites, ships their checkpoints so an evaluation can be re-run rather than
merely re-read, and records what each group supports.

WHAT IS SELECTED, AND WHY EACH GROUP EXISTS
  sweep12k     the 12,000-episode sweep -- the primary tables (RQ1)
  sweep4k      the 4,000-episode sweep -- the training-budget appendix
  federation   arms A and E plus the coordination baselines (RQ3)
  transfer     the answer-rate fleet (RQ2)
  attribution  the identifiability grid and scaling runs (RQ4)
  budget       the retrained cells behind the training-budget finding
  checkpoint   the best-vs-final-vs-argmax audit

Checkpoints ship as `_best.pt` and `.pt` only. Intermediate `u*.pt` are training artefacts and
are excluded except `u0500`, which is the 8,000-episode point the budget appendix reports.

    python scripts/build_submission.py           # assemble
    python scripts/build_submission.py --check   # verify nothing has drifted
"""
from __future__ import annotations
import argparse, glob, hashlib, json, pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "submission"

GROUPS = [
    ("sweep12k", "The 12,000-episode sweep. Primary tables for RQ1: window size, federation "
                 "size, contended fraction and budget multiplier, three seeds per cell.",
     ["results/sweep12k/k*_s?.json"], True),
    ("sweep4k", "The original 4,000-episode sweep. Reported beside the re-run in the "
                "training-budget appendix, never mixed into a table with it.",
     ["results/sweep/oracle/k*_s?.json"], True),
    ("federation", "RQ3. Arm A is the federated system; arm E removes the information and "
                   "optimiser partitions. Coordination baselines are scored inside each run.",
     ["results/central/v2_k*_?_s?.json", "results/central/shd_*.json"], True),
    ("transfer", "RQ2. The answer-rate fleet: seven partial-oracle rates, three seeds each, "
                 "evaluated under genuine finite-sample evidence.",
     ["results/power/rho/rho*_s?.json", "results/power/rho/CURVE.json",
      "results/power/confirm/*.json"], True),
    ("attribution", "RQ4. The identifiability grid, the matched-budget control, the coverage "
                    "series and the scaling runs to k=50.",
     ["results/attr_ceiling*.json", "results/attr_scale_final.json", "results/attr_reach.json",
      "results/attr/transfer_k12s50n04b200_s?.json"], False),
    ("budget", "The retrained cells behind the training-budget finding, and the learning-rate "
               "probe that ruled out an unstable step size.",
     ["results/longcheck/*_long_s?.json", "results/longcheck/*_conv_s?.json",
      "results/longcheck/shd_*.json", "results/lrcheck/*.json"], True),
    ("checkpoint", "The checkpoint audit: the same cells scored at the selected checkpoint, "
                   "the final update, and under argmax.",
     ["results/ckpt/*.json", "results/sweep12k/shd/*.json",
      "results/sweep12k/shd_final/*.json", "results/sweep12k/shd_argmax/*.json",
      "results/sweep12k/shd_u0500/*.json"], False),
]

# Checkpoints worth shipping: the two the thesis reports, plus the 8,000-episode point.
CKPT_SUFFIXES = ("_best.pt", ".pt", "_u0500.pt")


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    lines = ["# Submission manifest", "",
             "Assembled by `scripts/build_submission.py`. Every file here is cited by the",
             "dissertation. `results/` in the working repository holds the full 2.1 GB of",
             "working data including superseded builds; this is the subset the text depends on.",
             "", "`--check` re-hashes every file against its source and reports drift.", ""]
    total_bytes = drift = missing = 0

    for name, why, patterns, want_ckpt in GROUPS:
        data_dir, ck_dir = DEST / name / "data", DEST / name / "checkpoints"
        files = []
        for pattern in patterns:
            files += [pathlib.Path(f) for f in sorted(glob.glob(str(ROOT / pattern)))]
        if not files:
            print(f"  !! {name}: no files matched")
            continue
        if not args.check:
            data_dir.mkdir(parents=True, exist_ok=True)
            if want_ckpt:
                ck_dir.mkdir(parents=True, exist_ok=True)

        n_ck = 0
        for src in files:
            dst = data_dir / src.name
            if args.check:
                if not dst.exists():
                    missing += 1
                elif sha(dst) != sha(src):
                    print(f"  DRIFTED {name}/{src.name}"); drift += 1
            else:
                shutil.copy2(src, dst)
            total_bytes += src.stat().st_size
            if not want_ckpt or args.check:
                continue
            for suffix in CKPT_SUFFIXES:
                ck = src.with_name(src.stem + suffix)
                if ck.exists():
                    shutil.copy2(ck, ck_dir / ck.name)
                    total_bytes += ck.stat().st_size
                    n_ck += 1

        lines += [f"## `{name}/`", "", why, "",
                  f"{len(files)} result files"
                  + (f", {n_ck} checkpoints" if want_ckpt else " (engine output; no policy "
                                                              "checkpoint needed)"), ""]
        print(f"  {name:12s} {len(files):4d} files"
              + (f", {n_ck:4d} checkpoints" if want_ckpt else ""))

    if args.check:
        print(f"\n{drift} drifted, {missing} missing")
        return 1 if (drift or missing) else 0
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "MANIFEST.md").write_text("\n".join(lines))
    print(f"\ntotal {total_bytes/1e6:.0f} MB -> {DEST.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
