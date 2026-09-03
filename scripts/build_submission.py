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
    ("federation", "RQ3 at 12,000 episodes. Arm A is the federated system; arm E removes the "
                   "information and optimiser partitions. Coordination baselines are scored "
                   "inside each run. The k=12 arms are the 12,000-episode retrains: the "
                   "4,000-episode originals in `results/central/` are NOT shipped, because "
                   "their one significant seed was an unconverged centralised run measuring "
                   "0.00263 which measures 0.00000 once trained. k=20 comes from "
                   "`results/central/` because those six runs were always at 12,000.",
     ["results/central12k/v2_k12_?_s?.json", "results/central/v2_k20_?_s?.json",
      "results/rerows/ladder12k_?_best.json", "results/rerows/ladder12k_?_final.json"], True),
    ("transfer", "RQ2. The answer-rate fleet: seven partial-oracle rates, three seeds each, "
                 "evaluated under genuine finite-sample evidence. The per-cell paired "
                 "evaluations come from `deterministic/`, which carries the per-episode rows, "
                 "so the 15/15 count and every paired standard error can be recomputed rather "
                 "than taken on trust. The pre-fix copies in `results/power/rho/xfer_*.json` "
                 "are deliberately NOT shipped: they were scored before the evaluation RNG was "
                 "seeded and do not reproduce. `rho0.95_long_s?` is the doubled-training arm "
                 "for the rho=0.95 pivot and is listed separately from the seven-rate fleet so "
                 "a reader counting training runs gets 21 for a 21-cell grid, not 24.",
     ["results/power/rho/rho[01].[0-9][0-9]_s?.json", "results/power/rho/CURVE.json",
      "results/power/rho/deterministic/xfer_rho*_s?.json",
      "results/power/rho/DETERMINISTIC_COMPARE.json",
      "results/power/rho/rho0.95_long_s?.json"], True),
    ("generator", "The generator control: the advantage is not a scale-free artefact. Three "
                  "ER seeds at the principal cell, both conventions identical; the myopic rule "
                  "is the arm the family change breaks.",
     ["results/generator12k/er_s?.json", "results/generator12k/shd_er_*.json"], True),
    ("credit", "Turn-aware credit under pooled and federated optimisation, measured. The "
               "recorded-field interaction (18x, federation-only) does not exist: 15.1x pooled "
               "against 13.2x federated.",
     ["results/credit/k*_s?.json", "results/credit/shd/*.json"], True),
    ("inregime", "The answer-rate grid's second reading: each policy measured in its own "
                 "regime (21/21), plus the rebuilt fixed-policy sweep and the finite-sample "
                 "cell. All seeded-path measurements.",
     ["results/power/rho/inregime_det/rho*_s?.json",
      "results/power/rho/evalsweep_det/fixed_rho*_s?_evalp*.json",
      "results/sampled_det/nint200.json",
      "results/sampled_ref/k08s50n04b150i0200_s?.json"], True),
    ("attribution", "RQ4. The identifiability grid, the matched-budget control, the coverage "
                    "series and the scaling runs to k=50.",
     ["results/attr_ceiling*.json", "results/attr_scale_final.json", "results/attr_reach.json",
      "results/attr/transfer_k12s50n04b200_s?.json"], False),
    ("budget", "The retrained cells behind the training-budget finding, and the learning-rate "
               "probe that ruled out an unstable step size.",
     ["results/longcheck/*_long_s?.json", "results/longcheck/*_conv_s?.json",
      "results/longcheck/shd_*.json", "results/lrcheck/*.json"], True),
    ("checkpoint", "The checkpoint audit: the same cells scored at the selected checkpoint, "
                   "the final update, and under argmax. The window-axis measurements are "
                   "`results/rerows/`; `results/ckpt/` holds the same cells scored before the "
                   "evaluation RNG was seeded and is NOT shipped.",
     ["results/rerows/k??_best.json", "results/rerows/k??_final.json",
      "results/sweep12k/shd/*.json",
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
            found = sorted(glob.glob(str(ROOT / pattern)))
            # PER-PATTERN, not per-group. A dead pattern inside a group whose other patterns
            # match was previously silent: `results/power/confirm/*.json` sat in the transfer
            # group matching nothing at all, and the group reported success because four other
            # patterns did match. A registry that quietly ships less than it claims is worse
            # than one that fails loudly.
            if not found:
                print(f"  !! {name}: pattern matched nothing -- {pattern}")
            files += [pathlib.Path(f) for f in found]
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
