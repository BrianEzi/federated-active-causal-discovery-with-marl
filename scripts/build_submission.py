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
      "results/power/rho/rho0.95_long_s?.json",
      # The argmax derivative of the whole grid: 21 cells, the convention half of C6's
      # boundary. CURVE_ARGMAX.json is its curve summary. Added 3 Sep once the grid landed.
      "results/power/rho/argmax_det/argmax_rho*_s?.json",
      "results/power/rho/CURVE_ARGMAX.json",
      # The measured in-regime diagonal ships with the inregime group; listed here as well
      # would double-copy, so it is not.
      ], True),
    ("generator", "The generator control: the advantage is not a scale-free artefact. Three "
                  "ER seeds at the principal cell, both conventions identical; the myopic rule "
                  "is the arm the family change breaks.",
     ["results/generator12k/er_s?.json", "results/generator12k/shd_er_*.json"], True),
    ("skeleton", "What the supplied skeleton assumption is worth, measured at the principal "
                 "cell (Appendix, app:skeleton). Three sweeps: the achievable ceiling against "
                 "sample size and CI level; the achieved-against-ceiling saturation test at "
                 "three sample sizes and two budgets; and the budget-limited alpha test. The "
                 "older `results/skeleton_ablation.json` is the 3-agent window-6 probe it "
                 "supersedes and is shipped beside it for provenance, NOT as the reported "
                 "measurement.",
     ["results/skeleton_sweep_principal.json", "results/skeleton_alpha_n60.json",
      "results/skeleton_ablation.json", "results/skel_alpha/a*_b*.json",
      "results/skel_saturate/n*_b*.json"], False),
    ("generator_probe", "Why the myopic rule collapses on Erdos-Renyi: its own decision "
                        "statistic, the undetermined marks touched, at each agent's first "
                        "decision on both families under identical episode seeds.",
     ["results/generator_decision_probe.json"], False),
    ("ladder_b070", "The federation ladder at beta=0.7, the constrained cell where both arms "
                    "are competent and the equivalence bound tightens to 44% and 23% of its "
                    "margin against 76% and 89% at beta=1.5. Six seeds per arm.",
     ["results/ladder_b070/b070_?_s?.json", "results/ladder_b070/scored/*.json"], True),
    ("credit", "Turn-aware credit under pooled and federated optimisation at k=8, RETRAINED "
               "to 12,000 episodes. At the converged budget the pooled arm is flat (1.1x) and "
               "the federated arm degrades 6.1x, so an interaction does exist; the 4,000-"
               "episode runs, which showed 15.1x and 13.2x and supported no ordering, ship "
               "beside them because the difference between the two budgets IS the finding. "
               "k=12 was not extended: ~70 hours against 3.3 for k=8, and its pooled cells sat "
               "on the measurement floor in both credit states.",
     ["results/credit12k/k*_s?.json", "results/credit12k/shd/*.json"], True),
    ("credit4k", "The SUPERSEDED 4,000-episode credit ablation, shipped because the "
                 "difference between the two budgets is itself the finding: at 4,000 both "
                 "optimisers degraded about equally (15.1x pooled, 13.2x federated) and no "
                 "interaction was supported. It is NOT the reported measurement. Kept in its "
                 "own group because its filenames are identical to the 12,000-episode runs "
                 "and flattening both into one folder silently overwrote them.",
     ["results/credit/k*_s?.json", "results/credit/shd/*.json"], False),
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
    ("budget_tight", "The constrained-budget axis, beta 0.5-0.9 at k=12, three seeds each.",
     ["results/budget_tight/*.json"], False),
    ("noisedist", "Noise-shape and mechanism robustness of the rho=0.5 policies.",
     ["results/noisedist/*.json"], False),
    ("epsgreedy", "The epsilon-greedy control at k=12 and k=30: is the learned policy "
                  "dithered greedy. Grid eps 0.05-0.3, 200 paired episodes per seed.",
     ["results/epsgreedy/*.json", "results/epsgreedy/sweep/w*.json",
      "results/epsgreedy/k12policy/*.json", "results/epsgreedy/k30policy/*.json"], False),
    ("nint_curve", "The sample-size axis: k=8 12,000-episode policies under sampled "
                   "evidence, n_int swept 10 to 10,000, three arms re-scored per value.",
     ["results/nint_curve/*.json", "results/nint_disclose/*.json"], False),
    ("checkpoint", "The checkpoint audit: the same cells scored at the selected checkpoint, "
                   "the final update, and under argmax. The window-axis measurements are "
                   "`results/rerows/`; `results/ckpt/` holds the same cells scored before the "
                   "evaluation RNG was seeded and is NOT shipped.",
     ["results/rerows/k??_best.json", "results/rerows/k??_final.json",
      "results/rerows/k20_u0249.json", "results/rerows/k30_u0249.json",
      "results/rerows/k20_u0249_recovery.json", "results/rerows/k30_u0249_recovery.json",
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

        # NAME COLLISIONS ACROSS SOURCE DIRECTORIES. A group whose patterns span several
        # directories can match the same basename twice -- `shd/`, `shd_final/` and
        # `shd_argmax/` all hold `k12s50n04b150.json` -- and a flat copy silently lands one
        # on top of the other. The checkpoint group, whose whole purpose is to compare those
        # three conventions, was shipping one file where it claimed three. Where a basename
        # is not unique within a group, the source directory is prefixed so every file
        # survives and its provenance is legible from its name.
        seen = {}
        for src in files:
            seen[src.name] = seen.get(src.name, 0) + 1
        collide = {n for n, c in seen.items() if c > 1}
        if collide and not args.check:
            print(f"     ({len(collide)} basename(s) appear in more than one source "
                  f"directory; prefixing those with their directory)")

        n_ck = 0
        for src in files:
            dst = data_dir / (f"{src.parent.name}__{src.name}" if src.name in collide
                              else src.name)
            if args.check:
                if not dst.exists():
                    missing += 1
                elif sha(dst) != sha(src):
                    print(f"  DRIFTED {name}/{dst.name}"); drift += 1
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
