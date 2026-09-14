# Data

What is in `results/` and `submission/`, and how provenance is kept.

## Two views of the same measurements

**`results/`** is the working tree: one JSON file per evaluated cell, holding the config the
run used, the per-arm means, and the per-episode vectors behind them. Scripts read from
here, so the analysis in this repository runs as written.

**`submission/`** is the curated subset the dissertation cites, arranged by experiment,
with the policy checkpoints needed to re-run any reported arm. `MANIFEST.md` names every
group, says what it supports, and records the file and checkpoint counts.

    .venv/bin/python scripts/build_submission.py            # assemble from results/
    .venv/bin/python scripts/build_submission.py --check    # re-hash every file

`--check` compares each shipped file against its source and reports drift or absence. It
should report zero of both.

Training checkpoints are not carried outside `submission/`. There are 5,751 in the working
tree and almost all are intermediate states no result depends on; the 542 the dissertation
cites are shipped, and the rest are on the `pre-squash-submission` tag. Checkpoints are
`_best.pt` (the reported policy), `.pt` (the final update, reported alongside it) and
`_u0500.pt` (the eight-thousand-episode point the training-budget appendix uses).

## Anatomy of a result file

Each file records enough to re-derive its own numbers:

- `config` — every field the environment was built with. `scripts/rescore_from_config.py`
  rebuilds an environment from this rather than from defaults, so a re-scoring cannot
  silently differ from the run it scores.
- `eval_*` — what the *evaluation* changed relative to the run: evidence regime, budget,
  interventional sample size, skeleton source, noise family, mechanism. A transfer
  measurement is a run evaluated under conditions it did not train in, and these fields are
  what make that legible.
- `means` — per-arm summary.
- `rows` — the per-episode vectors. Present for every arm, including where baselines were
  reused, so any paired standard error can be recomputed.
- `baseline_from` — set when the myopic and random arms were reused from another file
  rather than recomputed, naming that file.

## Why baselines are reused

For a fixed cell and seed the myopic and random arms do not depend on the trained policy,
so they replay identically. Across a sweep of training regimes they can therefore be
computed once per seed and reused, which is what `--baseline_from` does. It refuses a
mismatch rather than pairing against the wrong episodes: the stored baseline records its
cell, seed, episode count and evidence regime, and all four must agree.

One caution follows. Reuse is only valid while the *evaluation* environment is identical
across the cells being compared. Where a sweep varies the training budget per cell, the
evaluation must be pinned to one budget, or the baselines differ and the comparison moves
two things at once.

## Provenance guards

`scripts/mark_provenance.py --check` verifies that every registered data pattern resolves
to files that exist, and that no analysis reads from a directory recorded as superseded.
Several directories in the working tree hold measurements taken before a fix and are
deliberately excluded from the shipped set; the manifest says which and why.

`thesis_results/CLAIMS.md` is the only source for a reported value. It is generated from
the measurement files, and its boundary notes are hand-maintained.
`thesis_results/RETRACTIONS.md` records nineteen claims withdrawn or corrected during the
work, with what replaced them.

## Reproducing a measurement

`docs/REPRODUCING.md` walks one number end to end. In short: take the run file from
`submission/<group>/data/`, take its checkpoint from `submission/<group>/checkpoints/`, and
re-score with `scripts/global_shd_paired.py`. The seeded protocol means the numbers should
match to the digits reported.
