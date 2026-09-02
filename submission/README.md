# Submission contents

Everything the dissertation depends on, separated from the 2.1 GB working tree.

## `MANIFEST.md`

Every file group, what it supports, and how many results and checkpoints it holds. Regenerate
or verify with:

    python scripts/build_submission.py            # assemble
    python scripts/build_submission.py --check    # re-hash against the working tree

## Layout

    sweep12k/      the 12,000-episode sweep -- primary tables (RQ1)
    sweep4k/       the original 4,000-episode sweep -- training-budget appendix
    federation/    arms A and E plus coordination baselines (RQ3)
    transfer/      the answer-rate fleet (RQ2)
    attribution/   identifiability grid, matched-budget control, scaling to k=50 (RQ4)
    budget/        retrained cells and the learning-rate probe
    checkpoint/    the selected / final / argmax audit

Each group holds `data/` (result JSON) and, where a policy must be re-run rather than merely
re-read, `checkpoints/`. Checkpoints are `_best.pt` (the reported policy), `.pt` (the final
update, reported alongside it) and `_u0500.pt` (the 8,000-episode point the budget appendix
uses). Intermediate training checkpoints are excluded.

## Reproducing a number

Every structural figure in the dissertation comes from one command:

    python scripts/global_shd_paired.py <result>.json --episodes 200 --sample \
        --checkpoint best --out <out>.json

`--checkpoint` takes `best`, `final`, or an update tag such as `u0500`. `--sample` evaluates by
sampling at temperature 1, which is the convention throughout; omitting it uses argmax.

**A result file's own `global_hard_shd` field is NOT the quantity the dissertation reports.**
That field is each run's own evaluation at its final update. On a 12,000-episode run the two
differ by up to a factor of 300 on the same seed. Use the command above.

## Reproducing a figure

`notebooks/thesis_figures.ipynb` draws every figure from the raw files, one section each, with
the table that feeds each plot printed above it. `scripts/figures.py` renders the same figures
headlessly for the LaTeX build.

## Where the reasoning lives

`docs/FINDINGS_*.md` record each result and, where one was withdrawn, what refuted it.
`thesis_results/RETRACTIONS.md` collects the withdrawals in one place.
`thesis_results/CLAIMS.md` states what may be asserted, with the boundary of each claim.
