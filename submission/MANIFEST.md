# Submission manifest

Assembled by `scripts/build_submission.py`. Every file here is cited by the
dissertation. `results/` in the working repository holds the full 2.1 GB of
working data including superseded builds; this is the subset the text depends on.

`--check` re-hashes every file against its source and reports drift.

## `sweep12k/`

The 12,000-episode sweep. Primary tables for RQ1: window size, federation size, contended fraction and budget multiplier, three seeds per cell.

54 result files, 162 checkpoints

## `sweep4k/`

The original 4,000-episode sweep. Reported beside the re-run in the training-budget appendix, never mixed into a table with it.

60 result files, 120 checkpoints

## `federation/`

RQ3 at 12,000 episodes. Arm A is the federated system; arm E removes the information and optimiser partitions. Coordination baselines are scored inside each run. The k=12 arms are the 12,000-episode retrains: the 4,000-episode originals in `results/central/` are NOT shipped, because their one significant seed was an unconverged centralised run measuring 0.00263 which measures 0.00000 once trained. k=20 comes from `results/central/` because those six runs were always at 12,000.

22 result files, 54 checkpoints

## `transfer/`

RQ2. The answer-rate fleet: seven partial-oracle rates, three seeds each, evaluated under genuine finite-sample evidence. The per-cell paired evaluations come from `deterministic/`, which carries the per-episode rows, so the 15/15 count and every paired standard error can be recomputed rather than taken on trust. The pre-fix copies in `results/power/rho/xfer_*.json` are deliberately NOT shipped: they were scored before the evaluation RNG was seeded and do not reproduce. `rho0.95_long_s?` is the doubled-training arm for the rho=0.95 pivot and is listed separately from the seven-rate fleet so a reader counting training runs gets 21 for a 21-cell grid, not 24.

69 result files, 48 checkpoints

## `generator/`

The generator control: the advantage is not a scale-free artefact. Three ER seeds at the principal cell, both conventions identical; the myopic rule is the arm the family change breaks.

5 result files, 9 checkpoints

## `credit/`

Turn-aware credit under pooled and federated optimisation, measured. The recorded-field interaction (18x, federation-only) does not exist: 15.1x pooled against 13.2x federated.

32 result files, 48 checkpoints

## `inregime/`

The answer-rate grid's second reading: each policy measured in its own regime (21/21), plus the rebuilt fixed-policy sweep and the finite-sample cell. All seeded-path measurements.

55 result files, 3 checkpoints

## `attribution/`

RQ4. The identifiability grid, the matched-budget control, the coverage series and the scaling runs to k=50.

9 result files (engine output; no policy checkpoint needed)

## `budget/`

The retrained cells behind the training-budget finding, and the learning-rate probe that ruled out an unstable step size.

32 result files, 68 checkpoints

## `nint_curve/`

The sample-size axis: k=8 12,000-episode policies under sampled evidence, n_int swept 10 to 10,000, three arms re-scored per value.

21 result files (engine output; no policy checkpoint needed)

## `checkpoint/`

The checkpoint audit: the same cells scored at the selected checkpoint, the final update, and under argmax. The window-axis measurements are `results/rerows/`; `results/ckpt/` holds the same cells scored before the evaluation RNG was seeded and is NOT shipped.

71 result files (engine output; no policy checkpoint needed)
