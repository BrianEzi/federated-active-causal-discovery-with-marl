# Submission manifest

Assembled by `scripts/build_submission.py`. Every file here is cited by the
dissertation. `results/` in the working repository holds the full 2.1 GB of
working data including superseded builds; this is the subset the text depends on.

`--check` re-hashes every file against its source and reports drift.

## `sweep12k/`

The 12,000-episode sweep. Primary tables for RQ1: window size, federation size, contended fraction and budget multiplier, three seeds per cell.

51 result files, 153 checkpoints

## `sweep4k/`

The original 4,000-episode sweep. Reported beside the re-run in the training-budget appendix, never mixed into a table with it.

60 result files, 120 checkpoints

## `federation/`

RQ3. Arm A is the federated system; arm E removes the information and optimiser partitions. Coordination baselines are scored inside each run.

27 result files, 48 checkpoints

## `transfer/`

RQ2. The answer-rate fleet: seven partial-oracle rates, three seeds each, evaluated under genuine finite-sample evidence.

28 result files, 48 checkpoints

## `attribution/`

RQ4. The identifiability grid, the matched-budget control, the coverage series and the scaling runs to k=50.

9 result files (engine output; no policy checkpoint needed)

## `budget/`

The retrained cells behind the training-budget finding, and the learning-rate probe that ruled out an unstable step size.

32 result files, 68 checkpoints

## `checkpoint/`

The checkpoint audit: the same cells scored at the selected checkpoint, the final update, and under argmax.

61 result files (engine output; no policy checkpoint needed)
