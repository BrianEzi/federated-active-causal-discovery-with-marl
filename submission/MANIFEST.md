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

## `skeleton/`

What the supplied skeleton assumption is worth, measured at the principal cell (Appendix, app:skeleton). Three sweeps: the achievable ceiling against sample size and CI level; the achieved-against-ceiling saturation test at three sample sizes and two budgets; and the budget-limited alpha test. The older `results/skeleton_ablation.json` is the 3-agent window-6 probe it supersedes and is shipped beside it for provenance, NOT as the reported measurement.

19 result files (engine output; no policy checkpoint needed)

## `generator_probe/`

Why the myopic rule collapses on Erdos-Renyi: its own decision statistic, the undetermined marks touched, at each agent's first decision on both families under identical episode seeds.

1 result files (engine output; no policy checkpoint needed)

## `ladder_b070/`

The federation ladder at beta=0.7, the constrained cell where both arms are competent and the equivalence bound tightens to 44% and 23% of its margin against 76% and 89% at beta=1.5. Six seeds per arm.

16 result files, 36 checkpoints

## `credit/`

Turn-aware credit under pooled and federated optimisation at k=8, RETRAINED to 12,000 episodes. At the converged budget the pooled arm is flat (1.1x) and the federated arm degrades 6.1x, so an interaction does exist; the 4,000-episode runs, which showed 15.1x and 13.2x and supported no ordering, ship beside them because the difference between the two budgets IS the finding. k=12 was not extended: ~70 hours against 3.3 for k=8, and its pooled cells sat on the measurement floor in both credit states.

16 result files, 36 checkpoints

## `credit4k/`

The SUPERSEDED 4,000-episode credit ablation, shipped because the difference between the two budgets is itself the finding: at 4,000 both optimisers degraded about equally (15.1x pooled, 13.2x federated) and no interaction was supported. It is NOT the reported measurement. Kept in its own group because its filenames are identical to the 12,000-episode runs and flattening both into one folder silently overwrote them.

32 result files (engine output; no policy checkpoint needed)

## `inregime/`

The answer-rate grid's second reading: each policy measured in its own regime (21/21), plus the rebuilt fixed-policy sweep and the finite-sample cell. All seeded-path measurements.

55 result files, 3 checkpoints

## `attribution/`

RQ4. The identifiability grid, the matched-budget control, the coverage series and the scaling runs to k=50.

9 result files (engine output; no policy checkpoint needed)

## `budget/`

The retrained cells behind the training-budget finding, and the learning-rate probe that ruled out an unstable step size.

32 result files, 68 checkpoints

## `budget_tight/`

The constrained-budget axis, beta 0.5-0.9 at k=12, three seeds each.

20 result files (engine output; no policy checkpoint needed)

## `noisedist/`

Noise-shape and mechanism robustness of the rho=0.5 policies.

15 result files (engine output; no policy checkpoint needed)

## `epsgreedy/`

The epsilon-greedy control at k=12 and k=30: is the learned policy dithered greedy. Grid eps 0.05-0.3, 200 paired episodes per seed.

24 result files (engine output; no policy checkpoint needed)

## `nint_curve/`

The sample-size axis: k=8 12,000-episode policies under sampled evidence, n_int swept 10 to 10,000, three arms re-scored per value.

44 result files (engine output; no policy checkpoint needed)

## `checkpoint/`

The checkpoint audit: the same cells scored at the selected checkpoint, the final update, and under argmax. The window-axis measurements are `results/rerows/`; `results/ckpt/` holds the same cells scored before the evaluation RNG was seeded and is NOT shipped.

71 result files (engine output; no policy checkpoint needed)
