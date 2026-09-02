# Sampled evaluation was not reproducible, and the audit that found it also calibrates the error bars

2 Sep 2026, 21:2x.

## What was wrong

`scripts/global_shd_paired.py` fixed the environment seed for every episode, so all arms saw
identical worlds and the paired comparison was sound. It did **not** seed the global torch
generator, from which a learned policy draws its actions under `--sample`. Re-running the same
checkpoint at the same seed for the same number of episodes therefore returned different
numbers.

Found while re-running fourteen stored outputs to add the per-episode rows they were missing.
The re-run was intended as a formality.

## The measurement

Twenty-four arm-level comparisons, old against new:

* **The greedy and random arms reproduced exactly**, to the last digit, in every case. They draw
  from their own seeded generators.
* **The learned arm differed in 18 of 24**, by 0.10 to 2.22 of the reported paired standard
  error, median about 0.4.

| cell, seed | old | re-run | difference in SE |
|---|---|---|---|
| k04 best, seed 0 | 0.010238 | 0.014762 | 2.22 |
| k08 best, seed 2 | 0.002340 | 0.001862 | 0.87 |
| k12 best, seed 1 | 0.000068 | 0.000000 | 0.38 |
| k12 best, seed 0 | 0.000137 | 0.000114 | 0.10 |

## Two conclusions, and they point opposite ways

**The published intervals are honest.** Every re-run landed inside roughly two standard errors
of the original, with a median of 0.4. That is what a correctly sized error bar looks like, and
it is a stronger check on the reported uncertainty than anything else in this project: the
paired standard error was computed within a run, and the re-runs test it across runs. It passes.

**The numbers were not reproducible, which is separate and not acceptable.** A result that ships
with its checkpoints invites re-running, and a reader who re-ran would have found different
values with no explanation available.

## The fix

`play()` now calls `torch.manual_seed(seed)` from the same seed that fixes the episode
sequence, making an evaluation a pure function of checkpoint, seed, episode count and
convention. Verified: two consecutive runs of the same cell now return
`0.00106383` and `0.00106383`.

## Consequence for the text

Results produced before this change differ from a re-run by roughly one standard error. Rather
than re-run every number the day before freeze, the honest treatment is:

* state that sampled evaluation is stochastic and that the reported paired standard error covers
  run-to-run variation, with the 24-comparison audit as the evidence;
* note that the generator is seeded from the current commit onward, so anything produced after
  it reproduces exactly;
* keep the audit in the appendix, because it doubles as a calibration check on every interval in
  the thesis.

Re-running everything would change every number by less than the uncertainty already reported on
it, at the cost of invalidating the tables, the figures and the claims file the day before
submission. That trade is not worth taking.
