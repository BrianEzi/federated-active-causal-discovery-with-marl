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

---

# Correction and extension, 2 Sep 22:0x

Two things above are wrong. Both are corrected here rather than edited away, because the
reasoning that produced them is the part worth keeping.

## 1. The defect was never confined to one script

The section above names `scripts/global_shd_paired.py` as the site of the bug. An audit of
every script that loads a policy and rolls it out found **nineteen such scripts, of which
seventeen were unseeded**, and essentially all of them roll out with `deterministic=False`:

```
attr_score.py  budget_curve.py  checkpoint_sweep_window_rate.py  diversity_probe.py
eval_at_budget.py  greedy_fairness.py  ma_graph_examples.py  ma_iv_decomposition.py
mi_gate.py  power_window_rate.py  rescore_from_config.py  shd.py  shd_diagnose.py
trace_episode.py  transfer_behaviour.py  transfer_eval.py  vs_evaluate.py
```

So it is a repo-wide property of how sampled evaluation was written, not a slip in one file.
`global_shd_paired.py` was simply the first place it was looked for.

Seeded so far: `global_shd_paired.py`, `shd_by_pair_class.py`, `attr_score.py` — the three
that produce numbers Chapter 4 quotes. The rest are diagnostic or feed the appendix and are
listed here so that nothing is quoted from them as reproducible without checking first.

`scripts/rebuild_grid_deterministic.sh` calls `global_shd_paired.py`, so agent B's grid
rebuild is unaffected by this extension.

## 2. "Re-running would change every number by less than its uncertainty" was the wrong argument

The recommendation above was not to re-run, on the grounds that every number would move by
less than its own reported error bar. The first half of that is true and the conclusion does
not follow from it, for a reason worth stating plainly:

**Significance is a threshold, and a sub-standard-error shift can cross it.**

The $k_v=30$ cell is the case. Published, the three per-seed paired differences were
+0.00054 +/- 0.00063 (ns), -0.00040 +/- 0.00008 (SIG), -0.00026 +/- 0.00007 (SIG) — recorded
in `CLAIMS.md` C1 as "two seeds significant, one indistinguishable". Deterministic, they are
-0.00009 +/- 0.00034 (ns), -0.00040 +/- 0.00008 (SIG), -0.00005 +/- 0.00021 (ns).

Every individual number moved by less than one standard error, exactly as predicted. The
**claim** still changed: two seeds separating became one. A statement of the form "$n$ of 3
seeds" is not protected by an error bar, because it is a count of threshold crossings.

The second premise was also wrong on cost. Re-running the $k_v$ axis and the federation ladder
took about an hour, and the full 18-cell 12,000-episode fleet is a further two — not the
wholesale invalidation the section above assumed. The cheap option was available the whole
time.

## 3. What the re-run bought that is worth keeping

The re-measured axis produced the control the original audit only inferred. Across ten
measurements the myopic arm reproduced **to five decimal places at every $k_v$ and both
checkpoints** (0.00611, 0.00082, 0.00077, 0.00053, 0.00042). Because the scripted arms and the
environment are bit-identical while only the learned arm moves, the defect is isolated to the
action draw: the episode pairing, the graph sequence and the belief update were always sound.
The paired comparison was never invalid — only irreproducible.

That is a materially stronger statement than the calibration argument above, and it is the one
that should go in the appendix.

## 4. A mixed set is worse than a stale one

The fix landed at 21:15:49 while the 12,000-episode measurement fleet was running. It wrote
some outputs under the old code and some under the new, and **nothing in the numbers
distinguishes them**. A uniformly stale set announces itself the moment anything is re-run; a
mixed set survives a spot-check, because the file checked may be one of the clean ones.

All 36 such outputs are quarantined (moved to the session scratchpad, not deleted) and the
fleet relaunched under the fixed path. Any directory being written across that moment should
be treated as mixed rather than assumed stale.
