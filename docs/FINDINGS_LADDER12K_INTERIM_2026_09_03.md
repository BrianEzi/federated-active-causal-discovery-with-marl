# RQ3's ladder at 12,000 episodes: interim, three seeds of six

3 Sep 2026, 01:xx. **This is a partial read and is not the result.** It is recorded because it
bears on whether the RQ3 subsection can be written at all, and because the direction it points
is worth knowing before the remaining three seeds land rather than after.

## Why the retrains exist

The six-seed federation result in Chapter 4 is measured on **4,000-episode** policies. The
completed sweep showed the learned arm ahead of the myopic rule in 2 of 18 cells at that budget
and 16 of 18 at 12,000, with 14 cells changing winner. A statement that federation costs
nothing measurable, made about two arms neither of which has converged, is not defensible on
that evidence.

Twelve retrains at 12,000 episodes are running: `results/central12k/`, arms A (federated,
`local_epochs 4`) and E (pooled, `local_epochs 0`), six seeds each.

## Interim, seeds 0 to 2, selected checkpoint

| seed | federated | centralised | myopic | federated $-$ centralised | |
|---|---|---|---|---|---|
| 0 | 0.00075 | 0.00000 | 0.00080 | $+0.00075 \pm 0.00067$ | ns |
| 1 | 0.00000 | 0.00130 | 0.00068 | $-0.00130 \pm 0.00092$ | ns |
| 2 | 0.00121 | 0.00000 | 0.00082 | $+0.00121 \pm 0.00104$ | ns |

Across the three seeds: $+0.00022 \pm 0.00077$.

## What it changes, provisionally

**The 4,000-episode result rested on one seed, and that seed's cause has disappeared.** At
4,000 episodes seed 0's centralised arm measured 0.00263 against 0.00000--0.00066 everywhere
else, and it was the only seed of six to separate. At 12,000 the same seed's centralised arm
measures **0.00000**. The outlier was an unconverged run, which is what tripling the budget
would be expected to fix.

So the conclusion looks likely to survive with better support than it had: at 4,000 episodes
"no measurable cost" was a mean over five null seeds and one significant one pointing the wrong
way for a cost; on these three seeds nothing separates at all.

**Provisional and not to be written until six seeds land.** Three seeds is the sample size that
produced two of the retractions on this project.

## An observation that needs the final checkpoint before it means anything

Both arms are numerically worse at 12,000 than at 4,000 on these seeds: federated 0.00065 mean
against 0.00014, centralised 0.00043 against 0.00092. More training producing a worse selected
checkpoint is the signature of the checkpoint tail --- the MI gate runs once per checkpoint, so
three times the updates give it three times the opportunity to retain an exploratory policy,
and `CLAIMS.md` C2 documents a case where that cost a factor of 570 on one seed.

The measurement queued for the full six seeds therefore runs **both conventions**, which the
first pass did not. If the final-update column is uniformly better than the selected one here,
the ladder is a checkpoint-selection story and not a federation story, and the subsection has
to say which.

---

## The checkpoint-tail hypothesis above is refuted on these seeds

Written twenty minutes after the section above, from the final-update measurement it asked for.

| seed | A selected | A final | E selected | E final |
|---|---|---|---|---|
| 0 | 0.00075 | 0.00132 | 0.00000 | 0.00000 |
| 1 | 0.00000 | 0.00171 | 0.00130 | 0.00130 |
| 2 | 0.00121 | 0.00121 | 0.00000 | 0.00000 |
| mean | **0.00065** | 0.00142 | 0.00043 | 0.00043 |

Selection **helps** the federated arm here, by a factor of 2.2, and is inert for the pooled one.
The opposite of what the section above predicted. So the numbers being worse at 12,000 than at
4,000 on these seeds is not a checkpoint-selection artefact and needs another explanation, or
none: three seeds inside their own noise do not require one.

The gate's firing pattern says the same thing. It selected an earlier update on 2 of 4
federated runs and 1 of 4 pooled runs, and where it fired the two conventions still measure
identically on the pooled arm. Firing and mattering are separate events.

**Both conventions still go in the table**, for the reason `FINDINGS_CHECKPOINT_TAIL` gave and
not for the reason predicted here: they disagree unpredictably per cell, so reporting one is
choosing a convention that fails somewhere without knowing where.

One observation held at arm's length because four runs cannot carry it: `best_mi_ratio` is
higher for every pooled run than for any federated run (0.74--0.85 against 0.67--0.74). Pooled
optimisation reaching a higher training-health score is what one would expect and it is not
evidence of anything at this sample size.
