# The agent-count reversal does not survive adequate training

2 Sep 2026, 10:5x. **DO NOT QUOTE THE CONVERGED RATIOS. A confound was found at 10:4x and the
numbers below are being re-measured.**

The converged column uses `global_hard_shd` as recorded by each run's own evaluation pass,
which scores the FINAL policy. `FINDINGS_CHECKPOINT_2026_09_01.md` established that the final
policy degrades badly on long runs -- worth a factor of 2.3 at $k_v=20$ and 16 at $k_v=30$. So
comparing a 4,000-episode final policy against a 12,000-episode final policy confounds the
training budget with that degradation, and the direction of the confound is not fixed.

The first uniform cell to complete makes this concrete. At $K=5$, all three seeds retrained:

| seed | SHD at 4,000 (final) | at 12,000 (final) |
|---|---|---|
| 0 | 0.00009 | 0.00004 |
| 1 | 0.00006 | **0.01841** |
| 2 | 0.00131 | 0.00004 |

Seed 1 is 300x worse after three times the training, while its window rate stays at 0.957 and
its joint recovery at 0.895. That is late-training degradation of the final policy, not a
failure to learn. Two seeds improve by factors of 2 and 33; one collapses. A cell mean over
that is meaningless.

## Re-measured properly at $K=5$, 11:1x: the reversal is a training-budget artefact

Both budgets scored from the **selected** checkpoint, 200 paired episodes per seed, so the only
thing differing between the rows is how long the policy trained.

| budget | learned | myopic | ratio | per seed | significant |
|---|---|---|---|---|---|
| 4,000 episodes | 0.00057 | 0.00030 | **1.92** | 0.00019, 0.00006, 0.00146 | 1 of 3 |
| 12,000 episodes | 0.00002 | 0.00030 | **0.06** | 0.00002, 0.00004, 0.00000 | 2 of 3 |

At five agents, training three times as long moves the learned policy from roughly twice the
myopic rule's structural error to **one sixteenth of it**. The myopic column is identical by
construction, since the baseline does not train.

This is the same conclusion the confounded comparison reached, arrived at without the
confound. Seed 1, which read 0.01841 at 12,000 episodes on the final policy, reads 0.00004 on
the selected one -- a factor of 460, and a direct confirmation that the earlier anomaly was
late-training degradation rather than a failure to learn.

$K = 8$ and $K = 10$ have uniform-budget runs training now and will be measured the same way.
Until they land, the $K=5$ row is the only one quotable.

## The claim being tested

The sweep at 4,000 episodes shows the learned policy leading the myopic rule to four agents and
trailing it from five, with the learned-to-myopic ratio of structural distance rising to 6.75 at
ten agents. That was read as coordination load: more agents contending for the same variables
degrades the learned policy while the myopic rule is unaffected.

## Two of the runs carrying it had not converged

The competence floor of 0.70 admits runs that are far from finished. Both high-agent-count
cells have a seed that cleared it and was nowhere near converged:

| run | window rate at 4,000 ep | learned SHD at 4,000 | at 12,000 | factor |
|---|---|---|---|---|
| k12s50n08b150 s2 | 0.838 (PASSED) | 0.00290 | **0.00005** | 58x |
| k12s50n10b150 s2 | 0.804 (PASSED) | 0.00220 | **0.00001** | 220x |

Joint recovery on the same runs moves from 0.635 to 0.990 and from 0.610 to 0.995.

## The reversal disappears

Learned-to-myopic ratio of structural distance, below 1 meaning the learned policy is better:

| $K$ | as run | excluding seed 2 | with seed 2 converged |
|---|---|---|---|
| 2 | 0.12 | 0.12 | -- |
| 3 | 0.33 | 0.33 | -- |
| 4 | 0.10 | 0.12 | -- |
| 5 | 1.65 | 0.25 | pending |
| 8 | 4.24 | 1.82 | **0.89** |
| 10 | 6.75 | 2.17 | **1.00** |

At eight agents the learned policy becomes better than the myopic rule; at ten it ties. Nothing
in the axis then shows the learned policy losing at any agent count.

## What the honest claim becomes

Not a statement about achievable accuracy. A statement about **sample efficiency under
contention**: at a fixed budget of 4,000 episodes the learned policy degrades as agents are
added, and the degradation does not survive training to convergence. Adding agents makes the
problem take longer to learn rather than making it less learnable.

That is narrower than the original claim and better supported. It also removes the strongest
apparent limitation of the method, which is worth saying plainly rather than burying: the
result the chapter was going to report as the honest boundary of the contribution was an
artefact of how long we trained.

## The methodological point, which is the transferable one

**A run passing a competence threshold is not evidence that it converged.** A window rate of
0.838 cleared our floor while sitting 58x from its converged structural error. Any gate defined
on a saturating quantity has this failure mode, and a fixed episode budget across cells of
different difficulty guarantees that the hardest cells are the ones that hit it.

## Still open

* Uniform-budget cells at $K = 5, 8, 10$, all seeds at 12,000 episodes, are training. Until
  they land the converged column mixes budgets within a cell.
* Whether the contended-fraction reversal at $\sigma = 0.75$ has the same cause. Its seed 2
  sits at window rate 0.758 with joint recovery 0.660, which is the same signature.
