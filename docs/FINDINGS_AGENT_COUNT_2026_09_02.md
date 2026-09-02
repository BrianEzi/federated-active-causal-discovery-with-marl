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

**Both budgets are being re-measured from the selected checkpoint**, which is what the chapter
reports everywhere else. Until that lands, the only safe statements are the per-seed
window-rate and joint-recovery figures, which do not depend on the checkpoint convention.

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
