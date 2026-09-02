# MI checkpoint selection has a tail risk on long runs, and it is not an evaluation artefact

2 Sep 2026, 16:2x. **Bears on `sec:meth_ckpt` and on every 12,000-episode number.**
PARTIAL: two of seven cells audited.

## The problem

Chapter 4 reports the checkpoint maximising the mutual information between belief and action
over training rollouts. At 12,000 episodes that rule occasionally selects a much worse policy
than the final update.

`k12s50n02b150`, three seeds, 200 paired episodes, sampled evaluation:

| seed | selected checkpoint | final checkpoint |
|---|---|---|
| 0 | 0.00197 | 0.00312 |
| 1 | 0.00009 | 0.00013 |
| 2 | **0.07038** | **0.00000** |

Two seeds behave as intended, with the selected checkpoint better than the final. The third is
catastrophic and carries the cell mean to 0.02415 against the myopic rule's 0.00157, turning a
cell the learned policy wins into one it loses by a factor of fifteen.

**This is a tail risk, not a bias.** The rule is usually right and occasionally very wrong,
which is harder to caveat than a systematic offset.

## It is not the evaluation convention

The obvious benign explanation was that MI selects high-entropy policies, which sampling would
punish and argmax would forgive. That predicts the anomalous seed evaluating well under argmax.
It does the opposite.

| cell, seed | sampled | argmax | resolved (sampled / argmax) |
|---|---|---|---|
| k12s50n02b150 s2 | 0.07038 | **0.15808** | 0.867 / 0.779 |
| k12s75n04b150 s2 | 0.00083 | **0.00359** | 0.925 / 0.922 |

Argmax is worse, by a factor of two on the affected seed. Committing fully to the mode of a bad
policy is worse than sampling around it. So the selected checkpoint is a genuinely poor policy
and the fix is not a change of convention.

**A useful secondary observation.** On the seeds that behave, argmax and sampling agree closely
(0.00197 against 0.00184; 0.00009 against 0.00000). The choice of action selection is nearly
free for a good policy and expensive for a bad one, so an argmax/sampling gap is itself a
symptom of policy quality rather than a measurement decision.

## Why 12,000 episodes and not 4,000

Three times the updates gives three times as many checkpoints for the criterion to choose badly
from. Mutual information between belief and action is high for a policy still spreading
probability across actions, not only for one that has learned to choose well, so a long run
offers more opportunities to select an exploratory checkpoint.

## What this does and does not touch

**Touches.** Every 12,000-episode structural number, which is the entire re-run and the
retrained cells behind the training-budget findings. The direction is not fixed: at K=2 the
selected checkpoint is pessimistic, so the finding it feeds would be understated rather than
overstated, but that has to be checked per cell rather than assumed.

**Does not touch.** Window rate and joint recovery, which are recorded from training and do not
involve checkpoint choice. The undertraining result rests on those and stands. Attribution is
deterministic. Agent B's transfer work uses its own pipeline.

## What would fix it

Reporting both checkpoints per cell, which the chapter already does at the window axis, and
stating the tail explicitly. A selection rule that occasionally picks a bad policy is
defensible when both are shown; it is not defensible when only the selected one is quoted.
Five more cells are auditing.
