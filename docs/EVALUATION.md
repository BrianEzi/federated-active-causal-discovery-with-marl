# Evaluation

The protocol, the metrics, and the conventions a reported number depends on.

## Runs do not report their own numbers

A training run records a score from its own evaluation pass, at whatever policy it held
when it finished. That is **not** what the dissertation reports, and the two can differ by
a large factor on the same seed.

Every reported number is produced by re-scoring a saved checkpoint with
`scripts/global_shd_paired.py`, which rebuilds the environment from the run's recorded
config, replays seeded episodes, and writes per-episode vectors. Reading a run's own
`global_hard_shd` field instead of re-scoring is the single mistake this project made most
often, and it is why the re-scoring path exists.

## Paired episodes

Episodes are replayed from seeds, so two arms scored with the same seed face identical
graphs and identical observational draws. Differences between arms are therefore **paired**,
and the standard error is computed over per-episode differences rather than over
independent samples. A paired standard error is typically several times smaller than the
spread of either arm alone, which is what makes three seeds informative at all.

The per-episode vectors are stored in every result file, so any paired standard error in
the dissertation can be recomputed rather than taken on trust.

## Two checkpoint conventions

A policy can be scored at the checkpoint selected by the training criterion, or at the
final update. Both are reported everywhere, because they disagree — and on at least one
comparison they disagree in sign. Quoting one alone would be selecting a direction.

## The metrics

**Joint recovery rate.** The fraction of episodes in which every agent resolved every pair
its window required. It is a conjunction: no agent can be carried by another.

**Pooled structural distance (SHD on committed marks).** Over the pairs some window covers,
the fraction whose pooled belief — the intersection across sites — is not exactly the true
mark. Lower is better.

These two can move in opposite directions, and they did. A pair-averaged measure rewards
spreading effort thinly across many windows; a conjunctive one rewards finishing windows.
At the principal budget the myopic rule leaves *fewer* pairs undetermined than the learned
policy while winning *fewer* episodes outright. Both statements are true, and the
dissertation reports both because neither alone answers the question.

### What the distance metric actually measures

It charges a pair as an error when the pooled belief has not resolved it to the correct
mark, which folds together indecision and inaccuracy. Measured over 157,680 pooled pair
instances across four arms and three budgets, the inaccuracy half **never fires**: no pair
was ever resolved to a wrong mark, in any arm, including random.

So the reported quantity is residual ambiguity, not error. Seven alternative metrics were
tried — resolution rate, ambiguity rate, graded excess marks, a severity-weighted
composite, soft distance, error rate, disagreement rate — and all seven invert the same
way, because they are monotone transforms of one quantity, the number of surviving marks.
No reweighting reconciles a pair-averaged measure with a conjunctive criterion.

## Reading a mean

These metrics are per-episode means of a quantity that is zero on most episodes, so a mean
can rest on very few contributing episodes. A mean of 0.00022 over 600 episodes may mean
every episode is slightly wrong, or that 599 are perfect and one is a disaster. Those are
different findings.

`scripts/check_reporting.py` counts, for every arm of every reported cell, how many
episodes actually contribute a non-zero error, and flags any mean resting on five or fewer.
Where a figure draws such a point, it is marked and labelled with its count rather than
drawn as a confident value.

## The competence floor

A run is admitted only if its mean per-window recovery rate reaches 0.70 over its last ten
checkpoints. Excluded runs are reported as excluded rather than dropped silently;
`submission/` carries them.

## Baselines

The learned arm is compared against uniform random targeting, a myopic uncertainty rule
that intervenes on the variable touching the most unresolved pairs, a partitioned variant
of that rule, and an oracle that covers exactly the set each window requires. The last is a
per-window ceiling rather than a method, and it is uncoordinated across windows by design —
so a policy that allocates a shared budget well can beat it, and at tight budgets it does.
