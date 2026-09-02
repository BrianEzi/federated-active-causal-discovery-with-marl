# The window-size crossover is also a training-budget artefact

2 Sep 2026, 15:4x. **Bears on section 4.2 of the results chapter, the central claim of RQ1.**
PARTIAL: k=4 and k=8 are measured; k=12 is measuring; k=20 and k=30 already ran at 12,000
episodes. Do not rewrite 4.2 until the axis is one design.

## The claim being tested

Section 4.2 argues that a myopic uncertainty rule is sufficient while the window is small
enough to be solved one step at a time, and that its sufficiency degrades as the window grows.
The evidence was a sign change between $k_v = 8$ and $k_v = 12$ appearing on two criteria
measured over different episodes, which was described as locating the crossover twice
independently.

That reading requires the learned policy to lose at $k_v = 4$ and $k_v = 8$. It does not lose
there. It had not finished training.

## Retrained at 12,000 episodes

Structural distance from the selected checkpoint at both budgets, 200 paired episodes per seed,
three seeds:

| $k_v$ | 4,000 ep ratio | 12,000 ep ratio | seeds favouring learned | significant |
|---|---|---|---|---|
| 4 | 1.68 | **0.45** | 3 of 3 | 1 of 3 |
| 8 | 1.39 | **0.50** | 2 of 3 | 2 of 3 |

Joint recovery rate over the same runs, both sides being each run's own final-policy
evaluation and therefore indicative rather than the chapter's numbers:

| $k_v$ | 4,000 ep gap | 12,000 ep gap |
|---|---|---|
| 4 | $-0.075$ | $+0.085$ |
| 8 | $-0.025$ | $+0.040$ |

Both criteria change sign at both cells. The crossover the chapter reports is the point at
which 4,000 episodes stops being enough, not the point at which the problem outgrows a greedy
rule.

## Strength, stated honestly

$k_v=4$ is directionally clear and statistically thin: every seed favours the learned policy,
only one of three individually. $k_v=8$ is firmer at two of three significant. The recovery-rate
rows are final-policy on both sides and should not be quoted as they stand.

## What this makes the third of

* the agent-count reversal (`FINDINGS_AGENT_COUNT_2026_09_02.md`)
* the contended-fraction reversal (same document)
* the window-size crossover (here)

Three of the four structural claims in Chapter 4 are the same artefact: a fixed episode budget
applied across cells of unequal difficulty, so that the apparent effect of the swept parameter
is partly the effect of the harder settings needing more training.

## Consequence

The thesis's claim changes shape. Not "learning pays only past a complexity threshold", but
"learning beats the myopic rule at every window size once trained, and the thresholds visible
in the sweep are an artefact of holding the episode budget constant". That is a cleaner claim
and a more defensible one, and it is also a heavier revision: it argues for promoting the
12,000-episode design to the primary tables rather than reporting it beside them.

That decision is Brian's and should be taken once the window axis is complete at one budget.
