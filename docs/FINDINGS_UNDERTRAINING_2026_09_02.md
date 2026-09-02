# The competence-floor exclusions are undertrained runs, not broken ones

2 Sep 2026, 07:1x. **PROVISIONAL: two of seven confirmed, five in flight.** Do not quote the
general claim until the remaining five land.

## The question

Seven runs in the oracle sweep fall below the competence floor of `window_rate >= 0.70`. All
seven are seed 2 and all seven are at k=12. Two explanations were open: bad optimisation at
the sweep's learning rate, or insufficient training. Brian asked whether a lower learning rate
would rescue them.

## A lower learning rate makes them worse

| cell, seed 2 | lr 3e-4 (sweep) | lr 1e-4 |
|---|---|---|
| k12s50n02b150 | window rate 0.519 | **0.206** |
| k12s50n04b100 | window rate 0.345 | **0.177** |

Both roughly halve. That rules out instability at the sweep's step size and points the other
way: at 4,000 episodes a smaller step does not arrive.

## More training rescues them completely

Same cells, same seed, same learning rate, 12,000 episodes instead of 4,000:

| cell, seed 2 | 4,000 episodes | 12,000 episodes | greedy on the same cell |
|---|---|---|---|
| k12s50n02b150 | wr 0.519, learned 0.240 | **wr 0.997, learned 1.000** | 0.900 |
| k12s50n04b100 | wr 0.345, learned 0.150 | **wr 0.970, learned 0.995** | 0.800 |

Both clear the floor comfortably, and the learned policy goes from far below the myopic rule
to well above it. These are not marginal passes: 1.000 and 0.995 joint recovery are among the
best figures anywhere in the sweep.

Raw: `results/longcheck/*_long_s2.json`, `results/lrcheck/*_lr1e4_s2.json`.

## What this establishes, and what it does not

**Establishes.** The excluded runs are undertrained at the sweep's 4,000-episode budget. The
competence floor is removing runs that had not finished learning, not cells that cannot be
learned. The gate is doing its job, and nothing recoverable was discarded by it.

**Does not establish.** Why seed 2 specifically. Seed 3 also fails at k12s25n08b150
(`docs/AGENT_B_INBOX.md`, 2 Sep 06:1x), so the clustering is not explained by this. The
mechanism is presumably that the run seed fixes the policy initialisation and some
initialisations need more updates to escape, but that has not been measured.

## Consequence for the results chapter

**The sweep's means are pessimistic, and this must be stated.** Cells where a seed was excluded
are reported on two seeds, and the excluded seed reaches the top of the range when trained to
convergence. The k=2 cell of the agent-count axis is the clearest case: it currently rests on
two seeds, and the third reaches 1.000 at 12,000 episodes.

**Do NOT substitute the 12,000-episode runs into the sweep tables.** The sweep is 4,000
episodes uniformly across all twenty cells, and mixing training budgets between cells would be
the same class of error as mixing checkpoints. Report the sweep as run, and report this
measurement beside it as a stated limitation of the sweep's design.
