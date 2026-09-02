# The competence-floor exclusions are undertrained runs, not broken ones

2 Sep 2026, 07:4x. **Six of seven confirmed, one in flight** (k12s25n08b150, the eight-agent
cell, which is also the one where seed 3 fails). The six-cell claim below is complete as
stated; the seventh may not follow it.

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

| cell, seed 2 | 4,000 ep: wr / learned | 12,000 ep: wr / learned | greedy |
|---|---|---|---|
| k12s25n02b150 | 0.347 / 0.130 | **0.997 / 0.990** | 0.825 |
| k12s25n04b150 | 0.659 / 0.485 | **1.000 / 0.995** | 0.900 |
| k12s50n02b150 | 0.519 / 0.240 | **0.997 / 1.000** | 0.900 |
| k12s50n03b150 | 0.646 / 0.500 | **0.994 / 0.965** | 0.950 |
| k12s50n04b100 | 0.345 / 0.150 | **0.970 / 0.995** | 0.800 |
| k12s50n04b120 | 0.552 / 0.540 | **0.950 / 0.970** | 0.870 |

**Six of six clear the floor, at window rates of 0.950 to 1.000, and six of six finish above
the myopic rule on their own cell.** Joint recovery rises from a 0.130-0.540 range to a
0.965-1.000 range. These are not marginal passes; several are among the best figures anywhere
in the sweep.

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
