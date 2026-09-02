# The window axis is confounded with training budget, and Chapter 4 states the opposite

2 Sep 2026, 21:5x. Agent C. **Bears on `tab:axis_k`, `tab:checkpoint`, and the central claim
of RQ1.** Verified from run configs and update counts, not from a summary table.

## The measurement

`results/sweep/oracle/` holds 60 runs over 20 cells. They were not trained at one budget.

| `train_episodes` | runs | cells |
|---|---|---|
| 4,000 | 54 | the 18 cells at $k_v \in \{4, 8, 12\}$ |
| 12,000 | 6 | `k20s50n04b150`, `k30s50n04b150` |

Confirmed three ways: `config.train_episodes`, `len(history)` (250 updates against 750, at 16
episodes per update), and file mtime (the $k_v=12$ cells written 31 Aug 20:48, the $k_v=20$
cells 1 Sep 19:59). **No $k_v=20$ or $k_v=30$ run at 4,000 episodes exists anywhere in
`results/`.** The window axis has never been measured at a uniform budget.

## How it happened, and it was not an accident

`docs/AGENT_B_INBOX.md`, 1 Sep 18:50: agent B copied `results/sweep/oracle_long/`'s six
12,000-episode files over the 4,000-episode $k_v=20$ and $k_v=30$ files in
`results/sweep/oracle/`, to replace 1--2 seed coverage with 3. That was the right call for
seed coverage and it was announced. What did not follow was propagating it into the chapter.

## What Chapter 4 currently says

`sec:res_sweep`, final bullet:

> Not substituted into any sweep table, which holds training budget fixed at 4,000 episodes.

False as written. Two of the twenty cells are at 12,000, and they are the two that carry the
headline.

## Why it matters more than a corrigendum

The two variables are **perfectly collinear** on the window axis:

| $k_v$ | 4 | 8 | 12 | 20 | 30 |
|---|---|---|---|---|---|
| budget | 4k | 4k | 4k | **12k** | **12k** |
| learned leads? | no | no | yes | yes | yes |

Every cell where the learned policy leads on the window axis either sits at 12,000 episodes or
sits immediately adjacent to the ones that do. Three consequences.

1. **The monotone widening cannot be attributed to window size.** The recovery-rate gap
   $-0.075, -0.025, +0.058, +0.083, +0.125$ has its two most favourable points at three times
   the training of the three least favourable. As it stands the axis measures window size and
   budget together.

2. **The sign change itself survives, and is the part worth keeping.** $k_v=8$ and $k_v=12$ are
   both at 4,000 episodes, so the crossover between them is a clean within-budget comparison.
   Agent B's deterministic re-measurement firms it up rather than moving it: significantly
   worse on 2 of 3 seeds at $k_v=4$, mixed at $k_v=8$, significantly better on 3 of 3 at
   $k_v=12$ and $k_v=20$ (`AGENT_B_INBOX.md`, 2 Sep 21:3x).

3. **`tab:checkpoint` is confounded the same way, and this is the cleaner reading.** The table
   is offered as "checkpoint choice is inert below $k_v=12$ and decisive above it". Its inert
   rows are exactly the 4,000-episode cells and its decisive rows are exactly the
   12,000-episode cells. `FINDINGS_CHECKPOINT_TAIL_2026_09_02.md` independently finds that the
   MI selection rule misbehaves *specifically* at 12,000 episodes, because three times the
   updates give the criterion three times the chances to pick an exploratory checkpoint. The
   parsimonious explanation of `tab:checkpoint` is therefore training budget, not window size,
   and the checkpoint-tail document already supplies the mechanism.

## Interaction with the crossover finding

`FINDINGS_CROSSOVER_2026_09_02.md` argues the crossover is itself a budget artefact, because
retraining $k_v=4$ and $k_v=8$ to 12,000 episodes flips both. If that holds, the axis at a
uniform 12,000 episodes has the learned policy leading everywhere and no crossover at all.
Those retrains were measured on the pre-fix RNG path, quarantined at 2 Sep 21:15, and are
re-measuring now. **Until they land, both readings of the window axis rest on comparisons
across budgets.**

## What would settle it

Only one of these is needed, and the first is cheapest by far because 4 of 6 runs already
exist.

1. **Retrain $k_v=4$ and $k_v=8$ at 12,000 and report the axis at 12,000 throughout.** Six runs
   are already in `results/sweep12k/` for $k_v = 4, 8$; $k_v=12$ has the full 12k set; $k_v=20$
   and $k_v=30$ are the existing sweep files. The axis is then uniform at 12,000 with no new
   training at all, only re-measurement under the fixed path.
2. Retrain $k_v=20$ and $k_v=30$ at 4,000 and report the axis at 4,000 throughout. Six new
   runs, and it discards the seed coverage agent B bought.

Option 1 makes the 12,000-episode design primary for the window axis, which is the direction
`FINDINGS_CROSSOVER` already argues for on independent grounds. It also inherits the
checkpoint tail risk, which is the real cost and which `FINDINGS_CHECKPOINT_TAIL` says must be
handled by reporting both checkpoints per cell.

## What must not be written until this is settled

* Any sentence saying the sweep holds training budget fixed.
* Any reading of the window axis as monotone in $k_v$.
* `tab:checkpoint` as evidence about window size.
