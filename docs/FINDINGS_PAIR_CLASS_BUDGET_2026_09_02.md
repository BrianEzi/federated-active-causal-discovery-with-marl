# Training moves what the policy is paid for and leaves what it is not

2 Sep 2026, 23:4x. Nine runs on the agent-count axis ($K = 4, 8, 10$ at $k_v=12$, $\sigma=0.5$),
three seeds each, 200 episodes per run, selected checkpoint, seeded evaluation. Every arm plays
the same episodes. The same nine cells measured at 4,000 and at 12,000 training episodes.

## The measurement

Errors, not rates, because the unrewarded class is rare enough that a rate hides the sample.

| budget | arm | rewarded (private-incident) | unrewarded (shared--shared) |
|---|---|---|---|
| 4,000 | learned | **487** / 673,200 | **11** / 27,000 |
| 4,000 | myopic | 213 / 673,200 | 0 / 27,000 |
| 12,000 | learned | **125** / 673,200 | **11** / 27,000 |
| 12,000 | myopic | 213 / 673,200 | 0 / 27,000 |
| either | random | 18,141 / 673,200 | 28 / 27,000 |

The myopic and random arms are deterministic and identical at both budgets, which is what makes
this a controlled comparison: one arm trains, the others do not.

## What it says

**Tripling the training budget cuts errors on the rewarded class by a factor of 3.9 and leaves
the unrewarded class where it was.** 487 to 125 against 11 to 11.

At 4,000 episodes the learned policy is worse than the myopic rule on both classes (+274 and
+11). At 12,000 it is better on the class it is scored on (-88) and unchanged on the class it
is not (+11). Learning acts on the reward and only on the reward.

This is the reward-alignment asymmetry that ledger 1.3 asserted, retracted on 2 Sep for lack of
evidence, and it is now evidenced -- with a control arm, a dose, and a direction predicted in
advance.

## What it does not say, and the size is the reason

Eleven errors in 27,000 observations is 0.04%. The policy does not neglect unrewarded pairs in
any material sense; it fails on them at a rate four hundredths of one percent while the myopic
rule fails at zero. **The claim available is about the pattern, not about a cost.**

Two further limits:

* With eleven events the Poisson standard error is about 3.3, so "11 at both budgets" means
  **no detectable change**, not exactly none. A budget effect smaller than roughly 3 errors
  would not be visible here.
* The eleven errors sit in entirely different runs at the two budgets -- at 4,000 they are in
  $K=8$ seed 2 and $K=10$ seeds 0 and 2; at 12,000 they are in $K=4$ seeds 0 and 2. An
  identical total with a completely different distribution is what a low-rate stochastic
  process looks like, and it is a reason not to read anything into which cells carry it.

## Supersedes

`CLAIMS.md` C3a, which reported only the 12,000-episode half (11 errors, 7 of 9 runs at zero)
and drew no comparison against the budget. The 4,000-episode comparator is what makes the
finding a finding rather than an observation.

The blanket justification in the original retraction -- "shared--shared error is 0.00000 for
both arms" -- remains withdrawn. It holds for the six $k_v=20$ and $30$ runs and nowhere else.

## What may be written

1. Training reduces error on rewarded pairs by 3.9x between 4,000 and 12,000 episodes and
   produces no detectable change on unrewarded pairs.
2. The myopic rule, which targets uncertainty uniformly, commits no unrewarded errors at either
   budget.
3. The magnitude of the asymmetry is 11 errors in 27,000 observations and must be quoted with
   the claim.
