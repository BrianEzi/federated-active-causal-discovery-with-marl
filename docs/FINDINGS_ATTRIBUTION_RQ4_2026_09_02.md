# RQ4, narrowed: what attribution achieves, what bounds it, and one claim that describes an experiment nobody ran

2 Sep 2026, 22:xx. Written against Brian's scoping decision: attribution is a narrow research
question that establishes what is possible and under what conditions, and leads into future
work. Nothing here is an argument that attribution carries the thesis.

## 1. Soundness holds everywhere, and it is structural rather than statistical

Across 13 configurations spanning $k_v \in \{12, 20\}$, $K \in \{2,3,4,8\}$,
$\sigma \in \{0.25, 0.5, 0.75\}$ and budgets 30 to 240, over 14,076 observed latent groups:
**zero incorrect attributions.** Not one, in any cell, at any group size.

This is not a statistic that came out well. The engine either identifies the owner by the
atomicity rule or abstains, so a wrong answer would be a defect rather than an unlucky draw.
The number worth reporting is the count of abstentions, not the error rate.

Unaffected by the determinism defect: `attr_ceiling.py` drives with a deterministic
round-robin sweep, not a learned policy, so these numbers reproduce.

## 2. The two bounds separate on one comparison

The sharpest thing in the attribution data is a single pair of rows at seven peers.

| Budget | 2 children | 3 | 4 | 5 |
|---|---|---|---|---|
| 60 | 63/1344 | 0/658 | 0/210 | 0/63 |
| 120 | 965/1344 | 0/658 | 0/210 | 0/63 |

Doubling the budget moves two-child resolution from 4.7% to 71.8% and leaves every group of
three or more children at **exactly zero**. A resource bound responds to resources. An
identifiability bound does not. Both are present, and this comparison tells them apart without
needing an argument.

The budget curve at four agents says the same thing from the other side: 21 correct at budget
30, then 349 at 60, 120 and 240 — the identical count, not merely the same rate. Coverage
saturates and the remaining groups are not reachable at any budget.

## 3. The mechanism prediction is half right, and the half that fails matters

`attr_ceiling.py` states the test in its own docstring: with one peer there is no ownership
question, the owner is forced, so **every group should resolve regardless of size and the
cliff should vanish**. That cell was described as worth more than all the others together.

It does not vanish. It moves.

| children | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| resolved at one peer | 76/76 | 38/59 | 29/74 | 0/59 | 0/25 |

Complete at two children, 64% at three, 39% at four, and zero from five on. Forcing ownership
buys three and four-child groups that no other configuration resolves, and buys nothing at
five. So the boundary is not a two-child law. Resolvability falls with group size, and where
it falls to zero depends on how many partners there are.

The honest statement for the chapter is that a group is attributable when its ownership
hypothesis can be separated by a total response, that this is always true at two children, and
that additional partners tighten the constraint rather than a fixed size threshold governing
it.

## 4. A claim in the appendix and in Chapter 4 described an experiment that was never run

Both said, in effect, *training on the attribution reward does not help*, citing three runs at
0.400, 0.355 and 0.205 joint recovery against the myopic rule's 0.945, 0.955 and 0.935.

Those runs have `reward_criterion="claims"` and `observe_owner_channel=False`. The trainer
accepts two criteria, `claims` and `u14`, and neither scores attribution. A survey of every
result file in the repository found 435 runs, none with an attribution objective and none with
the owner channel enabled.

What the three runs vary is the **belief backend**: `component_attributed` against the sweep's
`factored`. So they measure the cost of carrying an attribution-capable belief while being
scored on structure, at 4,000 episodes — the budget that three other structural claims did not
survive.

Corrected at the generator (`scripts/build_appendix.py`), relabelled `tab:attrbackend`, and
the Chapter 4 bullet now carries the correction inline rather than being deleted, so that
anyone who read the earlier version finds out why it changed.

Brian raised exactly this in conversation: *without training an agent on the attribution task,
it is quite hard for us to say anything*. He was right, and the text had not caught up.

## 5. What RQ4 may claim

1. Attribution of a latent confounder to the peer whose private block contains it is **sound**:
   14,076 groups, zero errors, thirteen configurations.
2. What it achieves is bounded by coverage and by identifiability, and the two are separable by
   measurement rather than by argument.
3. The identifiability bound is a property of group size and partner count together.
4. Nothing may be said about a policy trained to attribute, because none exists. That is the
   future work this question leads into.
