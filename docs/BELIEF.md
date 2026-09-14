# The belief

What an agent knows, how sites combine it, and what the representation cannot do.

## Marks

For every pair of variables in its window an agent tracks which of four marks is still
possible:

    NONE   no edge
    FWD    the first is an ancestor of the second
    BACK   the second is an ancestor of the first
    BI     neither is an ancestor of the other, but they share a hidden common cause

A pair is **resolved** when one mark remains. The set is a version space: it starts wide
and only ever narrows.

## The update

An intervention on `x` reveals, for every other variable `y`, whether `x` is an ancestor of
`y`. That single fact is enough to prune the pair, because for an adjacent pair:

- if `x` is an ancestor of `y`, the edge is `x -> y`. It cannot be `y -> x`, which would be
  a cycle, and it cannot be bidirected, because a bidirected edge forbids either endpoint
  being an ancestor of the other;
- if `x` is not an ancestor of `y` and the test had the power to see one, `FWD` is removed.

The update never removes the true mark, so a resolved pair carries the truth. This is the
soundness property everything else depends on.

## Why it composes

Each site's mark set for a shared pair contains the truth. Therefore so does the
intersection of any number of sites' sets, and the intersection is computed from the sets
alone — no data, parameters, gradients or likelihoods cross a boundary.

No posterior representation has this property. A product of site posteriors is not the
posterior of the pooled data, and repairing it requires exactly the exchange the privacy
constraint forbids. This is the one structural argument the representation owns outright.

The pooled global result reported in the dissertation is that intersection. A pair counts
as an error when the intersection is not exactly the true mark, so an undetermined pair
counts against us.

## What it cannot do

**It does not quantify uncertainty.** A mark is possible or impossible, never probable, so
nothing downstream inherits a calibrated confidence.

**It is an outer approximation.** Because the belief factorises over pairs, joint
constraints that couple distant parts of the graph are dropped. The belief can hold a
combination of marks that no single structure realises, which means it never claims more
than the truth but may stay more uncertain than a joint representation would.

**Its evidence is binary.** A test either fires or it does not, so magnitudes are
discarded and an experiment that falls just short of the threshold contributes nothing at
all. This makes the engine most expensive exactly where data is short.

## Where soundness fails

Soundness is conditional on the tests being calibrated, and there is a regime where they
are not.

An agent's control rows include rounds in which a peer intervened on a variable that agent
cannot see. Those rows are not the unintervened comparison the test assumes. The resulting
false detections **grow with sample size** — 3.1% at 100 interventional rows rising to
17.3% at 10,000 — because a larger sample makes a contaminated contrast easier to detect,
not harder.

Disclosing a single bit, that some foreign intervention occurred, restores calibration on
identical episodes. That remedy is switched off in every reported run, so the headline
numbers carry the cost of privacy rather than assuming it away.

## The skeleton assumption

Agents are given the adjacencies of their own window. This is the strongest assumption in
the work, and its cost is measured rather than asserted.

Adjacency is an observational quantity, so what is supplied is what the adjacency phase of
a constraint-based algorithm recovers from rows the agents already hold. What is withheld
is everything an intervention is needed for.

The assumption is load-bearing for a specific structural reason. A pair the skeleton calls
non-adjacent is closed to `NONE`, and a closed pair is skipped by every subsequent update —
so interventional evidence cannot reopen it, however plainly it demonstrates a dependence.
That monotone narrowing is what makes intersection pooling sound, and it means the belief
has no mechanism at all for recovering from a missed edge.

Two consequences follow, both measured at the principal cell:

- **The estimator's two errors are not symmetric.** A missed adjacency is fatal and silent;
  a spurious one merely leaves a pair nothing can settle, costing budget but never
  recording a false mark. At the conventional operating point the estimator makes the fatal
  error almost exclusively, missing 2,104 true adjacencies of 7,920 pairs while inventing 2.
- **So the test should be tuned for recall, not accuracy.** Retuning its significance level
  from 0.01 to 0.3 lowers pooled structural distance from 0.20732 to 0.17120 on the same
  sixty observational rows, and the gain survives the real intervention budget rather than
  living only at the ceiling.

Starting from a fully connected graph is not the alternative. With every variable
intervened on, a true skeleton resolves all 2,084 absent pairs correctly and a fully
connected start resolves none of them. Two mechanisms cause this, and neither is
statistical: `NONE` and `BI` are interventionally identical, since both mean neither
variable is an ancestor of the other; and ancestry is transitive while adjacency is not, so
a non-adjacent pair joined by a directed path reads as a direct edge.

The skeleton is therefore the only source of both distinctions, and no intervention budget
substitutes for it. What the belief needs is a **generous** skeleton rather than an accurate
one, which is a change to a threshold rather than a demand for more data.

`scripts/skeleton_ablation.py` produces these measurements; `submission/skeleton/` holds
the output.
