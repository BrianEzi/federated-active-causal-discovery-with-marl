# What the training graphs actually are — and two confounds in them

Measured 30 Aug 2026, before the sweep launched. Both findings are about the *distribution
episodes are drawn from*, which nothing in the results files records and which no metric
would have surfaced.

## The generator, stated once

Every sweep run draws a fresh DAG per episode from `Topology.sample_dag`:

- **Scale-free**, preferential attachment, `m=2` parents per node, along a random
  topological order. `prior_p` is **dead** under `sf` — it is consumed only by the `er`
  branch.
- **Acyclicity is free**, not enforced by rejection: edges only ever run forward along the
  order, under both generators.
- **The jointly-visible mask** is the real structural restriction. An edge may exist only
  where *some single agent observes both endpoints*, which at the baseline topology
  (4 agents, 6 private each, 6 shared; d=30) forbids **432 of 870 ordered pairs (49.7%)** —
  every cross-private pair. Agent i's private node can never point at agent j's private one.
- **`exposed_have_no_private_parents` is False** — `federated_topology` never sets it, so
  private nodes *can* parent shared ones. That is the hub-confounder pattern the thesis is
  about, and it is correct, but it is a default rather than a per-run decision.
- **No density guard.** `--max_edges` defaults to None.
- **Connectivity is NOT a restriction.** `_is_connected` only sets a flag; disconnected
  graphs are trained on, and `run_arm` splits the reported metrics by it.

## Finding 1 — the `confounded` filter distorts the small-k cells and not the large ones

`--episode_mix confounded` redraws until some agent's window contains a bidirected pair.
How often that rejects is not constant along the k axis. It is not close to constant.

| cell | d | P(confounded) unfiltered | draws/episode | connected | bidirected pairs/ep |
|---|---|---|---|---|---|
| k04 | 10 | **28.3%** | **4.38** | **63.3%** | 3.3 |
| k08 | 20 | 66.7% | 1.52 | 90.0% | 7.1 |
| k12 | 30 | 95.0% | 1.05 | 95.0% | 14.8 |
| k20 | 50 | 98.3% | 1.02 | 100.0% | 44.2 |
| k30 | 75 | 100.0% | 1.00 | 90.0% | 96.6 |

**At k=4 the filter discards 72% of draws; at k=30 it discards none.** So the k axis moves
two things at once: the window size, which is intended, and how heavily the training
distribution is conditioned, which is not. k=4 trains on the atypically dense, atypically
confounded 28% tail of its own graph distribution; k=30 trains on the natural one.

Compounding it, **k=4 is 37% disconnected** against 90–100% elsewhere, and a disconnected
graph gives the agents independent subproblems — no cross-agent confounding, nothing to
coordinate about. The connected/disconnected split in `run_arm` handles the *reporting*;
it does not change what the policy was *trained* on.

This is a SECOND reason w04 sits off the line from the other rungs. `scripts/sweep.py`
already records the first: w04 was the only rung at sigma=0.75 while the rest were at 0.50.
Two independent confounds now stack on the smallest k cell, which is the cell the scaling
claim anchors at.

**What to do.** Not "fix the filter" — conditioning on confounding is deliberate and the
alternative wastes budget on episodes with nothing to learn. Instead, one of:
  - report the k axis with `episode_mix=any` as a robustness check on k=4 and k=8;
  - restrict k-axis *claims* to k >= 8, where the filter is near-inert (<= 1.5 draws);
  - report the k axis split by connectedness throughout, which the metrics already support.
The cheapest honest option is the third plus a stated caveat on k=4.

## Finding 2 — scale-free and Erdos-Renyi cannot be matched on more than one axis at a time

The roadmap lists an ER arm (E5) to bound the graph-model dependence. It is worth having,
but it is not a free swap, and choosing `prior_p` is a *design decision that changes what
the comparison means*. Measured at the baseline topology:

| | edges | bidirected/ep | connected | max private out-degree |
|---|---|---|---|---|
| **sf, m=2** (the sweep) | 53.3 | **14.3** | 95.0% | 5.67 |
| er p=0.24 — density-matched | 53.5 | 11.9 | **40.0%** | 4.27 |
| er p=0.30 | 67.0 | 12.0 | 66.7% | 4.98 |
| er p=0.40 — connectivity-matched | **89.1** | 8.3 | 93.3% | 6.10 |
| er p=0.50 | 110.7 | 5.7 | 100.0% | 6.93 |

Match the density and ER is 40% disconnected. Match the connectivity and ER is 67% denser.
There is no p that does both.

And the quantity the thesis is actually about moves the wrong way throughout: **bidirected
pairs per episode FALL as ER densifies** (11.9 -> 8.3 -> 5.7 -> 1.9), because a dense graph
makes pairs directly adjacent rather than confounded. **No ER setting reaches scale-free's
confounding level at any density.**

That is not a problem with the ER arm — it is the argument for the scale-free choice,
quantified. `ma/topology.py` states it qualitatively: under `er` every private node is a
weak interchangeable confounder, while under `sf` a private node can be a HUB parenting
many shared variables at once, which projects to a bidirected CLIQUE in every partner's
window. The numbers above are that claim measured: max private out-degree 5.67 under sf
against 4.27 under density-matched er, and 14.3 bidirected pairs against 11.9.

**How to report an ER arm honestly.** Not as a co-primary generator, and not as a mixture —
training on a mixed pool dilutes both regimes and answers neither question. As a BOUNDING
result: "the advantage shrinks as hub structure is removed", with the matched quantity
named explicitly and the two that moved acknowledged. Density-matched (p=0.24) is the
defensible choice, because density is what the budget normalisation is defined against, and
the connectivity gap is then reported via the split that already exists.
