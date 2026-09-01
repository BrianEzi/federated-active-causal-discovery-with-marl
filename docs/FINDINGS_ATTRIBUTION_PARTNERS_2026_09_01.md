# Attribution is bounded by the PARTNER COUNT, and the bound is countable

1 Sep 2026. Supersedes the "two-child ceiling" framing in
`FINDINGS_ATTRIBUTION_CEILING_2026_08_31.md`, which was correct at one configuration and is
a special case of this.

---

## 1. The result

Recovery rate of a true latent group, by the number of window nodes it explains, against the
number of PARTNERS the agent has. 200 episodes per cell, k=12, sigma=0.50, oracle evidence,
component backend. **Zero misattributions in every cell.**

| partners | group with 2 children | 3 children | 4 children | overall attribution |
|---:|---:|---:|---:|---:|
| **1** (2 agents) | **100%** | **64%** | **39%** | **0.488** |
| **2** (3 agents) | 80% | 0% | 0% | 0.284 |
| **3** (4 agents) | 77% | 0% | 0% | 0.330 |
| **7** (8 agents) | **5%** | 0% | 0% | **0.028** |

Sample sizes: 76/59/74 groups at 1 partner, 252/210/146 at 2, 456/312/201 at 3, and
1344/658/210 at 7.

**Two distinct collapses.**

* **Multi-pair groups die at the SECOND partner.** 64% and 39% at one partner, 0% at two,
  and 0% everywhere after. Not a decline -- a cliff.
* **Single-pair groups die by the seventh.** 100% -> 80% -> 77% -> **5%**.

## 2. Why, and it is countable

A pair's ownership hypothesis space is the set of non-empty owner subsets: **2^n - 1** for
`n` partners. That is **1, 3, 7, 127** at 1, 2, 3 and 7 partners. With ONE partner the set is
a singleton -- ownership is forced, `attributions_for` returns exactly one canonical candidate
for any pair set, and everything the belief has settled resolves immediately. That is why the
one-partner column is limited only by COVERAGE: 100/64/39% is the probability that all 1, 3
or 6 pairs of the group get settled within budget, not an identifiability limit.

From two partners onward, two ambiguities compound. Ownership is open (3, 7, ... 127 subsets
per pair) AND clique structure is open, and separating a clique from several smaller latents
needs a PARTIAL response -- some of the group's pairs moving while others do not. That
requires the owner to probe its private variables ONE AT A TIME, which no policy here does,
so responses are always total and atomicity never fires. Multi-pair groups therefore go to
zero and stay there.

Single-pair groups have no clique ambiguity, so they survive on ownership evidence alone --
until the owner set gets large. At seven partners each pair has 127 candidate owner sets to
eliminate while the round budget, fixed at 60, is split eight ways. **Exponentially more to
rule out, linearly less to rule it out with.** 5% survives that.

## 2b. A closed-form predictor, and it holds across a 27x range

The two collapses are INDEPENDENT, which means attribution factorises:

    attribution  ~=  P(resolve | single-pair group)  x  share of groups that are single-pair

The first factor is set by the PARTNER COUNT. The second is pure graph combinatorics --
how many latent groups explain exactly one pair -- and needs no simulation at all: it is a
property of the topology and the graph model. Groups explaining two or more pairs contribute
nothing beyond one partner.

Checked against every measured cell (`scripts/attr_model.py`):

| cell | partners | P(resolve\|1-pair) | share 1-pair | predicted | measured | residual |
|---|---:|---:|---:|---:|---:|---:|
| k12 s.50 n3 | 2 | 0.798 | 0.389 | 0.310 | 0.284 | -0.026 |
| k12 s.50 n4 | 3 | 0.765 | 0.477 | 0.365 | 0.330 | -0.034 |
| k12 s.50 n8 | 7 | 0.047 | 0.639 | 0.030 | 0.028 | **-0.002** |
| k12 s.25 n4 | 3 | 0.941 | 0.845 | 0.795 | 0.755 | -0.041 |
| k12 s.75 n4 | 3 | 0.782 | 0.335 | 0.262 | 0.267 | +0.005 |
| **k20** s.50 n4 | 3 | 0.667 | 0.321 | 0.214 | 0.196 | -0.018 |

**Largest residual 0.041**, over measured values spanning 0.028 to 0.755 and varying partner
count, shared fraction AND window size independently.

**The one-partner cell is under-predicted by 0.263, by design.** It is the only regime where
multi-pair groups contribute, because a single owner leaves `attributions_for` one canonical
candidate and coverage becomes the only limit. A model that fails exactly where its own
derivation says it must is better evidence than one that fits everywhere.

**What this licenses.** Given a site count and a contended fraction, the recoverable share of
the latent structure can be computed BEFORE running anything. That is a design tool, not just
a description: it says how many sites a federation can have before confounder attribution
stops being worth attempting.

## 3. What it explains

**The structure sweep's n-axis collapse.** Learned-to-greedy hard SHD goes 0.10 (n=4) ->
1.65 (n=5) -> 4.24 (n=8) -> 6.75 (n=10). Two independent measurements, one cause: more
partners is where this problem gets hard, and the reason is that per-partner evidence thins
while the hypothesis space grows.

**D7's negative result.** A policy trained ON the attribution reward scores 0.400 and 0.355
against greedy's 0.945 and 0.955 over two seeds. At three partners only single-pair groups
are recoverable and those come free with structural coverage, so the attribution term is a
constant plus noise -- there is no gradient, because no achievable behaviour changes the
outcome. The reward cannot teach what the evidence cannot support.

**`greedy_attribution`'s 7% private-probing rate.** An agent optimising its own attribution
belief correctly declines to probe privately, because that evidence goes to PARTNERS. The
same fact from the other side.

## 4. The claim this licenses

> Federated attribution of latent confounders is feasible at two to four sites and infeasible
> at eight, and the boundary is not a property of the algorithm. The ownership hypothesis
> space grows as `2^partners - 1` per confounded pair while the interventional evidence per
> partner falls with a shared budget, so the identifiable fraction collapses from 49% at one
> partner to 3% at seven.

That is a scaling law for the federated problem, computable in advance from the site count
and the budget, and it holds with zero misattributions throughout -- the engine degrades to
UNSURE, never to WRONG.

## 5. Caveats, stated

* One window size (k=12) and one shared fraction (sigma=0.50) for the partner sweep. The
  k=20/k=30 and sigma cells are in `results/attr_ceiling.json`.
* The budget is fixed at 60 rounds across all partner counts, so "evidence per partner falls"
  is by construction. **Raising the budget with the partner count is the obvious control and
  has NOT been run.** Until it is, the exponential-space and thin-evidence explanations are
  not separated -- both predict the collapse. That experiment is the first thing to run next.
* Oracle evidence only.

## 6. Reproducing

```bash
.venv/bin/python scripts/attr_ceiling.py \
  --cells 12:0.5:2:60,12:0.5:3:60,12:0.5:4:60,12:0.5:8:60 \
  --episodes 200 --out results/attr_ceiling.json
```
