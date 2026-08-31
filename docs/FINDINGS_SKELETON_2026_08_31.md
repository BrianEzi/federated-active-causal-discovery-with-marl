# The skeleton assumption — what it actually costs

Measured 31 Aug 2026, prompted by asking what performance would look like without starting
from the correct skeleton. The answer reframes the assumption rather than excusing it.

## 1. The skeleton is NOT oracle knowledge

`FactoredBackend.reset_marks` seeds each pair from `self.truth`: a pair is either
known-absent or open with marks {FWD, BACK, BI}. That reads as handing the agent the answer.
It is not, and the reason is the defining property of a MAG.

A MAG over an observed set is constructed so its adjacencies are exactly the pairs that **no
observed conditioning set can separate**. So the skeleton is recoverable from observational
data alone. `ma/projection.py::observational_skeleton` computes that infinite-data limit --
conditioning only on OBSERVED nodes, so nothing an observational method could not know leaks
through. Measured against the true MAG skeleton:

| topology | pairs | agreement | spurious | missed |
|---|---|---|---|---|
| k=6, 3 agents | 1,350 | **100.0%** | 0 | 0 |
| k=8, 4 agents | 3,360 | **100.0%** | 0 | 0 |

Zero errors of either kind. So what the backend "assumes" is precisely what **FCI's adjacency
phase** recovers from the observational rows every agent already holds (`n_obs=60` before any
intervention).

**How to state it.** Not "we give the agents the true skeleton", but: *skeleton recovery is a
solved observational preprocessing step, performed exactly rather than estimated, because the
thesis studies experiment SELECTION -- the part that requires interventions.* A confounded
pair stays ADJACENT in the observational skeleton, so detecting the confounding remains
entirely the interventions' job; only adjacency is supplied.

## 2. What genuinely remains a limitation

Three things, narrower than "the skeleton is unknowable":

1. **Finite samples.** We hand over the *infinite-data* answer. Real FCI at 60 observational
   rows makes adjacency errors, and constraint-based orientation propagates them. The honest
   cost is a finite-sample error rate, and it is measurable -- see §3.
2. **Compute.** FCI's adjacency search is exponential in the maximum conditioning-set size,
   and that is where the polynomial cost of constraint-based discovery actually lives.
   Assuming it away is a compute shortcut as much as an information one.
3. **It removes a class of coordination.** Because adjacency is fixed and correct, a
   confounded pair is settled by intervening on BOTH endpoints, so a partner clamping the
   confounder buys nothing -- measured, 61.8% either way. On an engine that must LEARN the
   skeleton, a live confounder makes the pair look adjacent and clamping genuinely rescues
   the partner. See `FINDINGS_CLAMP_2026_08_30.md` §Finding 4. **This is the deepest
   consequence and it should not be buried in a methods footnote.**

## 3. The ablation — RUN, and the assumption is doing enormous work

The question the assumption raises is not "what if the skeleton were unknown" but "what does
the FINITE-SAMPLE skeleton cost". So the ablation estimates the skeleton from `n_obs` rows
with a CI-test adjacency search and reports:

- **skeleton error rate** against the true MAG skeleton, as a function of `n_obs`;
- **spurious vs missed adjacencies separately** -- they have different consequences. A missed
  adjacency removes a claim the agent will never make; a spurious one creates a pair that can
  never be settled, so it caps identification outright.
- **downstream identification**, seeded from the estimated skeleton rather than the true one.

`assume_skeleton` was **stored and never read** -- a dead flag. `FactoredBackend.reset` now
takes a `skeleton=` override, and `reset_marks` keeps it rather than re-deriving adjacency
from truth partway through an episode, which would have silently handed the assumption back.

### The result

k=6, 3 agents, every node intervened on, so nothing but the skeleton differs:

| n_obs | alpha | skeleton acc | spurious | missed | claims right (est) | claims right (true) | identified |
|---|---|---|---|---|---|---|---|
| 60 | 0.01 | 65.9% | 2 | 182 | 57.2% | 100.0% | **0.0%** |
| 60 | 0.05 | 68.1% | 3 | 169 | 60.1% | 100.0% | 0.0% |
| 60 | 0.20 | 73.9% | 10 | 131 | 68.1% | 100.0% | 0.0% |
| 1000 | 0.01 | 85.2% | 2 | 78 | 81.6% | 100.0% | 8.3% |
| 1000 | 0.05 | 85.7% | 6 | 71 | 82.7% | 100.0% | 5.6% |
| 1000 | 0.20 | 84.3% | 30 | 55 | 83.7% | 100.0% | 5.6% |

**The assumption is worth everything.** With the true skeleton, full coverage identifies
100% of windows. With a skeleton estimated at the operating sample size (`n_obs=60`), it
identifies **none of them**, and claim-level accuracy falls from 100% to 57%.

Three things to read carefully:

1. **Identification is a cliff because it is zero-tolerance** -- one missed adjacency scores
   a claim WRONG and destroys the whole window. The claim fraction is the gradient underneath
   it, and it is the more informative number: 57% at n_obs=60 rising to 82% at n_obs=1000.
2. **The errors are overwhelmingly MISSED edges** (182 against 2 spurious at alpha=0.01),
   which is an underpowered test rather than a fundamental limit. Raising alpha trades them:
   at 0.20 and n_obs=1000, missed falls 78 -> 55 while spurious rises 2 -> 30. Neither
   direction rescues identification, because both error kinds break it.
3. **Sixteen times the data buys 24 points of claim accuracy** (57% -> 82%) and still does
   not restore identification. This is not a sample-size problem that goes away.

### What this means for the thesis

The honest framing is now precise and it is not comfortable:

> The skeleton is **recoverable in principle** from observational data -- exactly, at 100%
> agreement, in the infinite-data limit. At the sample sizes this work runs at, estimating
> it instead of supplying it costs essentially all identification. The thesis therefore
> studies experiment selection **given a known skeleton**, and that scope restriction is
> load-bearing rather than cosmetic.

That is a stronger and more defensible statement than either "we assume the skeleton"
(sounds like cheating) or "the skeleton is observationally recoverable" (true, and
misleading on its own). Both halves have to be stated together.

Caveat on the ablation itself: the adjacency search is PC-style with conditioning sets capped
at size 2, which is the bounded search a practitioner would run and is where the cost the
assumption avoids actually lives. A full FCI search with better small-sample CI tests would
do better than these numbers; how much better is unmeasured.

## 4. Note for the write-up

The 100% agreement result is worth stating explicitly, because without it the assumption
reads as the weakest part of the setup and with it the claim becomes precise. But it must be
paired with §2.3: the assumption is defensible on information grounds and still removes the
clamp/rescue coordination channel, which is a genuine narrowing of what the thesis can
measure about coordination.
