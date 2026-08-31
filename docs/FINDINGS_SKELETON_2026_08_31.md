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

## 3. The ablation — `scripts/skeleton_ablation.py`

The question the assumption raises is not "what if the skeleton were unknown" but "what does
the FINITE-SAMPLE skeleton cost". So the ablation estimates the skeleton from `n_obs` rows
with a CI-test adjacency search and reports:

- **skeleton error rate** against the true MAG skeleton, as a function of `n_obs`;
- **spurious vs missed adjacencies separately** -- they have different consequences. A missed
  adjacency removes a claim the agent will never make; a spurious one creates a pair that can
  never be settled, so it caps identification outright.
- **downstream identification**, seeded from the estimated skeleton rather than the true one.

`assume_skeleton` already exists in `FactoredBackend.__init__` and is **stored and never
read** -- a dead flag. The ablation makes it live.

## 4. Note for the write-up

The 100% agreement result is worth stating explicitly, because without it the assumption
reads as the weakest part of the setup and with it the claim becomes precise. But it must be
paired with §2.3: the assumption is defensible on information grounds and still removes the
clamp/rescue coordination channel, which is a genuine narrowing of what the thesis can
measure about coordination.
