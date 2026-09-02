# Results ledger — every recorded result, rated, with the theme it supports

1 Sep 2026, 20:20. For framing the Results and Discussion sections.

**Strength scale.** Combines replication, independence of confirmation, and whether the run
post-dates the defects fixed on 31 Aug / 1 Sep.

| | meaning |
|---|---|
| **A** | >=3 seeds, post-fix, and either replicated or confirmed by a second measurement path |
| **B** | >=3 seeds, post-fix, single measurement path |
| **C** | 1-2 seeds, or a known unresolved confound, or pre-dates a relevant fix |
| **D** | superseded or refuted -- listed so it is not re-derived |

---

## 1. STRUCTURE RECOVERY

### 1.1 The advantage grows monotonically with window size — **A**

Success (all required claims correct in a window), 3 seeds per cell:

| k | learned | greedy | **gap** |
|---:|---:|---:|---:|
| 4 | 0.808 | 0.883 | **-0.075** |
| 8 | 0.922 | 0.947 | **-0.025** |
| 12 | 0.977 | 0.918 | **+0.058** |
| 20 | 0.980 | 0.897 | **+0.083** |
| 30 | 0.968 | 0.843 | **+0.125** |

Perfectly monotone, crossing zero between k=8 and k=12. k=20 and k=30 are 12,000-episode
runs completed 1 Sep; k=4/8/12 are 4,000 episodes.

**Theme: planning horizon.** A myopic uncertainty rule is sufficient while the window is small
enough to be solved greedily, and its sufficiency degrades smoothly as the window grows. This
is the thesis question answered directly — active experiment selection is worth learning
exactly when the problem is too large to solve one step at a time.

### 1.2 SUPERSEDED 21:00 — see `docs/FINDINGS_CHECKPOINT_2026_09_01.md`

The row below is from the FINAL policy (`ma_train.py:503`). Re-measured from the
MI-selected checkpoint the SHD row is 1.68 / 1.40 / 0.19 / 0.00 / 0.90, which crosses
between k=8 and k=12 -- the same crossover as the success gap. **The advantage does
appear on structural error.** Kept here only so the old numbers are not re-derived.

### 1.2 (old) The advantage does NOT appear on average structural error — **A**

Hard SHD of the pooled global graph, learned / greedy:

| k | 4 | 8 | 12 | 20 | 30 |
|---|---:|---:|---:|---:|---:|
| L/G | 1.91 | 1.30 | **0.10** | **1.23** | **0.85** |

Non-monotone and noisy. At k=20 two of three seeds are PERFECT (0.00000) and one outlier
(0.00194) carries the mean above greedy.

**Theme: the criterion decides the verdict.** Success and SHD answer different questions, and
the thesis must say which it is asking. Success asks "did the agent complete the task it was
set"; SHD asks "how accurate is the whole recovered graph, including parts nobody was scored
on". This is a genuine methodological point, not a hedge -- see 1.3 for the mechanism.

### 1.3 RETRACTED 2 Sep — see `docs/FINDINGS_PAIR_CLASS_2026_09_02.md`

Re-measured at 200 episodes from `_best.pt` over six runs: shared-shared error is
**0.00000 for BOTH learned and greedy**, maximum across all six runs, over 90,000
pair-observations. The learned advantage is entirely on private-incident pairs
(0.00002 against 0.00051). The asymmetry below came from 60-episode runs and does not
survive. Kept only so the old numbers are not re-derived.

### 1.3 (retracted) The policy is better where rewarded and worse where not — **B**

Hard SHD split by pair class, 60 episodes, 4 runs:

| arm | private-incident (rewarded) | shared-shared (not rewarded) |
|---|---:|---:|
| learned | **0.00011** | 0.00036 |
| greedy | 0.00061 | **0.00000** |
| random | 0.02459 | 0.00347 |

At k=30 seed 0 the contrast is sharpest: learned 0.00044 vs greedy 0.00073 on rewarded pairs,
and 0.00143 vs 0.00000 on unrewarded ones.

**Theme: reward alignment, and the cost of a partial criterion.** The learned policy does
exactly what it was paid to do and neglects the rest; greedy, targeting uncertainty uniformly,
spreads effort evenly. This explains the success/SHD divergence mechanistically and is a
result about REWARD DESIGN in federated discovery, not a failure of the policy. It also
implies the criterion should be widened if the whole graph is the deliverable.

### 1.4 The advantage reverses with site count — **B**

L/G hard SHD at k=12: n=2 **0.12**, n=3 0.33, n=4 **0.10**, n=5 1.65, n=8 4.24, n=10 **6.75**.
Monotone degradation from five sites. Same collapse at sigma=0.75 (L/G 4.36) where
three-quarters of the window is shared.

**Theme: coordination load is the real limit.** Two independent axes -- site count and shared
fraction -- both degrade the learned policy monotonically, and both increase the number of
agents competing for the same nodes. This is the honest boundary of the contribution and the
strongest argument for the future-work direction.

### 1.5 The budget advantage peaks in the middle — **C**

L/G by budget multiplier at k=12: beta 1.0 **0.42**, 1.2 0.30, 1.5 **0.10**, 2.0 0.84, 5.0 0.40.
Two of five cells have only 1-2 gated seeds.

**Theme: RL earns its keep under scarcity.** The advantage is largest when the budget is tight
enough that choosing well matters and loose enough to complete the task. Directionally clear,
statistically thin.

---

## 2. ATTRIBUTION OF LATENT CONFOUNDERS

### 2.1 A latent explaining >1 pair is unrecoverable from two sites onward — **A**

Recovery by group size and partner count, 200 episodes per cell, ZERO misattributions anywhere:

| partners | 1 pair | 3 pairs | 6 pairs |
|---:|---:|---:|---:|
| 1 | 100% | **64%** | **39%** |
| 2 | 80% | **0%** | **0%** |
| 3 | 77% | **0%** | **0%** |
| 7 (matched budget) | 72% | **0%** | **0%** |

Holds at any budget -- doubling it at 7 partners does not move the zeros.

**Theme: identifiability, not resources.** Separating one hidden cause of {u,v,w} from three
separate causes needs a PARTIAL response, which needs a partner to probe its private variables
one at a time -- an experiment with no payoff for the partner performing it. **This is the
altruism gap of the thesis, and it is a hard boundary rather than a training failure.**

### 2.2 Coverage is a step function — **A**

k=12, n=4, from `results/attr_ceiling_budget.json`: budget 30 -> **21 of 1056**, budget 60,
120 and 240 -> **349 of 1056**, IDENTICAL counts.

**CORRECTED 2 Sep.** This entry previously read "5% -> 77%". Those were share OF THE
PREDICTED CEILING under a ceiling estimate of 0.4286 that the file no longer carries (it
now records 0.4767, giving 4.2% -> 69.3%). Quote the raw counts: they are what the file
contains and they make the step-function point without depending on a ceiling estimate.

**Theme: a design rule.** Full window coverage is necessary and sufficient; beyond it, extra
budget is provably inert because a repeat reveals nothing under exact evidence. One term
retro-explains every other observation, including two cells that reach 5% from opposite
directions (too many partners, too small a budget) at the same turns-per-agent.

### 2.3 The recoverable share is predictable in closed form — **A**

`attribution ~= 0.76 x (share of latents explaining exactly one pair)`, the share computable
from the topology with no simulation. Largest residual **0.041** across partner count, shared
fraction, window size and budget, on measured values spanning 0.196 to 0.755.

**Theme: the contribution is a predictive tool, not just a measurement.** Given a site count
and a contended fraction, you can say in advance what share of the latent structure is
recoverable -- and therefore whether attribution is worth attempting at all.

### 2.4 The engine reaches k=50 with zero errors — **A**

| k | 30 | 40 | 50 |
|---|---:|---:|---:|
| right / wrong | 21 / **0** | 33 / **0** | 27 / **0** |
| s/episode | 5.3 | 5.1 | 9.4 |

Cost tracks the settled-pair count, not k. The joint enumeration cannot run here at all
(8.4e10 hypotheses at k=20).

**Theme: the method contribution.** Component factoring is exact (set equality on 240 random
pair sets) and makes attribution tractable at sizes where the natural formulation is not.

### 2.5 Training ON the attribution reward does not help — **B**

3 seeds: learned 0.400 / 0.355 / 0.205 against greedy 0.945 / 0.955 / 0.935.

**Theme: a reward cannot teach what the evidence cannot support.** With 3 partners only
single-pair latents are recoverable and those arrive free with ordinary structural work, so
the attribution term is a constant plus noise -- no gradient. Reinforces 2.1.

### 2.6 The self-interested attribution baseline probes privately 7% of the time — **B**

`greedy_attribution` 0.07 private share against 0.38-0.61 for every other policy, and it
attributes WORSE than a generic uncertainty rule (0.185 vs 0.333 identified).

**Theme: altruism, stated as a measurement.** An agent optimising its own attribution belief
correctly declines to probe privately, because that evidence benefits partners. The clearest
single demonstration of the coordination problem the thesis is about.

---

## 3. FEDERATION

### 3.1 Turn-aware credit matters only under federation — **A**

Hard SHD at k=8, 3 seeds per cell:

| | credit on | credit off |
|---|---:|---:|
| pooled | 0.00160 | 0.00137 (no effect) |
| **federated** | 0.00106 | **0.01917 (18x worse)** |

**Theme: federation changes what the learning algorithm needs.** Pooling averages the phantom
rows away in one batch; FedAvg corrupts each client's update before averaging. A control arm
turns an observation into a mechanism, and it is the clearest federation-specific result.

### 3.2 The same comparison is inconclusive at k=12 — **C**

E4_credit 0.00082 (3 seeds) against E4_nocredit 0.00025 (1 seed).

**Theme: scope the claim.** State 3.1 at k=8 and say plainly that it does not replicate at
k=12 with the seeds available.

---

## 4. THE SAMPLED REGIME

### 4.1 Oracle-trained policies do NOT transfer to sampled evidence — **B**

Two independent tests. 27 Aug: transferred 0.171 against RANDOM's 0.208. 29 Aug: greedy wins
at w08 and w12, both significant, identification 0.000 for every arm.

**Mechanism, measured:** repeat rate greedy 0.247/0.331 against the learner's 0.110/0.138.
Under exact evidence a repeat is wasted, so the learner correctly learns never to repeat;
under sampled evidence a repeat is how you buy statistical power. **The learned rule is
actively wrong in the new regime.**

**Theme: what RL learns is regime-specific.** This is a genuine and quotable finding about
idealised training environments, not merely a limitation.

### 4.2 The ENGINE spans both regimes even though the policy does not — **B**

Sampled belief is a superset of the oracle's 97.8% of the time, converging to it (1.41 vs 1.40
survivors at 4000 rows); truth retention **99.2%** at alpha=1e-3.

**Theme: the machinery is validated across the regimes; only the policy is not.** This is what
makes the oracle work the foundation rather than a detour.

### 4.3 Sampled feasibility: n_int is the binding parameter — **B**

Greedy-vs-random separation appears only at n_int>=100 (0.16 vs 0.00) and is clear at 400
(0.48 vs 0.00). At n_int=20 nothing separates.

**Theme: the noisy regime is feasible but expensive**, and it sets the sampled sweep's design.

### 4.4 Power-limited oracle evidence as a substitute — **D, contested**

Every arm comparison was unpaired until 1 Sep 16:00 (a free-running RNG meant arms saw
different withholding patterns). Post-fix, channels-on does not help (mean -0.090), the
learned-greedy gap is unexplained, and four hypotheses are refuted (sparsity, repeats,
observation blindness, starvation).

**Theme: do not report as a result.** At most a methods note that the shortcut was tried and
did not work, with the reason still open.

---

## 5. ASSUMPTIONS AND ABLATIONS

### 5.1 The oracle skeleton is load-bearing — **B**

With the true skeleton, full coverage identifies **100%** of windows. With one estimated at
n_obs=60, it identifies **0%**, and claim accuracy falls 100% -> 57%.

**Theme: the honest limitation.** Name it in Limitations as an assumption the thesis inherits,
with the number attached.

### 5.2 The MI gate was invalid and is replaced — **A**

MI tracks final entropy, not competence. It excluded runs solving 95-100% of windows, and two
runs with identical MI (0.032) had window rates of 0.145 and 0.992. Replaced by a competence
gate (window rate >= 0.70), which excludes 6 runs rather than 14.

**Theme: methodological care**, and worth a short subsection -- it changed the k=30 verdict
from "collapses" to "beats greedy" purely by fixing the gate.

---

## 6. WHAT WAS RETRACTED

Each refuted by a measurement queued to test it. Worth a short subsection: it is evidence of
how hard the results were pushed on.

| claim | what killed it |
|---|---|
| Attribution precision collapses 98% -> 59% with k | two engine defects; zero errors at every size after |
| The component engine gains precision by skipping cross-component pruning | the probe found ZERO such messages |
| The site-count collapse is exponential hypothesis growth | matched-budget control -- it was coverage |
| Probe diversity explains attribution performance | lowest-coverage policy ties the highest |
| The learned policy attributes worse than random | one seed at 2 SE; the next two reversed it |
| Power-limited evidence closes the transfer gap | training had not converged; then the RNG was unpaired |
| Learned beats greedy 5-11x on SHD at k=20/30 | full 3-seed replication at 12,000 episodes |

---

## 7. SUGGESTED SPINE FOR THE DISCUSSION

1. **Planning beats myopia exactly when the problem outgrows myopia** (1.1) -- the central claim,
   monotone across five window sizes.
2. **But the criterion decides the verdict** (1.2, 1.3) -- and the policy optimises what it is
   paid for. A methodological contribution about reward design.
3. **Coordination load, not problem size, is the limit** (1.4) -- the honest boundary.
4. **Some structure is unrecoverable at any budget** (2.1, 2.2, 2.3) -- identifiability, with a
   closed-form predictor and a design rule.
5. **The experiment that would fix it has no selfish payoff** (2.5, 2.6) -- the altruism gap,
   demonstrated from two directions.
6. **Federation changes what the optimiser needs** (3.1) -- credit assignment as an interaction.
7. **What RL learns is regime-specific** (4.1, 4.2) -- and the engine, not the policy, is what
   generalises.
