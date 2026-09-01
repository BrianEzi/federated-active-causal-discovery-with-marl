# Results — draft for supervisor, 1 Sep 2026

Every number here is from a committed result file and reproducible by the command in the
corresponding findings document. Where a claim was retracted during the work, the retraction
is stated rather than the earlier version quietly removed.

---

## 1. Structure recovery: reinforcement learning wins on window size and loses on site count

**Setup.** 20 cells x 3 seeds = 60 runs, all complete. Oracle evidence, factored belief,
scale-free graphs, `vary`-only action space, FedAvg with turn-aware credit and 4 local
epochs. 4,000 training episodes, 200 evaluation episodes per arm.

**Metric.** Hard structural Hamming distance of the POOLED GLOBAL GRAPH -- each covered pair
counted once across sites, and a pair is wrong unless the pooled belief settled on exactly the
true mark. Reported instead of episode success because success is the all-agents conjunction:
it saturates (every k=12 cell has two arms between 0.88 and 0.99) and amplifies (a per-window
rate of 0.98 against 0.90 becomes 0.94 against 0.61 at eight sites). One run scored success
0.035 while recovering the graph to hard SHD 0.0143 against random's 0.0537.

**Gate.** A run enters the tables if its mean per-window solve rate over the final ten
checkpoints is at least 0.70. 50 of 60 clear it; the 10 excluded are named per cell. An
earlier mutual-information gate was discarded: it tracks final policy entropy, not competence,
and was excluding runs that solved 95-100% of windows.

### 1.1 The window-size crossover

| window k (4 sites) | 4 | 8 | **12** | **20** | **30** |
|---|---:|---:|---:|---:|---:|
| learned hard SHD | 0.0117 | 0.0011 | **0.0001** | **0.0000** | **0.0001** |
| greedy | 0.0061 | 0.0008 | 0.0008 | 0.0006 | 0.0005 |
| random | 0.0886 | 0.0541 | 0.0391 | 0.0247 | 0.0177 |
| **learned / greedy** | 1.91 | 1.30 | **0.10** | **0.08** | **0.10** |

**There is a crossover between k=8 and k=12.** Below it a myopic uncertainty rule is
near-optimal and the learned policy's residual stochasticity is pure cost. Above it the
learned policy recovers the graph **10-12x more accurately**, and holds that margin to k=30.

Confirmed independently by a paired per-episode test on identical worlds (100 episodes,
`scripts/global_shd_paired.py`): k=20 `learned - greedy = -0.00050 +/- 0.00016`; k=30
`-0.00052 +/- 0.00011`; k=12 at high budget `-0.00027 +/- 0.00032`, inside two standard
errors. Two measurement paths, same conclusion.

### 1.2 The site-count boundary

| sites n (k=12) | 2 | 3 | **4** | 5 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| **learned / greedy** | 0.12 | 0.33 | **0.10** | 1.65 | 4.24 | 6.75 |

Monotone degradation from five sites. The same collapse appears at sigma=0.75 (L/G 4.36),
where the window is three-quarters shared and contention dominates.

**Reading.** The learned policy wins where the problem is large enough to require planning and
the coordination load is low; it loses where coordination dominates. Both halves are measured,
and the second is a stated boundary rather than an omission.

### 1.3 Budget

| beta | 1.0 | 1.2 | **1.5** | 2.0 | 5.0 |
|---|---:|---:|---:|---:|---:|
| **learned / greedy** | 0.42 | 0.30 | **0.10** | 0.84 | 0.40 |

The advantage is largest at moderate budget and shrinks when the budget is generous enough
that covering everything is feasible.

### 1.4 A caution about the aggregate

The single-number headline over all 50 gated runs is `learned - greedy = -0.016 +/- 0.118`,
which reads as a null. It is not informative: it averages a 10x win at four sites against a
6.75x loss at ten. **The result lives in the per-axis tables.**

---

## 2. Attribution of latent confounders: a three-term law

**The object.** Not "these two variables are confounded" but "AGENT j holds a hidden variable
that confounds exactly this set of my variables". Correctness is judged up to renaming, which
is also the privacy claim.

**Engine.** A version space over (structure, attribution) whose ownership half is factored
over connected components of the bidirected graph. The factoring is exact -- the candidate set
equals the product over components, verified as set equality on 240 random pair sets -- and
runs 1.5-2x faster than the joint enumeration with identical decisions.

**Zero misattributions in every cell of every experiment reported here.**

### 2.1 The three terms

| term | effect | evidence |
|---|---|---|
| **Coverage** | step function: below full window coverage, collapse; above it, nothing | budget sweep, 4 cells x 200 episodes |
| **Site count** | gentle once coverage is matched: 100 / 80 / 77 / **72%** at 1 / 2 / 3 / 7 partners | matched-budget control |
| **Group size** | absolute: **0%** for any latent explaining two or more pairs, from two partners, at any budget | every cell measured |

**Only the third is an identifiability barrier.**

### 2.2 Coverage is a step function

k=12, 4 sites, 200 episodes per cell:

| budget | turns per site | positions reached | single-pair recovery |
|---:|---:|---:|---:|
| 30 | 7.5 | ~7 of 12 | **5%** |
| 60 | 15 | 12 of 12 | **77%** |
| 120 | 30 | 12 of 12 | 77% |
| 240 | 60 | 12 of 12 | 77% |

The 60, 120 and 240 cells return identical counts (349 of 1056). Under oracle evidence a
repeat reveals nothing, so budget beyond full coverage is provably inert.

This one term retro-explains every other observation: n=8 at budget 60 (7.5 turns) gives 5%
and at budget 120 (15 turns) gives 72%; k=30 at budget 100 (25 turns for 30 positions) gives
30%. Two cells reach 5% from opposite directions -- too many sites, and too small a budget --
at the same turns-per-site.

### 2.3 The identifiability barrier

A latent explaining ONE confounded pair is recoverable: ownership is the whole question and a
single partner message settles it. A latent explaining a CLIQUE is not, from two sites onward,
at any budget. Separating "one hidden cause of {u,v,w}" from "three hidden causes, one per
pair" requires a PARTIAL response -- some pairs moving while others do not -- which requires
the owner to probe its private variables ONE AT A TIME. No policy in this work does so, so
partner responses are always total and the discriminating rule never fires.

Measured: 64% and 39% recovery for three- and four-child groups at ONE partner, falling to
**0%** at two partners and remaining 0% at seven with the budget doubled.

### 2.4 A closed-form predictor

    attribution  ~=  P(resolve | single-pair latent)  x  share of latents that are single-pair

The second factor is pure graph combinatorics, computable from the topology with no
simulation. With coverage adequate the first is nearly constant at ~0.76.

| cell | predicted | measured | residual |
|---|---:|---:|---:|
| 2 partners | 0.310 | 0.284 | -0.026 |
| 3 partners | 0.365 | 0.330 | -0.034 |
| 7 partners (matched budget) | 0.459 | 0.424 | -0.035 |
| sigma=0.25 | 0.795 | 0.755 | -0.041 |
| sigma=0.75 | 0.262 | 0.267 | +0.005 |
| k=20 | 0.214 | 0.196 | -0.018 |

Largest residual **0.041** across site count, shared fraction, window size and budget. The
one-partner cell is under-predicted by 0.263 **by design** -- the only regime in which
multi-pair latents contribute.

### 2.5 Reach

| k | pairs in window | right | wrong | s/episode |
|---:|---:|---:|---:|---:|
| 30 | 435 | 21 | **0** | 5.3 |
| 40 | 780 | 33 | **0** | 5.1 |
| 50 | 1225 | 27 | **0** | 9.4 |

Cost tracks the SETTLED PAIR COUNT rather than k, which is why k=40 is no dearer than k=30.
The joint enumeration cannot be run here at all: the attribution space alone is 8.4e10
hypotheses at k=20.

### 2.6 Training on the attribution reward does not help

Three seeds, k=12, 4 sites, reward criterion requiring attribution:

| | learned | greedy |
|---|---:|---:|
| seed 0 | 0.400 | 0.945 |
| seed 1 | 0.355 | 0.955 |
| seed 2 | 0.205 | 0.935 |

**Explained by 2.3.** At three sites only single-pair latents are recoverable and those come
free with structural coverage, so the attribution term of the reward is a constant plus noise:
no achievable behaviour changes it, so there is no gradient. The reward cannot teach what the
evidence cannot support.

The same fact from the other side: the baseline that greedily optimises its OWN attribution
belief spends **7%** of its moves on private nodes, against 38-61% for every other policy, and
attributes worse than a generic uncertainty rule. Private probes produce evidence for
PARTNERS, so an agent optimising its own belief correctly declines to make them.

---

## 3. Federation: turn-aware credit is an interaction, not an effect

Hard SHD at k=8, three seeds per cell:

| | credit on | credit off |
|---|---:|---:|
| pooled trajectories | 0.00160 | 0.00137 — no effect |
| **federated (FedAvg)** | 0.00106 | **0.01917 — 18x worse** |

Turn-aware credit matters **only under federation**. Under pooling the phantom rows -- stored
for agents that were forced to pass -- are averaged over the whole batch and wash out; under
FedAvg each client's local update is corrupted before averaging. The control arm makes this an
interaction with a mechanism rather than an observation.

Inconclusive at k=12 (three seeds against one surviving comparison run).

---

## 4. Limitations

* **The sampled regime is the open gap.** Every result above is oracle evidence. Policies
  trained under oracle evidence provably do NOT transfer to sampled data -- measured twice,
  with the mechanism identified: under oracle a repeat is wasted, under sampled it buys
  statistical power, so the learned rule is actively wrong in the new regime. A 66-task
  cluster sweep training directly under sampled evidence is in progress.
* **k=30 rests on one seed** locally after gating; a 12,000-episode re-run of the two
  under-trained seeds is in progress.
* **The skeleton is oracle-seeded.** With the true skeleton, full coverage identifies 100% of
  windows; with one estimated at n_obs=60 it identifies none. This is stated as an assumption,
  not a result.
* **"No policy probes one variable at a time"** is measured over the policies in this work,
  not proved impossible. Whether such an action is definable under the privacy constraint is
  the natural next question.

## 5. Claims retracted during this work

Recorded because the corrections are part of the evidence, not embarrassments to hide:

* *"Attribution precision collapses from 98% to 59% with window size"* — two engine defects,
  both fixed; there is no collapse and no misattributions at any size.
* *"The component engine gains precision by declining cross-component pruning"* — the probe
  built to test it found ZERO cross-component messages at any size.
* *"The site-count collapse is exponential hypothesis-space growth"* — the matched-budget
  control refuted it; the cause is coverage.
* *"Probe diversity explains attribution performance"* — instrumented and refuted; the
  lowest-coverage strong policy ties the highest.
* *"The learned policy attributes worse than random"* — one seed at two standard errors,
  reversed by the next two.
