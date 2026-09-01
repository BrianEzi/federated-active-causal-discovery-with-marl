# Results — draft skeleton, 1 Sep 2026

Every number here is measured and traceable to a result file. **Square brackets mark what is
still running or absent.** Written as a skeleton to edit, not as prose to submit: the claims
and evidence are settled, the framing is yours.

---

## R0. What is being measured, and with what

`k` window size, `sigma` the contended fraction (shared / k), `n` sites, `beta` the budget
multiple of the structural cover. Evidence is ORACLE unless stated: the belief asks the true
graph rather than the data, which is the infinite-data idealisation and where the ceiling and
the optimum are computable. [Sampled-evidence results: pending, section R2.]

**Primary metric: hard SHD of the pooled global graph.** Each covered pair counted once, wrong
unless the pooled belief settled on exactly the true mark. Episode `success` -- the
all-agents conjunction -- is reported second and should not lead: it saturates (two arms
between 0.88 and 0.99 in every k=12 cell) and it amplifies (a per-window rate of 0.98 against
0.90 becomes 0.94 against 0.61 at eight sites). Measured case: a run scoring `success` 0.035
recovered the graph to hard SHD 0.0143 against random's 0.0537.

**Gate.** A run counts if its mean window rate over the last ten checkpoints is >= 0.70.
The earlier mutual-information gate was discarded: it tracks final ENTROPY, not competence,
and it excluded two entire cells in which every seed solved 95-100% of windows. It is not
monotone in competence either -- two runs at MI 0.032 had window rates 0.145 and 0.992.

---

## R1. Active experiment selection: RL wins on window size, loses on site count

60 runs, 20 cells x 3 seeds, oracle evidence. 50 clear the gate.

**Hard SHD, learned / greedy. Below 1.0 means the learned policy recovers the graph better.**

| window k (4 sites) | 4 | 8 | 12 | 20 | 30 |
|---|---:|---:|---:|---:|---:|
| | 1.91 | 1.30 | **0.10** | **0.08** | **0.10** |

| sites n (k=12) | 2 | 3 | 4 | 5 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| | 0.12 | 0.33 | **0.10** | 1.65 | 4.24 | 6.75 |

| sigma (k=12) | 0.25 | 0.50 | 0.75 | | beta (k=12) | 1.0 | 1.5 | 2.0 | 5.0 |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| | 0.22 | **0.10** | 4.36 | | | 0.42 | **0.10** | 0.84 | 0.40 |

**R1.1 A crossover at k ~ 8-12.** Below it a myopic rule is enough and the learned policy's
stochasticity is pure cost; above it the learned policy recovers the graph 10-12x more
accurately and holds that to k=30.

**R1.2 The reverse in site count.** Fine to four sites, monotonically worse after. Same at
sigma=0.75, where the window is mostly shared and contention dominates.

**One sentence:** the learned policy wins where the problem is large enough to require
planning and the coordination load is low, and loses where coordination dominates.

**Verified independently.** A paired test on identical episodes with per-episode standard
errors, in the sweep's own evaluation condition: k=20 -0.00050 +/- 0.00016, k=30
-0.00052 +/- 0.00011. Both significant, both agreeing with the sweep to within noise.

**Sensitivity to policy extraction, disclosed.** Under ARGMAX rather than sampling the
k=20/k=30 advantage falls inside two standard errors, and at k=12/beta=5.0 the policy
collapses (+0.164 +/- 0.004). These policies keep entropy near 2.5 of a ~3.2 ceiling, so
their competence is in the distribution, not the mode. Sampling is the extraction that
matches training and the sweep; the argmax result is a caveat, not a headline.

[k=30 rests on one seed after gating; two under-trained seeds are being re-run at 12,000
episodes on the cluster.]

---

## R2. The sampled regime

[**No data yet.** 66 cluster tasks in flight.]

**Established meanwhile, and it constrains what R2 can claim:** policies trained under oracle
evidence do NOT transfer to sampled evidence. Two independent tests -- transferred policy
0.171 against RANDOM's 0.208 at four sites; and at w08/w12 greedy wins by +0.0346 +/- 0.0055
and +0.0292 +/- 0.0042, both significant.

**The mechanism is the repeat rule.** Under oracle a repeat is strictly wasted, so the learner
correctly learns never to repeat (repeat rate 0.110/0.138 against greedy's 0.247/0.331). Under
sampled evidence a repeat is exactly how statistical power is bought. **The trained rule is
actively wrong in the new regime.**

**What DOES transfer is the ENGINE.** The sampled belief is a superset of the oracle's 97.8%
of the time and converges to it (1.41 survivors against 1.40 at 4,000 rows), with truth
retention 0.992 at alpha=1e-3. So the oracle environment is the n -> infinity limit of one
family, not a separate world -- which is what makes R1 the foundation rather than a detour.

[Power-limited oracle evidence -- withholding a fraction of ancestry answers at oracle speed
to obtain the sampled input distribution 74-110x cheaper -- is under test. First attempt
inconclusive: the power levels starved the environment (greedy 0.95 -> 0.03), which the
control arm caught.]

---

## R3. Attribution: a scaling law for federated confounder ownership

The object: which SITE's hidden variable explains a confounded pair. A question single-agent
causal discovery cannot pose, because there is nobody for the answer to name.

**R3.1 Recovery is bounded by the PARTNER COUNT.** 200 episodes per cell, zero
misattributions anywhere.

| partners | 2-child group | 3-child | 4-child | overall |
|---:|---:|---:|---:|---:|
| 1 | **100%** | **64%** | **39%** | 0.488 |
| 2 | 80% | 0% | 0% | 0.284 |
| 3 | 77% | 0% | 0% | 0.330 |
| 7 | **5%** | 0% | 0% | 0.028 |

Two collapses. Multi-pair groups die at the SECOND partner and never return. Single-pair
groups die by the seventh.

**R3.2 Why, and it is countable.** A pair's ownership hypothesis space is the non-empty owner
subsets, `2^n - 1`: **1, 3, 7, 127** at 1, 2, 3, 7 partners. With one partner ownership is
forced and coverage is the only limit. From two partners ownership and clique structure are
open simultaneously, and separating a clique from smaller latents needs a PARTIAL response --
the owner probing its private variables one at a time, which no policy performs.

**R3.3 A closed-form predictor.**

    attribution ~= P(resolve | single-pair group) x share of groups that are single-pair

Residual <= 0.041 across partners 1-7, sigma 0.25-0.75, k 12-20, over measured values from
0.028 to 0.755. The second factor is pure graph combinatorics, computable with no simulation.
**Given a site count and a contended fraction, the recoverable share is known in advance.**

**R3.4 The coordination gap.** At four sites: 37% recoverable as a by-product of
self-interested structural work, 63% requiring an experiment with no selfish payoff.

**R3.5 Three things it explains.**
* A policy trained ON the attribution reward scores 0.400 and 0.355 against greedy's 0.945
  and 0.955. The reward cannot teach what the evidence cannot support -- the attribution term
  is constant plus noise when no achievable behaviour changes the outcome.
* `greedy_attribution`, optimising its own attribution belief, probes privately **7%** of the
  time against 38-61% for every other arm. Private probes produce evidence for PARTNERS.
* It mirrors R1.2's site-count collapse. Two independent measurements, one cause.

---

## R4. Federated optimisation: credit assignment is an interaction

k=8, three seeds, hard SHD:

| | credit | no credit |
|---|---:|---:|
| pooled | 0.00160 | 0.00137 (no effect) |
| **federated** | 0.00106 | **0.01917 (18x worse)** |

Turn-aware credit matters **only under federation**. Pooling averages the phantom rows away in
one batch; FedAvg corrupts each client's update before averaging. A control arm, not an
assertion. [Inconclusive at k=12: three seeds against one.]

---

## R5. Method

* **Attribution reaches k=30 with zero misattributions**, via a candidate set factored over
  connected components of the bidirected graph -- exact (set equality on 240 random pair
  sets), 1.5-2x faster than the joint enumeration, equal or more decisions.
* **The atomicity rule was unsound and is repaired.** Two sites may independently confound a
  pair, so a group can appear partially moved; the old rule refuted the TRUTH in 27 of 85
  oracle messages. Restricted to pairs a group explains exclusively: 0 of 85.
* The version-space guarantee -- settled implies settled correctly -- holds at every size.

---

## R6. Limitations

1. **The skeleton is oracle-seeded.** With an estimated skeleton at n_obs=60, full coverage
   identifies no windows and claim accuracy falls 100% -> 57%.
2. **Sampled evidence: R2 is unevidenced** pending the cluster.
3. **Oracle-trained policies do not transfer**; each regime needs its own training.
4. **The attribution budget is fixed across site counts**, so "evidence per partner falls" is
   partly by construction. [Matched-budget control running.]
5. Scale-free graphs throughout; Erdos-Renyi evaluation not run.
6. k=30 rests on one gated seed locally.
