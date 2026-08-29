# Compute map to freeze and submission — 29 Aug 2026

Every remaining compute-intensive job, costed, and ranked by which of the three thesis
results it serves. Costs are measured where the note says measured and extrapolated where it
says estimated. **Freeze 31 Aug morning · experiments to 2 Sep · write-up to 7 Sep 3pm.**

---

## 0. The prioritisation, stated first

The three results, and what each still needs:

| result | what it needs | compute |
|---|---|---|
| **1. Identification reduces to a forced set cover under perfect information** | F1 write-up + A5 oracle-cover planner as the optimal arm | **~zero** — A5 is eval-only, F1 is writing. Forced cover was already verified empirically today (both endpoints intervened ⟹ always resolved, 0.0000 at every rung) |
| **2. RL beats greedy under sampled inference, converging as n → ∞** | the sampled-evidence ladder | **107 core-hours.** This is the ONLY result with a large compute bill |
| **3. Maximise the cooperation gap, minimise the decentralisation gap** | C4 write-up; `gnn_portable` vs `gnn_solo` already measured | **~zero** — already measured. C2 FedAvg would strengthen it and costs a day+ |

**So: result 2 is the entire compute problem. Results 1 and 3 are writing.** Anything that
does not serve result 2, or is not eval-only, is a candidate for cutting.

**The tension worth naming.** Attribution is the NOVEL contribution and it is not in any of
the three results as currently stated. If the three results are the thesis, attribution is a
chapter that stands outside them. That is a framing decision, not a compute one.

## 1. Running now

| job | where | cost | status |
|---|---|---|---|
| sampled ladder w04/w08/w12 × 3 seeds | this laptop, 9 jobs | 14.7 core-h | **running**, ~3h |
| sampled ladder w20/w30 × 3 seeds | cluster (other agent) | 92 core-h, **lower bound** | handed over |

Measured basis: the same 160 episodes at w08 took **10.6 s oracle, 89.4 s sampled** — 8.4×.
Measured at k=8; sampled cost scales with CI tests ≈ k², so w20/w30 are lower bounds.

## 2. Everything else, costed

**Free — eval only, no training. Do all of these.**

| id | job | cost |
|---|---|---|
| A2 | convergence figure | data exists, just plot |
| A3 | budget-collapse plot | data exists (`results/cover/budget_curve*.json`) |
| A5 | oracle-cover planner arm | 1–2 h build, eval only. Turns every comparison into learned vs heuristic vs **optimal** |
| A4 | experiments-to-target, survival treatment | 1–2 h, `rounds_to_identification` already exists |
| E1 | evaluation protocol — the loader reseeds (`ma/policy.py:529`) | 2 h |
| E3 | record `vs_evidence` in attribution result files | 30 min |
| B1 | debug the attribution scorer (`cb/attribution.py:514`) | 1 h — **gates every attribution number** |
| D2 | strongest combined heuristic | 2 h |
| F1–F4 | theory write-ups | writing |

**Cheap training — hours, not days.**

| id | job | cost | serves |
|---|---|---|---|
| E5 | Erdős–Rényi replication, one rung × 3 seeds | 33 min oracle / **4.6 core-h sampled** | external validity. Zero ER runs exist |
| C1 | heterogeneous private sets | topology edit + 1 rung retrain | prerequisite for ANY federated claim |
| D1 | more private relative to shared | config + 1 rung retrain | result 2 |

**Expensive — each one is a day or more. Cut unless it buys a headline.**

| id | job | cost | verdict |
|---|---|---|---|
| C3 | solo with n× episodes | 4× ladder cost × 3 seeds ≈ 20+ core-h oracle | **cut** — nice-to-have on result 3, which is already measured |
| C2 | real FedAvg | 1 day build + training | **cut for the thesis**, name it as the missing middle |
| B3+B4+B5 | partner-need channel, global reward, adaptive baseline | 2h + 2h + 2h build, all need retrains | **cut** unless attribution becomes a headline |
| D4 | α-blend sweep | a sweep of retrains | **cut** |
| D5 | GRPO counterfactual groups | 1 day+ | **cut** — already refuted for the oracle regime |
| D6 | cost-heterogeneous experiments | 3 h + retrains | **cut** |
| D7 | self-assessed stopping rule | 3 h | **cut** — risks measuring calibration, not selection |

## 3. Factored attribution (B8) — it is CHEAPER, not more expensive

The slate filed B8 as "large build, 1–2 days, do not start". **On compute that is backwards.**

**Where the current `attributed` backend actually dies.** Two exponential walls, not one:

1. `equivalence_class` over whole-window structures — the same k ≤ 6 wall as `version_space`.
2. `_attributions` runs `product(owner_sets, repeat=n_confounded_pairs)`:

| agents | owner-sets | 3 pairs | 5 pairs | 8 pairs | 10 pairs |
|---|---|---|---|---|---|
| 3 | 3 | 27 | 243 | 6,561 | 59,049 |
| 4 | 7 | 343 | 16,807 | 5,764,801 | **282,475,249** |
| 6 | 31 | 29,791 | 28,629,151 | 8.5 × 10¹¹ | 8.2 × 10¹⁴ |

Hence `max_candidates=200_000` and the truncation path. So the wall is not "past 4 agents" as
the slate said — it is **past about 5 confounded pairs at any agent count.**

**What factoring costs instead.** One version space over the OWNER SET per pair — the exact
analogue of what `cb/factored.py` did for marks:

| window | state |
|---|---|
| k=8, 4 agents | 28 pairs × 7 owner-sets = **196 numbers** |
| k=20, 4 agents | 190 × 7 = **1,330 numbers** |
| k=30, 4 agents | 435 × 7 = **3,045 numbers** |

O(k²·2ⁿ) state and update — **the same order as the factored backend already in use, so
training an attributed ladder would cost about what the structural ladder costs.** The bill
is build time, not compute.

**What it gives up, and the fix.** Per-pair owners cannot represent *grouping*: one latent
parenting {u,v,w} versus three latents parenting {u,v},{u,w},{v,w} have the same per-pair
owners. But that is precisely the distinction `cb/attribution.py`'s own docstring says only
an INTERVENTION can settle — act on the single latent and all three associations move
together. So the co-movement is observable, and grouping can be recovered on top of the
factored owner map by **union-find over `response_signature` co-movement** rather than by
enumeration. Per-pair owner + co-movement union-find recovers what the enumeration recovers,
without the 282-million-candidate product.

**Validation is free and it already exists**: run factored attribution against the enumerated
`attributed` backend at k ≤ 6, where both are computable and the enumeration is exact.

**Estimate:** ~4–6 h build (mirror `cb/factored.py`), ~2 h validation against the enumerated
backend, ~0 marginal compute. Gated on **B1** — do not build on a scorer whose control fails.

## 4. Evaluation discipline

Sampled evaluation is not cheap. Measured: sampled `decompose` at w12, 50 episodes × 4 arms
took roughly 10 minutes — about 3 s per episode per arm. At 200 episodes × 4 arms that is
~40 min per rung-seed; across 5 rungs × 3 seeds, **~10 hours of evaluation alone.**

Rules, so evaluation does not eat the experiment budget:

1. **Gate before you measure.** Run `scripts/mi_gate.py` first. A rung that never trained is
   not a negative result, and evaluating it at 200 episodes is pure waste.
2. **One seed per rung at low episode count first.** Establish the ordering, then spend
   episodes only where the answer is close.
3. **Two arms in the exploratory pass** (learned argmax, greedy). Add `random_vary` and the
   sampled-policy arm only for the final table.
4. **Never re-evaluate a checkpoint you intend to retrain.**
