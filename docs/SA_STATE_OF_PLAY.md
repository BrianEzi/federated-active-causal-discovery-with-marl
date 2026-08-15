# Single-agent: state of play

Consolidated summary as of 2026-08-15. `docs/SA_EXPERIMENT_LOG.md` is the chronological
record with every null and self-correction; this is the readable version, and it states
what is established, what is not, and what is known to be wrong.

---

## 1. The headline result

**The agent beats the myopic greedy information-gain oracle.** That was the research
question — optimal sequential experiment design is not greedy design chained together, so
the oracle is beatable in principle — and success criterion S2 in the original plan.

`gap_closed` is `(random − agent) / (random − greedy)` on episode cost, with unsolved
episodes charged the full budget. 0 = random, 1 = greedy oracle, above 1 = better than the
oracle.

| setting | min gap over seeds | seeds | GATE 1 valid? |
|---|---|---|---|
| d=4, n_obs=5000 | **+1.283** | 3/3 | yes |
| d=5, n_obs=5000 | **+1.233** | 3/3 | yes |
| d=5, n_obs=20000 | **+1.060** | 3/3 | yes |
| d=5, n_obs=5000, 5 seeds (E3 control) | **+1.144** | 5/5 | yes |
| d=6, n_obs=1000 | +1.145 / +1.098 / +1.098 | 3/3 | **NO — see §5** |

Reported as the minimum across seeds, never the mean: a configuration is only as good as
its worst seed.

---

## 2. What made it work

**The architecture, and almost nothing else.** 61 configurations of the flat MLP failed
across every lever — entropy coefficient, learning rate, step cost, budget, prior,
intervention strength, and more. A supervised probe localised the cause: the flat network
cannot express the mapping, reaching only 0.42 accuracy at predicting the oracle's choice
*with full supervision*, where the task is not exploration or reward at all.

The oracle's score for node *i* is a function of node *i*'s own descendant structure — the
**same function for every i**. The flat network maps `d(d-1)` edge marginals through a
dense layer to `d+1` logits, so it has to learn each node's score as a separate function of
the whole vector, and discover from scratch that nodes are interchangeable.

`PerNodeActorCritic` states the structure directly: build node *i*'s features from its own
row and column of the edge-marginal matrix, push them through a **shared** MLP, and read
the output as node *i*'s logit. This makes the policy **permutation-equivariant** —
relabel the nodes and the logits permute with them, which is true of the oracle and was not
expressible before — and makes the network's width independent of `d`.

Measured worth of the architecture: **~2.7 gap closed** (per-node +1.23 against flat −1.86
at otherwise identical settings).

Supporting settings, all measured: `lr=1e-3`, `hidden=256`, `episodes_per_update=16`.
**Action memory** (appending per-node intervention counts to the observation) buys
**stability, not capability** — without it seeds ranged +1.043 to −1.766.

---

## 3. Diagnostics that turned out to matter

**Final policy entropy separated pass from fail better than any hyperparameter.** Failing
runs sat at 1.2–1.6 nats, passing runs at 0.5–0.7, against a uniform ceiling of ln(n_actions).
The quantity was already being logged; nobody was comparing it against its own ceiling.
Now canary **G1**.

**Supervised probes predict differences in KIND, not in DEGREE.** The probe correctly
found the flat-vs-per-node gap (worth ~2.7 gap closed). It did not transfer for depth:
depth lifted probe accuracy from 0.880 to 0.944 at d=4, replicated over 3 seeds and 4 data
sizes, and produced **no** RL improvement (§4). Worth carrying into the multi-agent work,
where screening designs by probe will be tempting.

---

## 4. Depth: a clean negative result

Pre-registered rule: if depth 2 or 3 beats depth 1 by more than 0.03 probe accuracy at
matched data size on both d=4 and d=5, carry it into RL. The rule **fired** —

| d | episodes | L1 | L2 | L3 |
|---|---|---|---|---|
| 4 | 9000 | 0.880 | 0.941 | 0.944 |
| 5 | 9000 | 0.801 | 0.863 | 0.864 |

— and E3 then measured it in RL, 5 seeds per arm:

| arm | min | median | max | seed spread |
|---|---|---|---|---|
| layers=1 | **+1.144** | +1.203 | +1.241 | **0.096** |
| layers=2 | +0.989 | +1.217 | +1.299 | 0.310 |

Depth 2 is +0.014 on the median — inside noise — with a worse minimum and 3.2x the spread.
**Keeping layers=1.** Both arms sit well above the oracle, so one reading is that both are
near this environment's achievable ceiling and depth has no room to show. That is a
hypothesis; separating "no headroom" from "headroom depth cannot reach" would need a
sequential-optimal reference, which does not exist.

---

## 5. GATE 1, and the caveat on d=6

**The task must require intervening.** Observational data cannot distinguish DAGs inside a
Markov equivalence class, so the fraction of episodes solvable without any intervention
should equal the fraction of DAGs alone in their class — a number computed exactly from the
graph space, not a judgement call.

This gate was pinned once at d=3 and **silently stopped holding from d=5 upward**. The
observational sample count must grow with `d`:

| d | n_obs needed | singleton fraction |
|---|---|---|
| 4 | 1000+ | 0.1087 |
| 5 | 5000+ | 0.0893 |
| 6 | 20000+ | 0.0810 |

**Consequence: the d=6 results in §1 are not currently valid.** All three used
`n_obs=1000`, where at d=6 the task does not require intervening. E4 is re-running them at
`n_obs=20000`, where GATE 1 is confirmed to pass (rate 0.0800 against target 0.0810).

Worse, `gate1` is recorded as `None` in every one of those result files — the gate was
checked in a separate job at matching `(d, n_obs)` and never stored beside the numbers it
qualified. That is exactly why canary **G5** now records it per run.

---

## 6. Infrastructure built because of specific past failures

Five canaries, recorded in every result file and printed at the end of every run. Each
encodes a failure that happened and went unnoticed; in every case the problem was not that
a check failed, but that nobody thought to run it.

| id | check | the failure it encodes |
|---|---|---|
| G1 | final entropy vs ln(n_actions) | all 61 flat-network failures |
| G2 | gap-closed anchored at 0 and 1, **and ordered** | anchors of 0.233 / 1.067 read and believed |
| G3 | informative-fraction floor | the retracted "99.4% oracle agreement", 93–98% vacuous |
| G4 | seed spread > 0.5 | +1.043 to −1.766 read as a success |
| G5 | GATE 1 recorded for THIS run | §5 |

WandB logging is offline by default (compute nodes have no outbound internet, so an online
init hangs rather than failing) and wrapped so it can never abort a run it is only
observing.

---

## 7. Performance work, 2026-08-15

Two independent bottlenecks, both fixed exactly rather than approximately.

**BGe was needlessly dependent on sample count.** The score depends on the data only
through `(n, column means, centred scatter)`, and subset statistics are submatrices of the
full ones — but the code re-read all `n` rows once per `(node, parent set)` pair, 160
passes per posterior at d=5. Hoisted to one pass per node, d=5 is now **flat in n_obs**:
49.0 / 48.0 / 49.3 ms at 1000 / 5000 / 20000.

**At d=6 the cost was never sample count.** It was two `n`-independent reductions over the
3.78M enumerated DAGs — the score gather (384 ms) and edge marginals (517 ms) against a
90 ms score table. Replaced with a precomputed flat index (bit-identical) and `d` bincounts
over parent-set ids (agreeing to 2.5e-15).

End to end: d=5/n_obs=5000 **72.5 → 42.1 ms**; d=6/n_obs=20000 **1850.6 → 845.7 ms**.
This is what made E4 affordable at 20x the sample count.

---

## 8. Subset DP: exact inference past d=6

Enumerating DAGs is super-exponential — 29,281 at d=5, 3,781,503 at d=6, ~1.14 billion at
d=7. A recurrence over **subsets of nodes** instead, decomposing each DAG by its sinks with
inclusion-exclusion, gives the same answers:

| quantity | d | difference from exact |
|---|---|---|
| log Z | 3, 5, 6 | 0.0 |
| log Z | 4 | 4.6e-13 |
| edge marginals | 4, 5, 6 | ≤ 7.2e-14 |

Not an approximation. Cost at d=6: partition function **294 → 2 ms**, edge marginals
**733 → 65 ms**. Reaches d=11 in 0.46 s for Z alone.

**"Identified" survives unchanged** — posterior mass on the true DAG is
`exp(score(G) − log Z)`, available directly from Z. It does *not* require reconstructing a
graph posterior from edge marginals, which is impossible in general since marginals discard
the correlations between edges.

**Open problem, and the real blocker:** the greedy EIG oracle needs each node's
**descendant-set** distribution, and reachability is not decomposable per node, so it does
not come out of this machinery. Z and edge marginals both worked, which makes this easy to
overlook.

---

## 9. Things I got wrong, and how they were caught

Kept because the pattern matters more than the individual errors.

| claim | what was wrong | caught by |
|---|---|---|
| "policy_loss ≈ 0 means the update is 5% of reward scale" | advantage normalisation cancels absolute reward scale; near-zero loss is structural | reasoning it through again |
| "the entropy bonus causes the collapse" | `entropy_coef=0.0` still gave entropy 1.596 | running the ablation |
| "per-node beats flat" (first version) | compared at 600 vs 3000 episodes | matched re-run; conclusion survived |
| per-node scorer is permutation-equivariant | it pooled neighbours in index order, so it was not | a test I wrote to check |
| "the 0.89 ceiling is not about multi-hop" | read from a partial grid, biased toward that answer | the completed grid |
| "depth probe pilot shows no signal" | 1 seed, 40 epochs, undertrained | the 3-seed grid |
| E4 costs ~2.7 h/seed | scaled a **Myriad** baseline by a **laptop** ratio | measuring it on a node |
| G2 catches swapped references | the 0/1 property is an identity that holds under a swap | the test passing for the wrong reason |
| subset DP cancels catastrophically | it was a scaling bug, not instability; growth ratio is below 1 everywhere | per-node score shifting |

Two structural lessons, both now encoded in code rather than intention: **a number must
travel with the check that qualifies it** (the canaries), and **a rule fixed in advance
must not be read early** (`analyse_depth` refuses to decide on a partial grid).

---

## 10. Open

1. **Oracle reachability under subset DP** (§8) — blocks scaling `d` past 6.
2. **The local score table** is `d · 2^(d-1)` BGe evaluations and becomes the bottleneck
   before the DP does — 24,576 small determinants per step at d=12.
3. **E4** — gate-valid d=6, running.
4. **Phase 2 E1×E2** — 66 configurations, which levers matter for the task versus which
   were compensating for a broken network.
5. **Multi-agent** — see `docs/MULTI_AGENT_DESIGN.md`.
