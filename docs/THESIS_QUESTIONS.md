# Federated Active Causal Discovery via Multi-Agent Reinforcement Learning
## The question hierarchy, and exactly where each question is answered

Written 30 Aug 2026. Every question below is tagged with its evidence and a status:
**✅ answered** (data in hand) · **▶ scheduled** (run defined, not yet complete) ·
**⚠ needs build** (no mechanism exists yet) · **✍ writing only**.

If a question has no evidence pointer, it does not belong in the thesis.

---

# MAIN QUESTION

> **Can decentralised agents, each holding a private slice of an overlapping system, learn to
> allocate a scarce experimental budget so that the federation recovers causal structure —
> including *whose* latent explains each confounded pair — that no single site could recover
> alone?**

Three sub-claims, and the thesis stands or falls on them jointly:
1. **Allocation is learnable** and beats a myopic rule at equal budget.
2. **Attribution is recoverable** across a boundary no raw data crosses.
3. **Decentralisation has a measurable price**, and it is small enough to pay.

---

# 1. BACKGROUND — what is known, and where the gap is

| # | Question | Where answered | Status |
|---|---|---|---|
| B1 | What do interventions buy that observation cannot, and what does a MAG/PAG represent? | standard exposition; `docs/GLOSSARY.md` | ✍ |
| B2 | What does "federated" mean in *causal discovery*, and what object is federated? | **The graph, not the network.** Two families: continuous/gradient (NOTEARS-ADMM, FedDAG) average a weighted adjacency; constraint-based (federated PC/FCI) aggregate CI statements. Neither federates a control policy | ✍ |
| B3 | What is the strongest existing framework for multi-context discovery, and what does it *not* do? | **JCI** (Mooij, Magliacane & Claassen 2020). Three gaps, all ours: it does **no experiment selection** (Table 4 has no such column); it **cannot handle different variables per site** (§4.3.7, and a minus in Table 4 for every JCI variant); it **assumes non-adaptive contexts** (§3.4.2 — a doctor who diagnoses before treating violates Assumption 1). `docs/BIBLIOGRAPHY.md` §19 | ✅ |
| B4 | Why is a myopic uncertainty-greedy rule the benchmark to beat? | Identification reduces to a covering problem, where greedy is near-optimal by construction. This *explains* the difficulty rather than reporting it | ✍ |
| B5 | What does MARL contribute, and what are its known failure modes here? | credit assignment; 1/n signal dilution; CTDE vs full decentralisation | ✍ |
| B6 | Is the owner of a latent identifiable at all? | One latent over {u,v,w} and three over the pairs induce **identical** bidirected edges — no observation separates them; an intervention does, and only the owner can perform it. `cb/attribution.py` module docstring | ✍ + ▶ empirical sketch |

**Background delivers:** the gap is the *intersection* of three things JCI does not do — not an asserted novelty.

---

# 2. METHODOLOGY — the design, and why each choice

| # | Question | Where answered | Status |
|---|---|---|---|
| M1 | What is the system, and what does each site see? | Linear-Gaussian SCM; **vertical partition** — every site sees all rows, only its own columns. Window = own private block ∪ shared block. `ma/topology.py`, `ma/scm.py` | ✅ |
| M2 | Which edges may exist, and why does that rule matter? | **An edge may exist only where one agent observes both endpoints.** Verified: 0 cross-private edges permitted at w20. This is what makes half of all pairs guaranteed non-edges and keeps the global graph recoverable in principle | ✅ |
| M3 | What are the three evidence regimes, and what exactly is idealised in each? | **deterministic** (oracle skeleton + oracle orientation) · **noisy** (oracle skeleton + **sampled orientation**) · **realistic** (CI tests for both, `constraint` backend). `cb/factored.py::reset_marks` seeds the skeleton from truth in *both* of the first two | ✅ |
| M4 | What is the belief, and what does factoring give up? | Per-pair version space, O(k²). Gives up joint constraints, so it is **conservative: never settles wrongly**. `cb/factored.py` | ✅ |
| M5 | What counts as identified, and why three outcomes not two? | right / wrong / **unsure** — a claim decided 7-of-12 is a coin flip, and majority voting would report it as knowledge. `cb/claims.py` | ✅ |
| M6 | How is experiment selection posed as MARL, and what makes the policy portable? | Per-node pointer head: `Linear(hidden, 1)` applied per node, so **parameter count is independent of k and n** and the same weights are meaningful at any site size. `ma/policy.py::PortableRoleActorCritic` | ✅ |
| M7 | How is training distributed? | **FedAvg over one network, no differential privacy.** Local epochs per site, server averages weights; no raw trajectory leaves a site | ⚠ **needs build** — current code concatenates raw buffers (`ma/policy.py`), which is *more* centralised than gradient sharing |
| M8 | What crosses the boundary, and what does it leak? | Partner intervention counts and per-round response signatures. The agent learns **which partner** acted, never **which variable** | ✅ |
| M9 | What assumption makes attribution work, and what does it cost? | The **local-disturbance assumption**: when a partner acts and pairs move, that partner's own latents are among the movers. False in general — a private node can sit above a third agent's latent through the shared block. Switchable via `attribution_local_disturbance`; measured cost: right **82 → 0** when dropped | ✅ |
| M10 | What is optimal, and how do we know? | **Forced cover** — a directed edge needs its tail, a confounded pair needs both endpoints — **closed-form at any k** under oracle evidence (0.757k at k=4 → 0.542k at k=30). Exact Bayesian optimum additionally available at k≤5 | ✅ theory, ⚠ **A5 planner needs build** |
| M11 | What are the baselines and why each? | `random_clamp` (primary floor), uncertainty-greedy, **partitioned greedy** (the coordinated control), probe-then-work, pass, forced-clamp, oracle-cover | ✅ except A5 |
| M12 | How is anything measured, and what can each metric **not** tell us? | `docs/METRICS.md` — nine groups, each with its blind spot stated | ✅ |

**Methodology delivers:** every design choice defended, and every idealisation named before results are shown.

---

# 3. RESULTS — the empirical core

## R1 · Deterministic case — can allocation be learned at all?
| | Question | Evidence | Status |
|---|---|---|---|
| R1.1 | Does the learner partition the experimental space rather than duplicate? | duplicate coverage vs **partitioned greedy**, read against `duplicate_coverage_floor` | ✅ metric ready, ▶ re-run |
| R1.2 | Does it beat a myopic rule at equal budget? | joint success + per-window solve, argmax, MI-gated | ✅ at several rungs |
| R1.3 | **How close to optimal?** | regret in rounds vs the forced cover (any k) and vs exact Bayesian optimum (k≤5) | ⚠ **A5** |

## R2 · Scaling — where does it hold, and where does it break?
| | Axis | Values | Status |
|---|---|---|---|
| R2.1 | Agents | 2, 3, 5, 8 (15 if compute allows) | ✅ 2/3/6/8 with `normalise_returns`; **5 and 15 new** |
| R2.2 | Private set size | 1, 2, 5, 10, 25 | ▶ partially (window ladder conflates private+shared) |
| R2.3 | Shared set size | 3, 5, 10, 25 | ⚠ never varied independently |
| R2.4 | Budget, as a multiple of minimum required | 1×, 1.2×, 1.5×, 2×, 5× | ⚠ **oracle-only** — `required_cover` refuses under sampled evidence |
| R2.5 | **Sample size** — the convergence axis | n_int 100 / 1,000 / 4,000 | ✅ 3 points: margin **+0.053 / +0.100 / +0.123** |

## R3 · Noisy case — reasoning under uncertainty
| | Question | Evidence | Status |
|---|---|---|---|
| R3.1 | Does the learner still beat greedy at fixed budget? | retrained under `vs_evidence=sampled` | ▶ w04 done ×3 seeds; w08/w12 in flight; w20/w30 need cluster |
| R3.2 | Does the margin **grow** with sample size? | R2.5 dial, extended | ▶ |
| R3.3 | Why can an oracle-trained policy not simply transfer? | **Measured mechanism:** repeat rate greedy 0.247/0.331 vs learner 0.110/0.138. Under oracle a repeat is wasted; under sampling it buys power | ✅ |

## R4 · Realistic case — full CI-test discovery
| | Question | Evidence | Status |
|---|---|---|---|
| R4.1 | Does a policy trained in the noisy regime transfer to the `constraint` backend? | evaluation-only; training is infeasible (CI tests dominate runtime) | ⚠ 12 runs exist, none current |

## R5 · Attribution — the novel contribution
| | Question | Evidence | Status |
|---|---|---|---|
| R5.1 | Can an agent recover *whose* latent explains its confounded pair? | `score_groups` right/wrong/unsure, now separating engine failure from misattribution | ✅ engine fixed (B1) |
| R5.2 | How often is a latent blamed on the **wrong** site? | false-attribution rate under noise | ▶ |
| R5.3 | Does attribution survive past the enumeration wall (~5 confounded pairs)? | **factored attribution** — 3,045 numbers at k=30 against 282M candidates | ⚠ **B8 needs build** |
| R5.4 | What is the smallest message that enables cross-boundary attribution? | minimal-disclosure analysis | ✍ |

## R6 · The price of federation
| | Question | Evidence | Status |
|---|---|---|---|
| R6.1 | What does FedAvg cost against raw-data pooling? | FedAvg vs pooled, matched budget | ⚠ **M7 build** |
| R6.2 | What does full decentralisation cost against one shared policy? | `gnn_portable` vs `gnn_solo`, measured at every rung. **Gap is +0.353 at w04 → −0.017 at 2 agents** | ✅ but ⚠ solo untrained at 6/8 agents (entropy ≈ max) — needs solo + `normalise_returns` |
| R6.3 | Does decentralisation cost **data** or **capability**? | **C3**: solo with n× episodes. Promoted because the gap *vanishes* at 2 agents — the 1/n signature | ⚠ **needs run** |

## R7 · What does coordination actually buy?
| | Question | Evidence | Status |
|---|---|---|---|
| R7.1 | Is the advantage coordination, or just a better myopic rule? | vs **partitioned greedy** — same rule, shared surface split by convention, no learning | ✅ control built |
| R7.2 | Does the federation's *pooled* graph beat any single site's? | `pooled_global_belief` — greedy 0.0060 vs random 0.0653 at w20, contradiction 0.000 | ✅ |

---

# 4. DISCUSSION — what the results mean, and what they cost

| # | Question | Substance | Status |
|---|---|---|---|
| D1 | Why is greedy so hard to beat? | Identification reduces to covering, where greedy is near-optimal **by construction**. This converts a negative into an explanation | ✍ |
| D2 | What did the *metrics* teach us? | Under oracle evidence the belief is structurally **incapable of being wrong** — soft SHD ≡ `1 − 1/|surviving marks|`, a count of residual ambiguity. And greedy's decision rule **is** that count (6,976 node-scores, 0 disagreements), so the baseline optimises the evaluation metric | ✅ |
| D3 | Where does the protocol do work for us? | Round-robin guarantees equal opportunity. Under random turns the margin falls **+0.14 → +0.01/+0.09**, and 29.9% of episodes become unwinnable by arithmetic | ✅ |
| D4 | Which assumptions are load-bearing, and what do they cost? | (i) local disturbance — worth *all* 82 settled claims; (ii) oracle skeleton in the noisy regime; (iii) homogeneous vary-only interventions, which switches off the de-confounding channel | ✅ |
| D5 | How does this sit against JCI? | JCI is the inference layer, we are the decision layer. Our contribution is the intersection of its three gaps | ✍ |
| D6 | What did we get wrong, and how did we find out? | The retraction record — reward scale mistaken for dilution; metrics that hid their own failure modes | ✍ |

---

# 5. FUTURE WORK & CONCLUSION

| # | Question | Why it is future work |
|---|---|---|
| F1 | Differential privacy on the FedAvg updates | Deliberately out of scope; the disclosure channel is characterised but not privatised |
| F2 | Heterogeneous site sizes (C1) | `federated_topology` cannot express unequal blocks; the portable head should already support it |
| F3 | Factored attribution at scale (B8) | If not reached in R5.3 |
| F4 | GRPO with counterfactual groups | Refuted under oracle (the exact difference reward is free); genuinely novel only in the sampled regime |
| F5 | Curriculum / mixed-regime training | Needs per-episode topology resampling, which does not exist |
| F6 | Erdős–Rényi and density variation | Zero ER runs; `sf_m` fixed at 2 throughout |
| F7 | Real consortium data | The target application, untouched |

---

# THE BUILD LIST THAT FALLS OUT

Only five items block a question above. Everything else is running or written.

| build | unblocks | size |
|---|---|---|
| **A5 oracle-cover planner** | R1.3 — the only "vs optimal" above k=5 | small, **eval-only**, retro-fits every existing checkpoint |
| **M7 FedAvg** | R6.1, and makes the Methodology claim true | ~50 lines in `update()` + a local-epochs field |
| **solo + `normalise_returns`** | R6.2 at 6 and 8 agents | one config field |
| **B8 factored attribution** | R5.3 | ~4–6 h; compute is free |
| **shared-size axis** | R2.3 | config sweep, no new code |

**Run list:** Rung 1 (deterministic, scaled) → re-score everything under the corrected
defaults → sampled ladder → attribution.
