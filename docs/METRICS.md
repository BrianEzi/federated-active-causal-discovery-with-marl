# Every evaluation metric in the project — 29 Aug 2026

What each one measures, where it lives, and what it CANNOT tell you. Written because we now
have enough metrics that quoting the wrong one is a live risk — twice this week a number was
reported against a criterion the environment was not paying.

**The rule this document exists to enforce:** state the criterion, the evidence mode, and the
evaluation policy (argmax or sampled) beside every number. Any one of the three left implicit
has produced a wrong claim on this project.

---

## 1. Success — the primary criteria

| metric | where | what it is |
|---|---|---|
| **Joint success** | `ma/evaluate.py::_claims_success` | Every agent's window identified, in the SAME episode. Binary, **zero tolerance** |
| **Per-window solve rate** | derived | Fraction of agents whose window is identified. The un-conjoined version |
| `identified` | `cb/claims.py::ClaimScore` | All REQUIRED claims right AND nothing anywhere settled wrong |

Joint success is what the reward pays and what the ladder reports. It is **all-or-nothing**,
so it is insensitive to how close a failure was — which is the entire reason SHD exists.

**Trap.** `_claims_success` must mirror `TwoAgentEnv._result` exactly. It did not, twice: it
used the default `require_all_types` instead of the configured one, and on the attributed
backend it scored the confounding claims that backend replaces.

## 2. Claim-level scoring — `cb/claims.py`

Three outcomes per claim, never two: **right / wrong / unsure**. A claim decided 7-of-12 is a
coin flip, and majority voting would report it as knowledge.

`n_right`, `n_wrong`, `n_unsure`, `required_right`, `required_total`, `required_wrong`, and
`fraction(penalty) = (right − penalty·wrong) / claims`.

Older DP-path equivalents, still in result files: `mass_credit`, `mass_exact` (posterior mass
on the true window / exact structure), `credit_fraction`.

## 3. Structural distance — `scripts/shd.py`, `scripts/shd_diagnose.py`

| metric | definition |
|---|---|
| **Soft SHD per pair** | Σ over pairs of (1 − P(true mark)), ÷ C(k,2) |
| **Hard SHD per pair** | MAP mark ≠ true mark, counted 1, ÷ C(k,2) |
| **WRONG / UNSETTLED / RESIDUAL** | soft SHD split by whether the belief is confidently wrong, diffuse, or right-with-leftover |
| **De-duplicated SHD** | each covered pair counted ONCE, not once per window |
| **Pooled SHD** | survivor sets intersected across windows first — the federated aggregation |

**What soft SHD cannot tell you, and this is not a caveat but a proof.** On the factored
backend under **oracle** evidence the belief never settles wrongly (`cb/factored.py`: each
update is individually sound), so

> soft SHD per pair ≡ 1 − 1/|surviving marks| ∈ {0, ½, ⅔, ¾}

It is a **count of residual ambiguity, not a distance to truth**, and the WRONG bucket is
exactly 0.0000 at every rung. Worse, `UncertaintyGreedyAgent`'s decision rule IS the count of
nonzero-SHD pairs incident to each node (6,976 node-scores compared, 0 disagreements), so the
baseline is a one-step descent on the metric. **Never report oracle SHD as "closer to the
true MAG".** See `docs/FINDINGS_SHD_2026_08_29.md`.

Under **sampled** evidence the WRONG bucket becomes non-zero — but only for ORIENTATION.
`FactoredBackend.reset_marks` seeds every pair from `self.truth`, so the skeleton is oracle in
both modes. And measured at w04, error was only 6% of the metric; `random_vary` had the
LOWEST error rate, because an arm that settles nothing cannot be wrong. **The error component
rewards inaction and must never be reported alone.**

## 4. Global / stitched graph — INCOMPLETE, and this is the gap

| metric | where | status |
|---|---|---|
| `union_acyclic` | `ma/evaluate.py::union_graph` | is the OR-stitched global graph a DAG |
| `union_matches_truth` | same | exact match against the true global adjacency |
| `union_equivalent` | same | MEC equivalence of the global graph |

**Three limitations, all live.**
1. `union_graph` reads `_Window.get(k).dags[map_index]` — it **enumerates**, so it dies above
   k=5. The whole factored ladder is out of its reach.
2. `_constraint_union` is the non-enumerating fallback, but it is **majority-vote binary
   adjacency** — no edge types, no bidirected marks.
3. Neither is reported on the **claims** criterion, which is what every ladder run uses.

**What should replace it.** `shd_diagnose.py --check global` already does the sound thing:
intersect each pair's surviving mark set across every window that contains it. Sound because
under oracle evidence every site's set contains the truth, so the intersection does too. That
is federated aggregation, and it should be promoted from a diagnostic to the reported global
metric with its own SHD. **~2 hours.**

Measured consequence already in hand: `pooled == dedup` to four decimals at every rung —
**aggregation buys nothing on structure**, because all sites already agree on shared-shared
pairs. The asymmetry is entirely in the private blocks, which is the argument that the
federated result has to live in ATTRIBUTION, not structure.

**Cross-private pairs are excluded and that is correct.** `Topology.allowed_edges` permits an
edge only where one agent sees both endpoints, so cross-private edges cannot exist (verified:
0 allowed at w20). They are ~half of all pairs and are guaranteed true non-edges — including
them adds exactly zero error and dilutes every difference by a constant.

## 5. Budget and efficiency

| metric | where | note |
|---|---|---|
| `rounds_to_identification` | `ma/env.py:1400` | right-censored at budget+1. **Censoring is severe — needs median or restricted mean, never a plain average** |
| regret vs optimum | `scripts/vs_evaluate.py` | exact only on enumerable backends; a BOUND on the factored path |
| success vs `budget / required_cover` | `scripts/budget_curve.py` | the normalisation that answers "you gave big windows a more generous budget" |
| `required_cover` | `scripts/required_cover.py` | measured forced-set size: 0.757k at k=4, 0.542k at k=30 |

## 6. Coordination and mechanism

| metric | where | what it shows |
|---|---|---|
| **Duplicate coverage** | `ma/env.py:1378` | shared nodes touched by more than one agent. **The mechanism behind the coordination claim** |
| Own coverage | `shd_diagnose --check decompose` | distinct window nodes this agent intervened on |
| Union coverage | `--check spend` | window nodes touched by anyone |
| **Shared vs private spend** | `--check spend` | share of moves on the shared surface. Explains most of the oracle SHD gap |
| **Hub targeting** | `--check targeting` | mean true degree of intervened nodes ÷ window mean |
| Repeat rate | `--check decompose` | **reverses meaning with evidence mode**: waste under oracle, statistical power under sampled |
| Clamp share | same | — |

## 7. Training-health gates — run BEFORE quoting anything

| metric | where | rule |
|---|---|---|
| **I(S;A)/H(A)** | `scripts/mi_gate.py` | **MANDATORY.** Did the policy condition on its observation at all? A rung at the floor never trained, and an untrained policy is not a negative result |
| Policy entropy | training log | falling entropy means COMMITTING, not committing to something good |
| `collapsed` | result file | — |
| `first_success_episode` | result file | — |

**The MI gate is not a diagnostic, it is a precondition.** Agent rungs above 3 have never
passed it (a06 0.067, a08 0.035, a08long 0.109 against 0.41–0.62 on the window axis), so no
coordination claim at 6 or 8 agents is supportable.

## 8. Attribution-specific — `cb/attribution.py`

| metric | status |
|---|---|
| `score_groups` — right/wrong/unsure on latent groups | **BROKEN.** Its oracle control reports `wrong` at 0.075–0.113 where its docstring says wrong is impossible. Likely `group_frequency.get(group, 0.0)` scoring an absent group as WRONG (line 514). **Gates every attribution number in the repo** |
| False-attribution rate under noise | blocked on the above |
| `owner_channel` | the per-agent owner distribution the policy observes |

## 8b. The change list — what to fix before freeze, and how

Ranked by whether it changes a number we would report.

### Fix now — these make a reported number wrong

| metric | what is wrong | the fix | status |
|---|---|---|---|
| `never_acted_episodes` | Tests `max(interventions) == 0` — whether NOBODY acted, which is almost never true. It reads 0.0 under random turns while **29.9% of episodes leave some agent with no move at all**, and those episodes are unwinnable for any policy. It hid the entire random-turn failure mode. | Added **`some_agent_never_acted`** = `mean(min(...) == 0)`. Old field kept so historic files still parse. | **DONE** |
| `free_rider_index` | Two problems. The name is backwards — it is `min/max` interventions, so **higher is better**. And it is protocol arithmetic: under random turns per-agent counts are Multinomial(budget, 1/n), and pure sampling with no policy gives 0.158 against the measured 0.140. | Added **`effort_evenness`** (same number, honest name) and **`effort_evenness_null`** (what the protocol alone yields: 1.0 for round-robin and simultaneous, ~0.16 for random). **Quote the ratio to null, never the raw value, whenever protocols differ.** | **DONE** |
| `duplicate_coverage` | Past `len(exposed)` shared interventions duplication is FORCED, so an arm that spends more on the shared surface duplicates more however well it coordinates. Greedy spends 44–71% of moves there against the learner's 35–59%, so the arms were never on the same footing. | Added **`duplicate_coverage_floor`**. Report the excess over floor, or both side by side. | **DONE** |
| Joint success | Zero-tolerance across every window, so it silently absorbs structurally impossible episodes. | Added **`success_feasible`** — joint success restricted to episodes where every agent got at least one move. | **DONE** |
| soft SHD, per-window average | Counts a shared pair once PER AGENT. Under oracle that double-counting alone accounts for greedy's entire win at w04/w12/w20 and reverses it at w08. | `shd.py` now computes **`dedup_shd`** alongside and prints it first as the number to quote. | **DONE** |
| `shd.py` learned arm | Loaded with `deterministic=False`. Argmax was worth half the gap at w08/w12 and REVERSES at w04. | **Argmax is now the default**; `--sample` restores the old behaviour and `--both_policies` prints both. Report per rung, never as a blanket rule. | **DONE** |
| `score_groups` | Oracle control fails: reports `wrong` at 0.075–0.113 where its docstring says wrong is impossible. | Three defects found. Contradiction guards added to both elimination channels; `score_groups` returns UNSURE + `exhausted` instead of laundering an emptied version space into a wrong verdict. **Rule 1 of `consistent_with_partner` remains sound-LEANING** and is now documented as such: deleting it is sound but collapses `right` from 72 to 0. Net wrong 24 → 20. | **PARTIAL — cost documented** |
| `rounds_to_identification` | Right-censored at `budget+1`, severely. A plain mean reports the censoring horizon, not the policy. | Added **`survival_summary`** → `censored_fraction`, `median_rounds` (nan when fewer than half identify — undefined is the honest answer), `restricted_mean` (bounded by the horizon). Wired in as `time_to_identification`. | **DONE** |

### Fix the reporting, not the metric

| metric | the fix |
|---|---|
| soft SHD under **oracle** | It is a residual-ambiguity count, not a distance — the WRONG bucket is exactly 0.0000. **Call it residual ambiguity in oracle results and reserve "SHD" for sampled evidence.** Always publish the WRONG / UNSETTLED split rather than the composite. |
| soft SHD under **sampled** | Error was only ~6% of the metric at w04, and `random_vary` had the LOWEST error rate because an arm that settles nothing cannot be wrong. **Never report the error component alone** — it rewards inaction. |
| regret vs optimum | Exact only on enumerable backends. **Label it a bound on the factored path.** |
| I(S;A)/H | A precondition, not a diagnostic. **Run before quoting any learned number, and record it in the result file** so a reader can check it without re-running. |
| `connected` split | Already computed and already correct. **Promote it to primary** — disconnected graphs give independent subproblems, so pooling them dilutes the effect the project exists to measure. |

### Replace — broken at the scale we actually run

| metric | why | replacement |
|---|---|---|
| `union_graph`, `union_acyclic`, `union_matches_truth`, `union_equivalent` | Reads `_Window.get(k).dags[map_index]` — it **enumerates**, so it dies above k=5 and the whole factored ladder is out of reach. `_constraint_union` is the fallback but is majority-vote **binary adjacency with no edge types**, and neither runs on the `claims` criterion every ladder run uses. | The pooled survivor-set intersection in `shd_diagnose --check global`: sound because under oracle evidence every site's set contains the truth, so the intersection does too. ~2 hours to promote to a reported metric. |

### Leave alone

Three-outcome claim scoring (right / wrong / **unsure**); hub targeting and shared-vs-private
spend; per-window solve rate; `duplicate_coverage` itself as a belief-free transfer diagnostic.

## 9. Known reporting traps

1. **`scripts/shd.py` loads the learned arm with `deterministic=False`.** Argmax was worth
   half the gap at w08/w12 — and REVERSES at w04. Report per rung, never as a blanket rule.
2. **`scripts/shd.py` averages over windows**, so a shared pair counts once per agent. Under
   oracle that double-counting alone accounted for greedy's entire win at w04/w12/w20.
3. **Greedy's default `bar=0.7` vs the graded `claim_bar=1.0`** — worth +0.233 to greedy at
   four agents, enough to invert the attribution headline. Use `rescore_from_config.py`.
4. **The loader reseeds** (`ma/policy.py:529`), so every quoted CI reflects ONE fixed torch
   sample path. Not yet fixed (slate E1).
5. **`vs_evidence` is absent from attribution result files**, which is why nobody could tell
   they were all oracle.
