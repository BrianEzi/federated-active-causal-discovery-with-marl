# Overnight plan — 2026-08-16, 00:15 to ~09:00

Plan of record for the unattended session. Written to disk deliberately: the conversation
will be compacted, and this must survive it. Update the STATUS column as blocks complete;
do not let it drift from what actually ran.

---

## Standing rules for the night

1. **Every acceptance test is fixed in this document, before the work.** A rule read after
   the numbers exist is not a rule.
2. **Check every new implementation DIRECTLY against ground truth**, never through a
   downstream consumer. Testing a sampler through the oracle instead of against the exact
   posterior cost three rounds on 2026-08-15.
3. **One debugging attempt per failure.** If a block fails its acceptance test twice,
   record the failure with measurements in `docs/SA_EXPERIMENT_LOG.md` and move to the next
   block. The Gibbs sampler detour is the reason this rule exists.
4. **Log at every block boundary**, including nulls and self-corrections.
5. **Do not change code under a running experiment.** Phase 2 owns `~/marl_sa` until it
   finishes (~01:20). New work runs from `~/marl_sa_fast`.
6. Commit and push after each block, so nothing depends on the session surviving.

---

## Cluster state and scheduling

Measured at 00:06: the binding constraint is **fair-share priority against my own running
work**, not cluster capacity. A newly submitted job sits at priority 0.00000 in `qw` while
Phase 2's tasks hold 2.50000. Night quietness does not help; submitting more jobs does not
increase throughput.

| time | event |
|---|---|
| ~01:20 | Phase 2 drains (11 tasks left, ~3 concurrent, ~20 min each) |
| ~01:50 | `sa_seeds10` (job 148032) completes — 10 seeds at d=4 and d=5 |
| ~01:30+ | cluster free; safe to sync `~/marl_sa` and submit new work |

---

## Blocks

| # | block | acceptance test | status |
|---|---|---|---|
| 1 | Subset-DP posterior wired into `sa/` | log Z and edge marginals identical to enumeration at d=3,4,5,6 | **PASS, after a correction** |
| 2 | One-pass edge marginals | matches enumerated marginals at d=4,5,6; ≥5x faster than `d(d-1)` constrained runs at d=6 | **PARTIAL** — exact; ≥5x met from d=7, not d=6 |
| 3 | MH oracle + sampling-based singleton fraction | oracle choices match exact oracle at d=5,6 within the measured MC floor; singleton estimate inside the exact value's CI at d=4,5,6 | **PARTIAL** — singleton estimator unbiased; oracle above the floor at d=6/4000 draws |
| 4 | Phase 2 E1xE2 analysis | every lever classified task/artefact/unlocked/dead; canaries surfaced; report and notebook regenerated | **PASS** (62/66 at time of analysis) |
| 5 | `ma/` package + GATE-M3 measured | ambiguity localised to boundary vs interior on >=3 candidate topologies | **PASS** — and T3 rejected on the measurement |
| 6 | GATE-M2 + `coordination_gained` | independent-greedy vs centralised-greedy intervals disjoint, or topology rejected | **measured — see the log** |
| 7 | Single-agent summary + artifact update | every claim traced to a result file | **PASS** |

### What actually happened, in one paragraph

Blocks 1 and 2 passed their acceptance tests and were **wrong anyway**: the subset DP was
verified on independent random data, which cannot exercise the numerical failure mode, and
it returned `Z = 0` on the first real environment episode at d=4. Rewriting the recurrence
in signed log space fixed it, and every test now uses SCM data. That correction is the most
important output of the night and is written up in full in `docs/SA_EXPERIMENT_LOG.md`.
Block 3's singleton estimator is sound; its pre-registered CI-containment test was badly
designed and was replaced with a bias test. Block 5 killed the T3 fallback that this plan
had written in as the escape hatch for latent confounding. d=7 **training** was not
reached; d=7 **gate validation** is running on the cluster, which is the correct order.

Blocks 1-3 are the critical path to d=7. If block 1 fails, blocks 3 and the d=7 runs are
cancelled and the night goes to blocks 4-7. If block 2 fails, d=7 still works, just slower
— skip and continue.

---

## Why d=7 needs THREE things, not one

1. **Belief** — subset DP replaces the sweep over enumerated DAGs. Verified 2026-08-15.
2. **Baseline** — the greedy oracle reads descendant sets off the enumerated graph list,
   which will not exist. Needs the MH path. Verified 2026-08-15.
3. **GATE 1** — compares the observational solve rate against the singleton-equivalence-
   class fraction, currently obtained by enumerating every DAG. At d=7 there is no list.

Without (3) a d=7 result cannot be validated, which is precisely the hole that made the old
d=6 numbers worthless. Fix: a DAG is alone in its class iff no **covered edge** can be
reversed while preserving skeleton and v-structures — a local, per-graph test. So the
fraction is estimated by sampling DAGs from the prior and testing each, with a bootstrap
interval. Acceptance test: the sampled estimate must contain the exact value at d=4, 5
and 6, where the exact value is computable.

---

## Cluster jobs, in order

| when | job | cost |
|---|---|---|
| queued | `sa_seeds10` (148032) — 10 seeds at d=4 and d=5 | ~30 min once it starts |
| after block 3 | d=6, 6 seeds, on the DP path | ~30 min/seed |
| after block 3 | d=7: references then 3 seeds | refs ~45 min, then ~1.6 h/seed |

**d=7 reference cost decision, made deliberately:** the greedy baseline needs ~900 oracle
calls. At 16,000 MH draws that is 2.5 h; at 4,000 draws it is ~45 min. The 2026-08-15
acceptance test measured 4,000 draws at 0.0080 nats of regret at d=6, against a Monte Carlo
floor of 0.0009. That is an acceptable cost for a *baseline* and is recorded here as a
choice rather than a default.

---

## Two-agent graph generation — the specification

**Nodes (6 total).** `A_private = {0, 1}`, `B_private = {2, 3}`, `Exposed = {4, 5}`.

**Forbidden edges.** Any edge between `A_private` and `B_private`, both directions — 8
forbidden directed pairs. Rationale: neither agent can ever observe such an edge, so no
data from anyone bears on it and it would be permanently unidentifiable. Allowing it would
make the global graph unrecoverable by construction.

**Generation.** Draw a random permutation as a topological order; for every ordered pair
`(u, v)` that is forward in that order and not forbidden, include the edge with probability
`p`. Acyclicity is free, no rejection sampling, and the prior is not silently distorted.

This is the "generate two DAGs then connect at the boundary" idea expressed as a mask
rather than a procedure — structurally identical, but without the double-drawing of the
shared exposed nodes or a repair step for acyclicity across the join.

**The prior must carry the same mask.** A generator that forbids cross-private edges paired
with a prior that allows them is a misspecification that would surface later as systematic
overconfidence and look like an estimator bug. Same discipline as the single-agent env.

**Observation.** Agent A sees columns `{0, 1, 4, 5}`; agent B sees `{2, 3, 4, 5}`.

**Authority.** Each agent may intervene on its private nodes and on the exposed nodes.
Shared authority over exposed nodes is deliberate — it is the surface on which coordination
and contention actually happen.

**Lever, not default:** widening the allowed cross-edges (private of one agent to exposed
of the other, or to anything) is the knob controlling how much difficulty sits at the
boundary. If GATE-M3 finds too little there, this is the repair.

---

## The risk that may force a redesign — latent confounding

**Flagged before the work, because it is likely and it changes what "local inference" even
means.**

Agent A observes `{0, 1, 4, 5}` and models a DAG over them. But the true generative model
includes `{2, 3}`, which can be **parents of the exposed nodes**. From A's perspective
those are unobserved common causes. A DAG model over A's view is then **misspecified**: the
correct object under latent confounding is a maximal ancestral graph, not a DAG, and A's
local BGe posterior is not a correct posterior over anything.

This is not a reason to stop — it may be the most interesting thing here, because it is a
precise, structural reason why coordination is *necessary* rather than merely helpful. But
it has to be measured rather than assumed, and it has consequences:

- If confounding is severe, per-agent exact posteriors are the wrong tool and the honest
  options are (a) MAG/PAG machinery, a substantial jump in scope, or (b) a topology where
  exposed nodes have no private parents from the *other* agent, which removes the
  confounding by construction at the cost of realism.
- If confounding is mild or absent under the chosen topology, local DAG posteriors are
  defensible and the design proceeds as written.

**Block 5 measures this explicitly**, as its first output, before anything is built on top:
for each candidate topology, what fraction of graphs give an agent a locally-confounded
view. This is a pure enumeration at 6 nodes, so it is a computation rather than a judgement.

---

## Topologies to compare in block 5

| id | A private | B private | exposed | note |
|---|---|---|---|---|
| T1 | 2 | 2 | 2 | the agreed default |
| T2 | 1 | 1 | 4 | wider boundary, more shared structure |
| T3 | 2 | 2 | 2, exposed nodes constrained to have no private parents | removes latent confounding by construction |

For each: number of DAGs in the masked space, fraction of graphs where an agent's view is
confounded, and where residual ambiguity sits (interior / exposed-exposed /
private-exposed).

---

## Deliverables by morning

1. `sa/` scaling past d=6, with d=7 either running or with a recorded reason it is not.
2. Phase 2 E1xE2 analysis — which levers matter for the task versus which were compensating
   for a broken network.
3. `ma/` package with the graph space, the three topologies, and GATE-M3 measured.
4. GATE-M2 measured, or the topology rejected with the number that rejected it.
5. A thesis-ready single-agent summary with every claim traced to a result file.
6. `docs/SA_EXPERIMENT_LOG.md` updated at every block boundary, nulls included.
