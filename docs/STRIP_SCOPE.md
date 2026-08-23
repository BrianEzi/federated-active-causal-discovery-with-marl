# Strip scope for the constraint-based worktree

**Scoped 2026-08-23. NOT EXECUTED. Needs sign-off before anything is deleted.**

Worktree `.claude/worktrees/constraint-based`, branch `explore/constraint-based`, cut from
`main` at `c58543e`.

The request: strip to a minimal core so new work is not mixed with legacy. Everything below is
measured from the tree rather than assumed, because "this looks old" is how live code gets
deleted.

---

## 1. What is actually here

    directory        tracked files    note
    results/                   646    74.8 MB
    legacy/                    145    self-contained; nothing outside it imports it
    docs/                       47
    tests/                      35    3,764 lines
    scripts/                    29
    sa/                         20    single-agent package, 3,891 lines
    ma/                         11    multi-agent package, 3,309 lines
    cluster/                     4

`results/` alone is **74.8 MB across 646 files**, and most of it is superseded:

    ma_fixed                 178      pre-rebuild
    raw                       76      pre-rebuild
    phase2                    66      pre-rebuild
    gnn_budget_exact          42      pre-rebuild
    ma_train                  40      pre-turn-taking
    ma_turntaking_v1_void     39      named "void" by whoever wrote it
    sa_split / sa_derived     40      CURRENT -- the 2026-08-22 policy A/B
    rung0                     20      CURRENT -- rung 0 pass

## 2. The real source of confusion is not file count

Deleting files will help, but it is not the thing that will actually bite. **There will be two
belief engines in this tree**, and `ma/env.py` is hard-wired to one of them:

    ma/env.py  ->  ma/belief_dp.py  ->  sa/dp.py, sa/score.py

The Day-1 gate *requires* both engines to coexist, because the only way to catch a silent
correctness bug in a new inference engine is to cross-check it against the exact one at `k=4-5`
where both run. So "delete the Bayesian path" is exactly wrong.

**The fix is a boundary, not a deletion.** See §5. Without it, no amount of tidying prevents the
confusion the strip is meant to avoid; with it, much less deletion is needed.

## 3. Dead or superseded code, with evidence

Measured by grep across `ma/`, `tests/`, `scripts/`.

| file | lines | referenced by | verdict |
|---|---|---|---|
| `ma/coordination.py` | 236 | **0 files** | **Dead.** Delete. |
| `ma/score_regimes.py` | 241 | 2 test files only | Superseded by `belief_dp`. Delete with its tests. |
| `ma/confounding.py` | 167 | `coordination` (dead) + 2 tests | Overlaps `ma/projection.py`. Delete **after** confirming `projection` covers its use in `tests/test_ma_topology.py`. |

**Nine `sa/` modules are unreachable from `ma/`** — the single-agent-only path:

    sa/env.py  sa/env_dp.py  sa/policy.py  sa/evaluate.py  sa/baselines.py
    sa/uncertainty.py  sa/tracking.py  sa/dag_samplers.py  sa/sampler.py

That is roughly 2,100 lines. **But see §8 before deleting them** — `sa/policy.py` contains
`PerNodeActorCritic`, which is unported work we may still want.

## 4. The minimal core — what stays, and why

**Data and structure**
- `sa/scm.py` — linear-Gaussian SCM, per-node noise. The data source.
- `sa/priors.py` — `connectivity_prior_p`, the `2 ln(d)/d` rule.
- `ma/topology.py` — n-agent topologies, visibility, intervention authority. **No internal
  dependencies.**

**Ground truth and validation** — this is what makes the Day-1 gate possible
- `ma/projection.py` — d-separation, ancestor matrices, latent projection, `bidirected_pairs`.
  **147 lines, zero internal dependencies, used by 8 files.** It is the oracle a
  constraint-based engine is checked against. Highest-value file in the tree for this work.

**Environment and learning**
- `ma/env.py` — needs the §5 change.
- `ma/policy.py` — PPO. Depends only on `ma/env`.
- `ma/baselines.py`, `ma/evaluate.py` — greedy/random baselines and the metrics.
- `sa/graphs.py` (7 references from `ma/`), `sa/oracle.py` (2), `sa/gates.py` (bootstrap CIs).

**Cross-check only — frozen, not extended**
- `ma/belief_dp.py`, `sa/dp.py`, `sa/score.py`, `sa/scoretable.py`, `sa/posterior.py`

**New**
- `cb/` — the constraint-based engine. CI test, skeleton search, orientation, bootstrap.
  `scripts/cb_feasibility.py` already holds a working skeleton search to lift from.

## 5. The one structural change: a belief backend boundary

`ma/env.py:_refresh()` currently calls `window.belief.edge_marginals(...)` directly. Introduce a
minimal protocol both engines satisfy:

    update(samples, known_intervened, clean) -> [k, k] edge beliefs
    identified(true_adjacency, confounded_pairs) -> bool

Then `MAConfig` carries `belief_backend: "exact" | "constraint"`, and the arms differ in exactly
one place — the same discipline the disclosure arms use.

**Why this is worth doing before any deletion:**
- it makes the cross-check gate mechanical rather than a manual comparison
- it stops `cb/` growing tendrils into `env.py` that have to be untangled later
- it is the difference between "two engines" and "two engines tangled together"

**Cost: half a day.** It is the highest-value item in this document and it is not a deletion.

## 6. Execution order

Each stage ends with `pytest tests/ -q` green. Stop at the first red.

**Stage 0 — safety.** Confirm `main` has everything (`git log origin/main..HEAD` empty for
deleted paths). Nothing here is unique to this branch, so every deletion is recoverable by
`git checkout main -- <path>`.

**Stage 1 — results.** Delete the six superseded directories (461 files, most of the 74.8 MB).
Keep `rung0/`, `sa_split/`, `sa_derived/`, `structural_ceiling.json`, `disclosure_scaling.json`,
`cb_feasibility.json`. **Highest value, zero risk** — no code imports results.

**Stage 2 — `legacy/`.** 145 files, verified self-contained. Delete wholesale.

**Stage 3 — dead `ma/` modules.** `coordination.py`, then `score_regimes.py` with its two tests,
then `confounding.py` once its test usage is repointed at `projection.py`. Run tests between each.

**Stage 4 — the boundary (§5).** Before any `cb/` code exists.

**Stage 5 — `sa/` single-agent modules.** Last, and only after §8 is resolved. Lowest value,
highest risk of hitting something live.

**Stage 6 — scripts and cluster.** Keep `ma_train.py`, `ma_structural_ceiling.py`,
`cb_feasibility.py`, `ma_evaluate`-adjacent, and the reporting scripts. Delete the pre-rebuild
sweeps.

## 7. Merge-back strategy — do not skip this

This branch **may never merge**. It is an exploration whose deliverable is an assessment plus a
prototype.

**Keep deletions in their own commits, separate from any additions.** If the exploration
succeeds, cherry-pick the `cb/` package and the §5 boundary onto `main` — do **not** merge the
branch, or 74 MB of experimental records and the entire single-agent path vanish from `main`.

Those results are thesis evidence. Losing them because an exploration branch got merged would
be an unforced error of exactly the kind this project has already paid for twice.

## 8. Traps — resolve before deleting

**`sa/policy.py` contains `PerNodeActorCritic`** — the permutation-equivariant message-passing
network with multi-round aggregation. It is *unported* to `ma/` and the student's standing rule
is that `ma/` must inherit `sa/`'s wins. It sits in the "unreachable from `ma/`" list, so a
reachability-driven deletion would remove it. **Do not delete `sa/policy.py`.**

**`ma/confounding.py` vs `ma/projection.py`** overlap but may not be identical.
`confounding.latent_projection_pairs` and `projection.bidirected_pairs` look like the same
computation; confirm before assuming.

**`tests/sa/` guards modules `ma/` depends on** — `sa/graphs.py`, `sa/score.py`, `sa/priors.py`.
Deleting `tests/sa/` wholesale removes coverage of live code. Prune per-file, not per-directory.

**`ma/env.py`'s multi-hidden-node guard is deliberate.** It refuses topologies hiding more than
one node from an agent. It is correct *for the Bayesian engine*. A constraint-based engine may
not need it — but relaxing it is a decision to take explicitly, with a test, not a line to
quietly remove. A subagent removed this guard once already and inverted its regression test.

## 9. What this buys

- ~600 fewer files, ~74 MB smaller, `legacy/` gone
- ~640 lines of dead `ma/` code gone
- one authoritative belief boundary instead of a hard-wired dependency
- a test suite that only guards what remains

**What it does not buy:** correctness. The strip makes the tree legible; it does nothing about
the risk in §2 of `docs/BRIEF_CONSTRAINT_BASED.md`. The cross-check gate is what addresses that,
and §5 is what makes the gate cheap.

## 10. Recommendation

Stages 1–4 before Day 1 — roughly half a day, and Stage 4 is most of it. **Defer Stage 5
entirely**; the nine single-agent modules cost nothing to leave in place, and one of them holds
work we still want.

If time is tight, Stage 1 and Stage 4 alone capture most of the value: the bulk removal and the
boundary that actually prevents the confusion.
