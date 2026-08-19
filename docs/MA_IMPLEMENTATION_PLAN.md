# Two-Agent Implementation Plan

Companion to `docs/MA_PROBLEM_STATEMENT.md`, which defines *what* is being built. This
defines *the order it gets built in, and what must be true before each step is allowed to
proceed*. Written 2026-08-19.

**The discipline being applied.** The last two-agent build accreted: environment, scoring
rules, disclosure protocol, and policy all landed before anything was gated, so when the
numbers came out strange it took days to work out which layer was lying. Every phase below
ends in a **check that can fail**, and a failing check stops the phase rather than being
noted and worked around.

---

## Phase 0 — Freeze the reference *(half a day)*

Before replacing the belief, capture what the current code produces so the replacement can
be proved equivalent rather than assumed to be.

**Build**
- `tests/fixtures/ma_reference_posteriors.npz` — for ~200 seeded episodes at `(1,1,3)`,
  the full 543-vector posterior per agent per round, under all four scoring rules, plus
  the sampled data that produced them.

**Check**
- Regenerating the fixture twice gives bit-identical output. If the current environment is
  not deterministic under a fixed seed, that is a bug to fix now, not after the rewrite.

*Why first: the enumerated posterior is the only ground truth available for the DP, and it
stops existing once we delete it.*

---

## Phase 1 — DP-backed belief with explicit confounding *(2–3 days)*

The core replacement. Exact enumeration goes; the subset DP arrives **[U11]**.

**Build** — `ma/belief_dp.py`
- Wrap `sa/dp.py` for an agent's window of size `k`.
- Confounding as an outer loop: for each subset `S` of the `|X|(|X|−1)/2` candidate
  bidirected shared pairs, run the DP over DAGs compatible with `S`, then marginalise `S`
  out. Cost `DP(k) × 2^(|X| choose 2)` — 8 passes at `|X|=3`.
- Port the regime split (`clean` / `dirty` row masks) from `ma/score_regimes.py` into the
  DP's local-score computation, keeping `JOINT_CONF` semantics exactly.
- Keep the per-row log-sum-exp shift. The global shift underflowed whole rows to `-inf` and
  silently deleted hypotheses; that bug must not be reintroduced in the rewrite.

**Checks — all three must pass**
1. **Equivalence.** At `k=4`, DP edge marginals and DP log-partition match the Phase 0
   enumerated reference to **1e-10 across every episode, round, and rule**. Same standard
   the regime-scorer fast path was held to.
2. **Confinement is enforced, not assumed.** Assert no candidate bidirected pair involves a
   private node. Proved in `ma/projection.py`; the assertion is the guard that it stays true
   as topologies grow.
3. **Scaling is real.** Time a `k=10` window and confirm it completes. If `k=10` is not
   tractable, the DP has not bought what it was chosen for and the plan stops here.

**Then, and only then**: delete the enumeration path.

---

## Phase 2 — Environment rebuild *(2 days)*

**Build** — `ma/env2.py`, replacing `ma/env.py`
- `n_obs=100` default; `n_obs` and `n_int` as swept parameters, not constants.
- Actions: `(target, mode)` over own-private ∪ shared, `mode ∈ {VARY, CLAMP}`, plus `PASS`.
  Both modes retained **[U7 as clarified, M]**.
- Simultaneous step; clamp wins collisions on a shared target.
- Separate per-agent budgets **[U9]**.
- Observation: `k(k−1)` edge marginals + normalised budget + disclosure bits. Features on a
  common `[0,1]` scale — raw counts were a real bug once already.
- Disclosure: shared-node targets after acting **[U10]**; regime bit **[U13]**.

**Checks**
- **Leak test.** An agent's observation is a deterministic function of its own window only.
  Assert by perturbing a hidden private value and confirming the observation is unchanged
  except through legitimate causal influence on observed columns.
- **Determinism** under fixed seed, as Phase 0.
- **Disclosure timing.** Assert the round-`t` observation cannot contain round-`t` targets —
  only round `t−1`. This is the "before or after acting" question made into a test.

---

## Phase 3 — Gates, before any RL *(1 day)*

The single-agent rebuild exists because gates were skipped once. Three here, and a failure
stops the project rather than being caveated.

- **GATE 1 — the task must require intervening.** Observational-only joint identification
  rate equals the prior-weighted singleton-equivalence-class fraction, within CI.
- **GATE 2 — choices must matter.** Random clearly worse than greedy on *unconfounded*
  episodes, non-overlapping intervals.
- **GATE 3 — coordination must be necessary and sufficient.** On confounded episodes: a
  never-clamping pair fails; a forced-clamping pair succeeds. The gap is the headroom any
  learned policy is competing for. If there is no gap, there is no coordination problem and
  the two-agent case collapses to two independent single-agent cases.

*Note: the budget sweep now running is effectively a preview of GATE 3 — greedy's
`clamp_fraction 0.000` and flat 0.25 solve ceiling is the never-clamping arm, already
visible.*

---

## Phase 4 — Baselines *(1 day)*

`ma/baselines2.py`: `pass`, `random-vary-only`, **`random-that-clamps`**, `greedy`.

**Check**
- `random-that-clamps` must be reported as the primary floor **[U16]**. The earlier
  comparison used a random policy whose clamping was incidental; making it explicit is what
  keeps the coordination claim honest.
- Every comparison holds the belief rule fixed. Cross-rule numbers are void — a
  `joint_conf`-trained policy scored under `subset` collapses below random **[M]**.

---

## Phase 5 — Learning *(1 week, and the risky one)*

`ma/policy2.py`: independent PPO, no CTDE, shared scalar reward **[U15, supervisor]**.

**The first task is not training — it is the 1-in-10 seed collapse.** sd 0.154 on a median
of 0.312, with one seed in ten degenerating into passing immediately. Ordered hypotheses:

1. **Entropy collapse** — policy entropy falls before the reward signal arrives. Check the
   entropy trace on a collapsed seed against a healthy one; if it separates early, the fix
   is an entropy floor or a warmup.
2. **PASS is too attractive early.** With a step cost and a low initial solve rate, passing
   dominates until the policy is good enough for the `+1` to be reachable. Check by masking
   PASS for the first `N` updates.
3. **Reward sparsity** — the `+1` is never sampled at all on the collapsed seed. Check the
   first-success episode index.

Run 20 seeds, not 3. Three seeds cannot distinguish a real effect from this variance, and
quoting a median over three was a mistake made once already.

**Check**
- Collapse rate below 1 in 20, or the instability is reported as a headline limitation
  rather than buried.

---

## Phase 6 — Evaluation *(2 days)*

Implements the three-part success criterion **[U14]**.

- **Private recovery** — own private substructure as a DAG.
- **Shared recovery** — shared structure to CPDAG resolution.
- **Global consistency** — the union resolves to the true global graph, **with an explicit
  acyclicity check**.

**Why the acyclicity check is not redundant.** The current `ma/env.py` comments that global
identification "needs nothing extra", because two *fully correct* induced DAGs union to the
true graph. That is right — but it is an argument about full DAG recovery, and **[U14]**
relaxes the shared part to CPDAG. Two agents that each orient `X` differently within the
same equivalence class can union into a cycle. The check earns its place precisely because
of the relaxation, and this is a real correction to the existing code's reasoning.

Evaluate on the **joint** object, not edge marginals: ~10% of posterior mass can sit on a
wrong skeleton while every marginal looks correct **[M]**.

---

## Phase 7 — Scaling *(open-ended)*

Ladder, per the standing scaling plan: `(1,1,3) → (2,2,3) → (2,2,5) → (3,3,5)`.

The binding constraint is **`|X|`, not `d`** — `2^(|X| choose 2)` DP passes. `|X|=3` is 8,
`|X|=4` is 64, `|X|=5` is 1024. Growing the shared boundary is where this design breaks
first, and that is the honest scaling statement to put in the thesis.

---

## Sequencing summary

| phase | output | gate that can stop it |
|---|---|---|
| 0 | reference posteriors | determinism |
| 1 | `ma/belief_dp.py` | 1e-10 match, confinement, `k=10` tractable |
| 2 | `ma/env2.py` | no leak, disclosure timing |
| 3 | gates | task requires intervening; choices matter; coordination has headroom |
| 4 | baselines | random-that-clamps is the floor |
| 5 | independent PPO | collapse rate < 1 in 20 |
| 6 | evaluation | union acyclicity |
| 7 | scaling | `2^(|X| choose 2)` |

## Open items not on the critical path

- **`n_obs` and `n_int` sweeps** **[U6]** — parallel to phases 3–4, no dependency.
- **Budget** — the sweep running now; feeds the Phase 3 gate design.
- **The regime bit needs Mirco's ruling** **[U13]**. It is load-bearing for Phase 2's
  disclosure protocol, so if he rejects it the design changes at Phase 2, not later. Worth
  asking before Phase 2 starts.
- **Principled MCMC fix** — deferred indefinitely. The DP removes the need up to `k≈15`.
