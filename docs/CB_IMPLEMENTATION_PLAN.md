# Constraint-based engine — implementation plan

**Written 2026-08-23. Freeze 31 August, dissertation 8 September.**

Budget: **2–3 days build, 3 days experiments.** Decision point at the end of Day 1 — if the
gate fails, fall back to the hybrid (constraint-based confounding test, Bayesian DAG posterior)
described in `docs/STRIP_SCOPE.md` and the session record.

---

## 0. Where things stand

Already done, 2026-08-23:

- Worktree `constraint-based` on `explore/constraint-based`, cut from `main` at `c58543e`
- **Guard removed** from `ma/env.py` with a note in its place; three tests converted, and the
  defect it protected is now *demonstrated* by
  `tests/test_env_turns.py::test_clean_fraction_cannot_say_WHICH_node_was_clamped`
- `ma/coordination.py` deleted — 236 lines, zero references
- `requirements.txt` rewritten: the real dependency set is **numpy, scipy, torch, pytest**.
  The old file pinned the dead JAX/AVICI stack and omitted torch entirely
- Feasibility measured — `scripts/cb_feasibility.py`, `results/cb_feasibility.json`
- 476 tests green

**Deferred by instruction:** `results/` stays tracked for now; reduce to minimal and archive the
rest at project wrap-up.

**Still outstanding from the strip:** `sa/` dissolution, `confounding`→`projection` convergence,
test-suite pruning, `docs/archive/` (19 superseded design docs). None blocks Day 1.

## 1. What we are building

An engine that answers the same question as `ma/belief_dp.py` — *what does this agent believe
about the structure of its window?* — by running independence tests instead of scoring graphs.

    cb/
      citest.py     independence test. Fisher-z partial correlation first; the interface is
                    what matters, because swapping in a kernel test IS the nonlinear story
      skeleton.py   PC/FCI adjacency search. Lift from scripts/cb_feasibility.py, which
                    already has a working one
      orient.py     v-structures and bidirected detection. NOT full FCI -- we need the
                    latent-common-cause verdict, not every orientation rule
      bootstrap.py  B resampled runs -> edge-type frequencies. This is where uncertainty
                    comes from, and it is the load-bearing piece
      backend.py    the adapter satisfying the same protocol as the exact engine

## 2. Why this is worth the disruption — three measured reasons

**Speed.** At a realistic 30-node/5-agent window (`k=9`, `|X|=4`) the exact path pays 543
assignments × 0.102 s — tens of seconds per belief update. Constraint-based with `B=50` is
1.17 s, and it parallelises. *At our current small scale the switch is 6× slower; the crossover
is near `k=6–7`.*

**Generality.** BGe is linear-Gaussian, full stop. The independence test is a plug-in.

**It unblocks rung 1.** Three agents with one private node each hides two nodes from every
agent — which the removed guard refused, because the exact engine's scalar clean-fraction
cannot say *which* hidden node was clamped. A constraint-based engine has no clean/dirty
mixture, so the defect does not exist. **This is the strongest argument and it is structural,
not a speed-up.**

## 3. Day 1 — build to the gate

**Morning.**
1. `cb/citest.py` — Fisher-z, plus the interventional rule: rows where a node was **clamped**
   carry no information about that node's *parents*, but remain valid for its *children*
   (Cooper & Yoo 1999). This is the same rule the BGe path already implements, so it is a
   translation rather than a new decision.
2. `cb/skeleton.py` — lift the working search from `scripts/cb_feasibility.py`.
3. **Validate against ground truth.** `ma/projection.py` gives exact d-separation on known
   graphs. Assert the recovered skeleton matches the true one at large `n`, on the topologies
   we actually run. This is the "verify directly, not through a consumer" rule — check the
   skeleton itself, not the metric downstream of it.

**Afternoon.**
4. `cb/orient.py` — v-structures plus bidirected detection, checked against
   `projection.bidirected_pairs` on graphs where the answer is known.
5. `cb/bootstrap.py` — `B` resampled runs → `[k,k]` edge-type frequencies, the drop-in shape
   for `edge_marginals`.
6. **Metric reachability**, per the standing rule: confirm the identification criterion can be
   *earned* on confounded AND unconfounded episodes. A metric that is structurally unearnable
   once passed 529 tests in this project.

### ▸ THE GATE — end of Day 1

| | must hold |
|---|---|
| **G1** | Bootstrapped skeleton recovers known structure at large `n`, on our topologies |
| **G2** | Bidirected detection agrees with `projection.bidirected_pairs` where the truth is known |
| **G3** | The identification metric is reachable in both regimes |
| **G4** | One belief update at `k=9` is under ~2 s with `B=50` |

**If any of G1–G3 fails, stop and take the hybrid.** G4 failing is a tuning problem (lower `B`,
parallelise), not a design failure — treat it separately.

## 4. Day 2 — make it an interchangeable backend

1. **The belief-backend boundary.** `ma/env.py:_refresh()` currently calls
   `window.belief.edge_marginals(...)` directly. Introduce a minimal protocol both engines
   satisfy — `update(...) -> [k,k]` and `identified(...) -> bool` — and put
   `belief_backend: "exact" | "constraint"` on `MAConfig`. **The arms then differ in exactly
   one place**, the same discipline the disclosure arms use.
2. **Cross-check.** Run both engines on identical seeded episodes at `k=4–5` where both are
   affordable. They will not agree exactly — different estimators — so the test asserts
   *agreement on the identification verdict*, not equality of marginals. Disagreement here is
   the single most likely place a silent bug surfaces.
3. Convert the removed guard into a **backend capability check** — the exact backend declares it
   cannot handle `widest_hidden > 1`, the constraint backend declares it can.
4. Baselines (`ma/baselines.py`) re-pointed at whichever backend is active.

## 5. Day 3 — wire, smoke, launch

1. Policy observation: bootstrapped edge-type frequencies. Shape-compatible, so `ma/policy.py`
   changes little.
2. Smoke train at the current small topology — does it learn *anything*? Not a result, a
   liveness check.
3. **Launch the real runs.** Rung 0 re-run on the constraint backend for comparability, then
   **rung 1 — three agents — which has never been runnable before.**

## 6. Days 4–6 — experiments

Analysis in parallel with compute, not after it. Log to `docs/logs/SA_EXPERIMENT_LOG.md` as
results land, keeping nulls and self-corrections.

**Priority order, because compute will not stretch to all of it:**
1. Rung 1 (three agents) — the new capability, and the clearest thesis contribution
2. Rung 0 constraint vs exact — establishes the two agree where both run
3. Scale: how far does `d` actually go before it hurts
4. Disclosure arms, *if* time allows — the design is engine-independent and maps onto
   constraint sharing more cleanly than onto the Bayesian prior

## 7. Risks, and what they look like

**Silent incorrectness.** This project's recurring failure is bugs that pass tests — the vacuous
oracle metric, the unearnable two-agent metric, the equal-variance leak. Mitigations: validate
against `ma/projection.py` ground truth directly (Day 1.3), and keep the exact engine alongside
for cross-checking (Day 2.2) rather than deleting it.

**Bootstrap cost.** `B=50` serial at `k=9` is 1.17 s. Per episode that is ~100 updates. Lower
`B`, or parallelise — it is embarrassingly parallel. Measure before optimising.

**Independence-test calibration.** `alpha` is a new knob with no principled default. Sweep it
once on known graphs on Day 1 and fix it; do not tune it against results later.

**Two engines disagreeing at small `k`.** Expected and informative — but if the *verdicts*
diverge, stop and find out why before spending three days of compute on the answer.

## 8. What is explicitly not in this plan

- Full FCI orientation rules — we need the confounding verdict, not a complete PAG
- COmbINE-style SAT integration across agents — that is the federated-combination question, and
  it is a separate piece of work
- The `sa/` dissolution and test pruning — worth doing, but not on the critical path
- Disclosure implementation — design is done (`docs/DISCLOSURE_SPEC.md`), and it is
  engine-independent, so it waits for a working engine
