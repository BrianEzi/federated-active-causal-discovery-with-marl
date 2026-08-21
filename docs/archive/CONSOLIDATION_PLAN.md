# Consolidation Plan — repo cleanup and thesis handover

**Written 2026-08-20, not yet executed.** Agreed to be run after the supervisor report is
finished, roughly 15:00 onward. Draft for tweaking, not a fixed contract.

**Why now.** `feat/single-agent-clean` has accumulated two generations of two-agent code side
by side, a pre-rebuild source tree, and a large number of one-off scripts. Someone cloning
the repo today cannot tell which of `ma/env.py` and `ma/env2.py` is real. Separately, a
different agent will write the thesis background and the start of the methodology, and it
needs the whole record in one branch — with an explicit statement of what has been RETRACTED,
because the logs are chronological and contain claims we have since withdrawn.

## Measured starting state (2026-08-20)

    31  submit_*.sh at the repository root
    52  files in scripts/
    24  files in docs/
    45  test files
    30  directories under results/
        ma/ holds BOTH generations: env.py + env2.py, baselines.py + baselines2.py,
        policy.py + policy2.py, score_regimes.py + belief_dp.py, gates.py, coordination.py
        plus src/ (pre-rebuild project), prototypes/, notebooks/, stray logs_*.txt in root

---

## Phase 0 — Freeze before touching anything (15 min)

Full test suite green, everything committed and pushed, then **tag**
`pre-consolidation-2026-08-20`.

Every later phase becomes revertible to a known-good point. A cleanup that loses a result is
strictly worse than the mess it removes.

## Phase 1 — Retire the superseded generation (45 min)

- Move superseded modules to `legacy/ma_v1/` with a README naming what replaced each and why.
- **Rename `env2 -> env`, `policy2 -> policy`, `baselines2 -> baselines`, `evaluate2 ->
  evaluate`.** The `2` suffixes are meaningless to anyone reading this in three weeks,
  ourselves included.

**`legacy/ma_v1/env.py` MUST BE KEPT, NOT DELETED.** It generated
`tests/fixtures/ma_reference_posteriors.npz`, and that fixture is the independent ground
truth the subset DP was validated against to 1e-10. Regenerating the fixture from the new
code would make the Phase 1 gate circular and destroy its entire value.

**GATE: full suite green immediately before and immediately after.** The rename touches ~45
test files, and this is the phase most likely to go wrong.

## Phase 2 — Docs: one authoritative set (45 min)

Several docs actively contradict each other (`MA_DESIGN.md` against
`MA_PROBLEM_STATEMENT.md`).

- **Promote:** `MA_PROBLEM_STATEMENT.md`, `MA_IMPLEMENTATION_PLAN.md`, `THEORY_NOTES.md`
- **Move to `docs/logs/`:** `SA_EXPERIMENT_LOG.md`, `MA_BUILD_LOG.md`, preserved VERBATIM.
  They are the primary record and their value is that they were written as things happened,
  including the wrong turns.
- **Move to `docs/archive/`** behind a superseded banner: `MA_DESIGN.md`,
  `MULTI_AGENT_DESIGN.md`, `NEW_EXPERIMENTAL_SPEC.md`, and the rest of the pre-rebuild set.
- **New:** `docs/DECISIONS.md`, `docs/RESULTS.md`, `docs/STATE_OF_TRUTH.md`,
  `docs/PARAMETERS.md` (Phase 5).

## Phase 3 — Scripts and cluster jobs (30 min)

- `cluster/` for the submit scripts that reproduce REPORTED results; the rest to
  `legacy/cluster/`.
- Same triage across `scripts/` — most are one-off (`generate_kaggle_*`, `sweep_stage*`).
- `src/`, `prototypes/`, `notebooks/`, root `logs_*.txt` -> `legacy/`.

**Rule: a script stays in the main tree only if it reproduces a number we cite.**

## Phase 4 — Merge to main (20 min)

Merge with **history preserved, NOT squashed**. The commit messages carry the reasoning
behind most decisions and are a genuine asset for the thesis — squashing would discard the
audit trail we spent a week building.

Then a `README.md` that orients a cloner in one screen: what the project is, where the
single-agent and two-agent cases live, and how to reproduce one headline result end to end.

## Phase 5 — The thesis handover pack (60 min)

**The trap this phase exists to avoid.** The logs are chronological and contain retractions.
An agent reading them cold will find "the greedy oracle never clamps" and "with-bit median
0.583" and cite them. Both are withdrawn. The pack therefore needs an explicit supersession
layer sitting above the logs.

### `docs/STATE_OF_TRUTH.md`
Three sections, every entry carrying an evidence pointer:
- **Established** — claim, evidence, where the number lives
- **Retracted** — claim, why it was wrong, what replaced it. This section exists specifically
  to stop a retracted claim being resurrected by someone reading the logs forwards.
- **Open** — with what would settle each one

### `docs/DECISIONS.md`
Register: *decision · basis (citation or measurement) · status · what would overturn it*.
Covering at least: BGe score-equivalence (Geiger & Heckerman 2002), the Cooper & Yoo (1999)
interventional rule, the Robinson sink recurrence, exact sampling (Talvitie, Vuoksenmaa &
Koivisto, UAI 2019), potential-based shaping (Ng, Harada & Russell 1999), independent
learners (de Witt et al. 2020), and the CTDE exclusion as a supervisor constraint.

### `docs/PARAMETERS.md`
Every parameter with its value and whether it is **measured, derived, or asserted**. As of
2026-08-20 the asserted list includes `step_cost=0.05` (decisive and never swept), `n_int=100`
(never justified), `intervene_scale=2.0`, `prior_p=0.5`, `identify_threshold=0.7` (derived for
the single-agent MEC tie caps and inherited untested), and the MA PPO hyperparameters. This
document is what stops a viva question landing badly.

### `THEORY_NOTES.md` verification pass
Several citations are marked UNVERIFIED. They must be checked against source before a thesis
agent cites any of them.

---

## Estimated total: ~3.5 hours

## Sequencing constraint

**`docs/RESULTS.md` cannot be written until the corrected two-agent runs land.** As of
2026-08-20 every two-agent number is provisional: the reported metric could not score
confounded episodes until today's fix, so the overnight figures are unconfounded-only.
Write the STRUCTURE during consolidation, fill the numbers when the corrected seeds finish.
Shipping the handover pack with withdrawn numbers in it would defeat its purpose.

## Risks

1. **The rename (Phase 1)** is the most dangerous step — it touches ~45 test files. The tag
   plus green-suite-either-side is what makes it safe.
2. **Pruning `results/`** risks deleting evidence for a cited claim. Prune only what is
   superseded or retracted, and record what was removed and why.
3. **Deleting `legacy/ma_v1/env.py`** would silently invalidate the Phase 1 validation gate.
   Called out separately because it looks like dead code and is not.
