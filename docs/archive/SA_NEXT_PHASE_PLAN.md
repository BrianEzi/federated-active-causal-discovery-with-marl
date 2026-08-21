# Next phase — plan of record

Agreed 2026-08-15, after the overnight run established that the agent beats the greedy
information-gain oracle at d=4, d=5 and d=6 with a permutation-equivariant per-node scorer.

This file is the plan of record. Update it as phases complete; do not let it drift from
what is actually running.

---

## Verified facts this plan depends on

- `wandb==0.17.9` installs into `~/envs/sa_env` and does **not** break numpy 1.26.4 /
  scipy 1.13.1 / torch 2.6.0+cpu. Newer wandb (0.28.x) has no wheel here and needs Go to
  build from source — **keep the pin**.
- **Myriad compute nodes have no outbound internet.** A `wandb.init()` in online mode will
  hang. Confirmed by submitting a curl test to a compute node: no HTTP status returned.
  Therefore: offline mode on the node, sync afterwards from the login node.
- GATE 1 passes at d=4 (n_obs 1000+), d=5 (n_obs 5000+), d=6 (n_obs 20000+).
- d=6 costs ~4.7h/seed at n_obs=1000. Cost at n_obs=20000 is **unmeasured** and must be
  measured before committing a long job.

---

## Phase 0 — Instrumentation (no experiments yet)

Everything here is infrastructure. It must be built and tested before any compute is spent,
because its whole purpose is to make the experiments interpretable.

### W1 · WandB, offline and non-fatal

- New module `sa/tracking.py`. A thin wrapper, not a dependency the rest of the code knows
  about.
- **Offline by default** (`WANDB_MODE=offline`), because compute nodes have no internet.
- **Never fatal**: every wandb call wrapped so an import failure, a disk problem or a bad
  config degrades to a no-op with a printed warning. The JSON result files stay the source
  of truth — wandb is a second view, never the record.
- Logs per training update: entropy, solve rate, mean episode length, policy/value loss.
  Logs at the end: the full metric set and the pass/fail verdict.
- Grouping: `group=<tag>`, `job_type=<arch>`, `name=<tag>_s<seed>`, tags for d / n_obs /
  arch, so 200+ runs stay navigable.
- `scripts/sync_wandb.py` — run on the **login node** after jobs finish, walks the offline
  run directories and syncs them.
- Flag: `--wandb_project` (default off; passing it enables logging).

### G1–G5 · Gates and canaries

All five live in `sa/gates.py` unless noted, are checked automatically, and are **recorded
in the result JSON** so a number can never be read without its checks.

| id | check | why |
|---|---|---|
| G1 | **Entropy canary** — warn when final entropy > 65% of ln(n_actions) | Would have flagged all 61 overnight failures immediately; free, already logged |
| G2 | **Anchor assertion** — random must score exactly 0.0 and greedy exactly 1.0 | Silent metric corruption already happened once (read 0.233 / 1.067 from stale RNG state) |
| G3 | **Informative-fraction floor** — refuse to report oracle agreement when too few steps were informative | The exact bug behind the retracted 99.4% figure |
| G4 | **Seed-spread warning** — flag when the seed range exceeds 0.5 gap-closed | Would have caught `pernode_best` being unstable rather than good |
| G5 | **GATE 1 precondition** — already implemented; extend to record in every result file | Checked once at d=3 and silently stopped holding at d>=5 |

Each gate needs a test asserting it **fires when it should**. A gate that only ever passes
is decoration — that is the standard the GATE 1 precondition tests already set.

### Definition of done for Phase 0

- Full test suite green (currently 158 tests).
- One short local run with `--wandb_project` set produces an offline run directory and a
  result JSON containing all five gate records.
- Venv on Myriad still imports numpy/scipy/torch/wandb.

---

## Phase 1 — Depth probe (cheap, and it decides Phase 2)

**The question.** The per-node scorer does **one** round of neighbour aggregation. The
oracle's score depends on each node's *descendants* — reachability — which is inherently
multi-hop. Probe accuracy topped out at 0.89 rather than ~1.0, and this is the leading
explanation.

**P1.** Extend `PerNodeActorCritic` with a `layers` parameter: repeat the message-passing
round `k` times, so node embeddings can carry information from `k` hops away. `layers=1`
must reproduce the current network exactly — assert this in a test, since the whole d=4/5/6
result rests on that architecture.

**Run supervised, not RL**: probe accuracy for `layers ∈ {1, 2, 3}` at d=4 and d=5, across
data sizes {300, 1000, 3000, 9000}. 24 tasks, minutes each.

**Decision rule, fixed in advance:** if depth 2 or 3 beats depth 1 by more than 0.03
accuracy at matched data size on both d=4 and d=5, carry the best depth into Phase 2's RL
runs. Otherwise keep depth 1 and record that the ceiling is *not* about multi-hop
reachability — which would itself be a finding worth having.

---

## Phase 2 — Experiments (parallel on Myriad)

All at `n_obs` where GATE 1 passes. All with WandB offline logging and all five gates.

### E1 · Lever sweep, per-node architecture, n_obs=5000, d=5
34 configurations × 3 seeds. The levers characterised on **the network we will actually
use**. This is the sweep whose conclusions get cited.

### E2 · Lever sweep, flat architecture, n_obs=5000, d=5
Same 34 configurations, old network. Requested explicitly as the comparison: it separates
"this lever matters for the task" from "this lever mattered because the network was
broken". E1 vs E2 on the same axes is a result in itself.

### E3 · Depth in RL
Best depth from Phase 1 vs depth 1, d=5, n_obs=5000, 5 seeds each. Only run if Phase 1's
decision rule fires; otherwise skipped and recorded as skipped.

### E4 · Gate-valid d=6
n_obs=20000, 3 seeds, per-node architecture.
**Measure first.** Cost at this sample count is unknown; d=6 took 4.7h/seed at n_obs=1000
and BGe scoring scales with sample count. Submit a short timing probe, then size the
walltime from the measurement rather than from a guess. (Sizing d=6 from a small sample is
a mistake already made once.)

---

## Phase 3 — Analysis and documentation

- Regenerate `results/all_runs.csv`, the report, and the notebook from the new data.
- Append to `docs/SA_EXPERIMENT_LOG.md` at **each phase boundary**, not only at the end —
  including nulls, skipped arms and self-corrections.
- Update the published artifact at its existing URL:
  `https://claude.ai/code/artifact/0cfb3c99-e7a4-43fe-b51a-c67e60a7a0ad`
- Sync the wandb offline runs from the login node.

---

## Ordering and why

Phase 0 before anything, because instrumentation added after the fact cannot explain runs
that already happened. Phase 1 before Phase 2, because it is minutes of compute that decides
what Phase 2 should be — the pattern that worked overnight, where a supervised probe
answered in minutes what 61 RL configurations could not.

E1, E2 and E4 are independent and run concurrently. E3 waits on Phase 1.

## Known risks

- **WandB may still misbehave under 34 concurrent offline writers.** Non-fatal by
  construction; if it causes trouble, disable and proceed — the JSON files are the record.
- **E4 timing is an extrapolation.** Now projected at ~2.7h/seed (see below), well inside
  the 10h walltime. The timing probe stays anyway: this extrapolates laptop CPU to Myriad,
  and the last two d=6 runtime predictions were both wrong.

## Superseded risk — hot path optimised 2026-08-15

Two risks listed above originally have been measured away. Full detail in
`docs/SA_EXPERIMENT_LOG.md`; the short version:

- **"E1/E2 will be ~2x slower at n_obs=5000" was wrong, and backwards.** The sample-count
  dependence was an implementation artefact: BGe needs only (n, means, centred scatter),
  and subset statistics are submatrices of the full ones, but the old code re-read all n
  rows once per (node, parent-set) pair — 160 passes per posterior at d=5. Hoisted to one
  pass per node, d=5 is now flat in n_obs (49.0 / 48.0 / 49.3 ms at 1000 / 5000 / 20000).
  E1/E2 at n_obs=5000 run at 42.1 ms/step, *faster* than the overnight runs they extend
  (d=5, n_obs=1000, 57.6 ms/step).
- **"E4 may be too slow" was diagnosed to the wrong cause.** d=6's cost was never sample
  count; it was two n-independent reductions over 3.78M enumerated DAGs (gather 384 ms,
  edge marginals 517 ms, against a 90 ms score table). Both replaced with exact
  equivalents. d=6 at n_obs=20000: 1850.6 → 845.7 ms/step, i.e. ~2.7h/seed at **20x** the
  sample count of the run that cost 4.7h.

All four changes are exact restatements, pinned by `tests/test_optimisations.py` (23 tests)
against the implementations they replaced. Full suite: 268 passed.
