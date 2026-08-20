# Session state — 2026-08-20, ~14:00

Written as a checkpoint so nothing in flight is lost. Read this first when resuming.

## Deadline context

Dissertation due **8 September**. User's plan: experiments finish ~27 August, then writing
only. Supervisor (Mirco) meeting **today at 15:00**; report wanted by 14:45.

---

## RUNNING RIGHT NOW

### Local (tracked background tasks)
- `withbit_fixed` seed 0 and seed 1 — two-agent training under the **corrected** metric,
  2000 episodes, budget 8, `n_obs=1000`. Writing to `results/ma_fixed/`. Started ~13:35,
  ~26 min each, contended so slower. **These are the first two-agent runs whose numbers are
  valid on confounded episodes.**

### Myriad
- **180127 `sa_d6`** — 20 tasks, d=6 at budgets 2 and 3, 5 seeds each. Locates where the
  learned agent stops beating greedy (it wins at d=5, loses at d=7 with scarce data).
  Writes to `results/d6_exact/` on the cluster.
- **180124 `ma_cost0`** — 20 tasks, both arms at `step_cost=0`. Separates "the regime bit
  makes it learnable" from "the step cost made passing optimal". Queued behind sa_d6.
  Writes to `results/ma_stepcost/`.
- Cluster results are NOT auto-synced. Pull with `scp` from
  `~/marl_sa_fast/results/<dir>/`.
- NOTE: Myriad's checkout has a `git stash` holding earlier result files
  (`cluster results before stepcost pull`). Do not lose it.

## NOT YET DONE — the report

The supervisor report was being drafted when this checkpoint was written. Generator and
template exist and work: `scripts/ma_report.py` + `scripts/ma_report_template.html`.
An earlier version is published at
https://claude.ai/code/artifact/8470339a-df3f-4e2f-88b8-213c41f90224
but **it states the pre-correction interpretation and must be updated or withdrawn.**

Recommended framing, agreed with the user: lead with **methodology and corrections**, not
two-agent numbers. Two seeds is a sanity check, not a result.

---

## WHAT IS SOLID

- Single agent **beats the myopic oracle at d=5**, most decisively at tight budgets —
  gap_closed +1.300 at budget 2, 3/3 seeds, both `n_obs` settings. (gap_closed: 0 = random,
  1 = greedy, >1 = beats greedy.) At d=7 with scarce data it does NOT (0.529).
- **The budget cliff**: greedy-vs-random discrimination is entirely a scarcity effect —
  +0.390 at budget 2, +0.015 at budget 8, 0.000 by 16.
- **Confounding is confined to shared pairs** — proved and exhaustively verified.
- Subset DP matches frozen enumeration to **1e-10**; exact sampler matches the DP's
  partition function to **1e-13** through an independent derivation.
- GATE 1 passes (0.0394 against a *predicted* 0.0402); GATE 3 passes (never-clamp
  structurally 0.000 against 0.184).
- **529 tests pass** (plus the new reachability guards, 39 in `tests/ma/`).

## WHAT IS RETRACTED — do not cite

- "The greedy oracle never clamps." Artefact: the old baseline's action list was hardcoded
  to VARY only. Real claim: myopic EIG is INDIFFERENT between the modes, so the tie-break
  decides; measured 0.526.
- **Every two-agent training number from 2026-08-19** — with-bit median 0.583, no-bit 0.055,
  the paired +0.013 over random. All are unconfounded-only, because the reported metric
  scored EXACTLY 0.000 on confounded episodes until today's fix.
- GATE 3 headroom "+0.393" — measured against a synchronised random baseline. Corrected
  value +0.184.

## OPEN / NEXT

1. Corrected two-agent runs at scale (≥20 seeds) — only 2 seeds in flight.
2. **GATE 2 unresolved.** Greedy is now equal to random, not worse. Live hypothesis: two
   greedy agents computing the same objective over overlapping authority collide (0.352 of
   rounds against random's 0.227).
3. **Parameter audit** — `step_cost=0.05` is decisive and never swept; `n_int=100` never
   justified; `intervene_scale`, `prior_p`, `identify_threshold` all inherited untested.
   See `docs/CONSOLIDATION_PLAN.md` Phase 5.
4. **Consolidation** — plan saved at `docs/CONSOLIDATION_PLAN.md`, to run after the report.
5. Scaling ladder: `(2,2,3)` then `(2,2,5)`. `|X|` is the binding axis (`3^(|X| choose 2)`).

## THE BUG FAMILY THAT COST THE MOST TIME

Four measurement bugs in two days, three of them the SAME confusion: a hypothesis is
(DAG H, confounding set P) with P's edges present in H, so the causal claim is `H \ P`, not
`H`. Asking about `H` gives a confounded agent exactly 0.000, always.

The guard is `tests/ma/test_metric_reachability.py`: **a metric can be well-formed and
still be unearnable, and the two look identical in a results table.** Every metric, in every
regime it claims to cover, needs an explicit "can this ever be earned here" case.
