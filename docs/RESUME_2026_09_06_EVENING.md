# Resume point — 6 Sep evening. Read this first after a compaction.

**Real deadline: Wed 9 Sep 16:00.** Brian's target: finished Monday night (7th).
Supersedes docs/RESUME_2026_09_03_0900.md and docs/ENDGAME_2026_09_07.md on scheduling.

## Division of labour, current
* **Agent C drafts Chapter 4.** Brief: `docs/AGENT_C_CH4_HANDOVER.md`. DO NOT edit
  `thesis/4 Results and Analysis.tex`. The ten `% >>> Analysis (Brian)` markers stay empty.
* **Chapters 5 and 6 are Brian's alone.** Material is staged as comments; never prose.
* **Agent A (this session):** measurement, figures, CLAIMS, gates, registries, coordination.
* **Agent B:** training fleets + Myriad. Inbox `docs/AGENT_B_INBOX.md`, reply every tick.

## The agreed Chapter 4 structure (settled 6 Sep)
4.1: sweep (4.1.1, unchanged) / eps-greedy control (4.1.2) / constrained budget (4.1.3) /
     generator control (4.1.4). Training Budget and Where-the-Advantage-Sits DEMOTED to appendix.
4.2: answer-rate grid (4.2.1) / fixed-policy (4.2.2) / robustness 2x2 (4.2.3) /
     sample-size + disclosure (4.2.4). Finite-Sample Evidence CUT (survives as app:evidence_cost).
     CRITICAL: the cut section motivated the partial oracle. Two sentences of that must move
     into the 4.2 section intro or 4.2.1 opens unmotivated.
4.3: unchanged. Abstract DELETED until the end (Brian's call; Report.tex input commented).

## Findings established 5-6 Sep (all in CLAIMS.md, all recomputed)
* **C10 constrained budget.** Ten-point beta axis 0.5-5.0 at k12. Learned margin over myopic
  PEAKS at beta=0.6 (+0.472), decays to +0.033 at beta=5. oracle_cover is BEATEN below
  beta~0.8 (it is per-window optimal, uncoordinated by design). At beta=0.5 avoidable
  duplication 0.017 learned vs 0.325 oracle_cover. **This is the coordination evidence.**
  MUST NOT say "beats the optimum" -- a coordinated optimum does not exist in this codebase.
  4.3.2's convention claim FLIPS at beta=0.9 and needs a qualifier.
* **C11 robustness 2x2.** noise{gaussian,uniform,t3} x mechanism{linear,tanh}, advantage
  holds 3/3 seeds in every corner, ratios 1.52x-1.70x. Uses the rho=0.5 k=8 fleet, NOT sweep
  policies. Robust because BLIND: the policy observes belief features, never raw data.
* **C9 eps-greedy**, all 20 cells: learned ahead 52/60 seeds. Four cells against us, and two
  are the two the chapter already names as exceptions.
* **C4a ladder equivalence bound**, 12 seeds. Both checkpoint conventions EQUIVALENT (TOST
  p=0.016, 0.023) but they DISAGREE IN SIGN. Bound is LOOSE (76-89% of the myopic gap) and
  says so. More seeds cannot fix it (48 for half, 192 for a quarter); the cell is saturated.
  Fix is a harder cell: beta 0.5-0.7.
* **C8a disclosure.** The n_int U-curve is contamination from undisclosed partner
  interventions; one bit restores monotonicity and the learned lead.

## THE METRIC FINDING (6 Sep, Brian's challenge — likely Discussion material)
`global_hard_shd` charges an UNDETERMINED pair as a full error, identically to a false one.
Measured over 157,680 pooled pair instances (4 arms x 3 budgets): **zero pairs ever resolved
to a wrong mark.** The reported quantity is residual ambiguity, not committed error.
Seven alternative metrics were built and tested (`scripts/metric_bakeoff.py`,
`results/metric_bakeoff.json`): resolution, error rate, ambiguity, excess marks, severity
composite, soft SHD, disagreement. **All seven invert against joint recovery at beta=1.5**
(myopic leaves fewer pairs undetermined, 0.00084 vs 0.00221, while winning fewer episodes,
0.933 vs 0.983). They are monotone transforms of one quantity, so no redefinition reconciles
them. Conclusion: two objectives, not one broken metric. Methodology now states the property
deductively; the measurement stays in Ch4/Ch5. Staged comment in `5 Discussion.tex`.

## Gates — run all three before any push
    .venv/bin/python scripts/check_mustnots.py    # forbidden phrases, incl. NOTEBOOK markdown
    .venv/bin/python scripts/check_style.py       # prose guidelines (NEW, 6 Sep)
    .venv/bin/python scripts/mark_provenance.py --check
check_style currently reports 40 issues across chapters (Ch6 clean); these are pre-existing
prose defects, not regressions. Ch2 (14) and Ch1 (8) are the worst.

## Standing rules that cost us when broken
1. **Never quote a run's `global_hard_shd`.** It is the run's own last-update self-eval.
   All SHD numbers come from `scripts/global_shd_paired.py`.
2. **Config plumbing is a whitelist.** `env_from_config` silently dropped `skeleton_source`
   on 5 Sep and voided a measurement. Every override now asserts after construction.
3. **Ask before implementing anything ambiguous** (memory: clarify-before-implementing).
   "Train with no skeleton" was misread and cost two training rounds.
4. **Don't push mid-discussion** (memory: discuss-first-push-later).
5. **Never exceed ~6 heavy processes**, renice 20. Nine parallel trainings took load to 244.
6. Methodology carries design constants, never measured outcomes (WRITING_GUIDELINES, 6 Sep).

## Open / in flight
* Agent B: rho12 fleet (21 runs, k=12 answer-rate grid at the PRINCIPAL cell, sweep-cell
  settings deliberately). I pipeline evaluations behind it with --baseline_from.
* Myriad PROVEN end to end; not worth it for this week, decisive for post-submission.
* Ceiling: unblocked (Brian's definition needs no Topology change) but parked; the cell to
  use is beta 0.5-0.7.
* Post-submission programme: joint adjacency+orientation version space; rank/kernel tests to
  exploit non-Gaussianity; JCI attribution; no-skeleton at 36k; k_v past 30.
