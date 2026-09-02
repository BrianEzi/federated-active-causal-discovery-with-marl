# Agent C session state — 2 September 2026

Written before a context compaction. Everything an agent C successor needs in order to
continue without re-deriving it. Companions: `docs/AGENT_C_METHODOLOGY_BRIEF.md` (the brief,
with four appended sections from agent A), `thesis/WRITING_GUIDELINES.md` (the style contract),
`thesis_results/CLAIMS.md` (the number source).

**Role.** Agent C owns the thesis prose: `2 Background`, `3 Methodology`, `4 Results`,
`1 Introduction` (last), and `Report.tex` structurally. Agent A owns tables, figures, number
verification and `thesis_results/`. Agent B runs training fleets.

---

## 1. THE RULES BRIAN SET THIS SESSION — these are binding

1. **`git pull` before any work.** Standing instruction. He edits on Overleaf and agent A
   commits concurrently; three collisions have already happened.
2. **Prose style is canonised** in `thesis/WRITING_GUIDELINES.md` under "Prose style". Two
   rules generate the rest: *say what we did and why, never what we tried*; and *structure
   follows the reader, not the build history*. Eleven banned constructions with greppable
   signatures, and markup budgets per 1000 words. A draft that breaks them gets sent back.
3. **Chapter 4 is bullets, tables and figures. No prose.** His words: "wholly too verbose and
   lacking in a clear narrative or brevity". He is deciding what goes where; treat the current
   content as a holding structure. Prose recoverable from `thesis` commit `2eb49ac`.
4. **Subsections, not subsubsections**, in both Ch2 and Ch3.
5. **The thesis must be self-contained.** Definitions are restored where something downstream
   depends on them. This overrode an earlier instinct to cut them as textbook.
6. **BGe and the Robinson sink recurrence stay cut.** Neither is behind a reported result.
7. **Introduction is written last**, after Results and Discussion. Its contributions list is
   still the stale one and he knows.
8. **Anything he thinks is irrelevant goes to the appendix**, not deleted. Applies to the
   eight equations added to Ch3 §Learning and §Evaluation.

## 2. WHERE THE CHAPTERS STAND

`thesis/` is a separate git repo, Overleaf remote, branch `main`, HEAD `9ce498f`.

| chapter | words | state |
|---|---:|---|
| 1 Introduction | 1,115 | Brian's four new RQs are in. **Contributions list is STALE** — still cites Clamp/Vary, BGe, "82--91% of clamps", 3.5x, the 1-bit regime signal. Deferred to last, deliberately. |
| 2 Background | 4,117 | Restructured gap-first, 24 subsections, 8 definitions. Done unless he asks for more. |
| 3 Methodology | 4,449 | Restructured, 8 new equations + 2 algorithms. Done unless he asks for more. |
| 4 Results | 3,228 | Bullets + 10 tables + 6 figures. Holding structure pending his placement. |
| 5 Discussion | 883 | **A SCAFFOLD AGAINST THE SUPERSEDED RQs.** Subsections are "RQ2 Budget Allocation and Altruistic Clamping" and "RQ3 Coordination Protocols and Signaling"; comments cite the 82--91% figure, `[U14]`, `identify_threshold`, the K=2 theorem. All retired. Only `sec:disc_validity` holds live content (the supplied-skeleton limitation). **Needs a full rewrite against RQ1--RQ4.** |

Chapter 3's structure, which took three attempts to settle:
`Problem Setting` -> `Environment Dynamics` -> `Attribution` -> `Learning` -> `Evaluation` ->
`Experiments`. Problem first, then how a step works, then training, then evaluation, then what
was run. Do not reorder it by how the work happened.

## 3. NUMBER SOURCES — get this right or repeat three retractions

**`thesis_results/CLAIMS.md` is authoritative. It supersedes
`docs/RESULTS_LEDGER_2026_09_01.md`.** The ledger is hand-maintained and has been wrong twice
in ways that reached a draft; CLAIMS.md is derived by `scripts/build_claims.py` from
`thesis_results/`. Read its **MUST NOT** lines before writing any sentence.

The MUST NOTs, as of this session:

* **C1** no ratio of means at `k_v=30`; two seeds significant, one indistinguishable.
* **C2** no agent-count reversal until K=10 lands. At K=5 it is one seed (1.65 all seeds, 0.25
  without seed 2). Do not claim the myopic arm improves along the axis — per-pair SHD divides
  by `global_pairs`, which runs 117 to 525.
* **C3** do not write the reward-alignment asymmetry. RETRACTED
  (`docs/FINDINGS_PAIR_CLASS_2026_09_02.md`): shared-shared error is 0.00000 for both learned
  and myopic over 90,000 pair observations.
* **C4** no direction for the federation cost; mean and median disagree. Quote the paired
  figure, -0.00017 +/- 0.00023 over six seeds.
* **C5** do not substitute the 12,000-episode retrains into any sweep table.

`WRITING_GUIDELINES.md` was corrected: it used to point every number at
`docs/STATE_OF_TRUTH.md`, which is dated 22 August and predates the constraint engine, the
factored belief, attribution, the partial oracle and the ladder. **Do not use that file.**

## 4. CORRECTIONS I MADE TO THE THESIS'S OWN CLAIMS

Each was a claim the thesis asserted and the code or data contradicted. Verify before writing;
this is the highest-yield thing agent C does.

1. **The agents do NOT start from the MEC.** They start from the **skeleton** with every
   orientation open. `cb/factored.py:300`: observational orientation rules are "deliberately
   NOT" applied, so the backend reports what pairwise interventional evidence alone can prove.
   `oracle_obs_structure` is False and requires the constraint backend, unreachable in the
   reported runs. Brian believed MEC; the correct framing is stronger in both directions,
   since we assume less and recover every orientation interventionally.
   Written at `sec:meth_skeleton`, priced at `sec:disc_validity`.
2. **The oracle answers ancestry, not conditional independence.** `_apply_ancestry` asks, having
   intervened on x, whether x is an ancestor of each y. The gloss said "conditional-independence
   query", which is the standard notion and not what this engine uses. Ch2 now defines the
   oracle generally; Ch3 `sec:meth_regimes` states the query first, then the three regimes.
3. **Undetermined pairs DO score a full SHD error.** The chapter said error accrues "only where
   the pooled belief committed to a mark other than the true one". False.
   `ma/evaluate.py::pooled_global_belief` sets hard = 1 whenever the pooled set is not exactly
   the true mark. Verified empirically over 1,875 pooled pairs: 1,827 resolved-and-agreeing at
   0, **4 undetermined-with-agreeing-truth at 1**, 44 mark-disagreement at 0 via a per-site
   fallback. The metric penalises indecision and cannot be gamed by abstaining, which is better
   than what I had described. **The name "SHD on committed marks" is now misleading** — flagged,
   not renamed, because it is used in Ch4, the glossary and CLAIMS.md.
4. **`sampled: true`, not argmax.** `PLAN_2026_08_28` F4 recommended argmax as primary.
   `results/ckpt/*.json` and `results/global_shd_paired_{argmax,sampled}.json` show argmax
   LOSES the result at all three cells and blows up 275x at k12b500 (45 SE). Mechanism is the
   determined fraction: 0.807 argmax vs 0.971 sampled, baselines unmoved — a deterministic
   policy cannot leave a state whose argmax action stopped being informative. **F4 is refuted.**
5. **The budget rule.** `bNNN` in a cell name is `beta*100`, NOT the budget: `k20s50n04b150`
   has budget 75. Rule is `ceil(beta * c(k_v) * k_v * K)` from `scripts/sweep.py:105`, with
   `c` interpolated between measured anchors (4, 0.757) and (30, 0.542) and clamped outside.
   Verified on five cells.
6. **Theorem 3.1 demoted.** Brian: "it's one of our assumptions... not rocket science." It is
   now three sentences under `Assumptions`, following from no-cross-private-edges plus
   no-exogenous-latent-confounding. No theorem or proof environment.

## 5. TERMINOLOGY — agent A's audit, applied in prose only, never to code

| prose | code |
|---|---|
| atomic intervention, $do(V=c)$ | `clamp` |
| randomised intervention, $do(V \sim \mathcal{N})$ | `vary` |
| determined / undetermined | `settled` / `unsure` |
| committed mark; decision threshold $\tau$ | `claim`; `claim_bar` |
| joint / per-window recovery rate | `success`; `window_rate` |
| partial oracle; answer rate $\rho$ | `evidence_power` |
| public good, public-goods problem | "altruism" |
| local variable set $\mathbf{V}_k$ (formal); window (informal) | `window` |

`oracle` is KEPT — standard constraint-based vocabulary. `Glossary.tex` carries the full
mapping; Brian deleted its inherited CV-template boilerplate.

**Symbols:** $c(k_v)$ is the required-cover fraction, $\rho$ is the partial oracle's answer
rate. **$\sigma$ is overloaded three ways** — SCM noise scale, intervention variance, contended
fraction — flagged and unresolved.

## 6. OUTSTANDING

Ranked.

1. **Chapter 5 Discussion.** Full rewrite against RQ1--RQ4. Currently a scaffold citing four
   retired claims. Largest single piece of work left.
2. **Chapter 4 placement.** Brian is deciding what goes where. Two sections are placeholders
   rather than stripped: `sec:res_transfer` (RQ2 part 3 awaits the answer-rate fleet) and
   `sec:res_negative` (tabulate ledger section 6, and add the two retractions from this
   session — reward alignment, and the K=5 reversal).
3. **Chapter 1 contributions list.** Stale, deferred to last by his instruction.
4. **Bibliography.** 123 entries. A 45-entry removal of uncited references was done and then
   REVERTED, because it was coupled to the Ch2 restructure Brian rolled back. Recompute the
   orphan list against the current Ch2 before removing anything, and ask first.
5. **Erdos-Renyi/Gilbert citation nuance.** The implemented generator is $G(n,p)$ (Gilbert
   1959), not ER's $G(n,M)$. Both cited, both annotated. Confirmed against `ma/topology.py`.

## 7. TRAPS

* **`scripts/build_results_skeleton.py` OVERWRITES Chapter 4 entirely, prose included.** Do not
  run it. Patch tables in place or copy the prose out first.
* **`pdflatex` and `bibtex` are NOT installed on this machine.** The mandated full build has
  never been run locally. Static checks only: citations resolve in both `references.bib` and
  `annotated_bibliography.md`, no `\citet` (forbidden — `ieeetr.bst` cannot populate it), every
  `\ref` defined, no duplicate labels, environments and braces balanced, British spellings.
  The real build happens on Overleaf.
* **`references.bib` and `annotated_bibliography.md` must stay 1:1.** Verify both directions
  after any bibliography edit.
* **Do not check numbers against the ledger.** See section 3.
* **Verify before launching compute.** I re-ran `shd_by_pair_class.py` at 60 episodes when
  agent A had already run it at 200; 60 was the underpowered configuration that produced the
  retracted claim in the first place. Check `thesis_results/` first.
* **Agent A edits Chapter 4 concurrently** despite having said prose is agent C's. Pull, and
  check `git diff` before assuming your last write is still there.

## 8. WHAT IS DONE AND SHOULD NOT BE REDONE

* Ch2 restructured gap-first with 8 definitions, including SHD (Tsamardinos, Brown & Aliferis,
  *Machine Learning* 65(1):31--78, 2006, verified by search) and latent projection — both were
  the thesis's primary metric and its central structural object, and neither had been defined.
* Ch3 restructured; 8 equations and 2 algorithms added across Learning and Evaluation, which
  had zero between them.
* The version-space algorithm moved from Ch2 to Ch3 as `alg:version_space_update`.
* `Report.tex` has `\input{Appendix}`; `app:excluded` resolves.
* Ch4 gained 6 figure floats. It previously referenced `fig:sweep_grid` with **no float at
  all**, so the backbone figure would have built as `??`. Every `\ref` in the document now
  resolves.
* `WRITING_GUIDELINES.md`: prose-style section canonised; attribution bullet updated to RQ4
  scope at agent A's request; the stale STATE_OF_TRUTH pointer corrected.
* `thesis/WRITING_CRITIQUE.md` holds ten annotated excerpts and the seventeen named tics.
  Brian has been editing it; leave it to him.

---

## Structural contract — 2 Sep, from Brian. Read before touching chapter structure.

`docs/THESIS_STRUCTURE_CONTRACT.md` is now the governing structural document and
overrides any earlier structural note in `docs/`, including anything in this file
above this line.

The short version:

- **One RQ drives one Results section.** Chapter 4 goes from eight sections organised
  by experiment to four organised by question, plus negative results last. The
  migration table is §3 of the contract.
- **Methodology owns all setup exposition.** Nothing in Results describes how an
  experiment was configured — it cites `sec:meth_ladder` / `sec:meth_eval` and goes
  straight to the number. §4.1 Measurement Protocol is cut entirely.
- **Chapter 5 and much of the Chapter 4 analysis are Brian's exclusively.** Do not
  draft them. If you think an interpretation is warranted, write it to a
  `docs/FINDINGS_*.md` note instead.
- **RQ4 (attribution) is narrow by design** — attribution is possible, under these
  conditions, leading to future work. Nothing more; no agent was trained on the
  attribution objective.
- Two repairs in `1 Introduction.tex`: RQ1's evidence-spectrum clause belongs wholly
  to RQ2, and the Contributions list is stale (BGe, two agents, 3.5x, 82--91% clamps
  — none of it current). Rewrite against `thesis_results/CLAIMS.md`.

Numbers come from `CLAIMS.md` and nowhere else, boundaries included.
