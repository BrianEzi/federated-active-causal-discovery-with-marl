# Agent C session state — 4 September 2026 (supersedes AGENT_C_STATE_2026_09_02.md)

Written before a context compaction. Companions: docs/THESIS_STRUCTURE_CONTRACT.md (governing,
updated 3 Sep), thesis/WRITING_GUIDELINES.md (style + caption rule), thesis/FIGURE_GUIDELINES.md
(figure sizing; agent A's), thesis/FIGURE_PLAN.md (all non-attribution figures DONE),
thesis_results/CLAIMS.md (numbers; C4 renamed to "pooled" by agent A via build_claims).

## 0. OPERATIONAL FACTS THAT MUST NOT BE RELEARNED THE HARD WAY

* **Agent A works in THIS SAME CHECKOUT, concurrently.** Their commits appear in this repo's
  reflog between mine. Always `git pull` both repos first; before editing a file, re-read it;
  after odd assert failures, suspect a concurrent edit, not corruption. Their in-flight work
  once referenced thesis figures (ladder.pdf/coordination.pdf) that did not exist -- my
  regeneration filled the gap; expect races like that.
* **Local rendering works**: tectonic (brew) + sips + pypdf (in .venv). Standalone TikZ:
  scratchpad/fig/; whole-chapter: scratchpad/ch4/driver.tex (PackagesX strips
  inputenc/fontenc/tipa/pstricks; body file with absolute figure paths). NEVER ship a figure
  or layout change unrendered. See memory local-tikz-render.
* Shell cwd resets between Bash calls unpredictably -- use absolute paths in scripts;
  quoted heredocs for LaTeX-bearing Python.
* Overleaf remote in thesis/, branch main; outer repo branch explore/constraint-based;
  push with GIT_ASKPASS=/bin/true.

## 1. BINDING DECISIONS SINCE THE 2 SEP STATE DOC (all Brian's)

1. **Three research questions.** Attribution demoted: lives ONLY in app:attribution
   (self-contained; generator scripts/build_appendix.py updated to emit it); appears nowhere
   else. RQ4 removed from Ch1; contributions and abstract cleaned.
2. **Appendices are COMMENTED OUT of the build** (Report.tex: Appendix, Results Tables,
   Negative Results) -- Brian unhappy with them, decision pending. Every live reference into
   them was reworded, with the original held in a % comment beside it for restoration.
   The abstract's nineteen-claims paragraph carries a comment that it needs the negative
   appendix restored.
3. **Ch4 shape**: three RQ sections; per subsection = one experiment paragraph + figure(s)
   (+ table only if very relevant) + "% >>> Analysis (Brian)" slot + commented bullets with
   verified numbers. Claim sentences (one per data-complete subsection) are in, sourced from
   CLAIMS.md with boundaries.
4. **Negative/withdrawn results**: condensed to app:negative (thesis/Negative Results.tex),
   on the chopping block but not deleted.
5. **Ch5**: two sections only -- The Results in Conjunction (Brian's, empty, comment prompts
   from CLAIMS) + Limitations (supplied-skeleton para + wrapped cost table; further
   limitations listed in comments). 5.2.1 is ONE paragraph by instruction.
6. **Captions at most two printed lines** (~190 chars); details go into a subsection
   paragraph. Canonised in WRITING_GUIDELINES.md. Fig 4.8 (ladder) caption is ~3 lines
   (agent A's) -- flagged, Brian may want it trimmed.
7. **Figure guidelines** (agent A's doc): author at print size, include with NO width
   argument (natural size); 5.40/3.60/2.70in menu. KNOWN DISCREPANCY: Report.tex geometry
   makes textwidth 6.25in, not the doc's assumed 5.40in; agent A's erratum accepts
   under-filling. My TikZ figures all fit 390pt so they are safe either way.
8. **"Pooled", not "centralised"**, for ladder arm E everywhere (prose, figures, CLAIMS C4).
   RQ3's §4.3 restructured: fig:ladder under 4.3.1 (fed vs pooled + paired panel),
   fig:coordination under 4.3.2 (random / myopic fixed partition / myopic uncoordinated),
   both defined in prose. **PENDING BRIAN'S GO: the true single-controller ceiling** --
   K=1, d=k_v, myopic-global needs no training, learned-global = 3 seeds x 12k episodes;
   spec in AGENT_A_INBOX 3 Sep.
9. **Float discipline**: \FloatBarrier at every Ch4 (sub)section; float-packing params in
   Packages.tex; [!htbp]. Ch4 renders ~12-14pp, no overflow, no drift (re-check after edits).

## 2. WORK COMPLETED THIS SESSION (do not redo)

* **Chronology pass Ch1->Ch3** (Brian-directed): n_int/oracle/sampled/sweep forward refs
  fixed by moving the intervention-mode grounds into 3.2.8; "mark" defined at first use
  (now in 2.1.3 after Brian's critique moved that paragraph out of 2.1.2 -- it also lost
  its premature "agents"); "window" anchored to local variable set (3.1.3) and glossed in
  RQ1; "mark marginals" defined at use (share of surviving marks, verified vs
  edge_marginals); "arm" introduced in 3.4.2; recovery defined at the reward; R-hat and tau
  introduced (tau = update's share of all returns seen -- Welford, NOT a fixed rate);
  PPO/GAE/MAPPO/IOD expanded; step-4 signpost; generator control added to 3.5;
  Ch2 metric description now matches eq:shd (unresolved pair = full error);
  "three-part criterion" and "confinement theorem" removed from Ch1/abstract (no referents).
* **3.2.8 tests fully specified** (verified in cb/citest.py): two-regime contrast excluding
  third-variable rows; Welch t (mean), Brown-Forsythe (variance -- the firing channel),
  Pearson on assigned values (randomisation); alpha=0.001; MIN_ROWS=20; power gate: pair
  powered iff variance test at alpha detects 1-r^2 shrinkage with prob 0.8. **Lineage
  named**: NOT FCI -- new subsection 2.5.4 sec:evidence_background (Fisher-z/PC/FCI for the
  skeleton; Tian-Pearl tian2001changes regime comparison; cooper1999causal pooling;
  fisher1935design randomisation; welch1947generalization; brown1974robust; Gaussian
  sufficiency = why mean+variance are exhaustive). Three NEW bib entries + annotations
  (files must stay 1:1).
* **Figures, all rendered before commit**: fig:mec (simplified by Brian: class+CPDAG only),
  fig:intervention (NEW, 2.1.3: the three class members under do(Y), severed parents grey-x,
  orange intervened, teal responders -- completes fig:mec's story), fig:latent_projection,
  fig:partition, fig:turns, fig:env (single panel, blue = agent 1's view + induced
  bidirected mark), fig:round, fig:vspace, fig:metric (three cases incl. undetermined=full
  error), fig:policy_net (coloured lanes), plus agent A's matplotlib suite (window_budget
  replaced figs 4.3-4.5; pair_class is now a table in-chapter).
* **eq:ppo split across two lines (multline)**; **eq 2.4 EIG variables defined** (last edit).
* **Review-readiness pass**: Ch1 restyled (motivation/problem formulation), contributions
  corrected against withdrawn claims (crossover bullet -> 16/18-at-convergence; C4 no
  direction; new C6 bullet), abstract pendings filled from C4/C6, Ch6 rebuilt as
  Answers + Future Work scaffold, repo path moved to comment, 2.2.3 CORE/CAASL beat added.
* **Ch4 writing pass**: no em dashes, no banned constructions; stale credit claim replaced
  by 15x/13x (both arms degrade; no ordering supported -- k12 panel differs, see fig).
* 5.2.1 one paragraph + p-column table (renders inside margins).

## 3. CURRENT NUMBERS DISCIPLINE

CLAIMS.md as of 3 Sep: C1 (k30 = ONE seed of three separates; k8 ambiguous-by-seed; axis
bounded at kv<=12 for the 4k trend), C2 (no agent-count reversal at 12k; K=2 is a
checkpoint-selection failure 0.07372-vs-0.00000, one seed), C3/C3a (rewarded-class 3.9x cut,
unrewarded 11/27,000 = 0.04% both budgets; magnitude MUST accompany), C4 (pooled;
-0.00000 +/- 0.00037; 0/6 seeds; no direction), C5 (exclusions undertrained; never
substitute), C6 (rho<=0.9: 15/15 beyond 2SE; rho>0.9: 0/6; boundary at 2SE not sign),
C7 (2/18 at 4k -> 16/18 at 12k, 14 flips, 1 tie -- NOT 15). Sweep is 18 cells at 4k + 2 at
12k (window-budget confound; within-budget trend only to kv=12).

## 4. OUTSTANDING, RANKED

1. Brian's analysis passes in Ch4 (slots marked), Ch5 synthesis, Ch6 prose. His alone.
2. Appendix decision (restore/rework/drop); then un-comment refs held beside their rewrites.
3. Single-controller ceiling: awaiting Brian's go (then agent A/B run it).
4. Fig 4.8 caption ~3 lines; trim if Brian confirms.
5. Introduction final pass once Results/Discussion settle (deferred by Brian's standing order).
6. Bibliography orphan sweep still not redone against current Ch2 (ask first).
