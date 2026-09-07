# Agent C state — 8 Sep 2026 (supersedes AGENT_C_STATE_2026_09_04.md)

Submission 8 Sep+. Pull BOTH repos before any work; agent A shares this checkout.
thesis/ pushes to Overleaf ONLY. Four gates before any thesis push:
check_mustnots.py, check_style.py, check_reporting.py --captions, mark_provenance.py --check.
Full local build: copy Report.tex->ReportLocal.tex swapping Packages for a variant stripped
of inputenc/fontenc/tipa/pstricks, tectonic -c minimal; count pages via pypdf outline.

## Where every chapter stands (all drafted by me at Brian's instruction; his edit final)
- Ch1 + Abstract: rewritten backwards from the finished thesis 7 Sep. Abstract two paras,
  bounded claims; intro has SOTA paragraph (tong2001active/hauser2012gies/agrawal2019abcd/
  sussex2021nearoptimal vs tillman2011iod/triantafillou2015combine/mooij2020jci/
  hahn2026fedci); contributions map 1:1 to CLAIMS with boundaries.
- Ch3: 3.3.5 One Episode End-to-End algorithm (airy format, refs as \Comment, [b] float);
  3.5 matches current Ch4; sec:meth_regimes defines the one-bit disclosure (semantics
  verified in cb/citest.py `foreign` mask: per-row, that-not-which).
- Ch4: Brian's template (headline figure -> experiment -> figure-walking analysis).
  4.3 fully rewritten 8 Sep after his critique (no project history, real interpretation,
  colons 23->13). 4.3.1 = two-budget equivalence bound (beta=0.7: 44%/23% of margin on 122
  episodes) + full-oracle scope (channels 0.771 vs 0.558 WINDOW RATE, never recovery).
  4.3.3 = interaction at 12k (pooled 1.1x, fed 6.1x, near-floor + seed-2 caveats).
  4.2.1 HOLDS for rho12on/rho12b fleets; when they land, k=8 grid demotes to appendix
  (Brian: "non-principal-cell results leave the thesis").
- Ch5: drafted from docs/AGENT_C_CH5_HANDOVER.md, then THREE tone iterations + accessibility
  pass + proportionality reframe (all rules canonised in WRITING_GUIDELINES.md). 5.3.1 now
  carries generous-not-accurate skeleton finding (alpha ceiling 61.0%->71.3%). Table 5.1
  REMOVED. Sample-efficiency paragraph REMOVED (Brian). GAP comments for pending fleets.
- Ch6: drafted, no section heading, five paragraphs, nothing new.
- Declaration: AI-usage statement added; name/date placeholders fixed.
- Appendix: 4 chapters, ~8pp, floats [H], long tables split side-by-side two-column,
  attribution one-page distillation (full chapter dormant in appendix_attribution()),
  Source Code placeholder link (app:source). In-Regime section CUT. Negative Results,
  Results Tables, aux metrics, per-seed robustness: repo-only.
- Page count: 96 local full build (was 104). Brian target <100, ideal 90-95. Front matter
  and \small-bib slimming were REVERTED at his request; savings came from appendix crunch,
  short LOF captions ([short] args), 44 uncited bib entries pruned (annotated 1:1 kept;
  I added missing annotations for shimizu2006lingam/hoyer2009nonlinear/peters2014continuous).

## Binding style law (all in thesis/WRITING_GUIDELINES.md, read before writing ANYTHING)
Brian's voice, calibrated over 7-8 Sep: accessible, low cognitive strain; sentence-initial
adverbs (more, not mechanical; six banned openers stay banned); short declaratives mostly
attached, occasional standalone for rhythm; colon "x: y" and copular "The X is Y, and Z"
are named tics (ceiling ~7 colons/2400 words); narration sparing, functional, never
self-insisting; past tense when recounting measurements; NO project history in prose
("say what we did, never what we tried" now enforced hard -- Brian called history-narration
out explicitly 8 Sep); limitations proportionate and constructive, never self-undermining.

## Open / pending
- rho12on (k=12 compensated answer-rate) + rho12b (channels ablation): agent A scoring.
  Then: rewrite 4.2.1, repoint fig_answer_rate at k=12, demote k=8 grid to appendix,
  fill 5.1 GAP comment.
- Brian may CUT parts of 4.3 ("considering cutting some") -- 4.3.3 weakest if so.
- PLACEHOLDER GitHub link in app:source needs filling at submission.
- Keywords set; abstract may need supervisor-required wording check on declaration.
- agent A audits land in AGENT_A_INBOX numbered sections; always re-read CLAIMS.md
  (MUST NOTs first) before touching any number. tab:12k tables carry per-cell truth.

## Numbers that changed recently (do not trust older prose)
C10 budget axis SHD is ragged (report both metrics or neither); C11 has SIX corners
(V-shaped adversarial, 0.98x); credit at 12k = interaction (supersedes "no interaction");
ladder margin beta=0.7 = 0.00386; skeleton sweep: generous > accurate.
