# Brief for agent C — rebuild Chapter 3 (Methodology)

1 Sep 2026, 23:5x. Written by the agent drafting Results, so the two do not collide.

## The one-line diagnosis

**Chapter 2 describes the project as it now exists. Chapter 3 describes the project as it was
in August.** Background already covers constraint-based equivalence classes, version spaces,
MAGs, federated optimisation, credit assignment and attribution. Methodology still describes a
two-agent, d=5, Clamp-only, BGe/subset-DP, disjoint-MLP-IPPO system that produced none of the
results being reported. Your job is Chapter 3. Background is not yours and is in good shape.

## Boundaries — three agents are working tonight

* **`thesis/` is a SEPARATE git repo** (Overleaf remote, branch `main`, clean at `e1ab35f`) and
  is gitignored by the outer repo. Work there. Commit and push to Overleaf as you go.
* **YOURS:** `3 Methodology.tex`, and `1 Introduction.tex` where it states superseded facts.
* **NOT YOURS:** `4 Results and Analysis.tex` and `5 Discussion.tex` — another agent is
  writing those tonight. Do not edit them, not even the comment scaffolds.
* **NOT YOURS:** anything under `results/` in the outer repo. Training jobs are running.
* `2 Background and Related Work.tex` — read it, cite into it, do not restructure it.

## Ground rules, learned the hard way tonight

1. **Never write a number from memory or from an earlier draft.** Every number comes from
   `docs/RESULTS_LEDGER_2026_09_01.md`, and its section 1.2 is **SUPERSEDED** — the current
   SHD row is in `docs/FINDINGS_CHECKPOINT_2026_09_01.md`. Check the ledger's section 6
   (retracted claims) before citing anything that looks plausible.
2. **Verify every system claim against a config, not against prose.** The command is
   `.venv/bin/python -c "import json;print(json.load(open('results/sweep/oracle/k20s50n04b150_s0.json'))['config'])"`.
   That file is a headline cell. If the thesis says X and the config says Y, the config wins.
3. **Read the code comment before describing a mechanism.** This codebase documents its own
   decisions in situ and several of tonight's errors came from not reading them first.
4. Cross-references: the chapter is `\label{Chap3}` and sections use `sec:meth_*`. Keep that
   scheme so existing `\ref`s survive.

## What is WRONG, with the evidence, so you do not have to re-derive it

Right-hand column is from `k20s50n04b150_s0`'s config.

| Chapter 3 claims | Reality |
|---|---|
| `K=2` throughout; theorem proved for two agents | `n_agents: 4`, swept 2-10 |
| topology (1,1,3), d=5, k_v=4 | k=12/20/30; d=50 at k=20 |
| "the default restricts every agent to CLAMP only" | `action_modes: ['vary']`; `--vary_only` in every sweep job |
| BGe scoring + Robinson sink subset-DP | `belief_backend: factored` (constraint-based); `vs_evidence: oracle`; `skeleton_source: true` |
| "two-layer MLP, 128 hidden, tanh" | `policy_arch: gnn_portable` (`PortableRoleActorCritic`) |
| "each agent operates a **completely disjoint** actor-critic pair" | ONE shared module for all agents — `ma/policy.py:641` |
| IPPO, no federation of the learner | FedAvg; `local_epochs: 4`; `turn_aware_credit: True` |
| "single cooperative team reward, shared identically" | `per_agent_reward: True`; `reward_criterion: 'claims'`; `claim_bar: 1.0` |
| U14 three-part success as the criterion | hard SHD of the pooled global graph is primary |
| RQ3: the 1-bit regime signal is necessary | `disclose_regime: False` in every sweep run |
| attribution of latent confounders | absent from the chapter entirely; it is the novel method |

**Two of these are load-bearing, not cosmetic.**

* **The Clamp narrative.** Section 3.2.2's altruism argument, RQ2, and the Introduction's
  "82-91% of clamps to private nodes" all rest on Clamp being the altruistic act. Clamp is not
  in the action space. The altruism result that EXISTS is ledger 2.6: `greedy_attribution`
  probes privately 7% of the time against 0.38-0.61 for every other policy, and attributes
  worse than a generic uncertainty rule. Same phenomenon, better evidence. `docs/MA_PROBLEM_STATEMENT.md`
  carries the identical stale framing and should be fixed in the same pass.
* **"Completely disjoint networks."** Stated as the thing that makes the setup federated, and
  contradicted by the code. The defensible framing is already written at `ma/policy.py:632`:
  PARAMETER SHARING WITH DECENTRALISED EXECUTION — each agent acts on its own observation
  only, no centralised critic, no pooled observations, and it "must be reported as" a
  departure from one-learner-per-agent. Use that framing and cite the constraint it preserves.

## What to KEEP

Section 3.1 (the environment formalism) and Theorem 3.1 are the strongest writing in the
chapter and both survive. **Generalise the theorem from K=2 to K agents** — the existing proof
already does the work, since with no cross-private edges a latent in ANY private block reaches
the window only through an observed shared node. The Discussion scaffold currently lists
"proved for K=2 only" as a threat to validity; generalising it retires that threat instead of
confessing it.

## Sequence

**Phase 0 — factual corrections (do first, ~30 min).** Fix only what is wrong regardless of
framing: action modes, belief engine name, policy architecture, the disjoint-network claim,
agent count, scale. Do not rewrite structure yet. Commit separately so the diff is reviewable.

**Phase 1 — section 3.6, Evaluation (do next; another agent is blocked on it).** Currently the
weakest fit and the thing Results must cite. It needs:
  * hard SHD of the pooled global graph as PRIMARY, success as secondary, and why (success
    saturates and amplifies variance; see ledger 1.2/1.3)
  * the paired-episode protocol (`scripts/global_shd_paired.py` — read its docstring, it
    explains why it exists and how it differs from `scripts/shd.py`)
  * the competence gate `window_rate >= 0.70` (`scripts/sweep_report.py:51`) and why the MI
    gate was rejected (ledger 5.2)
  * **checkpoint selection**: `best_mi_ratio` from TRAINING rollouts, why that is early
    stopping and not test-set leakage, both checkpoints reported. Worth 2.3x at k=20 and 16x
    at k=30 — a methods decision, not a footnote. Full detail in
    `docs/FINDINGS_CHECKPOINT_2026_09_01.md`.
  * **baseline definitions**: `greedy_uncertainty`, `random_vary`, `oracle_cover`,
    `greedy_partitioned`, `greedy_attribution`. They appear in every result and are defined
    nowhere. Source: `ma/baselines.py`.

**Phase 2 — sections 3.3-3.5, the rewrites.**
  * *3.3 Constraint-based belief.* Version space over marks; the two evidence regimes (oracle
    vs sampled with Fisher-z; `n_int` is the binding parameter, ledger 4.3); the true-skeleton
    assumption WITH its measured cost attached (100% -> 0% identification at `n_obs=60`,
    ledger 5.1); factorisation over connected components. Code: `cb/factored.py`,
    `cb/versionspace.py`.
  * *3.4 Attribution.* NEW, and it is the novel method. `LatentGroup(owner, children)`; the two
    pruning rules — atomicity (sound unconditionally) and local disturbance (a named, UNSOUND
    modelling assumption that must be declared as such); exact factoring over connected
    components; unit propagation to a fixpoint. This is what reaches k=50 where joint
    enumeration faces 8.4e10 hypotheses at k=20. Code: `cb/component_attribution.py`,
    `cb/attribution.py`. Background section `sec:attribution_background` already exists to
    cite into.
  * *3.5 Learning.* GNN with parameter sharing (framing above); FedAvg with `local_epochs`;
    turn-aware credit; per-agent reward. Federation of the OPTIMISER is a first-class design
    axis here — ledger 3.1 and tonight's ladder both measure it.

**Phase 3 — generalise Theorem 3.1 to K agents.**

**Phase 4 — BLOCKED on Brian's decision, do not start without it.** The Introduction's research
questions and contributions list. RQ1 survives and is stronger than when written. RQ2 asks
about a Vary/Clamp trade-off that no longer exists; RQ3 asks about a channel the runs have
switched off. Both need restating, and the restatement decides the shape of Discussion too.
Brian is being asked now; the answer will be appended to this file.

## If you find another contradiction

Add it to the table above and tell us in `docs/AGENT_B_INBOX.md` rather than resolving it
silently. Several of tonight's worst hours went into claims that were confidently wrong.

---

# Appended by agent C, 2 Sep — contradictions found beyond the table above

Every one of the eleven rows in the brief's table was verified against
`k20s50n04b150_s0`'s config and the code before being written up. **All eleven held.** The
following are additional, and are reported in `docs/AGENT_B_INBOX.md` with full detail.

| found | reality | where it landed |
|---|---|---|
| `PLAN_2026_08_28` F4 says argmax is primary | `results/ckpt/*.json` records `"sampled": true` — every reported SHD number is sampling | documented as sampling in `sec:meth_paired`; **needs a decision, may invalidate numbers** |
| `bNNN` in a cell name reads as the budget | it is `beta * 100`; `k20s50n04b150` has `budget: 75` | budget rule derived from `scripts/sweep.py:105` and verified on 5 cells, in `sec:meth_budget` |
| guidelines: federated training "explored, not adopted" | `local_epochs: 4` in every job — FedAvg is adopted | split: plain FedAvg adopted, server adaptivity explored (`server_optimiser: 'none'`) |
| guidelines: numbers trace to `STATE_OF_TRUTH.md` | that file is 22 Aug and predates the current engine | used the ledger + checkpoint doc per this brief |
| the 5.3x bidirected-triangle figure | measured, in no retraction list, but NOT in the ledger | used in `sec:meth_generator`; keep-or-drop question raised |
| clamp `+0.021` and turn-order `+0.028` CIs | both from the retired 22 Aug two-agent protocol, absent from the ledger | removed; direction asserted without a number |
| `--vary_only` justification | rests partly on `mode_at_scale.py`, which `PLAN_2026_08_28` §1 records as cut after 2 of 4 arms | worded as "no measured cost", not "Vary wins" |

## Status against the sequence

* **Phase 0 — done**, `thesis/` commit `c508599`. Factual corrections only, separate commit
  so the diff is reviewable.
* **Phase 1 — done**, `thesis/` commit `11f95bb`. Section 3.5 rewritten with six labelled
  subsections: `sec:meth_shd`, `sec:meth_paired`, `sec:meth_ckpt`, `sec:meth_gate`,
  `sec:meth_baselines`, `sec:meth_regimes`. Results can cite these now.
* **Phase 2 — next.** 3.3 version-space belief, 3.4 attribution in full, 3.5 learning.
  Labelled stubs for `sec:meth_versionspace` and `sec:meth_attribution` are already in place
  so nothing dangles.
* **Phase 3 — after that.** Generalise Theorem 3.1 to K agents.
* **Phase 4 — still blocked** on Brian. Nothing in the Intro's RQs or contributions list has
  been touched. Its Problem Formulation and Dissertation Structure are fixed.

**Caveat on everything above: `pdflatex` and `bibtex` are not installed on this machine, so
the mandated full build has NOT been run.** Static checks pass (citations resolve in both bib
and annotated bibliography, no `\citet`, British spellings, all `\ref`s defined, no duplicate
labels, environments and braces balanced). The build needs to happen on Overleaf.

---

# APPENDED 2 Sep, 00:2x — Phase 0b: terminology audit

Brian's instruction: the project uses coined words where standard research terms exist, and he
wants the standard ones. Do this in the SAME pass as Phase 0, because both are factual
corrections rather than restructuring.

**Rename PROSE ONLY. Do not rename code identifiers.** Renaming `clamp`/`vary`/`oracle` across
~50 files two days before the compute freeze is pure risk with no marking benefit, and it would
invalidate every config already written to disk. Instead, put the mapping in `Glossary.tex`
(it exists, 67 lines) so the gap between paper and repository becomes a stated convention:
"we write `vary` in configuration listings for the randomised intervention
$do(V \sim \mathcal{N}(0,\sigma^2))$". An examiner reading the code wants exactly that.

## Do NOT change these — they are already standard, and changing them would be wrong

**`oracle` is standard causal-discovery vocabulary.** PC and FCI correctness is proved with
respect to a CONDITIONAL-INDEPENDENCE ORACLE -- `spirtes2000causation`, already in the bib. It
needs a gloss on first use, never a rename:

> *oracle evidence* --- the infinite-sample limit, in which each conditional-independence query
> is answered exactly, as assumed in the correctness proofs of constraint-based discovery
> \citep{spirtes2000causation}.

This also makes RQ1's "from finite samples to the infinite-sample limit" land precisely, since
*oracle* and *infinite-sample* become one axis stated two ways.

Also standard, keep: **edge mark** (PAG vocabulary), **SHD**, **arm**, **budget**,
**free-riding**, **intervention target**, **latent projection**, **maximal ancestral graph**.

## Replace in prose

| coined | standard | source |
|---|---|---|
| \textsc{Clamp} | **atomic intervention**, $do(V=c)$ | `pearl2009causality` §1.3.1 |
| \textsc{Vary} | **randomised (stochastic) intervention**, $do(V \sim \mathcal{N}(0,\sigma^2))$ | `eberhardt2007interventions` |
| settled / unsure | **determined / undetermined** (or *invariant*, per `hauser2012gies`) | in bib |
| claim, `claim_bar` | **commit to a mark**; **decision threshold** $\tau$ | -- |
| success (the conjunction) | **joint recovery rate** | -- |
| window rate | **per-window recovery rate** | -- |
| hard / soft SHD | **SHD on committed marks** / **expected SHD under the posterior** | -- |
| probe | **intervention** / **experiment** | -- |
| sovereign | **autonomous**, or delete -- the no-sharing constraint is stated formally already | -- |
| altruistic, "the altruism gap" | **public-goods problem** -- the de-confounding experiment is a public good, costly to the provider and beneficial to peers | connects to the existing free-rider index |

**Both modes are hard (perfect, structural) interventions** -- parents severed in both cases.
They differ only in the interventional distribution assigned: a point mass against a Gaussian.
Stating that once in standard notation is clearer than either coined word, and it sharpens the
de-confounding argument: setting $V$ to a constant makes the latent's contribution degenerate
and breaks the covariance path, whereas assigning it an independent distribution substitutes
one exogenous source for another and preserves the induced dependence among the latent's other
children. Section 3.2 already argues target-not-value identifiability from `hauser2012gies`;
that argument reads better in standard notation, not worse.

## Coined and KEPT, but must be formally defined on first use

`attribution` (determining which agent owns a latent), `LatentGroup(owner, children)`,
`local disturbance`, `component factoring`, `contended fraction` $\sigma$. These name mechanisms
that have no standard name because they are the contribution. Coining is legitimate; using a
coined word undefined is not. `children` is already standard (children of the latent); only
`owner` needs a definition.

**`window`** is borderline -- 65 uses in the thesis, 70 files in code. The standard word for
what it is, is the **margin**: agent $k$ observes the marginal over $\mathbf{V}_k$ and the
latent projection is onto $\mathbf{V}_k$. Keep "window" as an explicitly defined shorthand, but
use "local variable set $\mathbf{V}_k$" in formal statements and in the theorem. Do not attempt
a global rename.

## The one worth changing urgently

**`evidence_power` / "power-limited evidence" collides with STATISTICAL POWER**, and the
collision is directly in the line of fire: the entire purpose of the mechanism is to imitate
finite-sample evidence, where statistical power is the actual technical quantity being
discussed. A reader meeting "power-limited evidence at power 0.85" has to work out that it has
nothing to do with $1-\beta$.

What the mechanism is: an oracle that answers only a fraction $\rho$ of queries and returns
*unknown* otherwise. Name it a **partial oracle** with **answer rate** $\rho$. Rename in prose,
leave the config flag alone. **Agent B is producing calibration results under the old name right
now** -- flag it to them in `docs/AGENT_B_INBOX.md` so the write-up does not fork.

---

# APPENDED 2 Sep, 04:1x — the RQs are settled. Phase 4 is UNBLOCKED.

Brian has restructured the research questions. The arc is now: does it work in an idealised
setting, does it survive a realistic one, what did our formulation of federation cost --- and
then a question of a different kind, what is knowable at all.

**The four RQs are already written into `1 Introduction.tex`** (`sec:research_questions`),
replacing the three superseded ones. Do not re-derive them; read them there. In summary:

* **RQ1** Federated active recovery --- the sweep, oracle evidence, four axes. Settled, 60 runs.
* **RQ2** Transfer to a realistic evidence regime --- version space and policy from exact
  conditional-independence answers to finite samples, and whether degrading the training
  regime closes the gap. Partly settled, partly live (agent B's answer-rate fleet).
* **RQ3** The price of this formulation of federation --- information, reward and optimisation
  partitions, and which coordination mechanisms recover the cost.
* **RQ4** Limits of latent attribution --- what share is attributable, what bounds it, and
  whether the bound is resources or identifiability.

## What this changes in Chapter 3

**Attribution is scoped, not central.** RQ4 makes it a research question, but a bounded one:
one Results section and one future-work paragraph, not a chapter. `sec:meth_attribution` should
be proportionate to that --- enough to define a latent group, the two pruning rules (atomicity,
sound; local disturbance, a named and UNSOUND modelling assumption that must be declared as
such), and the component factorisation that makes it tractable. It does not need the full
derivation.

`WRITING_GUIDELINES.md` currently says attribution is "not the thesis's centrepiece ...
Background gets a few sentences, not a subsection tree". That is now half-right and half-stale:
still not the centrepiece, but it is RQ4 and Background's existing `sec:attribution_background`
is proportionate. **Update that bullet** to "RQ4: one Results section, one future-work
paragraph, a Background section" so two agents are not reading opposite rules.

**RQ2 raises the bar on `sec:meth_regimes`.** Transfer is now a whole research question rather
than a limitation, so the methodology has to define both evidence regimes properly: the exact
regime, the finite-sample regime with Fisher-$z$ at `n_int`, and the **partial oracle** ---
an oracle that answers a fraction $\rho$ of conditional-independence queries and returns
*unknown* otherwise. That third one is new and currently undocumented. Its calibration
(`docs/FINDINGS_TRANSFER_2026_09_02.md` section 4) belongs in Chapter 3, not Chapter 4:
$\rho = 0.85$ matches genuine finite-sample belief resolution to within 0.0042 mean absolute
difference at $k_v=8$, optimal or tied through $k_v=20$, overtaken by 0.80 at $k_v=30$. Also
state its known limit: withholding is sound and can only decline to answer, so the proxy
reproduces the SPEED of belief resolution but not its FALLIBILITY.

**Do not write "power".** It is an answer rate $\rho$; "power" collides with statistical power,
which is exactly what the finite-sample regime is about. See Phase 0b above.

## Chapter 4 is now a generated skeleton --- read this before touching it

`4 Results and Analysis.tex` has been replaced with a section scaffold produced by
`scripts/build_results_skeleton.py`. Eight sections, each carrying a comment block that states
the CLAIM it must make, the DATA that carries it, and the BOUNDARY where the claim stops
holding. **The prose is yours to write. The tables are not.**

Every table is computed from `thesis_results/` at generation time and must never be edited by
hand --- if a number looks wrong, the data or the script is wrong, and editing the `.tex` only
hides it. Re-run the script after `scripts/collect_thesis_results.py` and the tables follow.

My own earlier prose draft is preserved at `thesis/DRAFT_results_prose_agentA.md`. Treat it as
raw material at best: measured against `WRITING_CRITIQUE.md` it carries 17 `\textbf`, 11
`\emph`, 16 em dashes, 17 "rather than" and 4 announced enumerations in 3,263 words --- a
markup event every 74 words, the same rate the critique flags in Chapter 3. **Do not paste it
in.** Mine it for facts and write the sentences yourself.

Three sections carry a PENDING marker and must not be written from memory: the pair-class table
(re-derive from `scripts/shd_by_pair_class.py`; the ledger figures pre-date the checkpoint
correction), and the two halves of RQ2 that depend on agent B's fleet.

---

# APPENDED 2 Sep, 06:4x — we collided on Chapter 4. It is yours from here.

We both wrote \S\ref{sec:res_attribution}. I drafted it, then found your version already in the
file below mine. **I removed mine and kept yours.** Yours is in your voice, which is the point
of the division of labour, and mine had a number wrong that yours had right.

**Chapter 4 prose is yours. I have stopped writing it.** What I will keep doing is generating
the tables and figures from data, verifying numbers against the raw files, and leaving CLAIM /
DATA / BOUNDARY comments in the sections that are still empty. If a number in your prose
disagrees with `thesis_results/`, I will correct the number and leave a marker rather than
rewrite the sentence around it.

## Two numbers in your attribution section corrected against the raw files

Both were ledger-sourced and the ledger is slightly off; `results/attr/transfer_*.json` are the
originals.

* attribution-greedy private share: `7%` -> **7.6%** (measured 0.0757)
* identified, attribution-greedy against the generic uncertainty rule: `0.185` against `0.333`
  -> **0.181** against **0.327** (measured 0.1808, 0.3267)

## And a correction to MY figure that your prose caught

You wrote the closed-form residual as **0.041**. My draft and `thesis/figures/attribution_law.pdf`
both said 0.040. Yours is right. `scripts/figures.py` was deriving the residual by subtracting
the printed predicted and measured columns, which are rounded to three places, instead of
reading the residual column that `scripts/attr_model.py` prints directly. Fixed, figure
regenerated, and it now reads 0.041.

## What is verified and safe to quote in \S\ref{sec:res_attribution}

Checked against `thesis_results/attribution/` tonight, not against the ledger:

* Matched-budget control, rounds per agent held at **15.0**: two-variable groups attributed at
  1.00, 0.80, 0.77, 0.72 at one, two, three and seven peers. Larger groups: **67 correct at one
  peer, then 0, 0, 0**. Zero misattributions in every row. This is the cleanest control in the
  thesis and the sentence that rules out resource starvation.
* Coverage: 21 of 1056 at budget 30; **349 of 1056 at 60, at 120 and at 240** -- the identical
  count, not merely the same rate.
* Group sizes are in VARIABLES, and a group spanning $n$ variables explains $\binom{n}{2}$
  pairs. At one peer, five- and six-variable groups are already at zero, which the ledger's
  three-column table does not show.
* Scale: 21, 33 and 27 correct at $k_v$ = 30, 40, 50; zero wrong; 5.3, 5.1 and 9.4 s/episode.

## Do not write these

* **The reward-alignment asymmetry** (ledger 1.3). Retracted:
  `docs/FINDINGS_PAIR_CLASS_2026_09_02.md`. Shared-shared error is 0.00000 for both learned and
  myopic over 90,000 pair observations.
* **The agent-count reversal beginning at five agents.** At $K=5$ it is one seed: the
  learned-to-myopic ratio is 1.65 with all seeds and 0.25 without seed 2. \S\ref{sec:res_scale}
  now states the reversal as beginning at eight, and reports both figures.

---

# APPENDED 2 Sep, 09:3x — `thesis_results/CLAIMS.md` is now the source for Chapter 4 numbers

Built by `scripts/build_claims.py`, regenerated from `thesis_results/` rather than maintained
by hand. Five claims so far (C1 crossover, C2 agent count, C3 pair class, C4 federation cost,
C5 undertraining), each with the number, the sample it rests on, and a boundary.

**Read the MUST NOT lines before writing any sentence in that section.** They are
hand-maintained, because a retraction is a judgement rather than a computation, and each one
marks a claim that was made in good faith and then refuted:

* **C1** -- do not quote a ratio of means at $k_v=30$; two seeds are significant and one is
  indistinguishable, and the ratio hides that.
* **C2** -- do not claim the myopic rule improves across the agent-count axis. Per-pair SHD
  divides by `global_pairs`, which runs 117 to 525; in raw counts that arm is close to flat.
  Also: at $K=5$ the reversal is one seed (1.65 with all seeds, 0.25 without), and two of the
  high-$K$ runs may be unconverged, which is under test now.
* **C3** -- do not write the reward-alignment asymmetry. Retracted.
* **C4** -- do not state a direction for the federation cost; mean and median disagree.
* **C5** -- do not substitute the 12,000-episode retrains into any sweep table.

**Prefer CLAIMS.md over `docs/RESULTS_LEDGER_2026_09_01.md`.** The ledger is a working document
and has now been wrong twice in ways that reached a draft: section 2.2 quoted a
share-of-ceiling as though it were a raw rate, and section 1.3 reported an asymmetry that a
larger re-measurement retracted. Both are corrected in place with banners, but the ledger is
maintained by hand and CLAIMS.md is derived, so where they disagree CLAIMS.md is right.

`scripts/collect_thesis_results.py --check` reports drift between `thesis_results/` and the
live `results/` tree. It reads 0 drifted, 0 missing as of this entry.
