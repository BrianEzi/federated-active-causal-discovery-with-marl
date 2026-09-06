# Chapter 4 handover to agent C — 6 Sep

Brian's instruction: **agent C drafts Chapter 4.** He will discuss the specific structure and
voice with you directly, then tweak on top. Chapter 5 (Discussion) and Chapter 6 remain his
alone; do not draft those. The ten `% >>> Analysis (Brian)` markers in Ch4 stay EMPTY unless
he tells you otherwise -- they are his argumentative slots, not descriptive gaps.

## 1. The agreed structure

### 4.1 Federated Active Recovery
| # | section | state |
|---|---|---|
| 4.1.1 | The Four-Axis Sweep (`sec:res_sweep`) | exists, unchanged; Brian is writing analysis here |
| 4.1.2 | The Epsilon-Greedy Control | NEW |
| 4.1.3 | Constrained Budget | NEW |
| 4.1.4 | The Generator Control | promote `fig:generator` from a floating figure into its own subsection |

DEMOTED TO APPENDIX (move prose + table, keep labels so refs survive):
`sec:res_budget` (Training Budget) and `sec:res_reward` (Where the Advantage Sits).
NOTE: the Abstract currently calls the training-budget finding the central empirical result.
Brian has DELETED the Abstract until the end, so there is no inconsistency to fix -- but do
not resurrect Abstract text that leans on it.

### 4.2 Transfer to a Realistic Evidence Regime
| # | section | state |
|---|---|---|
| 4.2.1 | The Answer-Rate Grid (`sec:res_rho`) | MOVED UP from 4.2.3 -- this is the headline |
| 4.2.2 | Fixed-Policy Decomposition (`sec:res_fixedpolicy`) | MOVED UP from 4.2.4 |
| 4.2.3 | Distributional and Mechanistic Robustness | NEW |
| 4.2.4 | The Sample-Size Axis and Disclosure (`sec:res_nint`) | MOVED DOWN from 4.2.2 |

CUT ENTIRELY: `sec:res_sampled` (Finite-Sample Evidence). Its content survives as
`app:evidence_cost` in the appendix (already generated).

**CRITICAL DEPENDENCY, do not lose this.** The cut section carried the motivation for the
entire partial-oracle idea: why would anyone withhold oracle answers deliberately? Because
training under genuinely sampled evidence fails outright -- 0.953 per-window solve rate under
oracle against 0.197 under sampled, matched configurations, three seeds each
(`app:evidence_cost`). Without two sentences of that in the **section 4.2 introduction**,
4.2.1 opens with "we withhold a fraction of answers" and reads as an arbitrary design choice.
That would badly weaken the chapter's strongest section.

### 4.3 The Price of This Formulation of Federation
Structurally unchanged (`sec:res_ladder`, `sec:res_convention`, `sec:res_credit`).
Two content changes coming from agent A, do not write around them:
- 4.3.1 gains an EQUIVALENCE BOUND replacing the current absence-of-evidence phrasing.
- 4.3.2's claim that "coordinating without communicating is worse than not coordinating at
  all" is now BUDGET-DEPENDENT and needs a qualifier -- see section 3 below.

## 2. What each new section rests on

**4.1.2 Epsilon-greedy control.** Myopic rule with probability 1-eps, uniform randomised
intervention otherwise; eps in {0.05, 0.1, 0.2, 0.3}; best eps per seed reported, a
convention that FAVOURS the control and must be stated wherever quoted. All 20 sweep cells,
3 seeds, 200 paired episodes. Learned ahead in 52 of 60 seeds, 1 tie, 7 behind. Data:
`results/epsgreedy/sweep/`, claim C9. Table `tab:epsgreedy` (k12/k30) stays in the chapter as
the headline; the full 20-cell grid goes to the appendix (agent A is building it).
FORBIDDEN: quoting SHD separations from this control at k30 -- all arms sit near the floor.
REQUIRED: state that eps-greedy BEATS plain myopic; the claim is that training buys the
remaining gap, never that exploration is worthless.

**4.1.3 Constrained budget.** Ten-point beta axis at k12s50n04, budgets 17 to 166, 3 seeds
each, `results/budget_tight/` plus the existing sweep cells. Joint recovery:

    beta   budget  learned  myopic  partitioned  oracle-cover  random
    0.5      17     0.602   0.165     0.280         0.140      0.000
    0.6      20     0.760   0.288     0.495         0.260      0.000
    0.7      24     0.913   0.507     0.678         0.583      0.002
    0.8      27     0.932   0.587     0.732         0.855      0.002
    1.0      34     0.973   0.815     0.742         0.997      0.005
    1.5      50     0.985   0.918     0.740         1.000      0.017
    5.0     166     0.980   0.947     0.740         1.000      0.795

Three facts the section must carry, all measured:
  (a) the learned margin over myopic PEAKS at beta=0.6 (+0.47) and decays to +0.03 at beta=5;
  (b) the oracle-cover arm -- optimal WITHIN each window, uncoordinated ACROSS them by design
      -- is BEATEN below beta≈0.8 and only becomes a true ceiling above it;
  (c) the mechanism is measured, not inferred: at beta=0.5 avoidable duplication is 0.017 for
      the learned arm against 0.325 for oracle-cover.
FORBIDDEN, and this one is important: never write "the learned policy beats the optimum".
Oracle-cover is per-window optimal and deliberately uncoordinated; a coordinated optimum does
not exist in this codebase and the true ceiling at tight budget is UNKNOWN.
Measured SHD at beta=0.5 (200 paired episodes, selected checkpoint): learned 0.00554, myopic
0.00977, random 0.10964; 2 of 3 seeds separate. Report it as the second, less discriminating
view -- the ratio is 1.76x against 3.6x on recovery -- and say so.

**4.1.4 Generator control.** Existing `fig:generator` and its findings doc. Scale-free against
Erdos-Renyi at the principal cell, 12,000 episodes, three seeds per family, densities
near-matched. No mechanism claimed for why the myopic rule collapses on ER.

**4.2.3 Distributional and mechanistic robustness.** NEW, data landing as this is written.
The rho=0.5 partial-oracle policies evaluated under sampled evidence across a 2x2: noise
{Gaussian, Student-t(3)} x mechanism {linear, tanh}. Complete so far (linear row):

    noise      learned   myopic   random   myopic/learned   learned ahead
    gaussian   0.02989  0.04846  0.05434       1.62x            3/3
    uniform    0.02936  0.04821  0.05599       1.64x            3/3
    t3         0.03168  0.04805  0.05654       1.52x            3/3

Nonlinear row (tanh) is still measuring; agent A will supply it. Two framings must both appear:
  ROBUSTNESS: the advantage survives every family tested, three seeds each, everything else
  held fixed, with the linear-Gaussian corner regression-checked against the stored baseline
  to every recorded digit.
  BOUNDARY: surviving a departure is not exploiting it. In the non-Gaussian and nonlinear
  families the structure is identifiable from observational data alone (Shimizu et al. 2006;
  Hoyer et al. 2009 -- both now in references.bib), and this engine reads two moments and a
  linear correlation, so it neither breaks nor benefits. See
  `docs/IDENTIFIABILITY_NOTE_2026_09_06.md`.
CRITICAL SCOPE NOTE: these runs use the rho=0.5 fleet, NOT the sweep policies. Different cell
(k=8), budget 70 not 50, belief channels and reprobe ON, 8,000 episodes not 12,000. Say so.

## 3. Corrections to EXISTING prose that must be made

- **4.3.2** currently states unconditionally that coordinating without communicating is worse
  than not coordinating at all (0.745 against 0.927 at k12). The budget axis shows the sign
  FLIPS: at beta<=0.8 the fixed partition BEATS uncoordinated myopic (0.280 vs 0.165 at
  beta=0.5), and only from beta>=0.9 does the published ordering hold. Add the qualifier.
- **The contended-fraction axis is confounded** and Brian may want this said. At a fixed
  window of 12, raising the shared fraction necessarily shrinks the private block (sigma=0.25
  gives 9 private, sigma=0.75 gives 3). Greedy recovery correlates -0.69 with private block
  size and only +0.44 with the shared fraction of the graph, so "contention makes it easier"
  is really "less solo work makes it easier". Do not assert a mechanism Brian has not signed
  off; flag it to him.

## 4. Guardrails -- these are not optional

1. Every number comes from `thesis_results/CLAIMS.md`, and the MUST NOT lines are read FIRST.
   Do not compute numbers yourself and do not copy them from findings docs.
2. NEVER quote a run file's `global_hard_shd` field. It is the run's own last-update
   self-evaluation, not the reported metric; it has misled this project seven times. All SHD
   numbers trace to `scripts/global_shd_paired.py` output.
3. `scripts/check_mustnots.py` must exit CLEAN before any push. It is currently clean.
4. Metric names are fixed: "joint recovery rate" and "SHD on committed marks"; any difference
   is "paired difference in SHD (A - B)". Arm E is "pooled", never "centralised".
5. Never run `scripts/build_results_skeleton.py` -- it overwrites Chapter 4 prose.
6. thesis/ pushes to Overleaf ONLY, never to GitHub.

## 5. Division of labour

AGENT A (me) is doing, so do not duplicate: the epsilon-greedy appendix table; the 4.1.3
tables and figure; the 4.2.3 robustness table once the nonlinear row lands; the 4.3.1
equivalence bound; CLAIMS entries; registry wiring; `fig_answer_rate` reduced to a single
panel (its right-hand panel with the argmax derivative is being retired to a holding section
Brian will resolve).

AGENT C (you): the chapter prose, the section ordering and the moves above. Talk to Brian
about structure and voice before drafting at length -- he has said he wants that conversation
first.
