# Chapter 5 (Discussion) and Chapter 6 (Conclusion) — handover to agent C

Written 7 Sep 2026 by agent A, at Brian's request.

## Read this first: an ownership change

Chapter 5 has been **Brian's alone** for the whole project. `docs/THESIS_STRUCTURE_CONTRACT.md`
and the header comment in `5 Discussion.tex` both say so, and agent C's scope was Chapter 4.
Brian has asked for this brief so C can draft it. **Confirm with Brian before drafting** — if
that reversal was not intended, this document is still useful to him as his own brief.

What has NOT changed: the `% >>> Analysis (Brian)` markers in Chapter 4 remain Brian's and
must never be filled by an agent.

## The structure, locked with Brian 7 Sep

Four sections in Chapter 5, **roughly one page each** (450–550 words). Chapter 6 is a **single
section, one page**. The scaffold is already in place in `5 Discussion.tex` and
`6 Conclusion.tex`, with the raw material filed under each heading as comments.

    5.1  The Results in Conjunction          sec:disc_synthesis
    5.2  What the Belief Representation Buys and Costs   sec:disc_representation
    5.3  Limitations                         sec:disc_validity
    5.4  Future Work                         sec:disc_future
    6    Conclusion                          sec:concl

The one rule that governs the whole chapter: **Chapter 4 owns per-result analysis, Chapter 5
synthesises across results.** If a paragraph could sit under a Chapter 4 subsection, it does
not belong here.

## What each section has to do

### 5.1 The Results in Conjunction

Open with the three RQ answers compressed to a sentence each. Spend the rest of the page on
the thread that unifies them: **the advantage is a scarcity effect, and under scarcity it is
coordination.** That thread carries three otherwise separate results — the sweep's apparent
structural thresholds dissolving into training budget, the margin collapsing as rounds are
added, and the partition/uncoordinated ordering inverting. Close by shutting the gap Chapter 2
opens: active *and* federated *and* agent-chosen is achievable, and this is its price.

### 5.2 What the Belief Representation Buys and Costs

Four movements: what it buys, what it costs, the conditionality, the positioning.

Buys: soundness, and **composition by intersection under privacy** — per-site sets combine
without sharing data, parameters or gradients. No posterior has this property; it is the one
structural argument the representation owns.

Costs: no uncertainty quantification (possible or impossible, never probable); the per-pair
factorisation is an outer approximation that overstates ambiguity; binary evidence discards
magnitudes and wastes sub-threshold experiments.

The conditionality, which must not be buried: soundness holds only where the tests are
calibrated, and contamination vacates it in exactly the realistic regime.

Positioning: against Bayesian experimental design this is a **trade, not a win**. Centralised,
graded-evidence methods likely dominate per sample. Federated, their posteriors do not compose.
Then the setting argument — unequal-variance linear-Gaussian is the one family where
interventions are strictly necessary — and its flip side, that nothing transfers automatically
to the easier families.

### 5.3 Limitations

Ordered by severity, each with what it costs and the measurement that prices it. A priced
limitation reads as strength; an unpriced one reads as a hole. The supplied-skeleton
subsection and `tab:disc_skeleton` are already written — do not rewrite them.

### 5.4 Future Work

Every item earned by a limitation above, in the same order where possible. **Keep "more seeds"
out** — that is a limitation, not a research direction.

### Chapter 6

Five short paragraphs: problem and approach; contributions; one line per RQ; one or two lines
pointing at the boundary; a closing forward sentence. **Nothing new** — no new evidence, no new
numbers, ideally no new citations. Limitations and future work are pointed at, never repeated.

## Verified numbers

Every figure below was recomputed from the measurement files on 7 Sep. Numbers not in this
table must come from `thesis_results/CLAIMS.md`, MUST NOT lines read first.

| quantity | value | source |
|---|---|---|
| Learned ahead of myopic, joint recovery | 2 of 18 cells at 4,000 episodes, **16 of 18** at 12,000; 14 cells change winner, 1 is an exact tie | C7 |
| Learned margin over the better myopic variant | **0.322** at β=0.5, 0.235 at β=0.7, 0.067 at β=1.5 | budget axis |
| Fixed partition minus uncoordinated | +0.115 (β=0.5), +0.171 (0.7), +0.004 (0.9), −0.178 (1.5) | `fig:coordination` |
| Partition flat above the crossover | 0.742 from β=0.9 onward; uncoordinated climbs to 0.918 | budget axis |
| Ladder, paired federated − pooled | **−0.00008 ± 0.00022** (selected), **+0.00027 ± 0.00016** (final), 12 seeds | C4a |
| Ladder equivalence margin and verdict | margin 0.00064; TOST p = 0.016 and 0.023; bound is 76% and 89% of the gap to the myopic rule | C4a |
| Ladder saturation | federated errs in **8** episodes of 1,200, pooled in **4**; 12 of 2,400 carry the comparison | `check_reporting.py` |
| Metric: wrong marks | **zero**, in all 12 arm-budget combinations (`error_rate = 0.0`) | `results/metric_bakeoff.json` |
| Metric: the inversion | at β=1.5 myopic leaves fewer pairs undetermined (0.00084 vs 0.00221) while winning fewer episodes (0.933 vs 0.983) | same |
| Alternative metrics | seven tried, all seven invert identically | same |
| Skeleton at n_obs=60 | 65.9% accurate; identification 100% → 0%, recovering to 8.3% at 1,000 rows | `tab:disc_skeleton` |
| Skeleton, coordination | identification 61.8% whether or not a peer intervenes on the confounder | `tab:disc_skeleton` |
| Cost of realistic evidence | per-window solve rate 0.197 sampled against 0.953 oracle, matched arms | `app:evidence_cost` |
| Contamination | false detections grow with n: 3.1% at n=100 to **17.3%** at n=10,000 | C8a |
| Robustness | myopic/learned ratio 1.52–1.70 across five corners, 3/3 seeds each | C11 |
| Adversarial corner | 0.98×, **0 of 3** seeds ahead; learned 0.183, myopic 0.179, random 0.217 | C11 |
| Detection channel, mean \|r\| | linear 0.782, tanh 0.654, V-shaped 0.051 | `ma/scm.py` |
| ε-greedy control | learned leads in 52 of 60 seeds; trails on the cell mean in 3 of 20 | `tab:epsgreedy_all` |

## Citations the chapter needs

- Shimizu et al. 2006 (LiNGAM) — linear non-Gaussian is observationally identifiable.
- Hoyer et al. 2009; Peters et al. 2014 — nonlinear additive noise, identifiable generically.
- Peters and Bühlmann 2014 — equal-variance Gaussian, identifiable. Already cited in Ch3.

The argument they support: **linear-Gaussian with unequal variances is the one family that is
not**, and `noise_range` spans a factor of three specifically to break equal variance. Use
`\citep{}` only — `\citet{}` is broken under `ieeetr.bst`.

## MUST NOTs specific to this chapter

- **MUST NOT** claim sample efficiency. 12,000 episodes to beat a zero-training heuristic that
  wins at 4,000. Say so.
- **MUST NOT** state "federation costs nothing". It is an equivalence bound at cells where 12
  episodes of 2,400 carry the comparison. "No cost detectable at these difficulty levels."
- **MUST NOT** claim the advantage holds in every corner, or quote the 1.52–1.70 ratio range as
  covering the grid. The adversarial corner is 0.98× with 0 of 3 seeds ahead.
- **MUST NOT** present the policy as exploiting non-Gaussianity or nonlinearity. It reads two
  moments and a linear correlation: robust because blind.
- **MUST NOT** claim a comparison against Bayesian experimental design that was not run.
- **MUST NOT** describe the primary metric as counting errors. It counts residual ambiguity.
- **MUST NOT** quote any number from a `square` (z²) mechanism. Those runs were numerically
  degenerate and were deleted.

## Data still landing, and how to write around it

Three fleets are running. **Do not write a sentence that depends on them**; write the sections
that do not, and leave a marked gap.

1. `rho12on` — the RQ2 answer-rate sweep at k=12 with compensated budget and belief channels
   on. Until it lands, RQ2's boundary sentence in 5.1 must say the k=8 grid trained every
   answer rate at one budget, so training pressure fell with the dial.
2. `rho12b` — the same fleet with channels off; the ablation showing the observation features
   are a precondition for clearing the competence floor.
3. `credit12k` — the turn-aware credit ablation retrained to 12,000 episodes at k=8. Until it
   lands, 5.3 should not lean on the credit result.

## Before pushing

Four gates, all from the repo root, all must pass:

    .venv/bin/python scripts/check_mustnots.py
    .venv/bin/python scripts/check_style.py
    .venv/bin/python scripts/check_reporting.py --captions
    .venv/bin/python scripts/mark_provenance.py --check

Style is `thesis/WRITING_GUIDELINES.md` and `thesis/WRITING_CRITIQUE.md`. The two rules that
catch drafts most often here: no announced enumeration ("this section makes three points"), and
"rather than" is capped at 2.0 per 1,000 words — state the positive claim.

`thesis/` pushes to **Overleaf only**, never GitHub.
