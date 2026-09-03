# Overnight, 2 September — what changed while you slept

Read this first. Three results got stronger, one central claim was withdrawn, and the machine
is still working.

---

## 1. The headline is now six seeds and it strengthened

$k_v = 20$, the principal cell, seeds 3--5 trained overnight with the config lifted verbatim
from the original sweep job so all six are one build.

    joint recovery   learned 0.9900 +/- 0.0100   myopic 0.8958   gap +0.0942
    SHD              learned 0.00000             myopic 0.00051
    5 of 6 seeds commit ZERO errors across 200 episodes each
    6 of 6 paired differences significant, 3.8 to 4.8 SE

The three-seed figures were 0.980 and +0.083. Doubling the seeds moved the headline toward the
claim. **This is the strongest result in the thesis and it is safe to build on.**

The sweep tables stay at three seeds per cell, because putting six into one cell would make the
window-size axis inhomogeneous in sample size. The six-seed result belongs beside the sweep as
a robustness check, not inside it.

---

## 2. The agent-count reversal does not exist — WITHDRAWN

This is the one to know about. Section 4.3 was going to report that the learned advantage
reverses as agents are added, with the learned-to-myopic ratio rising to 6.75 at ten agents,
and that this was the honest boundary of the contribution.

Two of the runs carrying it had **passed** the competence floor while nowhere near converged:

| run | window rate | SHD at 4,000 ep | at 12,000 ep | factor |
|---|---|---|---|---|
| k12s50n08b150 s2 | 0.838 PASSED | 0.00290 | 0.00005 | 58x |
| k12s50n10b150 s2 | 0.804 PASSED | 0.00220 | 0.00001 | 220x |

With those seeds trained out, the ratio goes 4.24 to **0.89** at eight agents and 6.75 to
**1.00** at ten. The learned policy is better at eight and tied at ten. Nothing on the axis
shows it losing at any agent count.

**The claim narrows to sample efficiency**: at a fixed 4,000-episode budget the learned policy
degrades as agents are added, and the degradation does not survive training to convergence.
Adding agents makes the problem slower to learn rather than less learnable. Narrower, better
supported, and it removes the method's apparent ceiling.

Section 4.3 carries a DO NOT WRITE marker. Seven uniform-budget runs (all seeds at K = 5, 8,
10) are training now, because the converged column above mixes one 12,000-episode seed into two
4,000-episode ones and cannot be quoted until the cells are uniform.

---

## 3. Your learning-rate question, answered

A lower step size makes the failing runs worse (window rate 0.519 to 0.206). More training
rescues them completely. **All seven competence-floor exclusions pass at 12,000 episodes and
all seven beat the myopic rule**, the extreme case going from 0.035 to 1.000 joint recovery.

The gate removes runs that had not finished learning rather than cells that cannot be learned,
so nothing recoverable was discarded. The sweep's means are pessimistic wherever a seed was
excluded.

---

## 4. Reward alignment — RETRACTED

Ledger 1.3 held that the learned policy is accurate where rewarded and neglects the rest.
Re-measured at 200 episodes from the selected checkpoint over six runs: shared-shared error is
**0.00000 for both learned and myopic** across 90,000 pair observations. The learned advantage
sits entirely on scored pairs, where it is 25x better. The asymmetry came from 60-episode runs.

---

## 5. Agent B: silent for four hours

No commits since 06:1x, ten sync attempts. Their transfer result stands (3/3 seeds at 200
episodes, all above 3 SE) and I confirmed it independently. The answer-rate fleet was at 5 of 21
when they last reported. **RQ2's part 3 depends on it and I cannot tell a healthy fleet from a
wedged one.** Worth a direct message when you wake.

---

## What is where

* `thesis_results/CLAIMS.md` — every claim with its number, sample, boundary, and a MUST NOT
  line. Prefer it over the ledger, which has been wrong twice in ways that reached a draft.
* `thesis_results/RETRACTIONS.md` — seventeen withdrawn claims, each naming what refuted it.
* `thesis/Appendix.tex` — the excluded runs. Needs `\input{Appendix}` in `Report.tex`.
* `scripts/figures.py` — all five figures, reproducible from data.
* Chapter 4: sections 4.1--4.4, 4.6, 4.7 drafted. 4.3 frozen pending the uniform retrains.
  4.5 blocked on agent B.

## Running now

Seven uniform-budget retrains at K = 5, 8, 10, plus a probe of whether the contended-fraction
reversal at sigma = 0.75 has the same cause. Its seed 2 sits at window rate 0.758 with joint
recovery 0.660, which is the same signature as the two above.

---

# 22:00--22:3x — the determinism rebuild, and three defects it turned up

## What the tick was for

Re-measure everything the unseeded evaluation RNG touched, and correct in writing anything I
had already told Brian or agent B that the new numbers contradict.

## 1. The defect is repo-wide, and my advice about it was wrong

Nineteen scripts load a policy and roll it out. Seventeen were unseeded. `global_shd_paired.py`
was simply the first place anyone looked.

Worse, I recommended against re-running on the grounds that every number would move by less
than its own error bar. **That is true and the conclusion does not follow.** At $k_v=30$ all
three per-seed differences moved by under one standard error, and the claim changed from two
significant seeds to one. Significance is a threshold; a sub-standard-error shift crosses it.
The cost estimate was also wrong: three hours, not the wholesale invalidation I assumed.

Corrected in `FINDINGS_DETERMINISM_2026_09_02.md` and `RETRACTIONS.md` rather than edited away.

## 2. What the re-measurement bought

The myopic arm reproduced **to five decimal places at every $k_v$ and both checkpoints**
(0.00611, 0.00082, 0.00077, 0.00053, 0.00042). Only the learned arm moved. So the episode
pairing, the graph sequence and the belief update were bit-identical throughout: the paired
comparison was always sound, and only its reproducibility was lost. That is a stronger appendix
statement than the calibration argument it replaces.

Agent B independently verified the fix is sufficient — two processes, all 180 per-episode values
identical including the learned arm.

## 3. Three defects found downstream, in order of how bad they would have been

**No figure had ever reached Overleaf.** `thesis/.gitignore` carries `*.pdf` for build
artifacts. It also matched `figures/*.pdf`, so all six `\includegraphics` targets were
untracked and the project has been compiling with every figure missing. Fixed with a
`!figures/*.pdf` exception; all six pushed. An ignored file never shows as missing in
`git status`, which is why this survived so long.

**An appendix table named an experiment that does not exist.** "Adding an attribution term to
the training reward", three runs, 0.400 / 0.355 / 0.205 against the myopic rule's 0.945. Those
runs have `reward_criterion="claims"` and `observe_owner_channel=False`. The trainer accepts
`claims` and `u14`; neither scores attribution. Across all 435 runs in the repository, **not one
was trained on an attribution objective**. What varies is the belief backend. Corrected at the
generator, relabelled `tab:attrbackend`, and the Chapter 4 bullet carries the correction inline
rather than being deleted. You raised exactly this objection two days ago; the text had not
caught up.

**The measurement fleet was mixed, not stale.** The fix landed at 21:15:49 while it was running,
so 36 outputs were written partly under each code path with nothing in the numbers to tell them
apart. A stale set announces itself on the first re-run; a mixed set survives a spot-check.
Quarantined to the scratchpad, not deleted, and the fleet relaunched.

## 4. Results that changed

**$k_v=30$:** one seed of three separates from the myopic rule, not two. `CLAIMS.md` C1 updated
and the old reading withdrawn.

**$k_v=8$:** a new MUST NOT. The three seeds disagree ($-0.00016$ ns, $-0.00059$ significant,
$+0.00096$ ns) and the mean is carried by seed 2, so no direction may be claimed there. The
crossover sits between 8 and 12, and 8 is where it is ambiguous rather than the last cell the
myopic rule wins.

**RQ3, six seeds, deterministic:** federated 0.00021 mean / 0.00013 median, centralised 0.00058
/ 0.00008, myopic 0.00068. Paired difference $-0.00037 \pm 0.00043$ across seeds. Five of six
seeds indistinguishable; the sixth favours federation and is carried by an outlying centralised
run rather than a strong federated one. At $k_v=20$ both arms are error-free across 600 episodes
and the cell settles nothing. Per-seed table now in the chapter.

**RQ4, verified and narrowed to your scope:** 14,076 observed latent groups across thirteen
configurations, **zero incorrect attributions**. The two bounds separate on a single comparison
— doubling the budget at seven peers moves two-child resolution from 63/1344 to 965/1344 and
leaves every group of three or more children at exactly zero. A resource bound responds to
resources; an identifiability bound does not.

One correction to the story we had: `attr_ceiling.py` predicted the cliff would *vanish* at one
peer, ownership being forced. It moves rather than vanishing — 76/76 at two children, 38/59 at
three, 29/74 at four, zero from five on. So there is no two-child law. Written up in
`FINDINGS_ATTRIBUTION_RQ4_2026_09_02.md`.

## 5. Written

* §4.8 negative and withdrawn results, in full — nineteen claims in five tables with the
  measurement that refuted each. Checked against `WRITING_CRITIQUE.md`: zero tics on the
  fourteen patterns it names.
* §4.6 RQ3 updated with the six-seed per-seed table.
* §4.7 RQ4: three defects corrected, and the by-group-size table added, which is the
  identifiability evidence and was missing.
* Introduction: the contributions list claimed BGe scoring, a two-agent partition, a 3.5x margin
  and an 82--91% clamp allocation. None of that describes this work. Rewritten against
  `CLAIMS.md`, seven items, every one traceable. **Emphasis is yours to set; I repaired facts.**
* RQ1's evidence-spectrum clause now points at RQ2 instead of overlapping it.

## 6. Still running at hand-off

* The 12,000-episode fleet, 12 of 18 cells at both conventions. Early and **not to be quoted**:
  $k_v=4$ appears to favour the learned arm at 12k where it favoured the myopic rule at 4k, and
  `k12s50n02b150` is an outlier at ratio 16. Both need the complete run.
* Pair-class at 12k, 8 of 9, plus a deterministic 4k comparator behind it. This tests whether
  the unrewarded-pair error is undertraining or reward alignment. The partial output points
  both ways, so I am holding.
* Agent B's 21-cell answer-rate grid, due about 00:20. `CLAIMS.md` C6 reads NOT YET AVAILABLE
  and will keep saying so until it lands. **RQ2 is a whole Results section with nothing in it.**

## 7. For you

The structure contract is `docs/THESIS_STRUCTURE_CONTRACT.md`, and agent C has been pointed at
it. Discussion and the interpretive parts of Results are untouched and left for you.

---

# Morning summary, 3 Sep 05:3x — where everything stands

## Results, by section

**RQ1 — complete.** Sweep at 12,000 episodes throughout, measured at three checkpoint
conventions, 2-of-18-to-16-of-18 as the headline (C7). Pair class measured at both budgets:
training cuts scored-pair errors 3.9x and leaves unscored pairs unchanged (C3a).

**RQ2 — data complete under the sampled convention; one boundary still open.** Agent B's
deterministic grid landed and verified here to the digit: 15/15 beyond 2 SE at rho<=0.90. The
fixed-policy decomposition rebuilt (295x against 27x). The in-regime diagonal measured 21/21:
the dial is a trade, and rho=0.95 is behind the myopic rule in its own regime on 3/3 seeds.
**Open: agent B's argmax grid shows the mid-rate advantage REVERSING under the deterministic
derivative of the policy** -- their full grid lands this morning, and C6 forbids finalising
4.2's boundary until it does. The framing that survives either way: the trained (stochastic)
policy wins below 0.95; its argmax derivative keeps only the low-rate advantage.

**RQ3 — complete, stronger than the draft ever claimed.** Ladder retrained at 12,000: zero of
six seeds separate, means and medians identical to five decimals (C4). The 4k version's one
significant seed was an unconverged run. Credit ablation measured: 15.1x pooled, 13.2x
federated -- the federation-specific mechanism never existed (the recorded field made it).

**Negative results — grown to its role.** The seventh `global_hard_shd` incident, the credit
mechanism, the attribution closed form, a misnamed metric, three generator/pipeline silent
failures: all recorded with what refuted them.

**Attribution — one self-contained appendix, per your instruction.** Three research questions.

## The two systems built tonight

`scripts/mark_provenance.py`: SUPERSEDED markers in every stale directory; both registries
fail if they read one; three deliberate exceptions listed with reasons. 41 patterns clean.

`thesis/FIGURE_GUIDELINES.md`: applied on your sign-off. Print-true sizes, 8 pt floor, the two
oversized grids now subfigure panels, titles off behind one switch. Nine chapter figures plus
the new in-regime one, all regenerated; notebook re-executed clean.

## Still running into the morning

* agent B's argmax grid (11 cells) -- the one thing 4.2's boundary waits on
* n_int=200 finite-sample re-measurement (slow by nature: CI tests per step)
* k=12 credit fill (seed 1 training) and its measurement watcher
* generator control measurement (ER trained 3/3; SF comparator already measured)
* then one `build_submission.py` rebuild under the provenance check

## Decisions waiting on you

1. §4.2's convention framing once the argmax grid lands (C6 has the MUST NOTs staged).
2. Figure guideline residuals: caption font `small` or 12 pt; Latin Modern in figures or not.
3. The analysis slots under every subsection -- the chapter is scaffolded and clean for you.
