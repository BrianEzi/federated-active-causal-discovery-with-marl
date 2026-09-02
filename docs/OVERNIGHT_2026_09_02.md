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
