# The SHD headline depends on which checkpoint is evaluated, and only at large k

1 Sep 2026, 21:00. Supersedes the SHD row in `docs/RESULTS_LEDGER_2026_09_01.md` section 1.2
and the 18:50 correction in `docs/AGENT_B_INBOX.md`.

## The defect

`scripts/ma_train.py:503` evaluates the FINAL in-memory policy:

    arms = {"learned": ppo.policies(deterministic=False)}

Line 496's own comment says the final policy is often worse than `_best.pt` when a better one
exists. So every SHD number the sweep has ever reported is from a checkpoint nobody selected.

## The measurement

`scripts/global_shd_paired.py`, 200 paired episodes, sampling, 3 seeds per cell, best and
final side by side. Raw: `results/ckpt/k{04,08,12,20,30}_{best,final}.json`.

Hard SHD of the pooled global graph, learned / greedy:

| k | 4 | 8 | 12 | 20 | 30 |
|---|---:|---:|---:|---:|---:|
| **best checkpoint** | 1.68 | 1.40 | **0.19** | **0.00** | 0.90 |
| final checkpoint | 1.42 | 1.00 | 0.19 | 2.34 | 16.1 |
| success gap (learned - greedy) | -0.075 | -0.025 | +0.058 | +0.083 | +0.125 |

## The three findings

**1. Two independent criteria cross in the same place.** SHD flips from learned-worse to
learned-better between k=8 and k=12; the success gap flips sign between k=8 and k=12. The
crossover is a property of the problem, not of the metric.

**2. The checkpoint choice is inert at k<=12 and decisive at k>=20.** 0.19 vs 0.19 at k=12;
0.00 vs 2.34 at k=20; 0.90 vs 16.1 at k=30. Late-training instability appears only in the
12,000-episode runs. This is the principled reason to quote the best checkpoint -- it is not
"it looked better", it is "the final policy is unstable exactly where training is longest".

**3. Which seed regresses is not stable.** Agent B's sweep saw k=20 seed 1 as the outlier;
this run sees seed 2 (0.00337). At k=30 final it is seeds 1 and 2 (0.01031, 0.00989). The
final policy's SHD is a lottery ticket.

## Why this is early stopping and not test-set leakage -- state this wherever it is quoted

Checkpoint selection is on `best_mi_ratio`, computed from TRAINING rollouts. It never sees
eval SHD. Two pieces of positive evidence, not just the argument:

* **At k=30 seed 0 the MI criterion picks the WORSE SHD checkpoint** (best 0.00108 against
  final 0.00012). If MI were covertly selecting on structural accuracy, that could not happen.
* **`resolved` is 0.959-0.975 for every arm at every k.** No arm wins by settling fewer pairs
  and thereby avoiding errors; they all settle the same fraction, and the best checkpoint
  gets more of them right.

Report both checkpoints in the chapter. The claim is about the selected policy, and the
selection rule has to be stated with it.

## Per-seed at the two cells that matter, because the means hide structure

    k=20 best   s0 -0.00053 +/- 0.00012   s1 -0.00062 +/- 0.00013   s2 -0.00044 +/- 0.00010
                3 of 3 seeds significantly better; learned is 0.00000 on all three
    k=30 best   s0 +0.00054 +/- 0.00063 (n.s.)  s1 -0.00040 +/- 0.00008  s2 -0.00026 +/- 0.00007
                2 of 3 significantly better, 1 indistinguishable

**k=30 must be reported as "2 of 3 seeds significantly better, one indistinguishable", not as
L/G 0.90.** The ratio is carried entirely by seed 0 and the ratio hides that.

## Consequence for the write-up

Section 1.2 of the ledger ("the advantage does NOT appear on average structural error")
is withdrawn as stated. The replacement is: the advantage appears on both criteria, with the
same crossover, once the selected policy rather than the final one is evaluated. Section 1.3
(the policy is accurate where it is rewarded) is unaffected and becomes a mechanism rather
than an excuse.
