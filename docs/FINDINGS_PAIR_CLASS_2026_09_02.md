# The reward-alignment explanation is refuted

2 Sep 2026, 05:5x. **Retracts section 1.3 of `docs/RESULTS_LEDGER_2026_09_01.md`.**

## The claim being tested

Ledger 1.3 held that the learned policy is accurate on the pairs its reward depends on and
neglects the rest: private-incident 0.00011 against greedy's 0.00061, and shared-shared 0.00036
against greedy's 0.00000. That asymmetry was the mechanism offered for the success/SHD
divergence, and it was quoted as "the policy does exactly what it was paid to do and neglects
the rest".

## The measurement

`scripts/shd_by_pair_class.py`, six runs (k=20 and k=30, three seeds each), **200 episodes**
each, loading `_best.pt` -- the checkpoint the thesis reports. Raw:
`results/shd_by_class_200.json`.

| arm | private-incident (scored) | shared-shared (unscored) |
|---|---:|---:|
| learned | 0.00002 | **0.00000** |
| greedy | 0.00051 | **0.00000** |
| random | 0.02302 | 0.00542 |

1,140,000 private-incident and 90,000 shared-shared pair-observations. The maximum
shared-shared error across all six runs, for both learned and greedy, is **0.00000** -- not a
small number, zero.

## What is actually true

1. **The learned advantage is entirely on private-incident pairs**, where it is 25x better
   than the myopic rule (0.00002 against 0.00051).
2. **Shared-shared pairs are solved by every competent policy.** Only random errs there. The
   claim that the learned policy neglects them is false.

## Why the old number differed

The ledger's figures were 60 episodes over four runs. At that episode count the shared-shared
denominator is roughly 1/3 the size, and a single unlucky episode moves the mean by more than
the effect being claimed. The current measurement uses the same script and the same checkpoint
convention, and is 3.3x larger.

## Consequences

* **Section 4.4 of the results chapter cannot be written as planned.** Its claim was the
  asymmetry. The replacement claim is the two facts above.
* **The success/SHD divergence no longer needs this explanation.** The divergence was largely
  an artefact of evaluating the final policy; from the selected checkpoint the two criteria
  agree (`docs/FINDINGS_CHECKPOINT_2026_09_01.md`). A mechanism was being offered for something
  that had already stopped needing one.
* **It sharpens the coordination-load result of section 1.4.** If shared pairs are always
  solved and the learned policy still degrades as agents are added, the degradation has to be
  on private pairs. Contention does not hurt by leaving shared pairs unresolved; it hurts some
  other way, and duplicated effort on the shared surface stealing budget from private
  variables is the obvious candidate. Queued as `results/shd_by_class_naxis.json` (K = 4, 8, 10
  at 200 episodes) to test exactly this.

## Standing lesson, again

This is the fourth claim tonight that read one way at low episode counts or on the wrong
checkpoint and the opposite way when measured properly. Every one was caught by re-deriving
rather than re-reading. Nothing enters the thesis on a 60-episode measurement.
