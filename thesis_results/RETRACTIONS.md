# Claims made and withdrawn

Source material for the negative-results section of Chapter 4. **Hand-maintained**, unlike
`CLAIMS.md`: a retraction is a judgement about evidence and cannot be derived from the data
files. Every row names the measurement that refuted it, so each is checkable.

The pattern worth stating in the chapter: with two exceptions, every claim below was refuted by
a measurement queued specifically to test it, and most were refuted by measuring the same thing
at a larger sample or on the correct checkpoint.

## Refuted by a larger sample or the correct checkpoint

| Claim | What refuted it |
|---|---|
| Learned beats myopic 5--11x on SHD at $k_v=20$ and $30$ | Full three-seed replication at 12,000 episodes. |
| Learned no longer beats myopic on SHD at $k_v=20$ | The sweep evaluates the final policy (`ma_train.py:503`). From the selected checkpoint the learned arm commits no errors on any seed. `FINDINGS_CHECKPOINT_2026_09_01.md` |
| The learned policy is accurate where rewarded and neglects the rest | 200 episodes over six runs: shared--shared error is 0.00000 for both learned and myopic across 90,000 pair observations. The asymmetry came from 60-episode runs. `FINDINGS_PAIR_CLASS_2026_09_02.md` |
| The learned policy attributes worse than random | One seed at 2 SE; the next two reversed it. |
| Power-limited evidence closes the transfer gap | Training had not converged, and the arms' RNG was unpaired so they never saw the same worlds. |
| Transfer beats the myopic rule on two of three seeds | 200 episodes instead of 40: three of three, none below 3 SE. The weaker claim was correct but understated. |
| The agent-count reversal begins at five agents | At $K=5$ it is one seed: ratio 1.65 with all seeds, 0.25 without. |

## Refuted by a control queued to test the mechanism

| Claim | What refuted it |
|---|---|
| Attribution precision collapses from 98\% to 59\% as $k_v$ grows | Two engine defects. Zero misattributions at every size after repair. |
| The component engine gains precision by skipping cross-component pruning | A probe for such messages found none. |
| The site-count collapse is exponential hypothesis-space growth | The matched-budget control, holding rounds per agent fixed. It was coverage. |
| Probe diversity explains attribution performance | The lowest-coverage policy ties the highest. |
| The competence-floor exclusions are broken runs | All seven pass at 12,000 episodes and all seven beat the myopic rule. `FINDINGS_UNDERTRAINING_2026_09_02.md` |
| A lower learning rate would rescue the excluded runs | It makes them worse: window rate 0.519 to 0.206, and 0.345 to 0.177. |
| The exclusions follow the seed and not the cell | Seed 3 also falls below the floor at eight agents and $\sigma = 0.25$. |

## Arithmetic and provenance errors, corrected rather than retracted

| Error | Correction |
|---|---|
| Coverage quoted as 5\% rising to 77\% | Those were shares of a superseded ceiling estimate. The file holds 21 of 1056 and 349 of 1056. |
| Closed-form residual quoted as 0.040 | 0.041. The figure was subtracting rounded columns instead of reading the residual column. |
| Ledger 1.3 and 2.2 | Both carry retraction banners in place. `CLAIMS.md` supersedes the ledger where they disagree. |

## What did not survive contact and was reopened

Power-limited oracle evidence was closed at grade D on a replication that had been gated on the
wrong metric. Re-gated on per-window recovery it passes 6 of 6 rather than 1 of 6, and the
approach now carries RQ2. A negative result was itself withdrawn.
