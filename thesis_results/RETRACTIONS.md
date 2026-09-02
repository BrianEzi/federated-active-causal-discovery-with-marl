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


---

# Added 2 September, afternoon and evening

The entries above were written at midday. The following day's work produced more, and the
pattern in them is different: where the earlier retractions came from too few episodes or the
wrong checkpoint, these came from **holding one variable fixed while forgetting another**.

## The training budget, three times over

| Claim | What refuted it |
|---|---|
| The advantage reverses as agents are added, rising to a ratio of 6.75 at ten | Two of the runs carrying it passed the competence floor at 0.838 and 0.804 while sitting 58x and 220x from their converged structural error. Retrained, the reversal does not survive. |
| The advantage reverses at high contention ($\sigma = 0.75$) | Same signature, same cause. Window rate 0.758 to 0.980 and joint recovery 0.660 to 0.990 with three times the training. |
| A myopic rule is sufficient at small windows; the crossover between $k_v=8$ and $k_v=12$ marks where the problem outgrows it | At 12,000 episodes the learned policy wins at $k_v=4$ and $k_v=8$ on both criteria. The crossover marks where 4,000 episodes stopped being enough. |

Three of the four structural claims in Chapter~4. What survived is the one whose cell already
trained at 12,000 episodes in the original sweep.

## Claims withdrawn about the withdrawals

Correcting a result is not the same as correcting it correctly.

| Claim | What refuted it |
|---|---|
| The competence-floor failures follow the seed and not the cell | Written off two of four runs. Seed 3 also falls below the floor at eight agents and $\sigma=0.25$. |
| The agent-count reversal disappears under convergence (ratios 0.89 and 1.00) | The comparison scored a 4,000-episode FINAL policy against a 12,000-episode FINAL policy, and the final policy degrades on long runs. Direction was right; method was not. Re-measured at a fixed checkpoint it holds. |
| Mechanism (b) for the transfer effect is refuted, because $\rho=0.95$ has the worst in-regime score and loses | $\rho=0.50$ has the second-worst in-regime score and the best transfer. A counterexample plus a supporting example is not a refutation. The defensible claim is that in-regime score does not ORDER transfer. |

## An external claim refuted by a prediction made in advance

| Claim | What refuted it |
|---|---|
| The partial oracle transfers because its belief-resolution trajectory matches genuine finite-sample evidence | Stated in advance: under calibration, $\rho=0.50$ should transfer worse than $\rho=0.70$, its distribution match being 19x worse. The curve is monotone straight through. Best transfer, worst match. The calibration measures a real quantity that is not the operative one; the mechanism is open. |

## Tooling that produced wrong numbers rather than failing

Each of these emitted something plausible instead of refusing.

| Defect | Consequence |
|---|---|
| `global_shd_paired.py` skips a seed whose checkpoint is missing, warning into a log nobody read | Three cells measured on two seeds while the report printed "/3" |
| The results collector copied result files without their checkpoints | Reused cells silently dropped from every measurement |
| The appendix generator averaged an empty list | `nan` reached Overleaf inside a table |
| The budget-comparison tool read each run's own `global_hard_shd` | A ratio of 20.79 printed for a cell that reads 0.06 when measured properly |
| A figure loaded $k_v=20$ and $k_v=30$ into a series labelled 4,000 episodes | Two 12,000-episode points on a line whose purpose was to expose exactly that error |
| The closed-form residual computed by subtracting rounded printed columns | 0.040 reported where the true value is 0.041 |

All now fail loudly: missing data drops a row and names it, reports print the real sample size,
and the residual is read from the column that carries it.

## What the pattern says

Every retraction on this page has one of two shapes. Either a quantity was measured on too
little data, or a comparison varied one thing while a second thing moved unnoticed --
checkpoint, training budget, episode count, action selection, seed count. The second kind was
invisible until each was varied deliberately, which is why the chapter now reports both
checkpoint conventions and holds the training budget explicit rather than assumed.

---

# Added 2 September, 22:0x — the determinism audit, and a retraction of my own reasoning

## Sampled evaluation was not reproducible

| Claim | What refuted it |
|---|---|
| The unseeded-RNG defect is a slip in `global_shd_paired.py` | An audit of every script that loads a policy and rolls it out: 19 such scripts, 17 unseeded, essentially all sampling. A property of how sampled evaluation was written throughout. |
| Re-running everything would change every number by less than the uncertainty already reported on it, so it is not worth doing | True of every number and false of the conclusion. At $k_v=30$ all three per-seed differences moved by under one standard error and the claim changed from two significant seeds to one. Significance is a threshold and a sub-standard-error shift can cross it. |
| $k_v=30$: two seeds separate from the myopic rule, one is indistinguishable | Deterministic re-measurement: one of three. `FINDINGS_DETERMINISM_2026_09_02.md` |

The second row is the one that matters. The argument for not re-running was cost against
benefit, and both halves were wrong: the benefit included a claim that changed rather than a
digit, and the cost was three hours rather than the wholesale invalidation assumed.

## What the re-run established that could not be inferred

Across ten measurements the myopic arm reproduced to five decimal places at every $k_v$ and
both checkpoints. Only the learned arm moved. The episode pairing, graph sequence and belief
update were bit-identical throughout, so the paired comparison was sound the whole time and
only its reproducibility was lost.

## Added to the tooling table

| Defect | Consequence |
|---|---|
| Sampled rollouts did not seed the torch generator | The same checkpoint at the same seed returned different numbers on every invocation, while the scripted arms reproduced exactly and hid it |
| The fix landed while a measurement fleet was running | 36 outputs written partly under each code path, indistinguishable from their contents. A mixed set survives the spot-check that a stale set fails. |

---

# Added 2 September, 22:5x

## A claim about the fix, withdrawn

| Claim | What refuted it |
|---|---|
| Seeding the evaluation RNG makes the reported intervals sound | It makes them reproducible. `ma/evaluate.py::run_arm_paths` records that the reseed was suppressed DELIBERATELY, because a fixed sample path makes every interval exclude policy stochasticity. Seeding restores that exclusion. Both properties are wanted; the seeded single-path evaluation has only one of them. |
| The 24-comparison audit shows the published intervals are honestly sized, the variation sitting inside them | Those re-runs were different policy sample paths over identical episodes, so their spread IS the component the within-path interval omits. A re-run 2.22 SE away shows a missing component of the interval's own order, not an interval that absorbs it. The inference was backwards. |
| The blanket justification for retracting ledger 1.3: shared--shared error is 0.00000 for both arms | True of the six $k_v=20$ and $30$ runs, false on the agent-count axis. At 12,000 episodes over 27,000 shared--shared observations the learned arm commits 11 errors and the myopic arm 0, with 7 of 9 runs at exactly zero. The retraction stands; its justification was too strong. |

## Two more tools that produced wrong output instead of failing

| Defect | Consequence |
|---|---|
| `build_claims.py` and `build_appendix.py` both wrote `thesis/Appendix.tex` | Whichever ran last won. Running them in the wrong order deleted three of the four appendices and left two `\ref` rendering as "??". Neither script errored; found by auditing labels against references. |
| `.gitignore`'s `*.pdf` rule matched `figures/*.pdf` | Six `\includegraphics` targets untracked, and the Overleaf project compiled with every figure missing for as long as figures have existed. An ignored file is never reported as missing. |
| `build_tables12k.py` wrote `thesis/Tables12k.tex`, which nothing `\input` | The 12,000-episode tables, the chapter's primary reporting, appeared nowhere in the document. |
| `measure_sweep12k.py` waited on `running[0]` rather than any worker | The fleet drained to 2 of 5 workers with 28 cells queued, throughput set by the slowest job in each batch. |

The pattern across all four: **output was produced, so nothing looked broken.** The failures are
visible only by checking that generated artefacts reach the document, which is now a label/ref
audit and an `\input` check rather than a habit.

---

# Added 3 September, 00:0x — a retraction withdrawn

## Ledger 1.3, partially restored

| Claim | Status |
|---|---|
| The learned policy is accurate where it is rewarded and neglects the rest | Retracted 2 Sep for lack of evidence. **The direction is now supported**; the strong reading is not. |

The retraction rested on 200 episodes over six runs at $k_v=20$ and $30$, where the
shared--shared error is 0.00000 for both arms. That measurement was correct and its
generalisation was not: on the agent-count axis the learned arm is nonzero and the myopic arm
is not.

Measured properly at both budgets over nine runs, 673,200 rewarded and 27,000 unrewarded pair
observations:

| budget | learned, rewarded | learned, unrewarded | myopic, unrewarded |
|---|---|---|---|
| 4,000 | 487 | 11 | 0 |
| 12,000 | 125 | 11 | 0 |

Tripling the budget improves the rewarded class 3.9x and produces no detectable change on the
other. That is a control (an arm that does not train), a dose (the budget) and a direction
predicted before it was measured, which is more than the original claim ever had.

**What stays withdrawn.** "Neglects the rest" implies a material cost. The cost is 11 errors in
27,000 observations, 0.04%, and with eleven events the Poisson standard error is about 3.3 --
so this is *no detectable change*, not exactly none. A sentence giving the direction without the
magnitude overstates the result, and `CLAIMS.md` C3a forbids it.

This is the only entry on this page to move in this direction. It is here because a retraction
made on too little evidence is the same error as a claim made on too little evidence, and the
page would be dishonest if it only recorded one of them.
