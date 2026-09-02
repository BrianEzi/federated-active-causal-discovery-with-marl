# Scripts and documents central to the thesis

Not an inventory of the repository. These are the files a reader would need to follow how a
number was produced or why a claim is stated the way it is.

## Measurement

| file | what it does |
|---|---|
| `scripts/global_shd_paired.py` | **The measurement every structural claim rests on.** Plays each arm over identical episode seeds and reports the standard error of the per-episode *difference*. Takes `--checkpoint best\|final\|u0500` and `--sample`. Its docstring explains why it exists separately from `scripts/shd.py` and how the two disagree in sign. |
| `scripts/shd_by_pair_class.py` | Splits structural error into pairs the acting agent is scored on and pairs it is not. |
| `scripts/measure_sweep12k.py` | Measures each 12,000-episode cell at all three conventions as its seeds land. Resumable. |
| `scripts/attr_model.py` | The closed-form attribution predictor and its residuals. |

## Engine

| file | what it does |
|---|---|
| `cb/component_attribution.py` | Exact attribution by factoring the candidate set over connected components. What makes k=50 reachable where the joint formulation holds 8.4e10 hypotheses. |
| `cb/attribution.py` | The two pruning rules: atomicity, sound; local disturbance, a declared modelling assumption. |
| `cb/factored.py` | The version-space belief, the two evidence regimes, and the partial oracle. |
| `ma/policy.py` | Training, FedAvg, and the batched rollout collection (`_act_many`), which batches forward passes but deliberately not sampling, so the RNG stream is unchanged. |
| `ma/baselines.py` | The comparison policies. `PartitionedGreedyAgent`'s docstring states why it exists: without it, "our baseline was uncoordinated" is not an answer. |

## Reproduction

| file | what it does |
|---|---|
| `notebooks/thesis_figures.ipynb` | Every figure, one section each, drawn from raw files with the feeding table printed above it. |
| `scripts/figures.py` | The same figures rendered headlessly for the LaTeX build. |
| `scripts/build_submission.py` | Assembles this directory and verifies it against the working tree. |
| `scripts/build_appendix.py` | Generates the appendix tables from data; drops any row it cannot compute rather than emitting a number. |
| `scripts/build_claims.py` | Generates `thesis_results/CLAIMS.md`. |
| `scripts/verify_batched_rollout.py` | Proves the batched rollout is action-identical to the sequential one at a fixed seed. |

## Records

| file | what it holds |
|---|---|
| `thesis_results/CLAIMS.md` | What may be asserted, with the sample behind it and a hand-maintained MUST NOT line per claim. **Preferred over the ledger where they disagree.** |
| `thesis_results/RETRACTIONS.md` | Every withdrawn claim and what refuted it, including three corrections to corrections. |
| `docs/FINDINGS_CHECKPOINT_2026_09_01.md` | Why the reported policy is the selected checkpoint and why that is early stopping rather than test-set selection. |
| `docs/FINDINGS_CHECKPOINT_TAIL_2026_09_02.md` | Why both checkpoint conventions are reported: at 12,000 episodes each has a tail the other does not. |
| `docs/FINDINGS_UNDERTRAINING_2026_09_02.md` | All seven competence-floor exclusions pass when retrained, and a lower learning rate makes them worse. |
| `docs/FINDINGS_AGENT_COUNT_2026_09_02.md` | The agent-count and contention reversals as training-budget artefacts. |
| `docs/FINDINGS_CROSSOVER_2026_09_02.md` | The window-size crossover, the third claim to resolve the same way. |
| `docs/FINDINGS_TRANSFER_2026_09_02.md` | The answer-rate curve, its verdict, and the calibration mechanism refuted by advance prediction. |
| `docs/FINDINGS_PAIR_CLASS_2026_09_02.md` | The retraction of the reward-alignment result. |
| `docs/AGENT_B_INBOX.md` | The full working correspondence between the two machines. Long, and the record of how each result was pushed on. |

## One caveat carried throughout

A result file's own `global_hard_shd` is that run's evaluation at its final update. It is not
what the dissertation reports, and on a long run the two differ by up to a factor of 300 on the
same seed. Three separate errors in this project came from reading it as though it were.
