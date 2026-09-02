# Turn-aware credit is not a federation-specific fix, and the claim that it was came from the wrong field

3 Sep 2026, 02:xx. Four cells at $k_v=8$, three seeds each, 200 paired episodes per seed at the
selected checkpoint, measured with `scripts/global_shd_paired.py`.

## What the chapter said, and what the data says

| configuration | recorded | measured | per seed, measured |
|---|---|---|---|
| Pooled, credit on | 0.00160 | **0.00025** | 0.00069, 0.00000, 0.00005 |
| Pooled, credit off | 0.00137 | **0.00376** | 0.00032, 0.00016, 0.01080 |
| Federated, credit on | 0.00106 | **0.00135** | 0.00090, 0.00021, 0.00293 |
| Federated, credit off | 0.01917 | **0.01773** | 0.01872, 0.00234, 0.03213 |

Recorded, removing credit made the pooled arm slightly *better* (0.00137 against 0.00160) and
the federated arm eighteen times worse. That asymmetry was the whole result: a correctness fix
that only bites under the federated optimiser.

Measured, removing credit costs the pooled arm **15.1x** and the federated arm **13.2x**.

**There is no federation-specific effect.** The interaction was an artefact of reading each
run's own `global_hard_shd`, which scores the last update rather than the reported checkpoint.
This is the sixth place that field has been read as the reported metric on this project.

## What may be claimed now

Turn-aware credit is worth roughly an order of magnitude under both optimisers. The claim is
about magnitude and not about which regime it applies to.

## What may not

That 15.1 and 13.2 differ. Both credit-off cells are carried by a single seed --- 0.01080 of
0.01128 in the pooled arm, 0.03213 of 0.05319 in the federated one --- and three seeds cannot
separate two ratios of that size. Any sentence ordering the two arms is unsupported.

## The lesson, which is now repeated rather than new

A recorded quantity that resembles the reported metric produced a clean, publishable,
mechanistically satisfying asymmetry which does not exist. It survived because nobody
recomputed it. Every SHD figure in the thesis now traces to `global_shd_paired.py`, and the
remaining risk is any table sourced before that rule was applied.
