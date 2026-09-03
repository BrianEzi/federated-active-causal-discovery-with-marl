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

---

## Update, 3 Sep 07:xx — the k=12 replication lands, and it must not be over-read in either direction

The two missing `E4_nocredit` seeds trained and all four $k_v=12$ cells are measured
(`results/credit/shd/k12*`), 200 paired episodes per seed, selected checkpoint, 4,000-episode
runs throughout:

| configuration | mean | per seed |
|---|---|---|
| Pooled, credit on | 0.00022 | 0.00000, 0.00066, 0.00000 |
| Pooled, credit off | 0.00003 | 0.00005, 0.00000, 0.00005 |
| Federated, credit on | 0.00079 | 0.00046, 0.00002, 0.00189 |
| Federated, credit off | 0.00683 | 0.00034, 0.00066, **0.01947** |

Ratios read naively: pooled $0.1\times$, federated $8.6\times$ -- which looks like the
federation-specific interaction this document retracted at $k_v=8$. It is not evidence of it:

* **The pooled cell is uninformative at $k_v=12$.** Both states sit at the floor (differences
  of $0.0002$ between arms whose $k_v=8$ effect was $0.0035$). A cell with no room to degrade
  cannot show whether credit matters; absence of effect and absence of headroom are
  indistinguishable here.
* **The federated degradation is one seed again.** 0.01947 of the 0.02047 total. Every
  credit-off degradation measured on this project -- both arms at $k_v=8$, federated at
  $k_v=12$ -- is carried by a single seed of three.

**What may be written:** removing turn-aware credit costs roughly an order of magnitude
wherever there is room to lose it (both optimisers at $k_v=8$; the federated arm at
$k_v=12$), the degradation is seed-concentrated everywhere, and no interaction between the
fix and the optimiser is established at either window size. All cells at 4,000 episodes,
stated with the claim.

**What may not:** reading the $k_v=12$ pooled null as the interaction returning. That null is
a saturated cell, and the $k_v=8$ measurement -- where both arms had headroom -- showed both
degrading alike.
