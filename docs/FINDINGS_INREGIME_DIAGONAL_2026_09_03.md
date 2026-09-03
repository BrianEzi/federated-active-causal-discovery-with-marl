# The in-regime diagonal, measured: the dial trades in-regime ground for transfer ground, and rho=0.95 is now anomalous in a measured quantity

3 Sep 2026, 05:0x. All 21 answer-rate policies evaluated in their own training regime with
`global_shd_paired.py`, 200 paired episodes per seed, selected checkpoint, seeded evaluation.
`results/power/rho/inregime_det/`. Until tonight the in-regime axis of every plot and
correlation was each run's own recorded `global_hard_shd`.

## First: how wrong was the recorded field here?

Barely -- Pearson **+0.956** against the measurement across the 21 cells. This grid is the
benign case: `best_update == final` everywhere, so the field's usual checkpoint mismatch did
not apply. The seventh appearance of the field was the only harmless one, and knowing that
required the measurement.

## The diagonal

| $\rho$ | in-regime $\Delta$ (learned $-$ myopic) | seeds beyond 2 SE | transfer $\Delta$ |
|---|---|---|---|
| 1.00 | $-0.00021$ | 0/3 | $+0.01090$ |
| 0.95 | $+0.00411$ | **3/3 worse** | $+0.00195$ |
| 0.90 | $+0.00005$ | 0/3 | $-0.00826$ |
| 0.85 | $+0.00009$ | 1 worse, 1 better | $-0.00936$ |
| 0.80 | $-0.00129$ | 2/3 better | $-0.01317$ |
| 0.70 | $-0.00340$ | 3/3 better | $-0.01766$ |
| 0.50 | $-0.00686$ | 3/3 better | $-0.01856$ |

Measured in-regime against measured transfer, 21 cells: Pearson $+0.609$, Spearman $+0.795$.
The direction agent B established on the recorded field survives measurement.

## Three statements the diagonal supports

1. **The dial is a trade, and both sides of it are now measured.** As $\rho$ falls the learned
   arm gains ground on the myopic rule in its own regime (from tied at $1.00$ to $-0.00686$ at
   $0.50$, 3/3 seeds) and gains far more at transfer. There is no rate where the learned arm
   pays in-regime without collecting at transfer, except one:

2. **$\rho=0.95$ is anomalous in a measured quantity, on every seed.** The learned arm is
   significantly WORSE than the myopic rule in its own regime on 3 of 3 seeds -- the only rate
   with that property -- while collecting no transfer benefit. This was flagged, retracted for
   want of a mechanism, and kept visible on agent B's insistence. It is now two measured
   quantities (in-regime deficit here, the window-rate dip at 0.95) plus the transfer curve
   all singling out the same rate. Still no mechanism; but it is no longer a wobble in one
   number.

3. **A single amplification factor does not survive measurement.** The old panel-3 annotation
   said transfer amplifies the rho effect "~2.9x". Measured, the ratio runs $\times 10.2$ at
   $\rho=0.80$, $\times 5.2$ at $0.70$, $\times 2.7$ at $0.50$, and is undefined at the mid
   rates where the in-regime delta is within noise of zero. The supportable sentence: the
   dial's effect is mostly invisible in-regime and large at transfer, and no single
   multiplier describes that.

## What this does NOT touch

All of this is the sampled convention. Agent B's argmax grid, landing now, shows the mid-rate
transfer advantage REVERSING under argmax; nothing in this note bears on that either way.
