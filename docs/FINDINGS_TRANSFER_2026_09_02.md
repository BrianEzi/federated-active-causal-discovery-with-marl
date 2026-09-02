# Partial-oracle training transfers to sampled evidence, and beats greedy there

> **Terminology.** This document says **partial oracle** and **answer rate rho**, never
> "power" -- statistical power is what the finite-sample regime is actually about and the
> collision would confuse a reader on the one page where both appear. The config flag remains
> `--evidence_power` and the paths remain `results/power/`; those are identifiers, not prose.

2 Sep 2026, 03:00. **Supersedes the conclusion of
`FINDINGS_POWER_LIMITED_EVIDENCE_2026_09_01.md`**, which closed this line at grade D on a
replication that turned out to have been measured through the wrong gate on the wrong metric.
That file's sections 2 and 3 should be read as retracted; its section 4 (distance-weighted
withholding fails) still stands.

All cells below are k=8, 4 agents, `factored` backend, `--turn_aware_credit --local_epochs 4
--normalise_returns`, scale-free graphs. Evaluation is `scripts/global_shd_paired.py` with
`--sample --override_evidence sampled`, 40 paired episodes, `_best.pt`.

---

## 1. The result

A policy trained under a **partial oracle** -- oracle ancestry answers withheld with
probability `1 - rho`, costing ~0.085 s/episode -- is evaluated under **genuine sampled
evidence**, the 6-9 s/episode regime it never saw in training. Hard SHD of the pooled global
graph, paired per-episode against greedy, **200 episodes, 7 rates x 3 seeds = 21 cells**:

| rho | n | learned | greedy | delta | seed SE | verdict |
|---|---|---|---|---|---|---|
| 1.00 | 3 | 0.05812 | 0.04846 | +0.00966 | 0.00759 | tied (control) |
| 0.95 | 3 | 0.05144 | 0.04846 | +0.00298 | 0.00188 | tied |
| 0.90 | 3 | 0.03918 | 0.04846 | -0.00927 | 0.00071 | **beats greedy** |
| 0.85 | 3 | 0.03945 | 0.04846 | -0.00901 | 0.00188 | **beats greedy** |
| 0.80 | 3 | 0.03555 | 0.04846 | -0.01291 | 0.00163 | **beats greedy** |
| 0.70 | 3 | 0.03184 | 0.04846 | -0.01661 | 0.00032 | **beats greedy** |
| 0.50 | 3 | 0.03060 | 0.04846 | -0.01785 | 0.00141 | **beats greedy** |

**15 of 15 seeds at rho <= 0.90 beat greedy, every one of them beyond 2 paired SE (weakest
-2.46, strongest -10.07). None of the 6 at rho >= 0.95 does.** Spread across rates 0.02752
against a typical seed SE of 0.00220 -- **12.5x the noise.**

The separation is stated in terms of significance because the two halves do not separate on
sign alone, and an earlier draft of this sentence hid that. Two of the six high-rate cells --
rho=1.00 seed 1 and rho=0.95 seed 2 -- are numerically ahead of greedy, at -1.57 and -0.29
paired SE. Both are inside noise, and two of the remaining four are significantly BEHIND, so
"no high-rate cell beats greedy" is true of what was measured and false of the raw sign. The
low-rate half needs no such qualification: all fifteen are ahead on sign and all fifteen clear
2 SE. Per-cell deltas, SEs and both counts: `results/power/rho/DETERMINISTIC_COMPARE.json`.

The curve is **monotone and saturating**: it improves all the way to rho=0.50 but the last
step is a fifth of the earlier ones. There is no interior optimum in the swept range, and no
floor is claimed since rho < 0.50 is unmeasured. The zero crossing lies between rho=0.95 and
rho=0.90.

Greedy scores identically (0.04846) at every rate because the baseline arms are the SAME
per-episode vectors, reused across rates -- an exact pairing, and a correctness check that the
comparison did not shift underneath.

Config: `--budget 70 --observe_belief_channels --observe_reprobe_signal --train_episodes 8000`,
k=8, 4 agents, `factored` backend. Only `--evidence_power` varies.

**Only the dial and the seed vary, across the whole grid.** Comparing all 21 configs field by
field against `rho1.00_s0`, exactly two fields ever differ: `ppo_seed` and `vs_evidence_power`.
Section 2 isolates the cause using the `p10`/`p07` pair; this extends that isolation to every
cell in the grid rather than to one pair of runs.

**One cell of 21 fails the project's own competence floor, and it is at the best-transferring
rate.** `rho0.50_s2` averages 0.637 per-window identification over its last ten evaluations,
below the `WINDOW_FLOOR = 0.70` that excludes runs from the four-axis sweep. Reported rather
than quietly kept, because the rate it sits at is the endpoint carrying the strongest claim.
It does not carry that claim: its transfer delta is -0.016862, the middle of the three seeds,
and dropping it moves rho=0.50 from -0.01785 to **-0.01835** -- slightly stronger, not weaker.
The count becomes 14 of 14 rather than 15 of 15 and the curve's shape is unchanged. Two further
cells sit near the floor (`rho0.50_s0` at 0.795, `rho0.50_s1` at 0.766); in-regime competence
falls as the answer rate falls, which is what section 3 is about, and is a reason to read the
floor as an in-regime measure rather than a transfer one.

**Checkpoint selection is inert here, which closes a confound without a control run.** The grid
is scored from `_best.pt`, and `_best.pt` is selected on `mi_ratio` -- I(S;A)/H(A), a
training-health gate, deliberately not reward and not `success`. The gate is nonetheless
evaluated inside the training regime, so it is measured on a different state distribution at
every rate, and a reader is entitled to ask whether low rates simply got better-chosen
checkpoints. They did not: **`best_update` is 499, the last update, in all 21 runs.** `best`
and `final` are the same policy at every cell, so selection cannot vary with rho because no
selection happened. A `--checkpoint final` control would reproduce the grid exactly and is not
worth the hour it would cost.

**The separation survives argmax evaluation; the effect SIZE does not.** The grid is scored by
sampling at temperature 1, the project's convention, and the action-selection control was run
at two rates -- the pivot and a clear winner -- but its result was never written down here.
200 paired episodes, same checkpoints, learned arm scored by argmax instead:

| rho | argmax delta | sampled delta | argmax per-seed significance |
|---|---|---|---|
| 0.95 | +0.02658 +/- 0.00268 | +0.00298 | 3/3 significantly WORSE than greedy |
| 0.70 | -0.00512 +/- 0.00052 | -0.01661 | 2/3 significantly better (-2.57, -2.06, -1.96) |

The direction holds at both rates and the separation between them widens, so the qualitative
claim is not a convention artefact. The magnitude is: at rho=0.70 the advantage shrinks by a
factor of 3.2 and one seed of three drops just under the 2 SE bar. **Any effect size quoted
from this grid is a sampled-evaluation number and must be labelled as one.** The control covers
2 of 7 rates; the shape of the curve under argmax is unmeasured.

The same table carries a limitation. `mi_ratio` was still rising at the last update in 21 runs
out of 21, so none of these policies is converged on the training-health criterion at 8,000
episodes. What the grid compares is 21 equally-undertrained policies, which is a fair
comparison and not a converged one. Its best value also falls as the answer rate falls (0.38 to
0.51 at rho=1.00, 0.27 to 0.36 at rho=0.50) -- the arms that transfer best score worst on the
gate. Across all 21 cells the association between gate score and transfer delta is r=+0.42,
p=0.06, and it is confounded with rho; within a rate there are three points per correlation,
which is not a measurement. **No claim is made that the gate predicts transfer in either
direction.**

## 2. The answer rate is the cause, isolated

`results/power/p10.json` and `p07.json` differ in **exactly one config field**, verified
field by field: `vs_evidence_power` (1.0 against 0.7). Same budget 35, same 4000 episodes, no
channels, no reprobe signal, identical credit/FedAvg/normalisation settings. Greedy scores
identically in both transfer tests (0.06649 hard SHD), which confirms the pairing.

| training regime | learned - greedy under sampled evidence |
|---|---|
| `vs_evidence_power = 1.0` (plain oracle) | **+0.02686 +/- 0.00806** -- loses, 3.3 SE |
| `vs_evidence_power = 0.7` | -0.00399 +/- 0.00435 -- tied |

**Turning the answer rate alone moves transfer from significantly losing to tied, a swing of
~0.031.** The plain-oracle row independently reproduces `FINDINGS_2026_08_27`'s finding that
oracle-trained policies do not transfer, which is what makes the comparison meaningful rather
than circular.

## 3. In-regime score PREDICTS transfer -- and the full oracle is the only arm that degrades

> **REWRITTEN 2 Sep, 20:00.** This section previously claimed in-regime performance
> anti-predicts transfer. **That was wrong and is retracted.** It compared in-regime `success`
> -- the all-agents conjunction, which this project demoted in August for saturating -- against
> transfer hard SHD delta. Two different metrics on the two sides of the comparison. Measured
> consistently they are POSITIVELY correlated. The corrected analysis follows.

Learned-minus-greedy hard SHD on both sides, three seeds per rate:

| rho | in-regime delta | transfer delta | change on moving to sampled |
|---|---|---|---|
| 1.00 | -0.00014 | +0.00966 | **+0.00980 WORSE** |
| 0.95 | +0.00337 | +0.00298 | -0.00039 unchanged |
| 0.90 | +0.00011 | -0.00927 | -0.00938 better |
| 0.85 | +0.00053 | -0.00901 | -0.00954 better |
| 0.80 | -0.00071 | -0.01291 | -0.01220 better |
| 0.70 | -0.00397 | -0.01661 | -0.01264 better |
| 0.50 | -0.00606 | -0.01785 | -0.01179 better |

**Pearson +0.703, Spearman +0.786.** In-regime predicts transfer.

**The finding is the ASYMMETRY, not an inversion.** The full oracle is the only arm that gets
worse when moved to sampled evidence; every partial-oracle arm gets better relative to greedy,
and rho=0.95 sits exactly at the pivot. The move also amplifies the effect: in-regime deltas
span 0.0094, transfer deltas span 0.0275, a factor of **2.9**.

**Why the wrong version was seductive.** rho=1.00 scores 0.980 on `success` and looks
dominant in-regime; on hard SHD delta it is merely TIED with greedy at -0.00014. Its apparent
in-regime supremacy was a property of the saturating metric, not of the policy. A cross-metric
comparison manufactured a reversal that a single-metric comparison does not support.

**Methodological consequence, corrected.** In-regime score is a WEAK but positive predictor of
transfer, not a misleading one. Selecting checkpoints on in-regime performance is defensible;
selecting on `success` specifically is not, because it saturates exactly where the arms differ.

## 4. The proxy is calibrated -- but calibration is NOT the mechanism

> **QUALIFIED 2 Sep, 17:50.** Everything measured in this section stands. What it does NOT
> support is the implication that distribution matching explains the transfer result. The
> completed rho sweep refutes that: **the best-transferring rate has the WORST distribution
> match in the sweep, by a factor of 19.**
>
> | rho | MAD vs sampled | transfer delta |
> |---|---|---|
> | 0.85 | **0.0042** best match | -0.00901 |
> | 0.70 | 0.0310 | -0.01661 |
> | 0.50 | **0.0807** worst match | **-0.01785** best transfer |
>
> If matching the belief-resolution trajectory were the operative mechanism, transfer would
> peak at rho=0.85 and fall away either side. It does not; it improves monotonically as the
> match degrades. This section measures a real quantity that turns out not to be the one
> driving the effect. Agent A predicted the opposite before rho=0.50 landed, on record, which
> is what makes the refutation worth something.


`scripts/power_vs_sampled_distribution.py` plays a belief-independent `RandomAgent` against
both evidence regimes on matched seeds, so only the evidence rule differs.

**Resolution speed matches.** Mean absolute difference between the partial-oracle and genuine
sampled belief-resolution trajectories, by window size:

| k | p=0.90 | p=0.85 | p=0.80 |
|---|---|---|---|
| 8 | 0.0090 | **0.0042** | 0.0084 |
| 12 | 0.0089 | **0.0060** | 0.0076 |
| 20 | 0.0057 | **0.0032** | **0.0032** |
| 30 | 0.0137 | 0.0111 | **0.0085** |

(k=8 also swept 1.0/0.95/0.7/0.5; the curve is U-shaped with a clear minimum.) `p = 0.85` is
optimal or tied-optimal through k=20 and is overtaken by 0.80 at k=30, so **the optimal answer
rate drifts downward as the window grows** and should be recalibrated per scale rather than fixed.

**Fallibility does not match, and cannot.** Sampled evidence settles pairs on a WRONG mark at
a rate rising to a ~2% plateau; the partial oracle produces exactly 0.000 error at every
answer rate and every round, by construction -- withholding is sound, it can only decline to
answer. So the proxy reproduces the SPEED of sampled belief resolution but not its
FALLIBILITY.

Why this does not undermine the result: a settled-wrong pair is observationally identical to
a settled-right one (both read as a clean 1.0/0.0), so no policy can perceive or learn to
recover from it -- **and greedy, reading the same belief, is equally blind.** The ~2% is a
shared cost on every arm under sampled evidence, not a penalty specific to the training
regime. It does set an absolute floor on achievable SHD for any policy.

## 5. What is NOT established

* **The mechanism is unconfirmed.** Four candidates were tested today; two are excluded by
  measurement:
  * *calibration / distribution match* -- **EXCLUDED**, section 4: best transfer has the worst
    match by 19x.
  * *repeat rate* -- **EXCLUDED**: flat at 0.720-0.749 across every rate, Spearman +0.107,
    p=0.82. The "withholding makes re-probing pay" story does not appear in the behaviour.
  * *MI ratio* -- **PARTIAL**: the control sits at 0.452 against a partial-oracle band of
    0.29-0.36, which matches the threshold but is flat across the slope.
  * *private-node coverage* -- **NOT SEPARATED FROM THE DIAL** (was LIVE; revised 2 Sep 23:2x).
    Coverage falls 0.93 -> 0.80 as withholding rises while total moves stay flat, so effort
    REALLOCATES from private to shared nodes (14% -> 29% of moves). The earlier Spearman of
    +0.929, p=0.0025 was computed on the seven RATE MEANS. On the 21 per-seed points it is
    +0.791, p<0.0001 -- still strong, and still confounded, because coverage tracks the answer
    rate itself at +0.760. **With the rate partialled out, coverage vs transfer falls to
    +0.353, p=0.12: it adds nothing the dial does not already explain.** Repeat rate (-0.193,
    p=0.40) and moves per episode (+0.037, p=0.87) are dead on the same test.

    **This does not exclude coverage as the mechanism, and the distinction matters.** If
    coverage is the CHANNEL through which the answer rate acts, partialling out the rate
    removes precisely the variation that carries the effect, and a mediator is
    indistinguishable from a proxy in this design. What the test does establish is that
    coverage is not an INDEPENDENT predictor: there is no residual coverage effect at fixed
    rate to build a story on. Separating mediator from proxy still needs an intervention on
    coverage at fixed rho, exactly as this line said before.
* **The full effect is not attributed to the answer rate alone.** The configuration changes
  rate, budget, channels, reprobe signal and episode count together relative to `p10`. Section
  2 isolates the rate; the -0.018 win as a whole is not isolated.
* **"Substitutes for sampled training" is unproven.** Every comparison is against GREEDY under
  sampled evidence. A sampled-TRAINED arm at k=8 exists (`results/sampled_ref/`) and loses to
  greedy on 3/3 seeds -- suggestive that the expensive path is not a free win, but confounded
  by budget 35 vs 70, n_int 200 vs 20, 4000 vs 8000 episodes, and channels off. A matched arm
  costs 40-60 core-hours and was deliberately not run.
* **Scale.** Everything here is k=8. Calibration reaches k=30; the transfer result does not.
  The thesis headline cells are k=20 and k=30.
* **Which observation feature earns the improvement is unknown.** The channels-vs-reprobe
  ablation was measured on window rate, which cannot resolve it -- see section 6.
* **rho=0.95 is not a special point.** It sits 1.5 SE from the straight line through its
  neighbours; the apparent in-regime "cliff" is 2.5 SE, uncorrected for seven comparisons,
  against a metric where rho=0.80's own three seeds span twice that. Doubling its training to
  16,000 episodes left in-regime success unchanged (0.497 -> 0.497) while doubling its variance,
  and moved transfer from +0.003 to +0.014 on one seed's collapse. **It is a noisy point near
  the zero crossing, not a threshold.**

## 6. A metric caveat that invalidates several earlier comparisons

Per-episode window rate is the mean over agents of a BINARY per-window identified flag. At 4
agents it can take five values and in practice takes two (0.75 and 1.0). Measured paired
standard deviation is 0.169, so:

| gap to resolve at 2 SE | episodes required |
|---|---|
| 0.004 | 7,146 |
| 0.010 | 1,143 |
| 0.050 | 46 |

At 40-60 episodes this metric supports claims about gaps of roughly 0.05 and larger, and
nothing finer. **The channels-vs-reprobe ablation (differences of 0.04-0.08 at 40-60
episodes) is therefore retracted and needs redoing on hard SHD**, which is continuous, pairs
cleanly, and resolved a 0.012 effect on 40 episodes.

## 7. Files

* Transfer grid (21 cells): `results/power/rho/xfer_rho*_s*.json`
* Training (21 cells): `results/power/rho/rho*_s*.json`
* Curve summary: `results/power/rho/CURVE.json`; figure `results/power/rho/rho_curve.png`
* Answer-rate isolation: `results/power/transfer_p{10,07,05}.json`, `results/power/p{10,07,05}.json`
* Calibration: `results/power/dist_compare_k8_b35_with_error.json`, `dist_compare_k{12,20,30}.json`
* Mechanism probes: `results/power/rho/repeat/` (coverage + repeat rate, 3 seeds x 7 rates)
* Argmax diagnostic and control: `results/power/rho/argmax/`
* rho=0.95 doubled-training arm: `results/power/rho/rho0.95_long_s*.json` and its transfer
* Sampled-trained reference (confounded, see section 5): `results/sampled_ref/`
* Tooling: `scripts/run_rho_fleet.sh`, `scripts/rho_transfer_daemon.sh`,
  `scripts/rho_curve_report.py`, `scripts/plot_rho_curve.py`,
  `scripts/power_vs_sampled_distribution.py`, `scripts/diversity_probe.py`,
  `scripts/power_window_rate.py`, `scripts/keep_awake.py`

## 8. Methods note: how the four wrong claims in this document were made

Four claims were stated and retracted on 2 Sep. They share one cause and it is worth naming,
because the same shape will recur in any sweep of this sort.

| retracted claim | what was actually compared |
|---|---|
| an interior optimum in the transfer curve | four rates, before the fifth and sixth landed |
| a rho=0.95 anomaly | one point against six, no multiple-comparison correction |
| "weak policies" explaining that anomaly | argmax gap and transfer -- two measures of ONE quantity |
| in-regime anti-predicts transfer | in-regime `success` against transfer hard SHD |

**Every one came from comparing across a condition that was not held fixed** -- number of
seeds, position on a noisy curve, two proxies for the same underlying property, and two
different metrics. None came from a coding error, and all four survived casual review because
the numbers involved were individually correct.

Two practices caught them, and both are cheap:

1. **Plot the spread, not the mean.** The rho=0.95 "cliff" is invisible as a defect until the
   seed error bars are drawn, at which point rho=0.80's own three seeds visibly span more than
   the dip being pointed at. `plot_rho_curve.py` panel 2 now draws them for this reason.
2. **State the falsification before the data lands.** Six predictions were registered in
   `AGENT_B_INBOX.md` before their measurements; four were refuted. Without the prior
   registration, at least two of those would have been quietly reinterpreted as support.

The verdict logic in `rho_curve_report.py` also had to be given a seed guard after it printed
DOSE-RESPONSE SUPPORTED on two single-seed rates, via `np.nanmean` silently dropping their
missing standard errors. A verdict function that can fire on absent data is worse than none,
and it fired in the direction its author wanted.
