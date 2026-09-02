# Power-limited oracle training transfers to sampled evidence, and beats greedy there

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

A policy trained under **power-limited oracle evidence** -- oracle ancestry answers withheld
with probability `1 - p`, which costs ~0.085 s/episode -- is evaluated under **genuine
sampled evidence**, the 6-9 s/episode regime it never saw during training. Hard SHD of the
pooled global graph, paired per-episode differences against the greedy baseline:

| seed | learned | greedy | random | paired learned - greedy | verdict |
|---|---|---|---|---|---|
| 0 | 0.04628 | 0.04388 | 0.05559 | +0.00239 +/- 0.00609 | tied |
| 1 | 0.03511 | 0.04707 | 0.06090 | **-0.01197 +/- 0.00495** | beats greedy, 2.4 SE |
| 2 | 0.03404 | 0.04468 | 0.05452 | **-0.01064 +/- 0.00524** | beats greedy, 2.0 SE |

**Two of three seeds beat greedy significantly under evidence they never trained on. The
third ties. None lose.**

Config: `--budget 70 --evidence_power 0.85 --observe_belief_channels
--observe_reprobe_signal --train_episodes 8000`.

## 2. The power dial is the cause, isolated

`results/power/p10.json` and `p07.json` differ in **exactly one config field**, verified
field by field: `vs_evidence_power` (1.0 against 0.7). Same budget 35, same 4000 episodes, no
channels, no reprobe signal, identical credit/FedAvg/normalisation settings. Greedy scores
identically in both transfer tests (0.06649 hard SHD), which confirms the pairing.

| training regime | learned - greedy under sampled evidence |
|---|---|
| `vs_evidence_power = 1.0` (plain oracle) | **+0.02686 +/- 0.00806** -- loses, 3.3 SE |
| `vs_evidence_power = 0.7` | -0.00399 +/- 0.00435 -- tied |

**Turning the power dial alone moves transfer from significantly losing to tied, a swing of
~0.031.** The plain-oracle row independently reproduces `FINDINGS_2026_08_27`'s finding that
oracle-trained policies do not transfer, which is what makes the comparison meaningful rather
than circular.

## 3. In-regime score does not predict transfer, and appears to anti-predict it

Paired per-episode window rate, 150 episodes, in the training regime:

| run | greedy | learned | gap | +/-1 SE | significant |
|---|---|---|---|---|---|
| 8000ep seed 0 | 0.943 | 0.808 | -0.135 | 0.024 | yes |
| 8000ep seed 1 | 0.955 | 0.923 | -0.032 | 0.015 | yes |
| 8000ep seed 2 | 0.957 | 0.843 | -0.113 | 0.021 | yes |
| 4000ep seed 0 | 0.943 | 0.895 | -0.048 | 0.018 | yes |
| 4000ep seed 1 | 0.955 | 0.705 | -0.250 | 0.028 | yes |
| 4000ep seed 2 | 0.957 | 0.785 | -0.172 | 0.024 | yes |

**Every policy is significantly behind greedy in its own training regime, yet two of them
significantly beat greedy at transfer.** Seed 2 is the sharpest case: -0.113 in-regime,
-0.0106 at transfer. Seed 0 is the reverse -- strongest of the 4000ep runs in-regime, and the
only one that fails to beat greedy at transfer.

Consequence for methodology: **in-regime performance is not a valid selection signal for
transfer.** Any procedure that picks checkpoints or seeds on in-regime score is selecting
against the property being claimed.

## 4. The proxy is calibrated, and its limits are known

`scripts/power_vs_sampled_distribution.py` plays a belief-independent `RandomAgent` against
both evidence regimes on matched seeds, so only the evidence rule differs.

**Resolution speed matches.** Mean absolute difference between the power-limited and genuine
sampled belief-resolution trajectories, by window size:

| k | p=0.90 | p=0.85 | p=0.80 |
|---|---|---|---|
| 8 | 0.0090 | **0.0042** | 0.0084 |
| 12 | 0.0089 | **0.0060** | 0.0076 |
| 20 | 0.0057 | **0.0032** | **0.0032** |
| 30 | 0.0137 | 0.0111 | **0.0085** |

(k=8 also swept 1.0/0.95/0.7/0.5; the curve is U-shaped with a clear minimum.) `p = 0.85` is
optimal or tied-optimal through k=20 and is overtaken by 0.80 at k=30, so **the optimal power
drifts downward as the window grows** and should be recalibrated per scale rather than fixed.

**Fallibility does not match, and cannot.** Sampled evidence settles pairs on a WRONG mark at
a rate rising to a ~2% plateau; power-limited evidence produces exactly 0.000 error at every
power value and every round, by construction -- withholding is sound, it can only decline to
answer. So the proxy reproduces the SPEED of sampled belief resolution but not its
FALLIBILITY.

Why this does not undermine the result: a settled-wrong pair is observationally identical to
a settled-right one (both read as a clean 1.0/0.0), so no policy can perceive or learn to
recover from it -- **and greedy, reading the same belief, is equally blind.** The ~2% is a
shared cost on every arm under sampled evidence, not a penalty specific to the training
regime. It does set an absolute floor on achievable SHD for any policy.

## 5. What is NOT established

* **The full effect is not attributed.** The winning configuration changes power, budget,
  observation channels, the reprobe signal and episode count together relative to `p10`. The
  power dial is isolated by section 2; the -0.012 win as a whole is not.
* **"Substitutes for sampled training" is unproven.** Every comparison here is against
  GREEDY under sampled evidence. Establishing substitution needs a sampled-TRAINED policy at
  the same cell, which is exactly the cost the method exists to avoid. Partial k=8 sampled
  checkpoints exist in `results/sampled_learned/` and are the closest available proxy.
* **Scale.** Everything here is k=8 (calibration reaches k=30, the transfer result does not).
  The thesis headline cells are k=20 and k=30.
* **Effect sizes are 2.0-2.4 SE at 40 episodes.** Real but marginal; more episodes would firm
  this up cheaply.
* **Which observation feature earns the improvement is unknown.** The channels-vs-reprobe
  ablation was measured on window rate, which cannot resolve it -- see below.

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

* Transfer: `results/power/TRANSFER_seed{0,1,2}_final.json`
* Power isolation: `results/power/transfer_p{10,07,05}.json`, `results/power/p{10,07,05}.json`
* Calibration: `results/power/dist_compare_k8_b35_with_error.json`, `dist_compare_k{12,20,30}.json`
* Checkpoint sweeps: `results/power/ckptsweep_{4k,long}_s{0,2}.json`
* Tooling: `scripts/power_vs_sampled_distribution.py`, `scripts/power_window_rate.py`,
  `scripts/checkpoint_sweep_window_rate.py`, `scripts/diversity_probe.py`
* Feature: `--observe_reprobe_signal` (`ma/env.py::_reprobe_signal`), opt-in, off by default
