# Why the learned arm's SHD is worse — 29 August 2026

Answers the open question left in `docs/SESSION_STATE_2026_08_29.md` §1. The short version is
that **the question was malformed**: under oracle evidence on the factored backend, soft SHD
cannot measure structural error at all, and the baseline it is being used to compare against
is a direct greedy optimiser of it.

Reproduce: `scripts/shd_diagnose.py`. All numbers below are 100 paired episodes, seed 0, on
the existing `results/ladder/w*_s0.pt` checkpoints. No retraining.

---

## 1. The metric never registers an error. Not once.

Decomposing every pair's contribution to soft SHD into WRONG (the belief is confident and
confidently wrong), UNSETTLED (nothing reaches the bar) and RESIDUAL (right, with leftover
mass):

| rung | arm | soft SHD | wrong | unsettled | residual |
|---|---|---|---|---|---|
| w08 | learned (argmax) | 0.0166 | **0.0000** | 0.0166 | 0.0000 |
| w08 | greedy | 0.0107 | **0.0000** | 0.0107 | 0.0000 |
| w12 | learned (argmax) | 0.0099 | **0.0000** | 0.0099 | 0.0000 |
| w12 | greedy | 0.0053 | **0.0000** | 0.0053 | 0.0000 |
| w20 | learned (argmax) | 0.0119 | **0.0000** | 0.0119 | 0.0000 |
| w20 | greedy | 0.0049 | **0.0000** | 0.0049 | 0.0000 |
| w30 | learned (argmax) | 0.0203 | **0.0000** | 0.0203 | 0.0000 |
| w30 | greedy | 0.0154 | **0.0000** | 0.0154 | 0.0000 |

`random_vary` is also exactly 0.0000 in the WRONG column, at ten times the soft SHD. Every
unit of measured "distance to the true MAG" is residual ambiguity. None of it is error.

**This is structural, not a small-sample accident.** `cb/factored.py`'s own docstring states
the property: *"it stays unsure where the enumeration would have settled, and never settles
wrongly, because each update is individually sound."* Mass is spread uniformly over each
pair's surviving mark set, and under oracle evidence the true mark never leaves that set.
Therefore

> **soft SHD per pair ≡ 1 − 1/|surviving marks|**, taking values in {0, ½, ⅔, ¾}.

Verified rather than assumed: over 2,800 pairs per arm at w08 and 6,600 at w12 — including
`random_vary`, whose belief is far more diffuse — every per-pair value landed on that grid
(0 off-grid) and the true mark was in the survivor set every single time (0 violations). The
distribution is a count, not a distance:

| surviving marks | w08 learned | w08 greedy | w12 learned | w12 greedy | w12 random |
|---|---|---|---|---|---|
| 1 (resolved) | 97.25% | 98.50% | 98.36% | 99.24% | 86.58% |
| 2 | 2.14% | 1.36% | 1.29% | 0.56% | 7.89% |
| 3 | 0.61% | 0.14% | 0.35% | 0.20% | 5.53% |

So the sentence in `results/cover/` — "greedy's belief is closer to the true MAG" — is not
what was measured. Greedy's belief is **less ambiguous**. Under oracle evidence neither
belief is ever wrong about anything.

## 2. Greedy is one-step steepest descent on that exact quantity

`UncertaintyGreedyAgent._unsure_touching` scores each node by the number of unsure claims
incident to it and intervenes on the argmax. At `bar=1.0`, "unsure" means "more than one mark
survives" — which is precisely the set of pairs with nonzero soft SHD.

Checked pair-for-pair against the belief's own survivor sets on real rollouts:
**6,976 node-scores compared, 0 disagreements.** The two counts are the same function.

The baseline therefore descends the support of the evaluation metric by construction, while
the learned policy is trained on identification. The comparison was never neutral. (Greedy
descends the *count* of unresolved pairs; soft SHD weights a 3-mark pair above a 2-mark one,
so the alignment is exact on the support and approximate on the mass.)

## 3. The mechanism has two channels, and both are measured

### 3a. Degree — greedy buys more pairs per move inside a window

Because greedy's score is a count of incident unresolved pairs, on a scale-free graph it is
degree-weighted: one intervention on a hub resolves every pair incident to it.

| rung | arm | mean degree of intervened nodes | window mean degree | ratio | top-quartile hit rate |
|---|---|---|---|---|---|
| w12 | learned (argmax) | 4.472 | 3.720 | 1.202 | 0.474 |
| w12 | greedy | **4.851** | 3.720 | **1.304** | **0.572** |
| w20 | learned (argmax) | 5.972 | 4.468 | 1.337 | 0.514 |
| w20 | greedy | **6.288** | 4.468 | **1.407** | **0.596** |
| w20 | random_vary | 4.538 | 4.468 | 1.016 | 0.325 |
| w30 | learned (argmax) | 9.643 | 5.077 | 1.899 | 0.849 |
| w30 | greedy | **9.752** | 5.077 | **1.921** | **0.898** |

Paired, learned − greedy: degree −0.379 ± 0.052 (w12), −0.315 ± 0.034 (w20), −0.108 ± 0.046
(w30); top-quartile rate −0.099 ± 0.011, −0.082 ± 0.008, −0.049 ± 0.009. All significant.
`random_vary` sits at ratio ≈1.0 — no preference at all — and pays for it with ten times the
residual. Note the channel **narrows** as k grows: at w30 both arms are strongly hub-seeking
(1.899 vs 1.921) and the gap is barely 2.4 standard errors. Degree alone does not carry w30.

### 3b. Shared versus private — greedy's moves count in every window at once

A move on a SHARED node lands in every agent's window simultaneously; a move on a PRIVATE
node helps only the mover. Soft SHD is averaged over all windows, so it prices a shared move
at roughly n times a private one. The arms allocate very differently:

| rung | arm | share of moves on shared nodes | union coverage per window | duplicate coverage |
|---|---|---|---|---|
| w12 | learned (argmax) | 0.349 | 0.592 | **0.043** |
| w12 | greedy | **0.438** | 0.596 | 0.072 |
| w20 | learned (argmax) | 0.365 | 0.500 | **0.001** |
| w20 | greedy | **0.499** | **0.532** | 0.072 |
| w30 | learned (argmax) | 0.589 | 0.353 | **0.000** |
| w30 | greedy | **0.705** | **0.384** | 0.026 |
| w20 | random_vary | 0.502 | 0.424 | 0.357 |

Paired, learned − greedy on shared share: −0.089 ± 0.010 (w12), −0.134 ± 0.010 (w20),
−0.116 ± 0.008 (w30). All significant. Union coverage follows: −0.032 ± 0.006 (w20),
−0.031 ± 0.004 (w30), both significant.

**This is the coordination result and the SHD deficit in one measurement.** The learner
barely duplicates at all — 0.000 to 0.043 against greedy's 0.026 to 0.072 and random's 0.193
to 0.470 — because it spends on the private nodes no partner will ever cover, which is
exactly what completing its OWN window requires. Greedy piles onto the shared surface, where
every move is worth n windows, and collects the multiplier nearly for free because its
duplication is low in absolute terms.

So the two objectives price shared-versus-private spend differently, and SHD uses greedy's
price. The learner is not failing at the metric; it is buying something the metric does not
sell.

**The forced-cover rule, confirmed empirically.** Splitting residual mass by how many of a
pair's endpoints anyone intervened on:

| rung | arm | neither endpoint | one endpoint | **both endpoints** |
|---|---|---|---|---|
| w12 | learned | 0.0036 | 0.0063 | **0.0000** |
| w12 | greedy | 0.0018 | 0.0036 | **0.0000** |
| w20 | learned | 0.0038 | 0.0081 | **0.0000** |
| w20 | greedy | 0.0023 | 0.0026 | **0.0000** |

A pair with both endpoints intervened is **always** fully resolved — 0.0000 for every arm at
every rung. That is the analytic forced-cover characterisation showing up in the data. At
w20 greedy also wins the coverage split itself (pairs with both endpoints covered: 0.275 vs
0.242, +0.033 ± 0.006), which is hub concentration paying off twice.

## 4. Part of the published gap is an evaluation artifact

`scripts/shd.py` loads the learned arm with `deterministic=False`. Switching to argmax:

| rung | learned−greedy, sampled | learned−greedy, argmax | joint success, sampled → argmax |
|---|---|---|---|
| w08 | +0.0122 ± 0.0030 | **+0.0059 ± 0.0026** | 0.320 → **0.450** |
| w12 | +0.0083 ± 0.0018 | **+0.0046 ± 0.0014** | 0.260 → **0.400** |
| w20 | +0.0080 ± 0.0012 | **+0.0070 ± 0.0012** | 0.090 → **0.140** |
| w30 | +0.0054 ± 0.0006 | +0.0049 ± 0.0007 | 0.000 → 0.000 |
| w04 | −0.0003 ± 0.0038 | +0.0142 ± 0.0055 | 0.840 → 0.730 |

Argmax roughly halves the gap at w08 and w12 and does not erase it anywhere at k≥8. **w04
reverses** — argmax is worse there — consistent with the standing retraction that "argmax as
primary" does not hold across the whole ladder. Report per-rung, never as a blanket rule.

## 5. What this retracts

**RETRACTED: "greedy wins SHD because joint success is zero-tolerance."** That was the
mechanism recorded in the session state and it is wrong. With argmax evaluation the learned
policy beats greedy on **per-window** solve rate too — 0.642 vs 0.625 (w08), 0.630 vs 0.588
(w12), 0.340 vs 0.307 (w20) — so it is not a joint-versus-marginal objective mismatch. The
learner is better at *identification* on both criteria and worse on *residual ambiguity*.
Those are different quantities and only the second is what SHD measures here.

(Under sampled evaluation the learner is worse per-window than greedy at every rung: 0.550,
0.575, 0.282. The per-window claim above is argmax-only and must be quoted that way.)

**RETRACTED: "the learned policy under-covers."** It covers *more* distinct nodes than greedy
at every rung — 0.338 vs 0.303 (w08), 0.299 vs 0.273 (w12), 0.239 vs 0.227 (w20), all
significant. It spends its budget on more nodes of lower degree.

## 6. Consequence for the thesis

**The SHD figure cannot be an error curve under oracle evidence — by construction, not by
accident.** The belief is incapable of being confidently wrong, so the y-axis is residual
ambiguity and the baseline is its greedy minimiser. Two options, and they are not exclusive:

1. **Run SHD under sampled evidence.** There the belief *can* be confidently wrong, the WRONG
   bucket becomes non-zero, and SHD becomes the structural-error metric everyone reads it as.
   This is now a much stronger reason to run the sampled-evidence version than the hunch it
   replaces — the oracle version is not a weaker version of the same measurement, it is a
   different measurement.
2. **Report the oracle SHD honestly as what it is** — a residual-ambiguity curve on which the
   baseline is the greedy optimiser — and lead on identification, where the learner wins.

Do not report the current table as "greedy's belief is closer to the true MAG."

## 6b. w30 is the cleanest case, and it isolates the mechanism

The w30 row was kept because both arms score exactly 0.000 on joint success there, so the
gap could not be dismissed as an objective mismatch. Decomposed, it says more than that.

At w30 the two arms intervene on the **same fraction of their own window** — 0.163 against
0.163, paired difference −0.0001 ± 0.0012, not significant — and neither solves anything
(per-window solve 0.003 for both). Budget, own-coverage, success and failure are all held
equal by measurement rather than by design. Greedy still leaves less residual ambiguity
(+0.0049 ± 0.0007, significant). With effort held equal the difference can only be **which**
nodes were chosen — and §3b names it: greedy puts 70.5% of its moves on the shared surface
against the learner's 58.9%, so its union coverage per window is higher (0.384 vs 0.353)
from the identical number of moves. The degree channel is nearly closed at w30 (1.899 vs
1.921); the shared-spend channel is not.

## 7. Still open

- **One seed per rung.** Everything here is `_s0`. The mechanism is structural and unlikely to
  move, but the magnitudes should not be quoted before the other seeds are in.
- **The sampled-evidence run** — item 1 above, unstarted.
