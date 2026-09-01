# Power-limited oracle evidence does not substitute for the sampled sweep

1 Sep 2026, overnight into midday. All results below are k=8/k=12, 4 agents, `factored`
backend, `--turn_aware_credit --local_epochs 4 --normalise_returns`, oracle evidence with
`--evidence_power < 1.0`, gated on `arms.greedy_uncertainty.success >= 0.85` (greedy is
oracle-informed, not learned, so its collapse indicates a starved environment, not a harder
learning problem).

## 1. The idea, and why it looked promising for one seed

Sampled evidence is ~74-110x slower to train under than oracle (0.085 s/episode against
6.3-9.4), which is why the sampled sweep needs a cluster and runs for days. `evidence_power`
withholds a fraction of ORACLE ancestry answers at oracle speed, reproducing the one thing
that matters about sampled evidence -- an unsettled belief -- without paying for the
statistical tests. If training under power-limited oracle evidence produced a policy that
transfers to genuine sampled evidence as well as one trained under sampled evidence, thesis
result 2 (sampled-evidence performance) would stop depending on the cluster job entirely.

A single seed at k=8, budget 70 (beta=3.0, double the sweep's own baseline budget of 35),
power 0.85 cleared all three checks: gate (greedy 0.89), mechanism (learned policy showed
higher private-node coverage AND repeat rate than both baselines under real sampled
evidence, matching the qualitative signature of genuinely sampled-trained policies), and
control (oracle-power performance tied with greedy, no cost). This looked like a result.

## 2. It does not replicate

Full 3-seed x 2-cell replication of that exact setting:

| cell | seed | greedy_uncertainty | gate |
|---|---|---|---|
| k=8  | 0 | 0.89 | PASS |
| k=8  | 1 | 0.82 | FAIL |
| k=8  | 2 | 0.83 | FAIL |
| k=12 | 0 | 0.81 | FAIL |
| k=12 | 1 | 0.77 | FAIL |
| k=12 | 2 | 0.80 | FAIL |

**1 of 6 seeds pass.** Seed 0 at k=8 was the outlier, not the pattern.

## 3. The full budget-boundary picture: a noisy asymptote, not a threshold

k=8, power 0.85, `greedy_uncertainty` success across the whole budget range tested (beta =
budget as a multiple of the sweep's own baseline; 1.5 is the untouched baseline budget):

| beta | budget | seeds (success) | mean | seeds passing gate |
|---|---|---|---|---|
| 1.5 | 35  | 0.49, 0.82             | 0.66 | 0/2 |
| 2.0 | 47  | 0.78, 0.75, 0.78       | 0.77 | 0/3 |
| 2.5 | 58  | 0.79, 0.83, 0.84       | 0.82 | 0/3 |
| 3.0 | 70  | 0.89, 0.82, 0.83       | 0.85 | 1/3 |
| 3.5 | 82  | 0.87                   | 0.87 | 1/1 |
| 4.0 | 93  | 0.82                   | 0.82 | 0/1 |
| 5.0 | 116 | 0.85                   | 0.85 | 1/1 (exactly at the line) |

The mean climbs smoothly with budget, which is consistent with a coverage-style hypothesis
(more turns per agent means more chances to answer a withheld question). **But no budget
tested gives a reliable pass**: per-seed spread of roughly +/-0.05-0.10 persists at every
budget from 47 up, and beta=4.0 (93) failing while sitting between two passing points
(beta=3.5 at 82, beta=3.0's best seed at 70) shows this is genuine noise, not a smooth
seed-independent curve. Even beta=5.0 -- nearly 5x the sweep's baseline budget -- only just
touches the gate on its one seed tested.

**There is no budget-only fix at k=8-12 that reliably clears a 0.85 gate.**

## 4. Distance-weighted withholding (rung 5) does not help either

Hypothesis: flat `evidence_power` treats a one-hop and a five-hop pair identically, but real
sampled evidence fails on WEAK, DISTANT effects specifically (`ma/env.py:220`'s own
objection to artificial uniform noise). Implemented `cb/factored.py::_window_hop_distances`
(BFS on the adjacency implied by the window's true marks) and scaled the withhold
probability as `evidence_power ** hop(x, y)` instead of a flat draw per pair
(`--distance_weighted_power`, opt-in, verified to degenerate exactly to flat power when every
pair is one hop apart, i.e. a fully-connected window; existing `tests/cb/test_versionspace.py`
and `tests/crosscheck/test_factored_attribution.py` unaffected).

| setting | flat power | distance-weighted |
|---|---|---|
| k=8, budget 35, power 0.85 (3 seeds) | 0.49 (1 seed, 31 Aug) | 0.49, 0.48, 0.58 |
| k=12, budget 100, power 0.85 (3 seeds) | 0.81, 0.77, 0.80 | 0.81, 0.77, 0.80 |

At k=12 the two are identical to two decimal places across all three seeds. The
implementation is verified correct (unit tests on a hand-built chain graph, a disconnected
graph, and the fully-connected degenerate case); this is a real result about the mechanism.
**Likely reason**: at k=8-12 most pairs within a window are 1-2 hops apart in the projected
MAG, so there is not enough long-range structure for hop-distance to redistribute probability
mass meaningfully. This might behave differently at k=20-30, untested.

## 5. Conclusion

Power-limited oracle evidence, as specified (flat or distance-weighted), does not reliably
substitute for training under genuine sampled evidence at the window sizes and budgets tested
here. The general shape of the coverage hypothesis (more budget helps, roughly) holds
directionally, but no operating point tested clears a hard competence gate reliably across
seeds. **The sampled sweep on Myriad remains the primary source for thesis result 2's
sampled-evidence claim.** This approach may be worth revisiting at a much larger budget or at
larger k (20-30, where distance-weighting has more room to matter), but that is a new,
larger search, not a finishing touch on this one.

All raw result files: `results/power/*.json` (training + gate), `results/power/oracle_*.json`
(control), `results/power/transfer_*.json` and `results/power/mechanism_*.json` (transfer and
diversity checks). Code: `scripts/diversity_probe.py` (new), `cb/factored.py`
(`distance_weighted_power`, opt-in, default off).
