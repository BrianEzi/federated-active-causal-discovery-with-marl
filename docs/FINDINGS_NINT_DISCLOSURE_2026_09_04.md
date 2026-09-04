# The n_int U-curve is the price of privacy non-disclosure, measured

4 Sep 2026. Brian challenged the U: sampled evaluation should converge to oracle as n_int
grows. It does not, and the reason is now measured, not hypothesised.

## The chain of evidence

1. `results/nint_curve/` (complete, 3 seeds x 7 values x 200 paired episodes): every arm's
   SHD traces a U in n_int. Learned min at n=100, 5x worse at 10,000. Random too -- so the
   engine, not the policies.
2. `scripts/nint_answer_probe.py` (new, committed): per-query answers vs window truth,
   greedy-only seeded episodes. False-detection rate GROWS with n: 3.1% (n=100) -> 4.5%
   (1,000) -> 17.3% (10,000) against nominal alpha 0.001. Powered-miss rate falls to ~0.
   The U is the false-positive channel.
3. Full-graph split, final (30 episodes, results/nint_curve/answer_probe_split.json):
   NONE of 1,100 false detections across n=100/1,000/10,000 (137/208/755) are real
   full-graph ancestries invisible to the window projection -- zero at every n. Not a
   projection artefact; entirely the contamination channel.
4. The A/B that closes it: same episodes, same seeds, n=1,000 --
       disclose_regime=False   false detects 40/1232  (3.25%)
       disclose_regime=True    false detects  2/1239  (0.16%)
   Under round-robin turn-taking, the rows where x was intervened are clean, but the
   CONTROL rows include rounds where a partner intervened on a node hidden from this agent
   -- invisible to its "no third variable" filter. The contrast is contaminated, the test
   attributes the partner's effect to x, and because the contamination is a real
   distributional difference, its detectability GROWS with n. cb/citest.py documented this
   in advance: "without disclosure the mask is all-False and the contamination is the
   honest price of privacy."

## What this makes of the result

Not a bug; a finding with a mechanism and a one-bit remedy. The sweep trains and evaluates
with disclose_regime=False, so the U prices non-disclosure. The calibrated rerun
(results/nint_disclose/, running 4 Sep, --override_disclose on the same checkpoints and
episodes) tests the convergence claim: with the one-bit disclosure, growing n_int should
take sampled evaluation monotonically toward the oracle-evaluation values, which are
already measured (sweep12k k08) and can be drawn as reference lines.

## Boundaries

- One cell (k_v=8), oracle-trained policies, greedy-only probe for the answer rates.
- The 0.16% disclosed rate is 8 episodes, one seed: indicative, not final. The 30-episode
  probe and the full disclosed sweep supersede these numbers when they land.
- Nothing here re-trains anything: disclosure at EVALUATION time only. Whether training
  under disclosure changes the policies is untested and out of scope before the deadline.
- n_int=100 minimises the learned arm's error only in the UNDISCLOSED regime; it is a
  property of the contamination, not a recommendable design point, and selecting the
  comparison point by the smallest learned-myopic gap would be selection on the outcome.
