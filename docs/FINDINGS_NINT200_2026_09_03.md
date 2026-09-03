# The finite-sample cell: what it can and cannot say, caught before it said the wrong one

3 Sep 2026, 06:1x. `results/sampled_det/nint200.json`: the three `sampled_ref` policies
(trained AND evaluated under sampled evidence, $n_{\text{int}}=200$, $k_v=8$) measured with
`global_shd_paired.py`, seeded, 200 paired episodes per seed.

## The numbers, and the fact that disqualifies the obvious reading

| seed | learned | myopic | paired | window rate (last 10) |
|---|---|---|---|---|
| 0 | 0.04457 | 0.01883 | $+0.02574 \pm 0.00303$ | **0.245** |
| 1 | 0.05537 | 0.01713 | $+0.03824 \pm 0.00319$ | **0.209** |
| 2 | 0.05585 | 0.01846 | $+0.03739 \pm 0.00308$ | **0.138** |

The learned arm loses by 8--12 SE on every seed -- and **every seed is below the 0.70
competence floor** that excludes runs from every other table in the thesis, with `mi_ratio`
at 0.03--0.11 against the rho fleet's 0.27--0.85. These are 4,000-episode runs that did not
finish training, under the noisiest evidence regime in the project.

## The claim this kills, and the claim it makes

**Not supportable:** "a policy trained under finite-sample evidence loses to the myopic rule."
By the thesis's own rule these runs are excluded from that comparison; writing it would repeat
the agent-count and contention mistakes with a fresh coat.

**Supportable, and better for RQ2's arc:** training under genuine finite-sample evidence at
this budget **fails to train** -- window rates 0.14--0.25 where the same cell under oracle
evidence reaches ~0.9, and where the partial-oracle policies at 8,000 episodes reach
0.73--1.00 and then transfer. The realistic regime is not a harder exam the policy fails; it
is a training signal too noisy to learn from at any budget tested here. That is the
motivation for the partial oracle stated as a measurement: degrade the cheap regime rather
than train in the expensive one.

The SHD numbers above are kept as the *description of the failure* (an untrained policy
commits errors at random-arm scale), not as a comparison of trained systems.

## Boundary

One $n_{\text{int}}$ value, one cell, 4,000 episodes. Whether in-regime training succeeds at
12,000 or more episodes is untested and is the obvious future-work line; the freeze forecloses
testing it tonight. \S4.2.1 says "at several values of $n_{\text{int}}$" -- the seeded runs
exist only at 200, and the subsection must say one value, not several.
