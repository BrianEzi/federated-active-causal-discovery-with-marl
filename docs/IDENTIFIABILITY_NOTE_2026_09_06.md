# Which SCM families are identifiable without interventions — and why ours is the hard one

Noted 6 Sep 2026 for the Discussion, at Brian's request. This reframes the linear-Gaussian
choice from a convenience into a deliberately adversarial one.

## The landscape

Observational data alone identifies the causal direction in every one of these families:

| family | identifiable observationally? | reference |
|---|---|---|
| linear, NON-Gaussian noise | YES, fully (LiNGAM) | Shimizu et al. 2006 |
| NON-linear, additive noise | YES, generically | Hoyer et al. 2009; Peters et al. 2014 |
| linear, Gaussian, EQUAL error variances | YES | Peters & Buhlmann 2014 (already cited) |
| **linear, Gaussian, UNEQUAL variances** | **NO — only up to the Markov equivalence class** | classical |

The last row is ours. `noise_range = U(0.5, 1.5)` spans a factor of three *specifically* to
break the equal-variance condition (already stated in Methodology, sec:meth_interventions),
which places the work in the single family where observational data cannot settle direction.

## Why this matters for the thesis

1. The setting is not a convenient special case; it is the ONE case where interventions are
   strictly necessary rather than merely helpful. That is an argument FOR the whole
   interventional framing, and it should be made explicitly in the Discussion.
2. It cuts the other way too, and honesty requires saying so: because the family is the
   hardest observationally, our results do not transfer automatically to the easier
   families. A method that exploits non-Gaussianity or non-linearity could need fewer
   interventions than ours on those problems, and our engine cannot exploit either --
   it reads only means, variances and linear correlations.
3. Two different axes must not be confused. Linear-Gaussian is the HARDEST family for the
   PROBLEM (no observational identifiability) and the EASIEST for OUR TESTS (the moment-based
   tests are exactly calibrated for it). Moving to another family makes the problem easier
   and the machinery worse at the same time.

## The transfer experiments this suggests (eval-only, existing checkpoints)

Both require a new mechanism/noise option in ma/scm.py; both are evaluation-only.

- NON-GAUSSIAN NOISE, same linear mechanism. Directly stresses test calibration, which is
  the failure mode already measured this week (contamination: miscalibrated tests write
  confident wrong marks, and the damage grows with sample size). Uniform is the mild,
  classic LiNGAM case; Student-t with 3 df is the adversarial heavy-tailed one.
- NON-LINEAR mechanism, Gaussian noise. Monotone (tanh) attenuates all three detection
  channels; symmetric (quadratic) zeroes the correlation channel by construction while
  leaving mean and variance shifts intact.

## Bibliography added

shimizu2006lingam, hoyer2009nonlinear, peters2014continuous -- entries added to
references.bib on this date; volume/page details verified against JMLR/NeurIPS listings.
