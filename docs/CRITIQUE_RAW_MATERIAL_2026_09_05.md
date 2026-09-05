# Critique of the methods and the work — raw material for the Discussion

5 Sep 2026, agent A, at Brian's request. Each item is labelled CONCEDE (say it plainly),
ANSWER (the thesis already carries the counter-evidence), or TURN (arguable as a thesis of
the Discussion). Numbers referenced live in thesis_results/CLAIMS.md.

## The problem construction
1. TURN -- The problem is constituted by its own assumptions. The ceiling degeneracy
   (0/2000 confounded episodes for a single controller vs 1821/2000 federated) proves the
   confounding exists BECAUSE of the partition. Turn: the partition is the object of study,
   and the degeneracy shows confounding-under-privacy is not reducible to centralized
   discovery -- measured, not assumed.
2. CONCEDE -- Synthetic linear-Gaussian SCMs, mostly SF(m=2) with one ER control, no real
   or benchmark data, skeleton supplied. The estimated-skeleton probe (65.9% accuracy;
   identification 100% -> 0% -> 8.3% at 1,000 rows) prices the strongest assumption.
   NOTE for the skeleton discussion: the version-space engine ADMITS unknown adjacency
   (the NONE mark exists in the pair sets; skeleton_source="estimated" exists in the env
   config) -- the barrier is the measured collapse, not the architecture. The assumption
   is priced modularity, not a hard-wired dependence.
3. CONCEDE (scope) -- Homogeneous intervention costs, window sizes, n_int, noise family;
   stationary SCMs; round-robin protocol; no latent confounding beyond partition-induced.

## The evaluation
4. ANSWER-PARTIAL -- The headline regime is the one where evidence is free (oracle);
   realistic evidence appears as transfer at ONE cell, where the learned arm trails the
   myopic rule almost everywhere, and the contamination finding shows even those transfer
   numbers measure a partly-broken engine. Counterweights: the partial-oracle result
   (15/15 beyond 2 SE at rho<=0.9), the disclosure mechanism + one-bit remedy, both
   measured. The asymmetry of where claims live must still be said plainly.
5. ANSWER -- Train-test coupling: joint recovery is the trained criterion; the selected
   checkpoint is a post-hoc choice baselines lack. Counterweights: SHD (untrained
   criterion) is the headline; both checkpoint conventions reported everywhere; the
   eps-greedy 2x2 (C9) shows the gap is not exploration noise.
6. CONCEDE + POINT -- Covered-pairs-only SHD ignores what the federation never examined
   (denominator 18-91% of pairs across configs). Complement: joint recovery covers every
   required mark; denominators are stated.
7. CONCEDE -- Three seeds (six for the ladder). Own measurement: significance counts flip
   within one SE across sample paths. Mitigations: paired designs, sign counts,
   conventions quoted beside numbers.

## The learning
8. TURN-PARTIAL -- 12,000 episodes to beat a zero-training heuristic by 0.05-0.09
   recovery; at 4,000 the learner loses. The thesis's own budget finding is this
   observation from the other side; exclude sample-efficiency claims explicitly.
9. CONCEDE -- The research process iterated on the cells it reports (k=8/12, SF); no
   held-out family. Counterweights: ER control, u0249 re-derivations, the retraction
   record, provenance-guarded numbers.

## The federated framing (the two sharpest)
10. TURN (must be argued, or an examiner argues it against us) -- No formal privacy
    guarantee, and the attribution appendix DEMONSTRATES leakage: a peer's private
    confounder can be located from public behaviour. Turn: attribution-as-leakage is a
    finding about the setting; connect it explicitly to the privacy framing.
11. CONCEDE (wording) -- "Federation costs nothing" is measured at saturated cells
    (k=20: zero errors both arms). Phrase as "no cost detectable at these difficulty
    levels"; C4's MUST NOTs already guard the strong form.

## The version-space representation (from the 5 Sep review; summarized)
12. ANSWER-PARTIAL -- Soundness is conditional on calibrated tests; contamination vacates
    it exactly in the realistic regime (17% false detections at n=10,000). Counterweight:
    boundary measured, mechanism named, one-bit remedy measured -- a defence posterior
    methods rarely have.
13. CONCEDE -- Intersection pooling inherits the conditionality; one confidently-wrong
    agent poisons the pool with no graded outvoting.
14. CONCEDE -- No uncertainty quantification: possible/impossible, never probable; no
    information-gain acquisition; no calibrated confidence for downstream use.
15. CONCEDE -- The per-pair factorization is an outer approximation (joint constraints
    discarded); belief size overstates true ambiguity.
16. CONCEDE (scope) -- Binary evidence is expensive in low data: sub-threshold experiments
    contribute nothing; magnitudes discarded. This is the price of the one property no
    posterior has -- composition by intersection under privacy (see 17).
17. TURN -- Against SOTA Bayesian experimental design (RL-CBED, CORE, amortized designs):
    centralized, they likely dominate on interventions-to-recovery (graded evidence);
    federated, their posteriors do not compose without sharing -- there is no off-the-shelf
    entry. The honest sentence: they are more efficient per sample; this belief is the one
    that composes under the privacy constraint. Future work: a federated Bayesian ED
    method against this version space on the same interface.
18. CONCEDE (scope) -- The RL results are belief-specific; no transfer shown to
    posterior-based or exact-DP beliefs beyond the small-k exact cells.

## Known result-gaps a critic can name
19. The partial-oracle finding (C6) is ONE cell, k_v=8. No axis sweep exists under
    partial-oracle training; whether the transfer result generalizes across window sizes
    is untested. (Extending it is one overnight fleet per cell; the boundary is stated
    in C6 either way.)
20. The rho grid remains at 8,000 episodes while headline cells moved to 12,000; advised
    extension was deprioritised on noise evidence (3 Sep).

## Strengths to set against the critique (verifiable, not rhetorical)
Seeded reproducible evaluations; provenance-guarded registries; adversarial controls run
against our own headline (generator family, eps-greedy 2x2, u0249 budget re-derivation,
answer-quality probe); a public retraction record; MUST-NOT-guarded claims file.
