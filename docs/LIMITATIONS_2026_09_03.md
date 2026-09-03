# Limitations — source material for Chapter 5's threats and Chapter 6's limitations

3 Sep 2026, morning. Brian asked for the key limitations and named three; this records the
full list with the evidence behind each, so the Discussion can cite receipts.
**The prose in Chapters 5 and 6 is his; this is the working record, not draft text.**

## The three Brian named, in their precise form

1. **Homogeneous intervention cost.** The budget counts interventions and prices them equally.
   No per-node cost, no observation-against-intervention asymmetry; budget allocation is
   combinatorial, never economic.
2. **Homogeneous windows.** Same size, same budget share, same reward, and the same network --
   `gnn_portable` is one shared policy. No agent is bigger, richer or better-informed.
3. **Latent confounding only where the design puts it.** Confounders exist on the shared
   interface and nowhere else. The confinement theorem describes the construction rather than
   surviving violations: no latents inside private blocks, none spanning arbitrary sets, and
   the environment never draws one the theorem does not cover.

## The rest, each with its receipt

4. **Linear-Gaussian everything.** Linear SCMs, Gaussian noise, Fisher-z tests. Nothing
   nonlinear or discrete; the belief's soundness leans on the test behaving.
5. **The oracle abstains but never lies.** The partial oracle withholds at rate rho and never
   returns a wrong answer; genuine finite-sample evidence errs in both directions. Measured
   consequence: sampled evidence fails as a training signal at every budget tested
   (window rates 0.14-0.25 against the 0.70 floor; FINDINGS_NINT200_2026_09_03.md). RQ2
   bridges exact-to-missing answers, not exact-to-noisy.
6. **The supplied skeleton.** Agents receive the observational skeleton free
   (`oracle_obs_structure`); the task is orientation and de-confounding, never adjacency
   discovery. Already in Chapter 5's threats; belongs in any limitations list too.
7. **Everything is budget-relative.** 14 of 18 cells changed winner between 4,000 and 12,000
   episodes (CLAIMS C7). The 8,000-to-12,000 movement is noise-sized (10 improve, 7 worsen;
   app:budget), which is evidence of a plateau and not proof of one.
8. **Three seeds, and threshold counts are fragile at three.** Path variance alone flipped a
   significance verdict (paths experiment, two cells, SD 0.21-0.41 of the within-path SE);
   every credit-off degradation is carried by one seed. Sign counts are robust; significance
   counts wobble.
9. **Convention-dependence, measured.** The RQ2 threshold moves under argmax (15/15 -> 3/15
   beyond 2 SE, 87% of the shift undetermined pairs; C6); checkpoint selection can be
   catastrophic on long runs (the 570x seed, C2). Results are about the stochastic policy
   evaluated by sampling, stated as such.
10. **Privacy is architectural, not formal.** No raw data crosses sites; no differential
    privacy, no leakage analysis of beliefs, claims or FedAvg gradients.
    docs/FUTURE_WORK_DIFFERENTIAL_PRIVACY.md holds the pointer.
11. **Cooperation is assumed, not incentivised.** All agents honest and obedient by
    construction; the de-confounding public good invites free-riding and misreporting
    questions the thesis does not test.
12. **Synchronous round-robin, fixed roster.** No asynchrony, dropouts, stragglers or
    mid-episode joins.
13. **Internal baselines only.** Myopic uncertainty, random, fixed-partition -- all ours. No
    head-to-head with published active-discovery methods; defensible because none handles
    this partition, and a reviewer will still ask.
14. **Scale wall.** d <= 75, K <= 10; k=30 training costs ~19x k=8, and the RQ2 grid at k=20
    was costed at ~43 wall-hours and abandoned (agent B, 3 Sep 01:35).
15. **Single-interface topology.** One shared boundary common to all agents; no pairwise
    overlaps, chains or hierarchies.

## The three most load-worthy for the Discussion, as suggested to Brian

#5 (bounds RQ2's realism claim), #7 (the thesis's own headline applied to itself), and
#10/#11 together (the word "federated" invites both questions and the thesis answers
neither).
