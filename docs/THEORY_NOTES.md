# Theory Notes

Running record of design choices and findings in this project that have real theoretical
grounding, with the citation **and** the reason it matters here.

Intended consumer: a later pass that folds these into `thesis/references.bib` and
`thesis/annotated_bibliography.md`. The "Why it matters here" paragraph is the part that should
become the annotation — the citation alone is not enough to work from.

**Verification status**: entries are marked `[verified]` if checked against the source, or
`[unverified]` if cited from memory. Volume/page numbers on `[unverified]` entries must be
confirmed before they enter the bibliography.

---

## 1. Equal error variances break Markov equivalence in linear Gaussian SEMs

**Peters, J. & Bühlmann, P. (2014).** "Identifiability of Gaussian structural equation models with
equal error variances." *Biometrika* 101(1):219–228. `[unverified]`

**Grounds:** `src/marl/bayes_optimal_estimator.py`, and the `--noise_scale` / `--sample_count`
environment configuration.

**Why it matters here:** This is the root cause of the oracle-metric failure found 2026-08-14. A
linear Gaussian SEM is normally identifiable only up to its Markov equivalence class from
observational data — which is the entire premise of this project, since it is what makes
interventions *necessary*. Peters & Bühlmann show that constraint lifts when all error variances
are **equal**: the DAG becomes fully identifiable observationally. Our SCM uses a single scalar
`noise_scale` for every node, so it sits exactly in that regime. Measured consequence: the
Bayes-optimal estimator recovers the true graph from observational samples alone with 98%
accuracy before any intervention is taken, which made the whole active-learning task degenerate.
Cite this when justifying the environment fix, and when explaining why the earlier oracle-agreement
numbers were retracted.

## 2. Score equivalence — why fitting the variance restores the intended difficulty

**Chickering, D.M. (2002).** "Learning equivalence classes of Bayesian-network structures."
*JMLR* 2:445–498. `[unverified]`

**Verma, T. & Pearl, J. (1990).** "Equivalence and synthesis of causal models." *UAI*. `[unverified]`

**Grounds:** the fix applied to `_fit_node_log_likelihood` — fitting per-node residual variance by
MLE instead of substituting the environment's known `noise_scale`.

**Why it matters here:** A Gaussian likelihood with **free** per-node error variances is
score-equivalent: every DAG in the same Markov equivalence class receives an identical score, so no
amount of observational data can separate them. Substituting a *known, shared* variance breaks
score equivalence and is precisely what handed the estimator its observational shortcut. Fitting
the variance restores MEC-limited identifiability, which is the regime the thesis assumes.
Empirically confirmed: observational-only accuracy drops from 0.98 to ~0.31–0.39, and 1/3 is
approximately the reciprocal MEC size for these spanning-tree topologies — i.e. the estimator
degrades to exactly "correct equivalence class, orientation undetermined," which is the
theoretically predicted behaviour rather than mere noise.

## 3. Bayesian scoring with interventional data

**Cooper, G.F. & Yoo, C. (1999).** "Causal discovery from a mixture of experimental and
observational data." *UAI*. `[unverified]`

**Grounds:** `_fit_node_log_likelihood`'s `self_intervened` masking.

**Why it matters here:** The canonical reference for the rule this code implements — under a hard
intervention, the intervened node contributes no likelihood term for its own structural equation
(that equation was replaced), but its realized value remains a valid conditioning value for its
children. Cite to justify the masking rather than presenting it as an implementation detail.

## 4. Interventional Markov equivalence — what an intervention actually buys

**Hauser, A. & Bühlmann, P. (2012).** "Characterization and greedy learning of interventional
Markov equivalence classes of directed acyclic graphs." *JMLR* 13:2409–2464. `[unverified]`

**Grounds:** conceptual framing for `src/marl/oracle_policy.py`.

**Why it matters here:** Formalizes how intervening refines the observational MEC into a smaller
I-MEC. The reachability-signature partition the oracle computes is a tractable proxy for exactly
this refinement, so this is the right citation for "what the oracle is approximating."

## 5. Bayesian optimal experimental design — the oracle's objective

**Lindley, D.V. (1956).** "On a measure of the information provided by an experiment." *Annals of
Mathematical Statistics* 27(4):986–1005. `[unverified]`

**Chaloner, K. & Verdinelli, I. (1995).** "Bayesian experimental design: a review." *Statistical
Science* 10(3):273–304. `[unverified]`

**Grounds:** `expected_discrimination` in `src/marl/oracle_policy.py`.

**Why it matters here:** Lindley is the origin of "choose the experiment maximizing expected
information gain about the parameter"; Chaloner & Verdinelli is the standard review and the better
citation for a related-work section. Establishes the oracle as a recognized design criterion rather
than a bespoke heuristic.

## 6. Generalized entropies — why a Gini criterion is legitimate

**DeGroot, M.H. (1962).** "Uncertainty, information, and sequential experiments." *Annals of
Mathematical Statistics* 33(2):404–419. `[unverified]`

**Grünwald, P.D. & Dawid, A.P. (2004).** "Game theory, maximum entropy, minimum discrepancy and
robust Bayesian decision theory." *Annals of Statistics* 32(4):1367–1433. `[unverified]`

**Grounds:** the `1 - Σ_g P(g)²` scoring rule in `expected_discrimination`.

**Why it matters here:** DeGroot shows *any* concave uncertainty function induces a coherent notion
of expected information, not Shannon entropy alone. That is what licenses the Gini/Simpson form as
principled rather than ad hoc. Grünwald & Dawid supply the decision-theoretic reading: Gini is Bayes
risk under the quadratic (Brier) score, Shannon under the log score.

**Additional note worth carrying into the thesis:** because the outcome (a reachability signature)
is a *deterministic* function of the hypothesis, `H(outcome | hypothesis) = 0`, so
`I(hypothesis; outcome) = H(outcome)`. Maximizing outcome entropy therefore **is** maximizing
expected information gain exactly. Under Shannon entropy the criterion would be exactly Lindley's;
the implemented Gini version is its Tsallis-2 analogue. Switching the scoring line to
`-Σ P(g) log P(g)` would make the criterion exactly Lindley-optimal with no approximation to defend
— a one-line change worth considering. The remaining idealization is that the experiment is assumed
to reveal the signature perfectly, which finite noisy samples do not; this makes the oracle an
*optimistic* reference.

## 7. Committee disagreement as an active-learning criterion

**Seung, H.S., Opper, M. & Sompolinsky, H. (1992).** "Query by committee." *COLT*. `[unverified]`

**Houlsby, N., Huszár, F., Ghahramani, Z. & Lengyel, M. (2011).** "Bayesian active learning for
classification and preference learning." arXiv:1112.5745. `[unverified]`

**Grounds:** `expected_discrimination`'s pairwise-disagreement formulation, and Track B's
uncertainty bonus in `src/marl/ppo_agent.py::compute_uncertainty_bonus`.

**Why it matters here:** The oracle's "probability two hypotheses drawn from the posterior disagree"
is literally Query-by-Committee. BALD is the modern mutual-information formulation and the closest
published analogue to Track B's per-edge `1 - |2p - 1|` bonus — the right citation for arguing the
uncertainty bonus is a known-good mechanism rather than an invention.

## 8. Potential-based reward shaping — constrains what the reward may contain

**Ng, A.Y., Harada, D. & Russell, S. (1999).** "Policy invariance under reward transformations:
theory and application to reward shaping." *ICML*. `[unverified]`

**Grounds:** `src/rewards.py::compute_ippo_rewards`, specifically the `dense` branch and any
added per-step penalty.

**Why it matters here:** The dense reward `(prev_shd - curr_shd)` is already potential-based
shaping with potential `Φ(s) = -SHD(s)`, so by Ng et al. it provably leaves the optimal policy
unchanged relative to the sparse terminal reward — a genuinely useful thing to be able to state,
since it means dense-vs-sparse is a pure learning-speed question, not a change of objective. The
theorem also constrains what may safely be added: a per-step term **not** of the form
`γΦ(s') - Φ(s)` (for instance a flat `-0.1 × SHD` holding penalty) is *not* policy-invariant and
can change what the optimal policy is. That is not automatically wrong — it converts the task into
a minimum-time formulation, which may be what we want — but it must be stated as an intentional
change of objective, not presented as neutral shaping.

## 9. Myopic (greedy) experiment selection

**Golovin, D. & Krause, A. (2011).** "Adaptive submodularity: theory and applications in active
learning and stochastic optimization." *JAIR* 42:427–486. `[unverified]`

**Grounds:** the single-step scope of `oracle_policy.py` (documented at its line 44).

**Why it matters here:** The standard justification for greedy sequential design, giving a `(1-1/e)`
guarantee **when** the objective is adaptively submodular. State the caveat honestly: expected
information gain is not adaptively submodular in general, so the guarantee does not automatically
transfer to our oracle. Cite for the framing, not for a bound we have not established.

## 10. Background / textbook anchors

**Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University
Press. `[unverified]` — truncated factorization / mutilated-graph semantics of `do()`, grounding
both the hard-intervention likelihood and the reachability model of an intervention's footprint.

**Spirtes, P., Glymour, C. & Scheines, R. (2000).** *Causation, Prediction, and Search* (2nd ed.).
MIT Press. `[unverified]` — constraint-based discovery and Markov equivalence.

**Peters, J., Janzing, D. & Schölkopf, B. (2017).** *Elements of Causal Inference*. MIT Press.
`[unverified]` — modern textbook treatment of SEM identifiability; convenient single citation for
background.
