# Theory notes and reading guide

Every mathematical idea this project rests on, what it is, where it appears in our code,
and what to read. Ordered as a reading path, not alphabetically.

**Bibliographic caveat.** Titles, authors and the substance of each entry are reliable;
years and venues are from memory and should be checked before anything is cited in the
thesis. Where confidence is lower it is marked *(verify)*.

---

## Tier 0 — if you read one thing

**Peters, Janzing & Schölkopf, *Elements of Causal Inference* (MIT Press, 2017).**
Free PDF from MIT Press. The best modern introduction: structural causal models,
interventions, identifiability, and the assumptions that make causal discovery possible at
all. Chapters 1–7 cover essentially everything this project assumes. Start here.

Alternatives if you want a second angle:
- **Pearl, *Causality* (2nd ed., 2009).** The canonical text. Heavier, more philosophical.
- **Spirtes, Glymour & Scheines, *Causation, Prediction and Search* (2nd ed., 2000).**
  Free from MIT Press. Constraint-based discovery; the origin of the PC algorithm.
- **Koller & Friedman, *Probabilistic Graphical Models* (2009).** Encyclopaedic reference
  rather than a read-through. The structure-learning chapters are the relevant ones.

---

## Tier 1 — the five ideas the project rests on

### 1. Markov equivalence — why the agent has to act at all

Several different DAGs can produce statistically identical observational data. They form a
**Markov equivalence class**, and no amount of passive observation separates them. This is
the entire reason interventions are necessary, and therefore the entire reason the task
exists.

The characterisation: two DAGs are Markov equivalent exactly when they share a **skeleton**
(the undirected version) and the same **v-structures** (patterns `A → B ← C` where A and C
are not adjacent).

- **Verma & Pearl (1990), "Equivalence and synthesis of causal models".** The
  characterisation above.
- **Andersson, Madigan & Perlman (1997)**, on the characterisation of equivalence classes
  and essential graphs *(verify)*.

*In our code:* `sa/graphs.py` computes skeletons and v-structures and groups DAGs into
classes. At d=5 there are 29,281 DAGs in 8,782 classes.

### 2. Score equivalence — the property that makes the task honest

A scoring function is **score-equivalent** if Markov-equivalent DAGs receive *identical*
scores. This is not a nicety: it is the formal statement that observational data cannot
distinguish them. A scorer that violates it is leaking orientation information, and will
appear to solve the problem without intervening.

**This is the single most important property to understand**, because violating it is what
invalidated the previous round of this project. An estimator was given the true noise
variance, which broke score equivalence, and 98% of graphs were then "recovered" from
observation alone with agents doing nothing.

- **Chickering (2002), "Optimal structure identification with greedy search", JMLR.**
  Score equivalence and the GES algorithm.
- **Chickering (1995), "A transformational characterization of equivalent Bayesian network
  structures".**
- **Heckerman, Geiger & Chickering (1995), "Learning Bayesian networks: the combination of
  knowledge and statistical data", *Machine Learning*.** Likelihood equivalence, parameter
  modularity, the BDe score.

*In our code:* tests assert that class members score identically. `KnownVarianceScore` in
`sa/score.py` is kept deliberately as a documented broken example.

### 3. The BGe score — how a graph is scored

**BGe** = Bayesian Gaussian equivalent. It is the *marginal likelihood* of the data under a
linear Gaussian model with a Normal–Wishart prior: rather than fitting the parameters and
plugging them back in, it **integrates them out**. That is what handles model complexity
automatically — a denser graph does not automatically score better, because extra
parameters carry a prior cost.

Why this mattered here: with an unpenalised profile likelihood at d=3, the six densest DAGs
tied at the top holding 67% of posterior mass while the true 2-edge graph ranked 9th of 25.
An unpenalised score does not merely blur the answer, it inverts it.

- **Geiger & Heckerman (2002), "Parameter priors for directed acyclic graphical models…",
  *Annals of Statistics*.** The derivation.
- **Kuipers, Moffa & Heckerman (2014), "Addendum on the scoring of Gaussian directed
  acyclic graphical models", *Annals of Statistics*.** Corrects the formula. Use this one —
  it is what our implementation follows.

*In our code:* `sa/score.py`, `BGeScore`.

### 4. Interventions in the likelihood

When you hard-intervene on a node, you replace its structural equation. So that sample says
nothing about that node's own parents — but its realised value is still valid data for its
*children*. Getting this rule right is what breaks Markov equivalence: once different nodes
are scored on different sample subsets, class members separate.

- **Cooper & Yoo (1999), "Causal discovery from a mixture of experimental and observational
  data", UAI.**

*In our code:* `sa/posterior.py`, `local_score_table` — rows where a node was intervened on
are dropped from that node's term only.

### 5. Equal error variances make the problem too easy

If all nodes share one noise scale, a linear Gaussian SEM becomes fully identifiable from
observational data. Get this wrong and there is nothing to discover.

- **Peters & Bühlmann (2014), "Identifiability of Gaussian structural equation models with
  equal error variances", *Biometrika*.**

*In our code:* `sa/scm.py` draws a per-node noise scale. Note the subtlety recorded there:
because BGe is score-equivalent by construction, per-node noise is defence in depth rather
than the load-bearing fix — the leak lived in the *estimator*, not the data.

---

## Tier 2 — the algorithms we implemented

### 6. Expected information gain, and why our oracle is simple

Bayesian experimental design scores an experiment by how much it is expected to reduce
uncertainty. In our setting the outcome (which nodes are downstream) is a *deterministic*
function of the graph, so `H(outcome | graph) = 0`, and the mutual information collapses to
`I(graph; outcome) = H(outcome)` — just the entropy of the outcome distribution. That is
why `sa/oracle.py` is a few lines of `bincount` and entropy.

- **Lindley (1956), "On a measure of the information provided by an experiment", *Annals of
  Mathematical Statistics*.** The origin.
- **Chaloner & Verdinelli (1995), "Bayesian experimental design: a review", *Statistical
  Science*.** The best single overview.
- **Murphy (2001), "Active learning of causal Bayes net structure"** *(verify)*, and
  **Tong & Koller (2001), "Active learning for structure in Bayesian networks", IJCAI**
  *(verify)*. Closest predecessors to what we are doing single-agent.

### 7. Why beating the greedy oracle is possible

Greedy selection has a `(1 − 1/e)` optimality guarantee **only** under *adaptive
submodularity*. Expected information gain does not satisfy this in general — so a myopic
policy can be strictly suboptimal, and a planner can beat it. This is the theoretical
justification for the whole single-agent research question.

- **Golovin & Krause (2011), "Adaptive submodularity: theory and applications in active
  learning and stochastic optimization", JAIR.**

### 8. Exact structure inference by dynamic programming

Enumerating DAGs is super-exponential (543 at d=4, 3.78M at d=6, ~1.14 billion at d=7).
Instead you can recurse over **subsets of nodes**. Two families:

- **order-based** — sum over topological orderings. Cancellation-free, but each DAG is
  counted once per linear extension, giving the *order-modular* prior, which is not uniform
  over DAGs. A real, known bias.
- **sink-based with inclusion–exclusion** — decompose each DAG by its sinks. Signed terms,
  but each DAG counted exactly once, so no prior bias. **This is what we implemented**, and
  it reproduced our enumerated answers to 1e-13.

- **Koivisto & Sood (2004), "Exact Bayesian structure discovery in Bayesian networks",
  JMLR.** The foundational subset DP for edge marginals.
- **Tian & He (2009), "Computing posterior probabilities of structural features in Bayesian
  networks", UAI** *(verify)*. Corrects the order-modular bias.
- **Silander & Myllymäki (2006), "A simple approach for finding the globally optimal
  Bayesian network structure", UAI** *(verify)*. DP for the MAP structure.
- **Robinson's recurrence** for counting labelled DAGs — the sink/inclusion–exclusion
  identity our implementation uses. Search "Robinson counting labelled acyclic digraphs".
- **Björklund, Husfeldt, Kaski & Koivisto (2007), "Fourier meets Möbius: fast subset
  convolution", STOC.** The route from our `O(3^d)` to `O(2^d d^2)`.

### 9. Sampling DAGs from a posterior

Needed because reachability (what is downstream of a node) is *not* decomposable per node,
so the DP cannot produce it. Sampling and computing descendants per sample is the way
round.

- **Madigan & York (1995)**, structure MCMC — single-edge moves *(verify)*.
- **Friedman & Koller (2003), "Being Bayesian about network structure", *Machine
  Learning*.** Order MCMC; much better mixing, at the cost of the order-modular prior.
- **Grzegorczyk & Husmeier (2008)**, a new edge-reversal move that greatly improves mixing
  *(verify)*. **Directly relevant** — our MH sampler currently has 6–12% acceptance.
- **Kuipers & Moffa (2017), "Partition MCMC for inference on acyclic digraphs", JASA**
  *(verify)*. Avoids the order bias while keeping good mixing.

### 10. PPO, GAE, and reward shaping

- **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018).** Free
  online. The foundation.
- **Schulman et al. (2017), "Proximal Policy Optimization Algorithms".**
- **Schulman et al. (2015), "High-Dimensional Continuous Control Using Generalized
  Advantage Estimation".**
- **Ng, Harada & Russell (1999), "Policy invariance under reward transformations", ICML.**
  Potential-based shaping is the only form that provably cannot change the optimal policy.
  Our `shaping_coef` uses it.

Worth internalising: **advantage normalisation cancels the absolute scale of the reward.**
I reasoned incorrectly from reward magnitudes early on because of this.

### 11. Permutation equivariance — the thing that actually made it work

A function is **equivariant** if relabelling the inputs relabels the outputs the same way;
**invariant** if the output does not change at all. Our scorer is equivariant in the node
logits (relabel nodes → logits permute) and invariant in the value and pass logit (how good
the state is does not depend on labels).

The mechanism is *pooling*: aggregating over neighbours with sum/mean/max destroys ordering
information, which is exactly what makes the function order-independent.

- **Zaheer et al. (2017), "Deep Sets", NeurIPS.** The characterisation of permutation-
  invariant functions as `ρ(Σ φ(x))`. Directly what `PerNodeActorCritic` implements.
- **Bronstein, Bruna, Cohen & Veličković, "Geometric Deep Learning" (2021).** The general
  framework of building symmetry into architectures.

A caution worth keeping: an earlier version of our scorer pooled neighbours *in index
order*, which is not equivariant at all — it only looked correct. A test caught it.

---

## Tier 3 — the multi-agent phase

### 12. Multi-agent RL

- **Albrecht, Christianos & Schäfer, *Multi-Agent Reinforcement Learning: Foundations and
  Modern Approaches* (MIT Press, 2024).** Free online. The current standard text; start here.
- **Foerster et al. (2018), COMA** — counterfactual credit assignment, and
  **Rashid et al. (2018), QMIX** — value factorisation. Both assume **CTDE**
  (centralised training, decentralised execution), which the supervisor's constraint rules
  out. Read them to understand *what we are giving up*.
- **Independent learners and non-stationarity** — search "independent Q-learning
  non-stationarity multi-agent". This is the regime we are actually in.

### 13. Federated learning and privacy

- **Kairouz et al. (2021), "Advances and open problems in federated learning".** The
  comprehensive survey; the non-IID sections are the relevant ones.
- **McMahan et al. (2017), FedAvg.** The baseline algorithm — note it assumes a central
  server, which we do not have.
- **Decentralised averaging / gossip protocols** — search "gossip averaging decentralized
  SGD consensus". This is the server-free alternative.
- **Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014).** Free.
  Read if privacy needs a formal guarantee rather than a structural argument.

### 14. Amortised causal discovery

- **Lorch et al. (2022), "Amortized inference for causal structure learning" (AVICI),
  NeurIPS** *(verify)*. A network trained on simulated data to map datasets to graphs.
  Scales, but gives edge probabilities rather than a calibrated posterior, and carries **no
  score-equivalence guarantee** — see Tier 1 §2 for why that matters here.

---

## Concepts you will be asked about in a viva

Short answers worth being able to give without notes.

- **Why can't you just observe?** Markov equivalence. Observational data identifies the
  class, not the member.
- **Why BGe rather than a likelihood?** It integrates parameters out, so complexity is
  handled by the mathematics rather than a bolted-on penalty; and it is score-equivalent,
  which a fitted likelihood is not.
- **Why is beating greedy possible?** Expected information gain is not adaptively
  submodular, so the `(1 − 1/e)` guarantee does not apply and myopic choice can be strictly
  suboptimal.
- **Why is your network shaped like that?** The oracle's score for a node is the same
  function of every node's local structure, so the policy should be permutation-equivariant.
  A dense layer has to learn that from scratch and empirically cannot.
- **How do you know the task is not trivial?** GATE 1: the observational-only solve rate is
  compared against the singleton-equivalence-class fraction, which is computed exactly from
  the graph space rather than guessed.
- **What is `gap_closed`?** `(random − agent) / (random − greedy)` on episode cost, with
  unsolved episodes charged the full budget so an agent cannot score well by abandoning hard
  instances.

## Training-algorithm alternatives to IPPO (raised 2026-08-19) -- ALL UNVERIFIED

Recorded as leads. Every one needs checking against source before it enters the thesis.

### Sequential Bayesian experimental design as an RL problem

- **Foster, Ivanova, Malik, Rainforth (2021), "Deep Adaptive Design: Amortizing Sequential
  Bayesian Experimental Design", ICML.** Trains a design policy against a differentiable
  lower bound (sPCE) on the TOTAL expected information gain across a horizon.
  *Why it matters*: this is our research question stated as an objective rather than hoped
  for via reward. It also supplies a dense per-step signal, which attacks the sparse-reward
  hypothesis behind the 1-in-10 seed collapse.
- **Blau, Bonilla, Chades, Dezfouli (2022), "Optimizing Sequential Experimental Design with
  Deep Reinforcement Learning", ICML.** The explicitly RL formulation of the same problem.
  *Why it matters*: closest published framing to what we are building; determines whether
  our contribution is the method or the federation of it.

### Reward shaping

- **Ng, Harada, Russell (1999)** -- already cited here for potential-based shaping.
  *New use*: with potential `Phi(s) = -H(belief)`, the shaped reward is realised information
  gain per step and the optimal policy is provably unchanged. Greedy EIG then becomes the
  MYOPIC OPTIMUM OF THE AGENT'S OWN REWARD, which sharpens the headline claim rather than
  weakening it.

### Independent learners, and the CTDE exclusion

- **de Witt et al. (2020), "Is Independent Learning All You Need in the StarCraft
  Multi-Agent Challenge?"** Cover for IPPO as a strong baseline rather than a compromise.
- **Yu et al. (2022), MAPPO**; **Rashid et al. (2018), QMIX**; **Sunehag et al. (2018),
  VDN**; **Lowe et al. (2017), MADDPG.** All CTDE, all therefore EXCLUDED by the supervisor
  constraint. Worth citing precisely as the scoped-out set -- a reader will otherwise ask
  why the obvious cooperative-MARL algorithms are absent.
- **Foerster et al. (2017), stabilising experience replay for deep MARL.** Relevant if IQL
  is added as a sample-efficiency arm, since independent Q-learners face non-stationarity.

### Planning

- **Silver & Veness (2010), POMCP.** Available to us -- exact belief update plus a simulator.
  Deprioritised on our own evidence, not on principle: two-step lookahead saves +0.103 (d=4)
  and +0.063 (d=5) with CIs spanning zero, and the required horizon grows as log2(d) against
  an inference wall at d~15-20. The curves never cross.
