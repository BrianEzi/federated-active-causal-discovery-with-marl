# Active Causal Structure Discovery

### A Graduate Text on Bayesian Metrics, Experimental Design, and Equivariant Reinforcement Learning

---

This is a self-contained mathematical treatment of the theory underlying the
single-agent active causal discovery system in [`sa/`](../sa/). It is written for a
reader with an undergraduate mathematics or computer science background — multivariate
calculus, linear algebra, measure-theoretic probability, graphical models, basic abstract
algebra, and computational complexity are assumed and used without apology. Algebraic
steps are not skipped.

Two things distinguish it from a survey. First, every theorem that the codebase depends
on is stated and proved, not gestured at. Second, every quantitative claim in the text is
produced by a script in [`verification/`](verification/) that runs against the actual
implementation; where theory and code disagreed, the disagreement is documented rather
than smoothed over. Several chapters are organised around real defects that this project
shipped and later had to retract — those are the parts most worth reading, because a
theorem's content is clearest when you can see what breaks without it.

---

## Contents

### Part I — Causal Graphical Models, Identifiability and Interventions

| | Chapter | Core results |
|---|---|---|
| 1 | [Structural Causal Models and d-Separation](part1/01-scm-and-d-separation.md) | SCMs, the Markov factorisation, d-separation, the global Markov property, faithfulness |
| 2 | [Markov Equivalence Classes](part1/02-markov-equivalence.md) | Verma–Pearl characterisation; CPDAGs and essential graphs; **exact proof of observational non-identifiability** |
| 3 | [Interventions and the Likelihood](part1/03-interventions-and-likelihood.md) | Graph surgery vs. do-calculus; the Cooper–Yoo factorisation; **how intervening isolates $\mathrm{Pa}(X_i)$**; interventional equivalence classes |
| 4 | [The Equal-Variance Identifiability Trap](part1/04-equal-variance-trap.md) | Peters–Bühlmann identifiability; why heterogeneous noise is required; the defect that voided the previous round |

### Part II — Score Equivalence and the BGe Metric

| | Chapter | Core results |
|---|---|---|
| 5 | [Score Equivalence, Likelihood Equivalence, Parameter Modularity](part2/05-score-equivalence.md) | Formal definitions; Chickering's covered-edge theorem; decomposability |
| 6 | [The Information-Leak Pitfall](part2/06-information-leak.md) | Why profile likelihoods, unpenalised AIC/BIC and `KnownVarianceScore` leak orientation |
| 7 | [The BGe Marginal Likelihood](part2/07-bge-derivation.md) | Full Normal–Wishart derivation; the Kuipers–Moffa–Heckerman correction; log-space evaluation |
| 8 | [A Worked Analytical Example](part2/08-worked-example.md) | Empty vs. chain vs. dense on a fixed $10\times 3$ dataset, every matrix shown |

### Part III — Active Bayesian Experimental Design

| | Chapter | Core results |
|---|---|---|
| 9 | [Expected Information Gain](part3/09-expected-information-gain.md) | Lindley's measure; EIG as mutual information; the decision-theoretic frame |
| 10 | [The Deterministic Reachability Collapse](part3/10-reachability-collapse.md) | **Proof that $H(Y\mid G,a)=0 \Rightarrow I(G;Y\mid a)=H(Y\mid a)$**, and why `sa/oracle.py` is ten lines |
| 11 | [Submodularity and the Failure of Myopic Search](part3/11-submodularity.md) | Submodularity, adaptive submodularity, the $(1-1/e)$ theorem, and **an explicit counterexample** |

### Part IV — Deep Reinforcement Learning for Active Discovery

| | Chapter | Core results |
|---|---|---|
| 12 | [Policy Optimisation: PPO and GAE](part4/12-ppo-and-gae.md) | Policy gradient theorem, trust regions, the clipped surrogate, GAE bias–variance |
| 13 | [Potential-Based Reward Shaping](part4/13-reward-shaping.md) | **Proof of policy invariance**; why non-potential shaping breaks it; advantage-normalisation scale invariance |
| 14 | [Symmetry: Group Actions on Graph Data](part4/14-symmetry-and-equivariance.md) | $S_n$ actions, invariance/equivariance, the Deep Sets theorem |
| 15 | [The `PerNodeActorCritic` Architecture](part4/15-pernode-architecture.md) | Equivariant policy heads, invariant value heads, **the index-order pooling bug** |

### Part V — Multi-Agent Systems, Privacy and Amortisation

| | Chapter | Core results |
|---|---|---|
| 16 | [Decentralised Multi-Agent Causal Discovery](part5/16-decentralised-marl.md) | CTDE and what forgoing it costs; independent learners; non-stationarity |
| 17 | [Federated Optimisation and Differential Privacy](part5/17-federated-and-privacy.md) | FedAvg, gossip consensus, $(\varepsilon,\delta)$-DP applied to structure telemetry |
| 18 | [Amortised Structure Learning](part5/18-amortised-discovery.md) | AVICI; why amortised edge probabilities are not a posterior and carry no score-equivalence guarantee |

### Part VI — Empirical Evaluation and Viva Preparation

| | Chapter | Core results |
|---|---|---|
| 19 | [The Benchmark and its Metrics](part6/19-benchmark-metrics.md) | Derivation of `gap_closed`, censoring, stratification, the metrics that were gameable |
| 20 | [Triviality Verification: GATE 1](part6/20-gate1-triviality.md) | The exact singleton fraction as a falsifiable target |
| 21 | [Viva Defence](part6/21-viva-defence.md) | Four core questions answered rigorously, with the objections an examiner will raise |

### Apparatus

- [Notation](notation.md) — symbols used throughout.
- [Bibliography](bibliography.md) — BibTeX and APA, every entry verified against the publisher of record.
- [`verification/`](verification/) — the scripts behind every number, and [`verification/RESULTS.md`](verification/RESULTS.md) with their output.

---

## How to read this

The dependency structure is not linear. Chapter 2 is the load-bearing one: almost
everything else exists because Markov equivalence makes observational identification
impossible. From there:

```
      Ch 1  SCMs, d-separation
        |
      Ch 2  Markov equivalence  <-- the reason the problem exists
       / \
      /   \
  Ch 3     Ch 5 -- Ch 6 -- Ch 7 -- Ch 8      (how a graph is scored)
interventions      |
      |            |
      Ch 4         |     (what makes the task non-trivial)
       \          /
        \        /
      Ch 9 - Ch 10 - Ch 11    (which experiment to run)
              |
      Ch 12 - Ch 13 - Ch 14 - Ch 15   (learning to choose)
              |
      Ch 16 - Ch 17 - Ch 18   (many agents)
              |
      Ch 19 - Ch 20 - Ch 21   (measuring it)
```

A reader preparing for an examination should read Chapter 21 first, then follow its
cross-references backwards into whichever proof is being demanded.

## Reproducing the numbers

```bash
cd <repo root>
PYTHONPATH=. python textbook/verification/v1_graph_space.py
PYTHONPATH=. python textbook/verification/v2_bge_identities.py
PYTHONPATH=. python textbook/verification/v3_worked_example.py
PYTHONPATH=. python textbook/verification/v4_information_leak.py
PYTHONPATH=. python textbook/verification/v5_eig_collapse.py
PYTHONPATH=. python textbook/verification/v6_submodularity.py
PYTHONPATH=. python textbook/verification/v7_equivariance.py
```
