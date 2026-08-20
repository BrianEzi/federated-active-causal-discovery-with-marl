# Bibliography

Every reference this project has used, in BibTeX-ready form, with an explicit **status** so
nothing goes into the thesis unchecked.

| status | meaning |
|---|---|
| **VERIFIED** | checked against the source; author, title, venue and year confirmed |
| **STANDARD** | canonical work, details taken from common usage — confirm the exact edition/page before submission |
| **UNVERIFIED** | recorded as a lead. **Do not cite until checked.** |

`docs/THEORY_NOTES.md` holds the *why it matters* for each; this file holds the citation.

---

## 1. Foundations — texts

```bibtex
@book{pearl2009causality,
  author    = {Pearl, Judea},
  title     = {Causality: Models, Reasoning and Inference},
  edition   = {2nd}, publisher = {Cambridge University Press}, year = {2009}}          % STANDARD

@book{spirtes2000causation,
  author    = {Spirtes, Peter and Glymour, Clark and Scheines, Richard},
  title     = {Causation, Prediction, and Search},
  edition   = {2nd}, publisher = {MIT Press}, year = {2000}}                           % STANDARD

@book{koller2009pgm,
  author    = {Koller, Daphne and Friedman, Nir},
  title     = {Probabilistic Graphical Models: Principles and Techniques},
  publisher = {MIT Press}, year = {2009}}                                              % STANDARD

@book{sutton2018rl,
  author    = {Sutton, Richard S. and Barto, Andrew G.},
  title     = {Reinforcement Learning: An Introduction},
  edition   = {2nd}, publisher = {MIT Press}, year = {2018}}                            % STANDARD

@book{albrecht2024marl,
  author    = {Albrecht, Stefano V. and Christianos, Filippos and Sch\"afer, Lukas},
  title     = {Multi-Agent Reinforcement Learning: Foundations and Modern Approaches},
  publisher = {MIT Press}, year = {2024}}                                              % STANDARD

@book{dwork2014privacy,
  author    = {Dwork, Cynthia and Roth, Aaron},
  title     = {The Algorithmic Foundations of Differential Privacy},
  publisher = {Now Publishers}, year = {2014}}                                         % STANDARD
```

## 2. Markov equivalence — why the agent must act

```bibtex
@inproceedings{verma1990equivalence,
  author    = {Verma, Thomas and Pearl, Judea},
  title     = {Equivalence and Synthesis of Causal Models},
  booktitle = {Uncertainty in Artificial Intelligence (UAI)}, year = {1990}}            % VERIFIED

@article{andersson1997characterization,
  author    = {Andersson, Steen A. and Madigan, David and Perlman, Michael D.},
  title     = {A Characterization of Markov Equivalence Classes for Acyclic Digraphs},
  journal   = {The Annals of Statistics}, volume = {25}, number = {2}, year = {1997}}   % STANDARD

@article{chickering2002ges,
  author    = {Chickering, David Maxwell},
  title     = {Optimal Structure Identification with Greedy Search},
  journal   = {Journal of Machine Learning Research}, volume = {3}, year = {2002}}      % VERIFIED

@inproceedings{chickering1995transformational,
  author    = {Chickering, David Maxwell},
  title     = {A Transformational Characterization of Equivalent {B}ayesian Network Structures},
  booktitle = {Uncertainty in Artificial Intelligence (UAI)}, year = {1995}}            % STANDARD
```

**Used for:** the definition of Markov equivalence as `(skeleton, v-structures)`, which is
implemented directly as `mec_signature` in `sa/graphs.py`.

## 3. Scoring a graph

```bibtex
@article{heckerman1995learning,
  author    = {Heckerman, David and Geiger, Dan and Chickering, David M.},
  title     = {Learning {B}ayesian Networks: The Combination of Knowledge and Statistical Data},
  journal   = {Machine Learning}, volume = {20}, number = {3}, year = {1995}}           % STANDARD

@article{geiger2002bge,
  author    = {Geiger, Dan and Heckerman, David},
  title     = {Parameter Priors for Directed Acyclic Graphical Models and the
               Characterization of Several Probability Distributions},
  journal   = {The Annals of Statistics}, volume = {30}, number = {5}, year = {2002}}   % VERIFIED

@article{kuipers2014addendum,
  author    = {Kuipers, Jack and Moffa, Giusi and Heckerman, David},
  title     = {Addendum on the Scoring of {G}aussian Directed Acyclic Graphical Models},
  journal   = {The Annals of Statistics}, volume = {42}, number = {4}, year = {2014}}   % VERIFIED

@article{schwarz1978bic,
  author    = {Schwarz, Gideon},
  title     = {Estimating the Dimension of a Model},
  journal   = {The Annals of Statistics}, volume = {6}, number = {2}, year = {1978}}    % STANDARD
```

**Used for:** `sa/score.py`. BGe is the default; the 2014 addendum is the corrected
formulation we implement. BIC is kept as an independent check.

## 4. Interventions

```bibtex
@techreport{cooper1999causal,
  author      = {Cooper, Gregory F. and Yoo, Changwon},
  title       = {Causal Discovery from a Mixture of Experimental and Observational Data},
  institution = {Uncertainty in Artificial Intelligence (UAI)}, year = {1999}}          % VERIFIED

@article{peters2014identifiability,
  author  = {Peters, Jonas and B\"uhlmann, Peter},
  title   = {Identifiability of {G}aussian Structural Equation Models with Equal Error Variances},
  journal = {Biometrika}, volume = {101}, number = {1}, year = {2014}}                  % VERIFIED
```

**Used for:** the interventional likelihood rule (drop the intervened node's own term, keep
its value for children), and the reason every node draws its **own** noise scale — equal
variances make the model identifiable observationally and delete the problem.

## 5. Experimental design and information gain

```bibtex
@article{lindley1956measure,
  author  = {Lindley, Dennis V.},
  title   = {On a Measure of the Information Provided by an Experiment},
  journal = {The Annals of Mathematical Statistics}, volume = {27}, number = {4}, year = {1956}}  % VERIFIED

@article{chaloner1995bayesian,
  author  = {Chaloner, Kathryn and Verdinelli, Isabella},
  title   = {{B}ayesian Experimental Design: A Review},
  journal = {Statistical Science}, volume = {10}, number = {3}, year = {1995}}          % STANDARD

@article{golovin2011adaptive,
  author  = {Golovin, Daniel and Krause, Andreas},
  title   = {Adaptive Submodularity: Theory and Applications in Active Learning and
             Stochastic Optimization},
  journal = {Journal of Artificial Intelligence Research}, volume = {42}, year = {2011}}  % VERIFIED

@inproceedings{murphy2001active,
  author    = {Murphy, Kevin P.},
  title     = {Active Learning of Causal {B}ayes Net Structure},
  year      = {2001}}                                          % UNVERIFIED -- tech report? check

@inproceedings{tong2001active,
  author    = {Tong, Simon and Koller, Daphne},
  title     = {Active Learning for Structure in {B}ayesian Networks},
  booktitle = {IJCAI}, year = {2001}}                                                  % STANDARD
```

**Used for:** the EIG oracle. Lindley grounds the claim that maximising the entropy of the
outcome partition *is* maximising expected information gain. Golovin & Krause is the reason
beating a myopic oracle is possible at all — the `(1−1/e)` guarantee needs adaptive
submodularity, which EIG does not satisfy in general.

## 6. Exact structure inference

```bibtex
@inproceedings{koivisto2004exact,
  author    = {Koivisto, Mikko and Sood, Kismat},
  title     = {Exact {B}ayesian Structure Discovery in {B}ayesian Networks},
  booktitle = {Journal of Machine Learning Research}, volume = {5}, year = {2004}}      % STANDARD

@article{tian2009computing,
  author  = {Tian, Jin and He, Ru},
  title   = {Computing Posterior Probabilities of Structural Features in {B}ayesian Networks},
  journal = {Uncertainty in Artificial Intelligence (UAI)}, year = {2009}}              % STANDARD

@inproceedings{silander2006simple,
  author    = {Silander, Tomi and Myllym\"aki, Petri},
  title     = {A Simple Approach for Finding the Globally Optimal {B}ayesian Network Structure},
  booktitle = {Uncertainty in Artificial Intelligence (UAI)}, year = {2006}}            % STANDARD

@inproceedings{bjorklund2007fourier,
  author    = {Bj\"orklund, Andreas and Husfeldt, Thore and Kaski, Petteri and Koivisto, Mikko},
  title     = {Fourier Meets {M}\"obius: Fast Subset Convolution},
  booktitle = {STOC}, year = {2007}}                                                    % STANDARD

@article{robinson1973counting,
  author  = {Robinson, Robert W.},
  title   = {Counting Labeled Acyclic Digraphs},
  journal = {New Directions in the Theory of Graphs}, year = {1973}}      % UNVERIFIED -- confirm venue
```

**Used for:** `sa/dp.py`. The sink recurrence with inclusion–exclusion is Robinson's; the
modern treatment and the `O(3^d)` framing come from Koivisto & Sood. This is what moves the
reachable size from ~6 variables to ~9.

## 7. Sampling DAGs

```bibtex
@inproceedings{talvitie2019exact,
  author    = {Talvitie, Topi and Vuoksenmaa, Aleksis and Koivisto, Mikko},
  title     = {Exact Sampling of Directed Acyclic Graphs from Modular Distributions},
  booktitle = {Uncertainty in Artificial Intelligence (UAI)},
  address   = {Tel Aviv, Israel}, year = {2019},
  note      = {Best Student Paper. Proceedings in PMLR vol. 115, dated 2020 --
               tooling renders this as talvitie20a; cite the conference year.}}         % VERIFIED

@article{madigan1995bayesian,
  author  = {Madigan, David and York, Jeremy},
  title   = {{B}ayesian Graphical Models for Discrete Data},
  journal = {International Statistical Review}, volume = {63}, year = {1995}}  % UNVERIFIED -- confirm

@article{friedman2003being,
  author  = {Friedman, Nir and Koller, Daphne},
  title   = {Being {B}ayesian About Network Structure},
  journal = {Machine Learning}, volume = {50}, year = {2003}}                            % STANDARD

@article{grzegorczyk2008improving,
  author  = {Grzegorczyk, Marco and Husmeier, Dirk},
  title   = {Improving the Structure {MCMC} Sampler for {B}ayesian Networks by Introducing
             a New Edge Reversal Move},
  journal = {Machine Learning}, volume = {71}, year = {2008}}                            % STANDARD

@article{kuipers2017partition,
  author  = {Kuipers, Jack and Moffa, Giusi},
  title   = {Partition {MCMC} for Inference on Acyclic Digraphs},
  journal = {Journal of the American Statistical Association}, volume = {112}, year = {2017},
  note    = {arXiv:1504.05006}}                                                          % VERIFIED
```

**Used for:** `sa/dag_samplers.py`. Talvitie et al. is the implemented exact sampler and is
the default oracle. Partition MCMC is implemented and **broken** (error ~0.5, unimproved by
200× burn-in) — report as a negative result, do not present as working.

## 8. Reinforcement learning

```bibtex
@article{schulman2017ppo,
  author  = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec
             and Klimov, Oleg},
  title   = {Proximal Policy Optimization Algorithms},
  journal = {arXiv:1707.06347}, year = {2017}}                                          % STANDARD

@inproceedings{schulman2015gae,
  author    = {Schulman, John and Moritz, Philipp and Levine, Sergey and Jordan, Michael
               and Abbeel, Pieter},
  title     = {High-Dimensional Continuous Control Using Generalized Advantage Estimation},
  booktitle = {ICLR}, year = {2016}, note = {arXiv:1506.02438, 2015}}                    % STANDARD

@inproceedings{ng1999policy,
  author    = {Ng, Andrew Y. and Harada, Daishi and Russell, Stuart},
  title     = {Policy Invariance Under Reward Transformations: Theory and Application
               to Reward Shaping},
  booktitle = {ICML}, year = {1999}}                                                    % VERIFIED

@inproceedings{zaheer2017deepsets,
  author    = {Zaheer, Manzil and Kottur, Satwik and Ravanbakhsh, Siamak and P\'oczos, Barnab\'as
               and Salakhutdinov, Ruslan and Smola, Alexander J.},
  title     = {Deep Sets}, booktitle = {NeurIPS}, year = {2017}}                        % STANDARD

@article{bronstein2021geometric,
  author  = {Bronstein, Michael M. and Bruna, Joan and Cohen, Taco and Veli\v{c}kovi\'c, Petar},
  title   = {Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges},
  journal = {arXiv:2104.13478}, year = {2021}}                                          % STANDARD

@inproceedings{silver2010pomcp,
  author    = {Silver, David and Veness, Joel},
  title     = {Monte-{C}arlo Planning in Large {POMDP}s},
  booktitle = {NeurIPS}, year = {2010}}                                                 % STANDARD
```

**Used for:** `ma/policy2.py` and `sa/policy.py`. Ng et al. is what licenses potential-based
shaping without changing the optimal policy. Deep Sets is the characterisation behind the
permutation-equivariant per-node scorer (the change that made the single-agent result work:
probe 0.814 against 0.528 for a flat network).

## 9. Multi-agent RL — including the deliberately excluded set

```bibtex
@article{dewitt2020independent,
  author  = {de Witt, Christian Schroeder and Gupta, Tarun and Makoviichuk, Denys and
             Makoviychuk, Viktor and Torr, Philip H. S. and Sun, Mingfei and Whiteson, Shimon},
  title   = {Is Independent Learning All You Need in the {S}tar{C}raft Multi-Agent Challenge?},
  journal = {arXiv:2011.09533}, year = {2020}}                                          % VERIFIED

@inproceedings{foerster2018coma,
  author    = {Foerster, Jakob N. and Farquhar, Gregory and Afouras, Triantafyllos and
               Nardelli, Nantas and Whiteson, Shimon},
  title     = {Counterfactual Multi-Agent Policy Gradients}, booktitle = {AAAI}, year = {2018}}  % STANDARD

@inproceedings{rashid2018qmix,
  author    = {Rashid, Tabish and Samvelyan, Mikayel and de Witt, Christian Schroeder and
               Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},
  title     = {{QMIX}: Monotonic Value Function Factorisation for Deep Multi-Agent
               Reinforcement Learning}, booktitle = {ICML}, year = {2018}}              % STANDARD

@inproceedings{sunehag2018vdn,
  author    = {Sunehag, Peter and others},
  title     = {Value-Decomposition Networks for Cooperative Multi-Agent Learning},
  booktitle = {AAMAS}, year = {2018}}                                                   % STANDARD

@inproceedings{lowe2017maddpg,
  author    = {Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter
               and Mordatch, Igor},
  title     = {Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  booktitle = {NeurIPS}, year = {2017}}                                                 % STANDARD

@inproceedings{yu2022mappo,
  author    = {Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu
               and Bayen, Alexandre and Wu, Yi},
  title     = {The Surprising Effectiveness of {PPO} in Cooperative Multi-Agent Games},
  booktitle = {NeurIPS Datasets and Benchmarks}, year = {2022}}                         % STANDARD

@inproceedings{foerster2017stabilising,
  author    = {Foerster, Jakob and Nardelli, Nantas and Farquhar, Gregory and Afouras,
               Triantafyllos and Torr, Philip H. S. and Kohli, Pushmeet and Whiteson, Shimon},
  title     = {Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning},
  booktitle = {ICML}, year = {2017}}                                                    % STANDARD
```

**How to use these in the thesis:** COMA, QMIX, VDN, MADDPG and MAPPO are all **CTDE** and
therefore all excluded by the supervisor constraint. Cite them precisely *as* the scoped-out
set — a reader will otherwise ask why the obvious cooperative-MARL algorithms are absent.
de Witt et al. is the cover for IPPO being a strong baseline rather than a compromise.

## 10. Federated learning

```bibtex
@article{kairouz2021advances,
  author  = {Kairouz, Peter and McMahan, H. Brendan and others},
  title   = {Advances and Open Problems in Federated Learning},
  journal = {Foundations and Trends in Machine Learning}, volume = {14}, year = {2021}} % STANDARD

@inproceedings{mcmahan2017fedavg,
  author    = {McMahan, H. Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth
               and Ag\"uera y Arcas, Blaise},
  title     = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  booktitle = {AISTATS}, year = {2017}}                                                 % STANDARD

@article{federated_causal_interventions,
  title   = {Federated Causal Discovery From Interventions},
  journal = {arXiv:2211.03846}}                              % VERIFIED 2026-08-20 -- see below
```

**Note:** FedAvg assumes a central server, which our setting forbids — worth stating
explicitly as the point of departure. `arXiv:2211.03846` looks like the closest related work
and **must be read** before the related-work section is written.

## 11. Latent confounding and MAGs

```bibtex
@article{richardson2002ancestral,
  author  = {Richardson, Thomas and Spirtes, Peter},
  title   = {Ancestral Graph {M}arkov Models},
  journal = {The Annals of Statistics}, volume = {30}, number = {4}, year = {2002}}     % VERIFIED
```

**Used for:** `ma/projection.py`. The definitions of MAG adjacency (no subset d-separates)
and orientation (`u ↔ v` when neither is an ancestor of the other) are taken from here, and
are what the "confounding is confined to the shared set" proof is stated in.

## 12. Scaling and graph density — the newest thread

```bibtex
@article{chevalley2025guarantees,
  author  = {Chevalley, Mathieu and Mehrjou, Arash and Schwab, Patrick},
  title   = {Theoretical Guarantees for Causal Discovery on Large Random Graphs},
  journal = {arXiv:2511.02536}, year = {2025}}                                          % VERIFIED

@article{vanderhofstad_percolation,
  author = {van der Hofstad, Remco},
  title  = {Percolation and Random Graphs},
  note   = {Survey chapter; also see arXiv:2512.15673, "Percolation on random graphs"}}  % UNVERIFIED
```

**Why this matters, and it is a live defect:** Chevalley et al. assume the sparse regime
**`p_e = Θ(1/d)`**. Our `prior_p` is a fixed **0.5**, which at `d = 4–5` lands at expected
degree 1.5–2.0 — inside the sparse regime *by accident* — and does not scale. At `d = 30` a
fixed 0.5 gives ~218 edges, expected degree 14.5, against the literature's ER-2/ER-4
benchmarks (60–120 edges). **`prior_p` must become a function of `d`** before any scaling
claim is meaningful.

The percolation link is **our framing, not a citation**: in `G(d, p)` the giant component
appears at `p_c = 1/d`, so `p_e = Θ(1/d)` *is* percolation-critical scaling. A fixed 0.5 sits
15× above threshold at `d = 30` — deep in the dense phase, one connected blob, where recovery
is neither realistic nor informative. State it as a framing device; do not attribute it.

## 13. Amortised causal discovery

```bibtex
@inproceedings{lorch2022avici,
  author    = {Lorch, Lars and Sussex, Scott and Rothfuss, Jonas and Krause, Andreas
               and Sch\"olkopf, Bernhard},
  title     = {Amortized Inference for Causal Structure Learning},
  booktitle = {NeurIPS}, year = {2022}}                                                 % STANDARD
```

**Note:** AVICI was used in an earlier generation of this project and is **not** part of the
current design. Mention only as an alternative to exact inference, and note the practical
finding that its released code breaks on newer JAX (`PositionalSharding` removed).

## 14. Sequential design as RL — the closest published framings

```bibtex
@inproceedings{foster2021dad,
  author    = {Foster, Adam and Ivanova, Desi R. and Malik, Ilyas and Rainforth, Tom},
  title     = {Deep Adaptive Design: Amortizing Sequential {B}ayesian Experimental Design},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series    = {PMLR}, volume = {139}, pages = {3384--3395}, year = {2021}}   % VERIFIED 2026-08-20

@inproceedings{blau2022rlboed,
  author    = {Blau, Tom and Bonilla, Edwin V. and Chades, Iadine and Dezfouli, Amir},
  title     = {Optimizing Sequential Experimental Design with Deep Reinforcement Learning},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series    = {PMLR}, volume = {162}, pages = {2107--2128}, year = {2022},
  note      = {arXiv:2202.00821}}                                       % VERIFIED 2026-08-20
```

**These two matter most for positioning.** They are the closest published framings of our
research question, and they determine whether the contribution is *the method* or *the
federation of it*. Both are currently **unverified** — reading them is a blocking task for
the related-work section, not an optional one.

---

## Before submission — the checklist

1. Verify every **UNVERIFIED** entry against the source. There are seven.
2. Read `arXiv:2211.03846` (federated causal discovery from interventions) — closest related
   work, and it is currently unread.
3. Read Foster et al. and Blau et al. properly; they set the positioning of the whole thesis.
4. Confirm editions and page numbers for the **STANDARD** book entries.
5. Fix the Talvitie citation key by hand if the bibliography tool renders `talvitie20a` —
   the conference was 2019, PMLR vol. 115 carries a 2020 date.

---

## Verification pass, 20 August 2026 — the three blocking entries

All three now **VERIFIED** against the publisher record.

**Foster, Ivanova, Malik & Rainforth (2021), _Deep Adaptive Design: Amortizing Sequential
Bayesian Experimental Design_.** ICML 2021, PMLR **139**:3384–3395; key `foster21a`. Authors
and title confirmed exactly as recorded. Amortizes sequential BOED: rather than solving a
design optimisation at every stage, a design network is trained once offline and maps history
to the next design in a single forward pass, so decisions take milliseconds at deployment.

*Relevance:* this is the standard answer to "why not just run greedy EIG at every step", and
our claim to beat myopic design has to be positioned against it. Note the difference in
motivation: DAD amortises for **speed at deployment**; we are after **non-myopia**, which is
a different axis and worth saying explicitly.

**Blau, Bonilla, Chades & Dezfouli (2022), _Optimizing Sequential Experimental Design with
Deep Reinforcement Learning_.** ICML 2022 **spotlight**, PMLR **162**:2107–2128,
arXiv:2202.00821; key `blau22a`. Reduces policy optimisation for sequential design to an MDP
and solves it with deep RL.

*Relevance:* **the closest published framing of our research question** — RL in place of
greedy EIG. It therefore decides whether our contribution reads as *the method* or as *the
federation of it*. The honest position is the latter: the federation, the private/shared
partition, and the latent confounding it induces are what is new here.

**Abyaneh, Scherrer, Schwab, Bauer, Schölkopf & Mehrjou (2022), _Federated Causal Discovery
From Interventions_ (FedCDI).** arXiv:2211.03846, submitted 7 Nov 2022, revised 11 Feb 2024;
arXiv preprint, no published venue recorded. **The author list was previously unrecorded and
is now confirmed.** Proposes a federated framework that exchanges *belief updates* rather than
raw data, with an intervention-aware aggregation rule covering shared and disparate intervened
variables.

*Relevance and the point of departure.* This is the closest related work and it is close:
federated, interventional, belief-passing rather than data-passing. **Two differences to
establish, and only one is confirmed so far.**

1. *Aggregation.* The abstract describes aggregating individual updates. Whether that
   aggregation requires a coordinating server — which our setting forbids outright — is
   **not resolvable from the abstract** and must be read out of the method section. If FedCDI
   does assume an aggregator, that is our sharpest point of departure. If it does not, the
   distinction has to be found elsewhere and the positioning gets harder.
2. *Latent confounding across the partition.* Our setting's defining difficulty is that an
   agent cannot see its partner's private variables, so a shared pair can be confounded by
   something structurally invisible to it. Whether FedCDI's partition creates the same
   problem is unknown and is the second thing to check.

**Reading the abstract is not reading the paper.** Both points above are positioning-critical
and neither is settled by what has been verified here.
