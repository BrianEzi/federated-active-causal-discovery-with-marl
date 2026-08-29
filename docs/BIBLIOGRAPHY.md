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

```bibtex
@article{ali2009markovmag,
  author  = {Ali, R. Ayesha and Richardson, Thomas S. and Spirtes, Peter},
  title   = {{M}arkov Equivalence for Ancestral Graphs},
  journal = {The Annals of Statistics}, volume = {37}, number = {5B}, year = {2009}}    % STANDARD
```

**Used for:** the definition of Markov equivalence as `(skeleton, v-structures)`, which is
implemented directly as `mec_signature` in `sa/graphs.py`.

**Used for (2026-08-26):** `ali2009markovmag` is the MAG-side characterisation -- same
adjacencies, same colliders with order, plus a discriminating-path condition. The
"same adjacencies" clause is what licenses `cb/versionspace.py::equivalence_class` to
enumerate orientations of the TRUE SKELETON (3^edges) instead of all mark assignments
(4^pairs). Verified against exhaustive search at k=4 in
`tests/cb/test_versionspace.py`. CONFIRM the discriminating-path clause against the paper
before citing it in the thesis -- the implementation does not rely on it, since candidates
are filtered by full m-separation signature rather than by the graphical criterion.

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

## 18. Version spaces and query learning — the deterministic backend's frame

```bibtex
@article{mitchell1982generalization,
  author  = {Mitchell, Tom M.},
  title   = {Generalization as Search},
  journal = {Artificial Intelligence}, volume = {18}, number = {2}, year = {1982}}      % STANDARD

@article{angluin1988queries,
  author  = {Angluin, Dana},
  title   = {Queries and Concept Learning},
  journal = {Machine Learning}, volume = {2}, number = {4}, year = {1988}}              % STANDARD

@inproceedings{seung1992qbc,
  author    = {Seung, H. S. and Opper, M. and Sompolinsky, H.},
  title     = {Query by Committee},
  booktitle = {Computational Learning Theory (COLT)}, year = {1992}}                    % STANDARD
```

**Used for:** the framing of `cb/versionspace.py`. A version space is the set of hypotheses
consistent with the evidence so far; learning is ELIMINATION, the space only shrinks, and a
proposition is known when every survivor agrees (Mitchell's classification rule -- which is
exactly why `claim_bar` must be 1.0 on that backend, and why settled-wrong is impossible
there). The observational version space over MAGs IS the Markov equivalence class, i.e. the
PAG. Interventions are QUERIES that partition the space, which is the halving-algorithm /
query-by-committee setting: choose the query the survivors most disagree about.

Also the source of the known WEAKNESS: version spaces are brittle under noise, because a
single incorrect elimination removes the truth permanently and the space cannot recover.
That is the argument for keeping the bootstrap belief in the statistical environment rather
than porting elimination to it.

## 19. Joint Causal Inference — the regime indicator, and why exclusion is a choice

```bibtex
@article{mooij2020jci,
  author  = {Mooij, Joris M. and Magliacane, Sara and Claassen, Tom},
  title   = {Joint Causal Inference from Multiple Contexts},
  journal = {Journal of Machine Learning Research}, volume = {21}, number = {99},
  pages   = {1--108}, year = {2020},
  note    = {arXiv:1611.10351; code at https://github.com/caus-am/jci}}                 % VERIFIED

@article{hauser2012interventional,
  author  = {Hauser, Alain and B\"uhlmann, Peter},
  title   = {Characterization and Greedy Learning of Interventional {M}arkov Equivalence
             Classes of Directed Acyclic Graphs},
  journal = {Journal of Machine Learning Research}, volume = {13}, pages = {2409--2464},
  year    = {2012}}                                                                     % VERIFIED

@article{hauser2014twostrategies,
  author  = {Hauser, Alain and B\"uhlmann, Peter},
  title   = {Two Optimal Strategies for Active Learning of Causal Models from
             Interventional Data},
  journal = {International Journal of Approximate Reasoning}, volume = {55}, number = {4},
  year    = {2014}}                                                                     % VERIFIED

@misc{hardinterventions2025,
  title = {Characterization and Learning of Causal Graphs from Hard Interventions},
  note  = {arXiv:2505.01037}, year = {2025}}                                            % UNVERIFIED
```

**Used for (2026-08-26):**

`mooij2020jci` is the formal basis for treating the disclosed regime bit as a VARIABLE
rather than a filter. JCI adds context variables as NODES alongside system variables, pools
all contexts into one dataset, and runs standard discovery on the joint set; it does not
require knowing intervention targets or types.

ASSUMPTIONS, now VERBATIM from the paper (read in full 2026-08-29; the earlier entry was a
paraphrase from the abstract and got JCI 3 WRONG):
- **JCI 0** ("Joint SCM", required): the data-generating mechanism is a simple SCM jointly
  modelling system and context, with graph on nodes `I union K`.
- **JCI 1** ("Exogeneity", optional): no system variable causes any context variable,
  `for all k in K, i in I : i -> k not in G(M)`.
- **JCI 2** ("Complete randomized context", optional): no context variable is confounded
  with a system variable, `for all k in K, i in I : i <-> k not in G(M)`.
- **JCI 3** ("Generic context model", optional): `for all k != k' in K : k <-> k' in G(M)
  AND k -> k' not in G(M)`.

**The JCI 3 correction matters.** The old entry said "no arrows among context variables".
The assumption is the opposite in spirit: every pair of context variables IS connected, by a
BIDIRECTED edge, and none by a directed one. That is why FCI-JCI123 does not remove edges
between context variables in its adjacency phase (Section 4.2.4). An implementation built
from the old paraphrase would have deleted exactly the edges the assumption asserts.

CAVEAT, and the paper states it itself (Section 3.4.2): our interventions are chosen
ADAPTIVELY from beliefs that are functions of past data, so the context variable is caused
by past system variables and JCI 1 fails. Their own example is a doctor who "first diagnoses
a patient before deciding on treatment", for which "JCI Assumption 1 would not apply", and
footnote 15 notes that sticking to a protocol FIXED BEFOREHAND is what excludes the
influence. Active experimental design is therefore outside JCI's exogeneity assumption by
construction. Treat each round as its own context, or condition on history, and say so.

TWO FURTHER POINTS THAT BEAR ON THIS PROJECT:
- **JCI cannot handle different variables per context.** Table 4's "Different variables in
  each context" column is a MINUS for Joint Causal Inference, FCI-JCI and ASD-JCI alike;
  Section 4.3.7 says it needs a strengthened faithfulness assumption, and Section 6 lists it
  as future work. Our vertical setting sits in that named gap. The methods that DO have a
  plus there are Claassen & Heskes (2010), Tillman & Spirtes (2011), Hyttinen et al. (2014),
  Triantafillou & Tsamardinos (2015) and Forre & Mooij (2018).
- **Multiple context variables beat one merged variable** (Section 4.3.5, Figures 20 and 23),
  and merging "typically loses information". Our `clean` is a SCALAR fraction per row batch,
  i.e. a merged context variable -- and `ma/env.py` already documents the symptom in its own
  words: the mixture "knows how MANY hidden nodes were clamped, never WHICH".

`hauser2012interventional` gives the characterisation used to settle clamp-vs-vary: two
DAGs are I-Markov equivalent iff, for every intervention target set S, the graphs with
incoming edges of S removed are Markov equivalent. The class depends on the TARGETS, not on
the intervened VALUES -- so a clamp and a randomised "vary" are equally informative in the
identifiability limit. The measured preference for vary (2026-08-24) is a FINITE-SAMPLE
estimator property: clamping leaves the target with zero variance, so the correlation
channel is uninformative and only mean/variance shifts remain. Conversely, clamping a
HIDDEN common cause cuts the path and makes the association vanish, which is why clamp is
the right mode for private nodes and vary for one's own target.

`hardinterventions2025` is a LEAD ONLY, recorded for the statement that hard interventions
may fix a value "deterministically ... or by stochastically assigning values drawn from an
independent distribution" -- i.e. our `vary` is a hard intervention, not a soft one. Read it
before citing.

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

*Read in full 2026-08-20 (v3, 24 Jun 2023). Every question below is now answered from the
method, not inferred.* Note the v3 title differs from v1: it is **"FED-CD: Federated Causal
Discovery from Interventional and Observational Data"**, and the method is called FED-CD,
not FedCDI. Cite the version you actually read.

1. **It requires a central server, explicitly.** Section 3: "Our proposed federated setup
   consists of a central node acting as a server S and K other nodes as clients... Each
   client is an independent processing unit, and can only communicate with the server."
   Clients never talk to each other. The server runs `proximity_based_aggregation`
   (Algorithm 1) and broadcasts the aggregated belief back each round. **This is our
   sharpest point of departure and it is now confirmed rather than hoped.**
2. **Every client observes every variable.** Section 3: the observational split is
   HORIZONTAL, and "we assume that clients are aware of all the dataset features, but might
   not have access to interventional data corresponding to each random variable." What they
   call a vertical split concerns only which variables a client may INTERVENE on. There is
   no restricted observation window anywhere in the paper.
3. **Latent confounding is assumed away.** Section 3: "we assume causal sufficiency of the
   CGM, i.e., all common causes of variables are included and observable." The problem that
   defines our setting does not arise in theirs.
4. **What is exchanged is a full `N x N` belief matrix** of independent Bernoulli edge
   probabilities (Definition 4.1), plus the server's aggregated belief on the way back.
5. **The local discovery method is ENCO** (Lippe et al. 2021), a continuous-optimisation
   learner; SDI and DCDI are named as alternatives. The contribution is the aggregation
   rule -- reliability scores computed by flowing hypothetical mass from a client's
   intervened nodes along paths in the current graph, then softmax-weighted across clients.
6. **Interventions are never CHOSEN.** The interventional data is given. There is no budget,
   no design, no policy, no sequential decision at all.

**Experiments:** `d = 20`, Erdos-Renyi ER-1/2/4/6, categorical data from randomly initialised
MLPs, SHD against ground truth, 20 seeds; real graphs Sachs, Alarm, Asia; up to ~10 clients.
Baselines GIES, IGSP, centralised ENCO, and an isolated non-collaborating client.

**Net effect on our positioning -- much better than feared.** FED-CD is *passive* federated
structure learning with a server, full observability at every client, and causal sufficiency.
We are *active* experimental design, serverless, with restricted per-agent windows and latent
confounding induced by exactly that restriction. The overlap is the word "federated". Their
`d = 20` ER-1..6 setup is a useful precedent for our scaling target, and their
five-clients-intervening-on-4-of-20-variables setting is the closest thing in the literature
to our five-agent goal.

### Foster et al. (DAD) and Blau et al. (RL-BOED) — read in full, 20 August 2026

**DAD, read from the PMLR v139 PDF.** The design network `pi_phi` is a **deterministic**
policy mapping history to the next design in one forward pass. Theorem 1 rewrites the total
EIG of a *policy* over `T` experiments as a single expectation, which removes intermediate
posteriors from the objective entirely; Theorem 2 gives the sequential PCE (sPCE) lower
bound, tight as `L -> infinity` at `O(1/L)`. The optimal policy is shown to be invariant to
the ORDER of the history, and the architecture is built around that symmetry.

*The limitations are stated by the authors and they matter to us.* The main gradient
estimator assumes a **continuous design space** and a **reparametrisable, differentiable**
likelihood. For discrete observations the exact gradient costs `O(|Y|^T)` and is usable only
when both the number of experiments and the number of outcomes are tiny; otherwise it falls
back to REINFORCE. **Our design space is discrete by construction** -- which node to
intervene on -- so DAD is not directly applicable to our problem, and that is a factual
statement about its assumptions rather than a criticism.

**RL-BOED, read from arXiv:2202.00821v3.** Formulates sequential design as a **Hidden
Parameter MDP** (Doshi-Velez & Konidaris 2016), because the model parameters are not
observable at test time and are fixed within an episode. Theorem 1 shows a terminal-reward
HIP-MDP already optimises sPCE; Section 3.2 then replaces it with a **dense** per-step
reward (Eq. 13) measuring each experiment's marginal contribution to cumulative EIG, and
Theorem 2 shows the dense form has the same optimum. Trained with **REDQ** (Chen et al.
2021) in Pyro + Garage. Their stated advantages over DAD are exactly the three we care
about: **discrete design spaces**, **black-box non-differentiable likelihoods**, and
**exploration** -- DAD's policy is deterministic and "a pure exploitation algorithm".

Their NAIVE-RL ablation isolates the reward design: 9.789 against 11.73 for the full method
on source location at `T = 30`, with DAD at 10.965 and random at 1.624. Same architecture,
different reward -- so the dense reward is doing real work, not the network.

**What neither paper does, and what that means for our claim.** Neither does *causal
structure* discovery: their problems are parameter inference (source location, constant
elasticity of substitution). Neither is federated, neither has restricted observability, and
neither has more than one agent. So the honest positioning is:

- **Method:** RL for sequential experimental design is Blau et al. We are not claiming it.
- **Ours:** the federation of it -- multiple agents with disjoint private views, no server,
  and latent confounding created by the partition itself, which none of the three addresses.

Two specifics worth carrying into the write-up. First, our **discrete** design space is
precisely the case that motivates RL over DAD, so Blau et al. supports our choice of PPO
rather than threatening it. Second, DAD's **permutation invariance** of the optimal policy is
the same structural argument behind our permutation-equivariant per-node scorer, and it
should be cited there rather than presented as our own idea.

## 15. Multi-agent / distributed greedy — the baselines our design was missing

Found 2026-08-21 while looking for a defensible multi-agent greedy. This literature does
exactly the coordination problem we have, and it arrives at turn-taking independently of
the supervisor.

```bibtex
@article{fisher1978analysis,
  author  = {Fisher, M. L. and Nemhauser, G. L. and Wolsey, L. A.},
  title   = {An Analysis of Approximations for Maximizing Submodular Set Functions---{II}},
  journal = {Mathematical Programming Study}, volume = {8}, pages = {73--87},
  year    = {1978}}                                          % STANDARD -- the 1/2 bound

@article{grimsman2018impact,
  author  = {Grimsman, David and Ali, Mohd. Shabbir and Hespanha, Jo{\~a}o P.
             and Marden, Jason R.},
  title   = {The Impact of Information in Distributed Submodular Maximization},
  journal = {IEEE Transactions on Control of Network Systems}, year = {2018},
  note    = {Earlier version, IEEE CDC 2017; arXiv:1807.10639}}   % VERIFIED 2026-08-21

@article{corah2019distributed,
  author  = {Corah, Micah and Michael, Nathan},
  title   = {Distributed Matroid-Constrained Submodular Maximization for Multi-Robot
             Exploration: Theory and Practice},
  journal = {Autonomous Robots}, volume = {43}, number = {2}, pages = {485--501},
  year    = {2019}}                                              % VERIFIED 2026-08-21
```

**Sequential greedy assignment (SGA).** Agents decide IN TURN, each conditioning on the
choices already made by earlier agents; `1/2` of optimal under a matroid constraint
(Fisher et al. 1978). This is the multi-agent greedy we should have had. Our current
`GreedyAgent` is NOT this -- it optimises its own window in isolation and conditions on
nothing -- and turn-taking is precisely the protocol that makes SGA implementable.

**Grimsman et al. (2018)** bound greedy's quality as a function of **how much of the earlier
agents' decisions each agent can see**, with performance degrading in the size of the largest
group deciding independently. That is our disclosure question stated formally: we already
disclose shared-node targets after acting, and this says what such disclosure is worth.

**Corah & Michael (2019)** give DSGA, which keeps SGA's bound plus **a penalty for assigning
several plans at once**. It therefore quantifies the cost of simultaneous decision-making
against sequential -- independent theoretical support for the turn-taking directive, from a
literature that had never heard of this project.

**THE CAVEAT, WHICH MUST BE STATED WHEREVER THESE ARE CITED.** All of these guarantees need
submodularity, and the adaptive sequential case needs **adaptive submodularity** (Golovin &
Krause 2011), which expected information gain does **not** satisfy in general. We borrow the
ALGORITHMS as baselines. We must not claim the BOUNDS.

**The agreed baseline set** (decided 2026-08-21):

| baseline | what it is | what it shows |
|---|---|---|
| random | uniform over legal targets | what pure exploration achieves |
| selfish greedy | own window only -- what we have | the honest myopic floor |
| sequential greedy | conditions on the partner's disclosed choices | the real decentralised baseline |
| joint greedy | one oracle scoring BOTH posteriors | upper bound; deliberately violates federation |

Beating sequential greedy makes the claim mean something; approaching joint greedy shows the
federation costs little. Random stays in because it is the only arm that says what
exploration alone buys.

---

## 16. Belief aggregation — opinion pooling, fusion, and constraint combination

Added 2026-08-23. Every entry below was **checked against the source on that date** while
scoping `docs/DISCLOSURE_SPEC.md`; these are VERIFIED in the strict sense of this file.

```bibtex
@article{stone1961opinion,
  author  = {Stone, Mervyn},
  title   = {The Opinion Pool},
  journal = {The Annals of Mathematical Statistics},
  volume  = {32}, number = {4}, pages = {1339--1342}, year = {1961}}                   % VERIFIED

@article{genest1984characterization,
  author  = {Genest, Christian},
  title   = {A Characterization Theorem for Externally {B}ayesian Groups},
  journal = {The Annals of Statistics},
  volume  = {12}, number = {3}, pages = {1100--1105}, year = {1984}}                    % VERIFIED

@article{genest1986combining,
  author  = {Genest, Christian and Zidek, James V.},
  title   = {Combining Probability Distributions: A Critique and an Annotated Bibliography},
  journal = {Statistical Science},
  volume  = {1}, number = {1}, pages = {114--135}, year = {1986}}                       % VERIFIED

@article{degroot1974consensus,
  author  = {DeGroot, Morris H.},
  title   = {Reaching a Consensus},
  journal = {Journal of the American Statistical Association},
  volume  = {69}, number = {345}, pages = {118--121}, year = {1974}}                    % VERIFIED

@inproceedings{julier1997covariance,
  author    = {Julier, Simon J. and Uhlmann, Jeffrey K.},
  title     = {A Non-divergent Estimation Algorithm in the Presence of Unknown Correlations},
  booktitle = {Proceedings of the American Control Conference},
  pages     = {2369--2373}, year = {1997}}                                             % VERIFIED

@article{grime1994decentralized,
  author  = {Grime, S. and Durrant-Whyte, H. F.},
  title   = {Data Fusion in Decentralized Sensor Networks},
  journal = {Control Engineering Practice},
  volume  = {2}, number = {5}, pages = {849--863}, year = {1994}}                       % VERIFIED

@book{pearl1988probabilistic,
  author    = {Pearl, Judea},
  title     = {Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference},
  publisher = {Morgan Kaufmann}, year = {1988}}                                        % VERIFIED (book)
```

**Why each matters.**

**Stone (1961)** is the linear opinion pool — the weighted arithmetic mean, i.e. the obvious
thing. Cite it to *reject* it: the linear pool is **not externally Bayesian**, so pooling then
updating differs from updating then pooling. For a belief updated every round that is
disqualifying.

**Genest (1984)** is the reason we do not have to argue about which pooling rule to use. With
unanimity and regularity, **logarithmic pooling is the UNIQUE externally Bayesian operator**.
It is a uniqueness theorem, not a list of nice properties. If this project ever pools opinions,
this citation settles how.

**Genest & Zidek (1986)** is the canonical survey, and it also catalogues the impossibility
results — no rule satisfies every desideratum, so some property must be given up deliberately.

**DeGroot (1974)** is decentralised consensus by iterated weighted averaging. Relevant because
it needs no coordinator, which is one of the few places our constraints are stricter than the
field's. The log-domain version converges to the logarithmic pool.

**Julier & Uhlmann (1997)** — covariance intersection — is the rigorous form of "take the least
confident estimate". It fuses estimates whose evidence overlap is *unknown* with a guarantee of
never becoming overconfident. It is a special case of the broader **Chernoff fusion rule**, and
the fusion literature's stated position is that Chernoff-family rules are what you need
specifically when network loops make double counting unavoidable. The price is that it
deliberately discards real information.

**Grime & Durrant-Whyte (1994)** is the exact alternative: track what each pair has already
exchanged and subtract the common part, recovering the centralised answer. It needs a loop-free
communication structure. **This maps onto our scaling ladder directly** — at two agents there is
one link and no loop, so exact incest-free fusion is available *now*; at three or more with
all-to-all broadcast, loops appear immediately and exactness becomes impossible. That is a
concrete cost of rung 1 we had not priced.

**Pearl (1988)** is noisy-OR — not an aggregation rule for opinions but a model of one variable
with several **independent sufficient causes**. That is why it fits our confounding claims, where
each agent reports on a different private set, and why it does not fit shared directed edges.
The book is VERIFIED; the specific section attribution for noisy-OR is standard usage and was
not confirmed at page level.

**The term to use in the write-up is "data incest"** (also "rumour propagation") — confirmed as
the distributed-fusion literature's own vocabulary for double counting. Its two remedies are
exactly the two above: bookkeep and subtract, or use a rule that cannot become overconfident.

**Ruled out: Dempster–Shafer combination.** It looks purpose-built for combining evidence from
independent sources, but Zadeh's counterexample shows it behaves pathologically under strongly
conflicting evidence — and conflict is our interesting case, not our edge case.

## 17. Federated causal discovery — the current landscape

Added 2026-08-23. This section exists because the field moved while we were building, and two
of these are close enough that the thesis must position against them explicitly.

```bibtex
@inproceedings{tillman2011overlapping,
  author    = {Tillman, Robert E. and Spirtes, Peter},
  title     = {Learning Equivalence Classes of Acyclic Models with Latent and Selection
               Variables from Multiple Datasets with Overlapping Variables},
  booktitle = {Artificial Intelligence and Statistics (AISTATS)},
  series    = {PMLR}, volume = {15}, year = {2011}}                                    % VERIFIED

@article{triantafillou2015combine,
  author  = {Triantafillou, Sofia and Tsamardinos, Ioannis},
  title   = {Constraint-based Causal Discovery from Multiple Interventions over
             Overlapping Variable Sets},
  journal = {Journal of Machine Learning Research},
  volume  = {16}, year = {2015}, note = {arXiv:1403.2150}}                              % VERIFIED

@article{hahn2026fedci,
  author  = {Hahn, Maximilian and Zajak, Alina and Heider, Dominik and Ribeiro, Adele H.},
  title   = {Federated Causal Discovery Across Heterogeneous Datasets under Latent Confounding},
  journal = {arXiv preprint arXiv:2603.05149}, year = {2026}}                           % VERIFIED

@inproceedings{wang2025nonidentical,
  author    = {Wang, Yunxia and Cao, Fuyuan and Yu, Kui and Liang, Jiye},
  title     = {Federated Causal Structure Learning with Non-identical Variable Sets},
  booktitle = {International Conference on Machine Learning (ICML)}, year = {2025}}     % VERIFIED

@inproceedings{baldo2026regret,
  author    = {Baldo, Federico and Assaad, Charles K.},
  title     = {Regret-Based Federated Causal Discovery with Unknown Interventions},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}, note = {arXiv:2512.23626}}                                        % VERIFIED

@inproceedings{ng2022federated,
  author    = {Ng, Ignavier and Zhang, Kun},
  title     = {Towards Federated {B}ayesian Network Structure Learning with
               Continuous Optimization},
  booktitle = {Artificial Intelligence and Statistics (AISTATS)},
  series    = {PMLR}, volume = {151}, pages = {8095--8111}, year = {2022}}              % VERIFIED

@article{fcdsurvey2026,
  author  = {{Authors not yet checked}},
  title   = {A Survey on Federated Causal Discovery and Inference},
  journal = {arXiv preprint arXiv:2606.23741}, year = {2026}}                           % UNVERIFIED
```

**Tillman & Spirtes (2011)** and **Triantafillou & Tsamardinos (2015)** are the two that match
our *partition* — overlapping variable sets, latent confounders. The second is closer: its
**COmbINE** algorithm additionally handles **multiple interventions**, converts dependencies and
independencies into path constraints, and solves the combination as a boolean satisfiability
instance. That is our problem with a different solver, and it is the source of the
constraint-combination argument in `DISCLOSURE_SPEC.md` section 7.

**Hahn et al. (2026)** is the closest published work and the one to position against. It
federates **IOD** (Integration of Overlapping Datasets — the Tillman & Spirtes line) and claims,
verbatim, to enable *"for the first time, federated causal discovery under latent confounding
across distributed and heterogeneous datasets"*. Two things follow. **We cannot claim that
phrase.** And two differences are ours to hold: it is **purely observational** — no
interventions, no experiment selection, no budget — and it shares **more than we do**, exchanging
regression sufficient statistics through a federated iteratively-reweighted-least-squares
procedure, against our one probability per shared pair. Its headline that federated performance
is *"comparable to fully pooled analyses"* is a bar worth quoting: federation costs them almost
nothing statistically.

**Wang et al. (ICML 2025)** matters for a subtler reason than its title. Its stated problem is
that *"non-overlapping variables may introduce spurious dependencies"* — which is our confounding
problem stated from the other side. Note which side they take: their method **removes** the
spurious dependency; ours **annotates** it as confounding. Both design choices now exist in the
literature and can be contrasted directly. Their aggregation is a two-level priority-selection
heuristic over local graphs rather than a principled rule, which is a weakness to cite rather
than inherit.

**Baldo & Assaad (ICML 2026)** looked like the nearest competitor and is not. Its interventions
are **unknown and externally imposed** (different hospitals' protocols happen to induce them) and
the method's job is to cope with interventions it cannot see or label. Ours are **chosen**, under
a budget. It is also **horizontally partitioned** — same variables, different samples — targeting
the CPDAG of the union of client graphs. Cite it to distinguish *chosen* from *observed*
interventions.

**Ng & Zhang (2022) — CORRECTION, recorded so it is not re-used wrongly.** This was cited on
2026-08-23 as evidence that federated structure learning addresses overlapping variable sets.
**It does not.** It is **horizontally partitioned**: every party holds the same variables over
different samples. Ours is the vertical, overlapping case. Real paper, wrong argument.

**The survey (arXiv:2606.23741)** is UNVERIFIED beyond existence — authors not checked. Its
abstract gives three axes (methodological paradigm, federation topology, structural scope) and
covers non-identical variable sets. It does **not** mention interventional or active discovery,
multi-agent settings, or reinforcement learning. That is suggestive of our gap but an abstract
omitting a category is not the survey omitting it. **Read the taxonomy section before using this
as a positioning argument.**
