# Writing handover — background, literature and methodology

For the writing agent. **A map, not a rewrite:** almost everything below already exists in
this repository, measured and sourced. The job is to turn it into prose, not to re-derive it.

Written 30 Aug 2026, after the method converged. Read section 0 before anything else.

---

## 0. READ THIS FIRST — documents that are WRONG

Four documents contain claims that measurement has since overturned. They are still in the
repo because the corrections are more legible beside the originals. **Do not write from
them without reading the correction.**

| document | what is wrong | correction |
|---|---|---|
| `MA_PROBLEM_STATEMENT.md` §2 | Frames coordination as the **clamp/vary sacrifice** — "for your partner, only CLAMP works... that trade-off IS the coordination problem". True of the *bootstrap* engine it was measured on; **false** of the factored backend the thesis runs. There is no sacrifice to make because there is nothing to rescue. | `FINDINGS_CLAMP_2026_08_30.md` §Finding 4. The coordination problem here is **allocation** — not duplicating effort on the contended surface. |
| `PLAN_2026_08_28.md` | "clamp-only **0.233** vs vary-only **0.589**" | Withdrawn. `mode_at_scale.py` was never committed, it was two of four arms, and the value collides with an unrelated finding. |
| `ROADMAP_AGENT_B_2026_08_28.md` | "Clamp on hub-heavy graphs? **Refuted.**" | Refuted *for this backend*, for a structural reason. Not refuted in general. |
| Any pre-30-Aug scaling claim | "identification collapses as the window grows" | Substantially a **budget-starvation** artefact. The window ladder gave k=30 a budget of 15 and k=20 a budget of 20. See §3.2. |

**The rule this project runs on, and it belongs in the methodology:** every number is quoted
with three things or not at all — **the MI gate, the evidence mode, and the evaluation
policy (argmax or sampled)**. Every wrong claim on this project came from one of the three
being left implicit.

---

## 1. Where everything already is

| document | lines | what it carries |
|---|---|---|
| `BIBLIOGRAPHY.md` | 937 | 19 thematic sections, entries verified with a status field. **The literature review's skeleton already exists here.** |
| `GLOSSARY.md` | 721 | Mechanism reference in 7 parts, plus "questions you are most likely to be asked" |
| `SCORING.md` | 354 | BGe, score equivalence, the regime split, worked example |
| `MA_PROBLEM_STATEMENT.md` | 255 | Formal setup — **see §0** |
| `METRICS.md` | 211 | Every evaluation metric and its traps |
| `ENGINES_AND_FLOW.md` | — | Backends, evidence modes, architectures, baselines, the flow of one experiment |
| `PARAMETERS.md` | 74 | Every parameter with status MEASURED / DERIVED / ASSERTED |
| `THESIS_QUESTIONS.md` | — | Question hierarchy by chapter |
| `FINDINGS_*.md` | — | Individually dated findings; the five from 29–30 Aug are the freshest |

---

## 2. BACKGROUND AND LITERATURE — what must be covered

### 2.1 Causal discovery foundations
- **MAGs and PAGs**; latent projection; bidirected edges as hidden common causes; m-separation.
  *Richardson & Spirtes 2002* (cited in `ma/projection.py`).
- **Inducing paths** — Verma & Pearl's characterisation, which is why ground truth scales:
  u and v are adjacent in the MAG over the observed set exactly when an inducing path exists,
  replacing a search over every conditioning subset (2^(k−2) per pair). *Verma & Pearl 1990*,
  cited three times in `ma/projection.py`.
- **Markov equivalence** — what observation alone cannot settle. Counts used as tests in
  `ma/graphs.py`: labelled DAGs 1, 3, 25, 543, 29281 (OEIS A003024); equivalence classes
  1, 2, 11, 185, 8782 (A007984).
- **Equal-variance identifiability** — *Peters & Bühlmann 2014*. This is why `noise_range`
  spans 3×: a single shared noise scale makes a linear-Gaussian DAG identifiable from
  observation alone and would hand the agents the answer for free. Cited in `ma/scm.py`.
- **Interventional equivalence** — Hauser & Bühlmann: interventions identify by TARGET, not
  by value. This matters for the clamp/vary discussion.

### 2.2 Joint Causal Inference — the closest published framing
`BIBLIOGRAPHY.md` §19. *Mooij, Magliacane & Claassen, JMLR 2020.* The thesis positions
against JCI on **three gaps**: no experiment selection, no per-site variable sets, and
non-adaptive contexts. Also relevant: our one-intervention-per-round protocol **is** JCI's
diagonal design, which is the favourable case — this retracted an earlier objection to
round-robin. See `FINDINGS_TURN_ORDER_2026_08_29.md`.

### 2.3 Federated learning
- **FedAvg** and why weight averaging is meaningful only for a portable architecture.
- **Client drift under non-IID clients** — *Karimireddy et al., SCAFFOLD 2020*. Our clients
  are non-IID by construction: different private blocks, different roles.
- **Server-side adaptivity** — *Reddi et al., Adaptive Federated Optimization, ICLR 2021*
  (FedAdam / FedYogi). **This is now load-bearing** — see §4.4.
- **Federated causal discovery landscape** — `BIBLIOGRAPHY.md` §17. The positioning point:
  in this literature the federation is in learning the **graph**, not in a neural network as
  in classical FL.
- **Differential privacy** — explicitly future work, not implemented.

### 2.4 Reinforcement learning
- PPO; GAE; entropy regularisation.
- **Deep Sets** — *Zaheer et al. 2017*, cited twice in `ma/nets.py`. Justifies mean+max
  pooling over neighbours and is what makes the architecture permutation-equivariant.
- **Potential-based shaping** — *Harada & Russell 1999*, cited twice.
- Multi-agent RL, **including the deliberately excluded set** — `BIBLIOGRAPHY.md` §9.
- **Skeleton recovery is assumed, not learned.** This is a deliberate simplification and
  needs citing to the algorithms that do recover skeletons (PC, FCI, GES). It is not a
  minor convenience: it is *why* rescue does not exist in our setting (§4.3), so it must be
  argued rather than mentioned.

### 2.5 Sequential experimental design
`BIBLIOGRAPHY.md` §14 — DAD (Foster et al.) and RL-BOED (Blau et al.), both read in full
20 Aug 2026, notes in the bibliography.

---

## 3. METHODOLOGY — the setup

### 3.1 The federated structure
- **Vertical partitioning**: every site holds all rows, its own columns.
- `federated_topology(n_agents, private_size, n_shared)` — n agents, `private_size` private
  nodes each, over `n_shared` shared. Window size k = private + shared.
- **The jointly-visible mask** (`ma/topology.py::allowed_edges`) — an edge may exist only
  where *some single agent observes both endpoints*. At the baseline topology this forbids
  **432 of 870 ordered pairs (49.7%)** — every cross-private pair. This is a structural
  assumption, not a nuisance: it is what makes each window a well-posed sub-problem.
- **σ = shared / k**, the contended fraction. The window ladder never varied it (0.50 at four
  rungs, 0.75 at w04), which is why w04 sits off the line.

### 3.2 The (k, σ, n, β) parameterisation — `scripts/sweep.py`
Varying `private_size` and `n_shared` independently confounds two things. The axes that mean
something are k (window size), σ (contended fraction), n (agents), β (budget as a multiple
of the required cover).

**β exists because a fixed budget-per-node confounds budget with window size.** The measured
consequence, and it is a headline correction: the ladder gave **k=30 a budget of 15 and k=20
a budget of 20**. Under β-normalised budgets greedy at k=30 goes from **0.000 to 0.760**. The
old scaling curve was in substantial part a budget-starvation curve.

### 3.3 The forced-cover characterisation
Under oracle evidence the belief is a deterministic function of the intervened SET, and the
required set is **forced** rather than chosen: a directed edge is settled by its TAIL, a
confounded pair needs BOTH endpoints. So the optimum is closed-form at any window size —
measured at **0.757k at k=4 falling to 0.542k at k=30**, i.e. sublinear in k. This is used
*predictively* (to set budgets) rather than fitted after the fact, and it is what makes an
optimal reference arm (`OracleCoverAgent`) exist at any k.

### 3.4 Belief backends — `ENGINES_AND_FLOW.md` §1
`exact`, `constraint`, `version_space`, `attributed`, `factored`. **The thesis runs
`factored`**: one small version space per pair, 4·C(k,2) numbers — 1,740 at k=30 against
~10^19 candidates. What is lost, and it must be stated: joint constraints (ancestrality,
maximality) that couple edges. The consequence is **conservative** — it stays unsure where
enumeration would settle, and never settles wrongly, because each update is individually
sound. Slower convergence bought scale. Also lost: the exact ceiling and exact optimal-rounds
figures become bounds, so "closed X% of the achievable headroom" is available at small k and
not at large k.

**The skeleton is oracle-seeded in BOTH evidence modes** (`reset_marks` reads `self.truth`).
Consequences in §4.3.

### 3.5 Evidence mode — orthogonal to the backend
`oracle` (prunes by true ancestry) vs `sampled` (Fisher-z tests on drawn data). Under oracle
the belief is a function of the intervened set alone; under sampling it is not, which is why
the required cover — and therefore the optimal arm — **does not exist** under sampling.

### 3.6 Policy architecture
`gnn_portable` — one shared network across all agents, per-node pointer head, so the
parameter count is independent of k and n. Permutation-equivariant via mean+max pooling
(Deep Sets). **This is what makes weight averaging meaningful**: every site's parameters
carry the same semantics, so a coordinate-wise mean is a policy rather than a mixture of
incompatible representations. `gnn_solo` is the fully decentralised comparison — one network
per agent, nothing shared.

### 3.7 The generator
Scale-free, preferential attachment, `m=2`, along a random topological order. Acyclicity is
free by construction. **Why scale-free and not Erdős–Rényi, quantified** — under ER every
private node is a weak interchangeable confounder; under SF a private node can be a HUB
parenting many shared variables, projecting to a bidirected CLIQUE in every partner's window.

Measured, and this is the argument for the choice: **no ER setting reaches SF's confounding
level at any density.** SF gives 14.3 bidirected pairs/episode at 53.3 edges and 95%
connected; density-matched ER (p=0.24) gives 11.9 but only 40% connected; connectivity-matched
ER (p=0.40) is 67% denser and gives 8.3. Confounding *falls* as ER densifies. See
`FINDINGS_GRAPH_DISTRIBUTION_2026_08_30.md` §Finding 2.

### 3.8 Parameters — `PARAMETERS.md`
Each carries MEASURED / DERIVED / ASSERTED. **`intervene_scale` was upgraded to MEASURED on
30 Aug** and its justification is not realism: an intervention that reproduces the
observational marginal is **uninformative by construction**, so the scale must sit outside
`noise_range=(0.5, 1.5)`. Detection of the attribution signal is a V-curve in the scale with
its minimum exactly where they match — 22.0% at 1.0 against 92.5% at 2.0. The curve should
be published; "why 2.0?" has no better answer than it.

---

## 4. THE FIVE FINDINGS THAT CHANGE WHAT IS WRITTEN

All measured 29–30 Aug, all settled, none dependent on any pending run.

### 4.1 Credit assignment — 75% of training rows were phantom
Under round-robin only the ACTIVE agent's action is applied, but with
`turn_aware_credit=False` every agent stored a transition every round. Measured at four
agents: **1200 of 1600 rows (75%)** are actions that were **discarded**, carrying reward
produced by another agent's move — mean +0.188 (sd 0.387) against +0.197 (sd 0.391) for real
rows, statistically indistinguishable. The observation has **173 features and none encode
whose turn it is**, so the policy cannot separate them even in principle. At 8 agents it
would be 87.5%.

Turning credit on: pooled entropy **1.224 → 0.598**, MI **0.425 → 0.700**. Every result
predating this is undertrained. It is also a candidate explanation for the agent-count
collapse (a06 MI 0.067, a08 0.035), since the phantom fraction is (n−1)/n.

### 4.2 FedAvg equals pooling per update; the gap is optimiser state
Measured directly on weights, no training: one FedAvg update matches one pooled update to
**0.9971 cosine and 0.99× displacement**. The update rule is not where the gap is. The only
thing FedAvg discards that pooling keeps is optimiser state — local Adam moments rebuilt from
zero every round. This is exactly the FedAdam/FedYogi problem.

### 4.3 "Rescue" does not exist on this backend
A partner clamping a confounder helps identification of the pair it confounds **exactly as
much as varying it: not at all** (oracle 100% both; sampled 61.8% both). Because the skeleton
is oracle-seeded, adjacency is known and a confounded pair is settled by intervening on both
endpoints — a third node's state cannot enter. On an engine that must *learn* the skeleton,
clamping genuinely rescues. **This is the strongest argument for vary-only**, and it makes
the oracle skeleton a limitation that removes a class of coordination rather than merely
making the task easier.

### 4.4 Server-side adaptivity beats data pooling
k=8, 3 seeds, credit on (greedy 0.950, ceiling 1.000):

| arm | success | entropy | MI |
|---|---|---|---|
| pooled | 0.958 | 0.598 | 0.700 |
| fedavg (plain) | 0.922 | 1.307 | 0.365 |
| **fedyogi @ server_lr 0.01** | **0.993** | **0.377** | **0.810** |
| solo (fully decentralised) | 0.930 | 1.030 | 0.481 |

**The federated method beats the centralised one.** Note the pooled path *concatenates raw
trajectories*, which is data pooling and is strictly more centralised than gradient sharing —
so this is not federation-as-compromise. `server_lr` is the whole story: 0.03 unstable, 0.01
right, 0.003 undertrained. **Caveat that must be written:** FedAdam/FedYogi were tuned over
three server rates while pooled and solo ran at defaults. *Pending confirmation at k=12.*

### 4.5 The `confounded` filter distorts small-k cells and not large ones
`episode_mix=confounded` discards **72% of draws at k=4 and 0% at k=30**, and k=4 is **37%
disconnected** against 90–100% elsewhere. So the k axis moves window size *and* how heavily
the training distribution is conditioned. This is a second, independent reason w04 sits off
the line. Limitation, with the k≥8 restriction as the honest mitigation.

---

## 5. NOT YET SETTLED — do not write these as results

- The sweep's own numbers: k, σ, n, β marginals and the σ×n interaction. **Not yet run.**
- Whether FedYogi holds at k=12 (running).
- The sampled-evidence (Rung 3) results.
- Attribution / Rung 1 exact.
- The ER arm.

## 6. Style notes that reflect how this project works

- Prefer "measured" to "shown"; every claim above has a file behind it.
- Where a finding **overturned** an earlier one, write both — the corrections are part of the
  method's credibility, not an embarrassment. Several are already documented that way.
- The MI gate, evidence mode and evaluation policy accompany every number. No exceptions.
