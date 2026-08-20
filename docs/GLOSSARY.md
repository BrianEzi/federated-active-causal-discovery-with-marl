# Glossary and mechanism reference

Every technical term and every method in this project, with how it actually works. Written
to be read before a supervision meeting and skimmed during one.

**How to use it.** Part 1 is the problem. Parts 2–3 are the inference machinery (the part
most likely to be probed, because it is where the real mathematics is). Part 4 is the
federation layer, Part 5 the reinforcement learning, Part 6 measurement. Part 7 is a list of
the questions most likely to be asked, with the short answer to each.

Notation: `d` = number of variables, `n` = number of samples, `G` = a DAG, `V` = the
variable set.

---

## Part 1 — The problem

### Causal discovery
Recovering the causal graph — who causes whom — from data. Not correlation: the claim
"X causes Y" means that *intervening* on X changes Y, whereas "X correlates with Y" does not.

### DAG (directed acyclic graph)
The object we are trying to recover. Nodes are variables, an arrow `i → j` means "i is a
direct cause of j", and no directed cycle exists. Represented as a `d × d` binary adjacency
matrix with `adjacency[i, j] = 1` for `i → j`.

- **Parents** of j: everything with an arrow into j. j's value is a function of these.
- **Children / descendants** of i: what i can reach by following arrows. Descendants are
  what an intervention on i propagates to, which is why they matter so much here.
- **Sink**: a node with no children. **Source**: a node with no parents. Both are used as
  decomposition handles by the algorithms in Part 2.

### Structural causal model (SCM)
The data-generating object. Each node has a *structural equation* saying how its value is
produced from its parents plus its own noise. Ours is **linear Gaussian**:

```
X_j  =  sum over parents i of  w_ij · X_i  +  epsilon_j ,     epsilon_j ~ N(0, sigma_j^2)
```

To draw a sample: visit nodes in topological order (parents before children), draw each
node's noise, compute its value. Edge weights `w_ij` are drawn per episode with magnitude in
[0.5, 2.0] and random sign — kept away from zero deliberately, because an edge with a
near-zero coefficient is *present in the graph but absent from the data*, which would make
the ground-truth label wrong rather than the task hard.

### Per-node noise scale — and why it is not a detail
Each node draws its own `sigma_j`, fresh each episode, from [0.5, 1.5].

This is the fix for the defect that invalidated an earlier round of results. If **all nodes
share one noise scale**, a linear Gaussian model becomes fully identifiable from
observational data alone (Peters & Bühlmann 2014) — the equivalence class collapses to a
single graph and **interventions stop being necessary**. The earlier codebase used one
scalar for all nodes and, as a direct consequence, roughly half its episodes were already
solved before the agent acted, some with the agent doing nothing at all. Drawing per-node
scales restores the intended regime: observational data pins down the equivalence class and
nothing more.

*If asked "why not equal variance, it's simpler?" — because it deletes the problem.*

### Intervention, `do(X_i = v)`
A **hard** intervention replaces node i's structural equation outright: i stops depending on
its parents and takes the assigned value. Effects propagate to i's descendants and to
nothing else. That asymmetry — parents unaffected, descendants shifted — is what makes an
intervention informative about *orientation*, which observation alone cannot give.

Two modes, and the distinction carries the whole two-agent story:

| mode | what it does | what it is for |
|---|---|---|
| **VARY** | sets i to a fresh random draw each sample, `N(0, sigma^2)` | orientation: a *varying* cause produces detectable variation in descendants |
| **CLAMP** | holds i at one constant for the whole batch | de-confounding: a constant transmits no variance, so it stops being a common cause |

Measured, and it is why both modes exist: a *randomised* `do()` does **not** cut confounding.
A randomly-varying confounder is still an invisible variance source — you have swapped one
latent common cause for another. Rescue rate 0.000 at intervention scale 2.0 and 1.0.
Clamping is what removes the pathway.

### Markov equivalence class (MEC)
The central obstacle. Several different DAGs can imply *exactly the same* set of conditional
independences, so no amount of observational data can tell them apart. Such DAGs form an
equivalence class.

Two DAGs are Markov equivalent iff they share (Verma & Pearl 1990):

1. the **skeleton** — the undirected edges, i.e. which pairs are adjacent at all; and
2. the **v-structures** — patterns `a → c ← b` where a and b are *not* adjacent. Also called
   colliders. These are detectable observationally, which is why they survive as a
   constraint.

Our `mec_signature(G)` returns exactly the pair `(skeleton, v-structures)`, and two graphs
are in the same class iff their signatures match.

Example at `d=3`: `A→B←C` is a v-structure and is **alone** in its class — observation alone
identifies it. The three non-collider orientations of the same skeleton (`A→B→C`,
`A←B←C`, `A←B→C`) are all equivalent and form one class of size 3.

- **CPDAG** — the standard summary of a class: edges that all members agree on are directed,
  edges they disagree on are left undirected. We work with signatures rather than CPDAGs
  because they are cheaper to compare, but the content is identical.
- **Singleton class** — a class with one member. The fraction of singleton classes is the
  ceiling on observational-only identification, and it is *computed* from the graph space,
  not fitted. That is what GATE 1 tests against.

Counts we rely on: `d=3` → 25 DAGs, 11 classes. `d=4` → 543 DAGs, 185 classes. `d=5` →
29,281 DAGs, 8,782 classes. `d=7` → 1.14 billion DAGs.

---

## Part 2 — Scoring a graph

### Structure score
A number saying how well a candidate graph explains the data. Ours is Bayesian: the score is
the **marginal likelihood** `P(data | G)`, with the edge weights and noise scales
*integrated out* rather than fitted.

### Why not just fit the parameters? (the profile-likelihood trap)
The earlier codebase fitted each node's regression by ordinary least squares, plugged the
fit back in, and evaluated. That is valid only if all candidate graphs have the same number
of edges — which was true there (all 8 candidates had exactly 3 edges) and is false the
moment all DAGs are allowed. **A graph with more parents always fits better, because it has
more free parameters.** Measured at `d=3` over all 25 DAGs: the six densest graphs tied at
the top holding **67% of the posterior mass**, while the true 2-edge graph ranked **9th**.
An unpenalised score does not blur the answer — it inverts it.

### BGe (Bayesian Gaussian equivalent) — the default score
The marginal likelihood of a linear Gaussian DAG under a Normal–Wishart prior on the
parameters (Geiger & Heckerman 2002; corrected formulation Kuipers, Moffa & Heckerman 2014).

Mechanically: for each node, take that node together with its parents, and evaluate a
closed-form expression in the sample count, the column means, and the centred scatter matrix.
The integral over parameters has an analytic solution, so no fitting happens at all.
Complexity is handled *by the mathematics* — a richer parent set has to average its
likelihood over a larger parameter space, which penalises it automatically — rather than by
a penalty term bolted on afterwards.

**Sufficient statistics.** BGe touches the data only through `(n, column means, centred
scatter matrix)`. Two consequences we exploit heavily:
- Statistics for a *subset* of columns are plain submatrices of the full ones. So the entire
  table of local scores is built from **one `O(n·d²)` pass**, instead of re-reading all n
  rows once per (node, parent-set) pair — which would be 160 re-reads at `d=5`.
- The score becomes independent of `n` after that pass, which is where a ~2× end-to-end
  speed-up came from.

### BIC — the check
`maximised log-likelihood − (k/2)·log n` (Schwarz 1978), an asymptotic approximation to the
above. Kept not because we use it but because it is simple enough to verify by inspection,
which makes it an independent check on the BGe implementation.

### Score equivalence — the property that makes the task real
Both scores are **score-equivalent** on observational data: every DAG in a Markov
equivalence class receives an *identical* score.

This is not a nicety. It is the formal statement that observational data cannot distinguish
within a class (Chickering 2002), and it is what makes interventions necessary. **A scorer
that violates score equivalence is leaking information** — it is silently ranking graphs the
data cannot rank. We test it directly: members of a class must score identically to
floating-point tolerance.

### Cooper & Yoo (1999) — how interventional data enters
A sample in which node j was itself set by intervention tells you nothing about **j's own
parents**, because the intervention replaced j's structural equation. So:

- that sample is **dropped** from j's local score term;
- that sample is **kept** as a parent value in every *other* node's term, because j's
  realised value is still a perfectly valid cause of its children.

This is precisely why score equivalence holds only on observational data. Once different
nodes are scored on different sample subsets, the equivalence classes **separate** — which is
the entire point of intervening.

### Decomposability / modularity
The score of a DAG is a **sum over nodes of a local term** depending only on that node and
its parent set:

```
log P(data | G)  =  sum over nodes j of  local(j, parents_G(j))
```

Equivalently `P(G) ∝ product of per-node factors` — a **modular** distribution. Everything in
Part 3 depends on this. It means we can build a table indexed by `(node, parent set)` —
`d · 2^(d−1)` entries, so 32 at `d=4` rather than 543×4 — and then any DAG's score is a
lookup and a sum.

---

## Part 3 — Computing the posterior

### Posterior over graphs
`P(G | data) ∝ P(data | G) · P(G)`. The prior `P(G)` is modular too (an independent
probability per edge), so the product stays modular. The normaliser `Z = sum over all DAGs`
is the hard part.

### Method A — exact enumeration
List every DAG, score each, normalise. Correct and simple. Used at `d ≤ 4` (543 graphs) and
for the two-agent windows (543 graphs at window size 4).

**It is a wall, not a slope.** 543 at `d=4`, 29,281 at `d=5`, 3.78 million at `d=6`, 1.14
billion at `d=7`. Every quantity computed by sweeping the graph list ceases to exist one node
past the current setting.

### Method B — the subset dynamic program (how we get past the wall)
Computes the *same exact posterior* without ever listing a DAG. Cost `O(3^d)` instead of
`O(#DAGs)` — which moves the reachable size from about 6 variables to about 9.

**The idea (Robinson's sink recurrence).** Every DAG has at least one **sink** — a node with
no children. So you can decompose any DAG by asking which nodes are sinks, and recursing on
what remains. The catch is double-counting: a DAG with three sinks gets counted once for each
non-empty subset of them. Inclusion–exclusion fixes that exactly:

```
f(A) = sum over non-empty S ⊆ A of  (−1)^(|S|+1) · f(A \ S) · product over i in S of alpha_i(A \ S)
```

where `alpha_i(B)` is the total weight of all parent sets for node i drawn from B, and
`f(V) = Z`. See Koivisto & Sood (2004) for the modern treatment.

- **`alpha` / the zeta transform.** `alpha_i(B)` sums a node's local weights over every
  subset of B. Computing it for all B at once is a **zeta (Möbius) transform** — a standard
  subset-sum-over-subsets done in `O(2^d · d)` rather than `O(3^d)` naively.
- **Backward pass.** A second sweep gives edge marginals and the probability of any specific
  DAG, not just `Z`.

### Signed log space — and the war story
The recurrence runs in **signed log space**: every quantity is stored as `(log magnitude,
sign)`, and additions use log-sum-exp / log-diff-exp. This is not defensive programming.

A first version ran in ordinary doubles, rescaling each node's weights by that node's own
maximum. It verified perfectly against enumeration at `d = 3, 4, 5, 6` — on data drawn from
independent normals. **On the first contact with real environment data it returned `Z = 0` at
`d = 4`.**

The reason is structural, not a rounding accident. Rescaling can only be done per node,
because that is the only thing that factorises. But the sum of per-node maxima is the score
of a configuration in which *every* node takes its unconstrained best parent set — and those
choices are jointly **cyclic**. No DAG attains it. The shortfall is the total information the
nodes share, and it grows with `d` and `n`:

| gap (nats) | n=1000 | n=5000 | n=20000 |
|---|---|---|---|
| d=4 | 834 | 4,612 | 18,233 |
| d=5 | 1,821 | 8,888 | 35,999 |
| d=6 | 3,892 | 19,404 | 78,306 |

A double underflows past 745. So the plain-arithmetic version **could not have worked at any
size actually used** — it looked correct only because independent columns make the gap
vanish, and independent columns are exactly what a causal discovery environment does not
produce.

*The recorded lesson: verifying against ground truth is not enough if the test data is
unrepresentative. The acceptance test was right; the test data was not.*

**Log-sum-exp** — how you add numbers stored as logarithms without leaving log space:
`log(e^a + e^b) = m + log(e^(a−m) + e^(b−m))` with `m = max(a, b)`. The shift is what
prevents overflow. A recurring bug in this project has been applying **one global shift**
instead of a **per-row shift** — it reintroduces exactly the underflow the technique exists
to prevent, and it has been fixed three separate times.

### Edge marginals
`P(edge i → j exists | data)` — a `d × d` matrix, obtained by summing posterior mass over all
DAGs containing that edge, computed via the DP's backward pass. This is the compressed belief
the scalable agent observes, because it is `d²` numbers regardless of how many DAGs exist.
It is **lossy**: it discards the correlations between edges, so the problem becomes partially
observed. Measuring that loss is a deliberate result, not a caveat.

### Sampling DAGs from the posterior
The oracle needs *sample graphs*, not just marginals.

**Structure MCMC (the original, now replaced).** Metropolis–Hastings over graphs: propose a
single-edge change, accept with a probability that makes the chain converge to the posterior.
- *burn-in* — discard the first k steps, before the chain has forgotten where it started
- *thinning* — keep every k-th sample, since consecutive states are correlated
- *acceptance rate* — fraction of proposals accepted; ours sat at **5.8% regardless of
  settings**, the signature of a bad proposal. Single-edge moves are simply wrong for a
  posterior whose effective support is ~172 graphs: almost every neighbour is far worse.
- *mixing* — how fast the chain covers the distribution. Ours mixed badly, and the damage
  was measured: **0.1116 nats** of information lost against an exact reference.

**Exact sampling by source-layer decomposition (the replacement).** Talvitie, Vuoksenmaa &
Koivisto, UAI 2019. Requires the distribution to be modular — which ours already is.

Decompose a DAG by **source layers** instead of sinks:
```
L1 = nodes with no parents
Li = nodes whose parents all lie in L1..L(i−1), with at least one in L(i−1)
```
Every DAG has **exactly one** such decomposition, so the layers partition the DAG space with
no double counting and therefore **no inclusion–exclusion**. The weight of appending layer M
after a placed set U whose last layer was L is

```
product over v in M of [ alpha_v(U) − alpha_v(U \ L) ]
```

The subtraction enforces "at least one parent in the previous layer". Because `alpha` is
monotone in its set argument, this difference is **non-negative by construction** — and that
is the entire point. Robinson's sink recurrence is an *alternating* sum whose terms can be
negative, and you cannot sample from a negative term. This decomposition trades a slightly
larger state space for terms that are all valid probabilities. Sampling is then a forward
walk: from `(U, L)`, draw the next layer with probability proportional to its term times the
value of the remaining subproblem.

**Measured payoff:** information lost against an exact reference, **0.0018 nats vs 0.1116** —
a factor of **62**. Doubling the draws changes nothing (0.0021 at 2000 draws), which is the
check that the residual is not a sample-size problem. Its partition function agrees with the
independently derived subset DP to **1e-13**.

**Partition MCMC** (Kuipers & Moffa 2017) is also implemented — MCMC over *node partitions*
rather than graphs, which mixes far better in principle. Ours is **broken** (error ~0.5,
unimproved by 200× burn-in) and unused. Recorded as a negative result, not hidden.

---

## Part 4 — The federation layer (the two-agent case)

### Topology `(1, 1, 3)`
Five variables, one system, two agents.
- Agent **A** owns 1 **private** variable; agent **B** owns 1 private variable.
- 3 variables are **shared** (also called *exposed*): both agents see them.
- Each agent's **window** = its own private variable + all shared ones = 4 variables.
- An agent **never** observes its partner's private variable. There is **no central server**.

`(1,1,3)` rather than `(1,1,2)` because at `(1,1,2)` the confounding rate is only 6.3% (13 of
207 graphs, always the same pair) — too rare to learn from.

### Latent projection, MAG, bidirected edge
What an agent can learn about its own window, given infinite observational data, is **not a
DAG**. Marginalising out the partner's private variable leaves a **MAG** (maximal ancestral
graph, Richardson & Spirtes 2002), in which a **bidirected edge `u ↔ v`** means "u and v have
an unobserved common cause", with no claim about what or where.

Constructed by the textbook definitions: u and v are adjacent in the MAG iff **no subset of
the remaining observed variables d-separates them** (equivalently, an inducing path exists);
oriented `u → v` if u is an ancestor of v in the true DAG, `u ↔ v` if neither is an ancestor
of the other. We build it by brute force over separating subsets — correct by construction,
and it is a *verification tool*, not an inference engine.

- **d-separation** — the graphical criterion for conditional independence. Two nodes are
  d-separated by a set S if every path between them is blocked by S (chains and forks are
  blocked by conditioning on the middle node; colliders are blocked *unless* you condition on
  the collider or its descendant).

### Confounding is confined to the shared set — the structural result
**Proved and then verified exhaustively over the graph space:** a bidirected edge can *never*
touch an agent's private node; confounding only ever appears between two **shared** variables.

This is what makes the whole belief representation tractable. It means an agent's belief is
"a DAG over my window, plus one flag per shared *pair*". The DAG part stays decomposable, so
the subset DP carries over untouched. Had it been false, the belief would have needed full
MAG machinery and the score would have stopped decomposing.

*(It also corrected the design document's confounding rates, which overcounted by roughly 3×.)*

### `joint_conf` — the belief the agents actually use
A hypothesis is a **pair**:

```
(  DAG H over the window  ,  set P of shared pairs declared confounded  )
```
with P's edges required to be present in H. The **causal claim is `H` minus `P`** — never `H`
itself. This distinction is the single most expensive thing in the project (see Part 7).

- Scoring: the **clean** regime scores `H \ P`, the **dirty** regime scores `H`. For a fixed
  P the score is modular, so it is one DP pass. Total `3^(number of shared pairs)` passes.
- At `|X| = 3` there are 3 shared pairs, so `3^3 = 27` assignments (each pair: absent,
  confounded one way, confounded the other). **2 of the 27 admit no acyclic completion** and
  would make the partition function zero, so they are excluded: **25** live assignments.
- **Why three states per pair, not two.** The previous code oriented each confounding edge
  "along the DAG's topological order" — but a DAG has many topological orders, and the code
  picked one by an arbitrary tie-break (lowest index). For two shared variables *incomparable*
  in the DAG, the orientation was therefore decided by **node numbering**, and the two
  orientations score differently under BGe. So the hypothesis being evaluated depended on an
  implementation detail. Making orientation part of the hypothesis and **marginalising it
  out** removes the arbitrary choice. Bonus: acyclicity comes free, because the DP only ever
  emits DAGs.

### The regime bit
One bit per round of disclosure: *"I clamped something you cannot see."* It does not name the
variable, so no private information crosses the boundary.

Its purpose: it partitions the agent's rows into **clean** (collected while the partner was
clamping, so the confounding pathway was off) and **dirty** (confounding active). Without the
partition the agent cannot tell a genuine shared edge from confounding through the partner.

Measured impact, structural: pooled rows → 0.000 identification; regime-separated → 0.162;
regime plus the agent's own interventions inside the clean regime → **1.000**.

Measured impact, 2026-08-20: the bit moves the **belief**, not just the policy. The no-bit
arm's own *random* floor is ~0.03 against ~0.19 with the bit — same policy class, same
budget. So most of its effect happens upstream of any learning.

*Note: an alternative disclosure — the `|X|²` ancestral-order bits proposed in the design
document — was measured at ~0.005 bits each. A correctness guard, not an enabler.*

### Why clamping is the coordination problem
Clamping *your own* private variable is what lets your **partner** distinguish confounding
from a real edge. It does nothing for you. And it costs you: a variable held constant teaches
you nothing about what drives it, and those boundary edges are part of *your own* success
criterion.

Measured: an agent that clamps every round never learns its own private node's parents —
agent A rose 0.368 → 0.814 while B, clamping every round, stayed at 0.04 and never identified.
**Pure altruism is strictly dominated.** So the problem is one of **timing**, not willingness:
the agents must discover a *mix*.

---

## Part 5 — The learning

### PPO (Proximal Policy Optimization)
The policy-gradient algorithm we train with. Ingredients:

- **Actor** — a network mapping observation → a probability distribution over actions.
- **Critic** — a network estimating the expected future return from a state; used to reduce
  the variance of the gradient.
- **Advantage** — "how much better was this action than the critic expected?" Positive
  advantage pushes the action's probability up.
- **GAE (generalised advantage estimation)** — a weighted blend of short- and long-horizon
  advantage estimates, trading bias against variance (`gae_lambda = 0.95`).
- **Clipping** — PPO's defining trick. The update is bounded so a single batch cannot move
  the policy too far: the objective is clipped once the new/old action-probability ratio
  leaves `[1−ε, 1+ε]` (`ε = 0.2`). This is what makes it stable enough to run unattended.
- **Entropy bonus** — a small reward for keeping the action distribution spread out
  (`entropy_coef = 0.01`), delaying premature commitment. Final entropy is also our best
  **collapse diagnostic**: ~0.02 means the policy has committed to one action, ~1.9 means it
  is still exploring.
- **Value loss** — the critic's own regression loss, weighted at 0.5.

Batching: 16 episodes per policy update, 2000 training episodes → 125 updates.

### IPPO (independent PPO), and the CTDE exclusion
Each agent has **its own** actor, critic, and optimiser, and sees **only its own
observation**. Nothing is shared but the scalar reward.

- **CTDE** = *centralised training, decentralised execution* — the standard cooperative-MARL
  design in which a shared critic sees the global state during training. **Excluded by supervisor
  constraint.** QMIX, VDN, MAPPO, MADDPG are all CTDE and therefore all out of scope. They
  are worth citing precisely *as* the scoped-out set, or a reader will ask why the obvious
  algorithms are absent.
- **The shared scalar reward is not CTDE**: no observations, parameters, or gradients cross
  the boundary. It is necessary — a selfish agent has no reason to clamp for its partner, so
  a *per-agent* reward would make the target behaviour strictly dominated.
- **Non-stationarity** — the cost of independence. From A's point of view the environment
  includes B, and B is changing, so A is chasing a moving target. This is the standard
  objection to independent learners; de Witt et al. (2020) is the cover for treating IPPO as
  a strong baseline rather than a compromise.

### Potential-based reward shaping (Ng, Harada & Russell 1999)
Adding `gamma·Phi(s') − Phi(s)` to the reward, for any function `Phi` of state, **provably
leaves the optimal policy unchanged**. Only the gradient becomes more informative. We use
`Phi(s) = −H(belief)`, which makes the shaping term the *realised information gain per step*.

Elegant consequence: under that shaping, greedy EIG becomes the **myopic optimum of the
agent's own reward**, so "beats greedy" sharpens into "beats the one-step optimum of its own
objective".

### Permutation-equivariant per-node scorer
The architecture that unlocked the single-agent result. A flat MLP over the belief has to
learn the same fact separately for every node index. The per-node scorer instead computes a
representation for each node from its own edge features, **pools over neighbours** (so node
order cannot leak in), and scores each node with shared weights.

Result: **relabel the variables and the logits permute the same way.** The probe that
motivated it scored 0.814 for per-node against 0.528 for flat — the architecture, not the
RL, was the bottleneck. (An earlier version used a fixed-order neighbour vector and was only
*invariant*, not equivariant; a test caught it.)

### Action space
Per round, each agent picks one `(variable, mode)` pair, or **passes**. `mode ∈ {vary,
clamp}`. Budget is the number of interventions an agent may spend per episode.

---

## Part 6 — Measurement

### The success criterion, `[U14]` — all three parts must hold
1. **Private** — each agent's posterior recovers every edge *touching a private variable*
   exactly, orientation included. Interventions on private nodes make this identifiable.
   ("Private-incident", so boundary edges count: 3 edges per agent at `(1,1,3)`.)
2. **Shared** — the rest need only be recovered to **Markov equivalence**. Demanding
   orientation within a class would be measuring floating-point tie-breaks, not estimation.
3. **Global** — the **union** of the two agents' answers must be **acyclic** and Markov
   equivalent to the true global graph.

The **credit set** is the set of DAGs satisfying (1) and (2); success requires ≥ **0.7** of
posterior mass on it, plus (3).

**Why the acyclicity check is not redundant.** For *full* DAG recovery it would be — two
correct DAGs union to the truth. But criterion 2 *relaxes* the shared part to equivalence,
and two agents can orient a shared edge differently within the same class and union into a
**cycle**. The check exists precisely because of the relaxation. Disagreements are resolved by
OR, which is the permissive choice: it can create cycles but never hide them, so the check
sees the worst case.

**Why posterior mass on a set, not the MAP graph.** Equivalence-class members score
*identically*, so `argmax` is decided by floating-point ordering. About 10% of posterior mass
can sit on a wrong skeleton while every edge marginal looks fine, so a marginal-based
criterion would pass on graphs that are jointly wrong. Everything is read as mass over
**sets of DAGs**.

### `gap_closed` — the single-agent headline metric
```
gap_closed = (agent − random) / (greedy oracle − random)
```
`0` = matched random. `1` = matched the myopic oracle. `> 1` = **beat** it. Normalising by the
oracle-minus-random gap makes cells with different difficulty comparable.

### Bootstrap confidence interval
Resample the episode outcomes with replacement 2000 times, recompute the mean each time, take
the 2.5th and 97.5th percentiles. Assumes nothing about the distribution. **Every reported
number carries one**, and "beats X" always means *non-overlapping intervals*, never a better
point estimate.

### The EIG oracle — the opponent
The baseline the agent must beat: at every step it picks the single intervention with the
highest **expected information gain**, using the exact posterior.

*How it works.* Under `do(X_i)`, exactly i's **descendants** shift. So two hypotheses are
distinguishable by that intervention precisely when their descendant sets from i **differ**.
Group the hypotheses by descendant set; the experiment's outcome tells you which group you
are in. The value of intervening on i is therefore the **uncertainty of that outcome** — the
Shannon entropy of the posterior-weighted partition.

*Why this is exactly EIG, not a proxy.* The descendant set is a deterministic function of the
graph, so `H(outcome | graph) = 0`, and therefore
```
I(graph ; outcome) = H(outcome) − H(outcome | graph) = H(outcome)
```
Maximising partition entropy **is** maximising expected information gain (Lindley 1956). (An
earlier version used a Gini/Simpson index — the Tsallis-2 analogue — which needed a defence
via generalised uncertainty measures. Shannon removes the approximation for one line.)

*Two honest limitations, stated rather than buried:*
- **Myopic** — the best *next* experiment, not the best *sequence*. The `(1−1/e)` guarantee of
  Golovin & Krause (2011) requires **adaptive submodularity** ("information gets less useful
  the more you already have", formalised), which expected information gain does **not**
  satisfy in general. That gap is deliberate: it is the headroom the learned agent is meant
  to find.
- **Optimistic** — it assumes the experiment reveals the descendant set *perfectly*. With
  finite noisy samples it does not, so it credits distinctions the agent may be unable to
  make.

### The other baselines
| baseline | what it does | what it isolates |
|---|---|---|
| **no-intervention** | never acts | how much is solvable observationally — the GATE 1 control |
| **random** | uniform over valid actions | the floor: does choosing well matter at all? |
| **random_vary** | random, but never clamps | isolates the *value of being able to clamp* |
| **pass** | always passes | the degenerate policy the step cost can make optimal |
| **forced_clamp** | always clamps its private node | the "pure altruism" arm — measured to be dominated |
| **edge-marginal greedy** | EIG on marginals only | the cost of the lossy, scalable belief |

Every agent gets its **own RNG stream** (`_agent_seed(seed, name)`). This is a fix, not a
detail: A and B were built from the same seed, so the random baseline was two *synchronised*
agents colliding on **78%** of rounds against an expected ~19%, which silently inflated every
comparison against it.

### The three gates — each blocks the phase after it
The gates exist because this project once skipped them and had to throw away a whole results
table. Each is a **stopping condition**: a benchmark that cannot discriminate still produces
numbers, and those numbers are worthless.

**GATE 1 — the task must require intervening.** Observational-only identification must equal
the **predicted** singleton-class fraction, which is computed from the graph space rather than
fitted. It is deliberately **two-sided**:
- *above* target = a **leak** — the task is solvable without acting; invalidates everything;
- *below* target = a **power problem** — the posterior cannot reach the 0.7 threshold even
  where the graph is uniquely identifiable; the task is *harder* than intended, not easier.

Two-agent: **passes** (0.0388 against a predicted 0.042). Single-agent at `d=5, 7`: **fails on
the below side** in every cell (0.040 against 0.089) — so policy comparisons stand (all arms
face the same environment), but absolute identification rates do not.

**GATE 2 — choices must matter.** Random must be clearly worse than the oracle, at a *tight*
budget. **Fails at two agents**, and the mechanism is now measured — see Part 7.

**GATE 3 — coordination must be necessary and available.** On confounded episodes, a pair that
*cannot* clamp must do worse than a pair that can. **Passes**: 0.012 vs 0.249, n=169,
headroom **+0.237**. That gap is the coordination value a learned policy competes for. Run at
a *larger* budget than GATE 2 on purpose — the two gates want opposite things from the budget,
and running both at one budget guarantees one is uninformative.

### The budget cliff — why gates run tight
Greedy-versus-random discrimination is **entirely a scarcity effect**. At `d=5`: +0.373 at
budget 2, +0.300 at 3, +0.127 at 5, +0.047 at 8. By budget 8 random already solves 95% of
episodes. **A generous budget produces a benchmark that passes while measuring nothing.**

### Canaries
Automatic checks attached to every run, so a dead run cannot look like a result: policy
entropy vs the uniform ceiling; baseline anchors; informative-sample fraction; seed spread;
GATE 1 status; and an **under-acting** canary (mean steps < 1.5 means the policy collapsed
into passing rather than learning).

### Reachability tests — the newest guard
**A metric can be perfectly well-formed and still be structurally unearnable, and the two look
identical in a results table.** The whole 529-test suite passed while the reported two-agent
metric could not be earned *at all* on confounded episodes, because every test asked whether
the metric computed the right number and none asked whether that number was **reachable**.

Every metric now carries an explicit reachability case per regime it claims to cover: build
the best achievable belief in that regime, and assert the metric can be earned.
`tests/ma/test_metric_reachability.py`.

---

## Part 7 — The questions you are most likely to be asked

**"Why is your posterior exact rather than variational or sampled?"**
Because at these sizes it can be. Exact enumeration to `d=4`, the subset DP to ~`d=9`. It
removes an entire class of confound: when a result moves, it is the policy or the
environment, never the inference. Beyond `d≈9` the DP stops and this becomes a real design
question — that is the honest scaling limit.

**"How do you know the DP is right?"**
Three independent checks. It matches **frozen enumeration to 1e-10** — and the fixture was
generated by the *previous generation of the code*, so the check is not circular. The exact
sampler, derived completely differently (source layers, no inclusion–exclusion), matches the
DP's partition function to **1e-13**. And 534 tests pass.

**"Why did you switch samplers?"**
The Metropolis–Hastings chain had a 5.8% acceptance rate regardless of tuning, because
single-edge proposals are wrong for a posterior supported on ~172 graphs. Measured cost:
**0.1116 nats** lost against an exact reference, versus **0.0018** for exact sampling — 62×.
Doubling the draws changed nothing, which rules out sample size as the explanation.

**"Isn't beating a greedy oracle trivial / impossible?"**
Neither. Greedy EIG is the standard tractable choice and is genuinely strong. But it is
**myopic**, and optimal sequential design is not greedy design chained together: expected
information gain is not adaptive submodular, so the `(1−1/e)` guarantee does not protect it.
That gap is exactly the headroom. Measured: median `gap_closed` clears 1.0 in **10 of 14**
cells, best at `d=5` budget 2 (1.300, all three seeds ≥ 1.2).

**"Why does the oracle fail in the two-agent case?"** *(the sharpest question we have an
answer to)*
Not by collision-as-bad-luck. We tested that: we gave the agents **opposite tie-breaking
conventions** — A takes the lowest-indexed tied action, B the highest, a purely local
convention with nothing crossing the federation boundary. It changed nothing (0.040 vs 0.064,
collisions 0.366 vs 0.372).

The diagnostic says why: a tie-break can only separate agents that **have** a tie, and at the
level of *which variable to target* they almost never do — a target-level tie in **6.8%** of
decisions, and **0 of 74 collisions** involved a tie for both agents. (The 2-element argmax
set present in ~94% of rounds is VARY and CLAMP on the *same* node — the known indifference
between modes, not a choice of target.)

So each agent independently computes a **unique** best target and it is the **same** target,
because the objective is identical and the shared variables are visible to both.
**Myopic design does not fail here by accident — it fails because a one-step information
criterion has no term for what the partner needs.** Greedy is therefore retired as the
two-agent reference; random is the floor.

**"What was the bug you retracted, exactly?"**
A hypothesis is `(H, P)` with P's edges *present in* H, so the causal claim is `H \ P`. The
reported metric compared **H** against the true *causal* graph. On a confounded episode the
truth then matched only under the **empty assignment** — the one hypothesis that refuses to
model the confounding at all. Reported success on confounded episodes was therefore
**exactly 0.000, always** — not low, structurally unreachable — against ~0.59 on unconfounded
ones. Since confounded episodes are the entire point of the design, every headline number was
silently conditioned on the cases that need no coordination. Four measurement bugs in two
days, **three of them this same confusion**, in the reward, the reported metric, and the
identification criterion.

**"Did correcting it flatter your results?"**
It moved GATE 3 the way I did **not** predict. I expected the headroom to shrink; it grew,
from +0.184 to **+0.237**. The old criterion demanded the exact true DAG *together with* the
exact confounding set — strictly harder than the criterion we specified — and it was
penalising the arm that *can* clamp more than the arm that cannot.

**"Is the regime bit doing real work, or is your reward design hiding a failure?"**
Answered by control. Set the step cost to zero, change nothing else: the no-bit arm **stops
collapsing** (8.4 steps vs 0.0, entropy 1.93 vs 0.02) — so the collapse really was the reward
design. But it **still learns nothing**: 0.047 against its own random floor of 0.053. Every
arm in that run sits at 0.06–0.08, greedy-with-exact-posterior included, so the ceiling is a
property of the **belief**, not of any policy in it. The step cost explains the collapse and
does not explain away the bit.

**"Why is the step cost 0.05?"**
It was inherited and never swept — a fair hit. At ~7.7 steps that is a charge of ~0.39
against a success reward of 1, which is enough to make **passing optimal** for a
random-quality policy. It is on the parameter-audit list with `n_int`, `intervene_scale`,
`prior_p`, and `identify_threshold`, all of which are currently **asserted rather than
measured**.

**"Isn't a shared reward really centralised training?"**
No. CTDE means a critic that sees the **global state**. Here each agent has its own actor,
its own critic, its own optimiser, and its own observation; no observation, parameter, or
gradient crosses the boundary. Only a scalar reward is common — and it must be, or clamping
for your partner is strictly dominated.

**"What are the limits you would state yourself?"**
Four. (i) Single-agent GATE 1 fails on the below-target side at `d = 5, 7` — comparisons
stand, absolute rates do not. (ii) `|X|` is the binding scaling axis: the confounding
enumeration is `3^(pairs)`, and the subset DP itself stops around `d ≈ 9`. (iii) Several
parameters are asserted rather than measured. (iv) The two-agent result is 10 seeds — ahead
on 10/10 by point estimate, separated on 8/10 by non-overlapping intervals, and the two that
overlap are the two lowest-acting seeds, so the residual failure mode is **under-acting**
rather than mis-acting.

---

## Citations, and their status

| work | used for | status |
|---|---|---|
| Verma & Pearl (1990) | Markov equivalence = skeleton + v-structures | verified |
| Chickering (2002) | score equivalence | verified |
| Geiger & Heckerman (2002); Kuipers, Moffa & Heckerman (2014) | BGe marginal likelihood | verified |
| Schwarz (1978) | BIC | verified |
| Cooper & Yoo (1999) | interventional scoring rule | verified |
| Robinson; Koivisto & Sood (2004) | sink recurrence / subset DP | verified |
| Talvitie, Vuoksenmaa & Koivisto (UAI 2019) | exact DAG sampling by source layers | **verified 2026-08-20** — note PMLR v115 is dated 2020, so tooling renders it `talvitie20a` while the conference was 2019 |
| Lindley (1956) | expected information gain | verified |
| Golovin & Krause (2011) | adaptive submodularity, the `(1−1/e)` bound | verified |
| Peters & Bühlmann (2014) | equal-variance identifiability | verified |
| Richardson & Spirtes (2002) | MAGs, latent projection | verified |
| Ng, Harada & Russell (1999) | potential-based shaping | verified |
| de Witt et al. (2020) | independent learners as a strong baseline | verified |
| Kuipers & Moffa (2017) | partition MCMC | implemented, **broken**, unused |
| Foster et al. (2021) DAD; Blau et al. (2022) RL-BOED | closest published framings | **UNVERIFIED** — do not cite until checked |

Full notes and the reason each matters: `docs/THEORY_NOTES.md`.
