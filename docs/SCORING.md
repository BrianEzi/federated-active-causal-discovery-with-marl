# How scoring works, end to end

A reference for the whole scoring stack: what a "score" is, why interventional rows are
handled differently, what the regime split does, and what the agent is finally judged on.
Written to be read top to bottom. References are collected in §11 and are all in
`docs/BIBLIOGRAPHY.md`.

---

## 1. The one-sentence version

A **score** is `P(data | graph)` — a single number saying how well one candidate graph
explains the data collected so far. We compute it for every candidate graph, normalise across
them, and that gives a **posterior over graphs**. Everything else in this document is detail
about how that number is computed and what has to change when someone intervenes.

---

## 2. Why marginal likelihood and not just "fit"

The obvious thing would be to fit each candidate graph to the data and see which fits best.
That fails: a denser graph always fits at least as well, so "best fit" always picks the
densest graph. You then have to bolt on a complexity penalty and argue about its size.

Instead we use the **marginal likelihood**, which integrates the parameters out:

```
P(data | G) = ∫ P(data | G, θ) P(θ | G) dθ
```

A complex graph has more parameters, so its prior probability is spread thinner over
parameter space, and it pays for that automatically. Occam's razor falls out of the
arithmetic instead of being added by hand. This is the standard Bayesian model-selection
argument (MacKay 2003, ch. 28).

**Measured, on this project:** at `d=3` with a profile likelihood (fit, not integrated), the
six densest DAGs tied at the top holding **67% of the posterior mass**, while the true
two-edge graph ranked **9th of 25**. That is what "always picks the densest graph" looks
like in practice, and it is why the rebuild specified BGe from the start.

---

## 3. BGe — the specific score we use

Our data is **linear Gaussian**: each variable is a weighted sum of its parents plus
independent Gaussian noise. The matching marginal likelihood is **BGe** (Bayesian Gaussian
equivalent), from Geiger & Heckerman (2002), with the corrections in Kuipers, Moffa &
Heckerman (2014) — the original paper has an error in the parameter of the local marginal
likelihood, and the addendum is the version everyone actually implements.

BGe places a **Normal–Wishart** prior on the mean and precision of the joint Gaussian. Our
settings, in `sa/score.py`:

- prior mean zero
- `alpha_mu = 1`
- `alpha_w = d + 2`, the smallest value giving a proper prior
- prior scale `t · I` with `t = alpha_mu · (alpha_w − d − 1) / (alpha_mu + 1)`

That last choice is not cosmetic: it makes the prior **marginally consistent across
subsets**, which is precisely the condition that yields score equivalence (§5).

### What it actually needs from the data

Only **sufficient statistics**: the row count `n`, the means, and the centred scatter matrix.
Once you have those, the score of any graph is pure arithmetic — the raw rows are never
touched again. This is why adding 100 new rows is cheap: update the statistics, rescore.

---

## 4. The score decomposes per variable

A DAG makes one claim per variable: *this variable is generated from its parents*. So the
score factorises:

```
log P(data | G) = Σ over variables i:  localScore(i, parents of i in G)
```

and each local term is a difference of two marginal likelihoods over subsets of variables
(`sa/score.py::local_score_from_stats`):

```
localScore(i, Pa) = logMarginal(Pa ∪ {i}) − logMarginal(Pa)
```

which is exactly `log P(variable i | its parents)`.

**Two consequences that the whole codebase depends on.**

**Caching.** A local term depends only on `(node, parent set)`, not on the rest of the graph.
There are `d · 2^(d−1)` distinct terms — 32 at `d=4` — against 543 whole graphs. So we build
a **local score table** once per belief update and every graph is then a sum of table
lookups.

**The subset DP.** Because the score decomposes this way, the sum over *all* DAGs can be
computed without enumerating them, via a sink-based recurrence over subsets (Robinson 1977;
see `sa/dp.py` and `docs/THEORY_NOTES.md`). That is what took us from `d ≤ 6` by enumeration
to `d ≈ 9` exactly. The arithmetic must be done in **signed log space** — inclusion–exclusion
produces negative intermediate terms, and per-node rescaling provably cannot work.

---

## 5. Score equivalence, and why it forces our success criterion

Some graphs are **indistinguishable from data, permanently**. `A → B` and `A ← B` imply
exactly the same set of conditional independences, so no amount of observational data can
separate them. Graphs sharing a **skeleton** and the same **v-structures** form a **Markov
equivalence class** (Verma & Pearl 1990).

BGe is **score-equivalent** (Chickering 2002): members of the same class receive *numerically
identical* scores. Not approximately — identically.

**This is why "did the agent pick the right graph?" is the wrong question.** Within a class,
`argmax` is decided by floating-point tie-breaking, so it measures the order of a loop rather
than the quality of an estimator. The project therefore scores **posterior mass**, not
argmax — and that decision is forced by score equivalence, not a matter of taste.

It is also why **interventions are necessary at all**. Intervening breaks the symmetry that
makes two orientations equivalent, shrinking the class (Hauser & Bühlmann 2012). A graph that
is alone in its class — a **singleton** — is identifiable from observation alone; anything
else is not.

---

## 6. Interventional rows

This is the part that trips people up.

**If you clamp variable X, then X is no longer generated by its parents — you overrode them.**
So the term "how well do X's parents explain X" is scoring a relationship you personally
destroyed. It must be dropped.

**But X's children are still generated normally**, and they are now responding to a value you
chose rather than one the system produced. Their terms still count, and they carry *more*
information than before, because you have broken the confounding between X and everything
upstream of it.

So the rule (Cooper & Yoo 1999) is:

> For rows in which variable `i` was intervened on, omit `localScore(i, ·)`.
> Keep every other variable's term for those rows.

In our code this is the caller's responsibility: `local_score` documents that *"rows where
`node` was intervened on are the caller's responsibility to remove"*, and the `known` mask
carried alongside `samples` is what records which rows those are.

**That asymmetry — drop the target's term, keep the children's — is the entire mechanism by
which acting beats not acting.** Everything the agent learns by intervening enters the score
through this door.

---

## 7. From scores to a posterior

```
P(G | data) ∝ P(data | G) · P(G)
```

The prior `P(G)` is Erdős–Rényi with edge probability `p`, which is **modular** — it factors
into a per-edge term — and that modularity is required for the subset DP to represent it
exactly. `sa/dp.py::for_prior` **refuses** non-modular priors such as scale-free rather than
silently approximating them, because a DP that quietly ignored the reweighting would produce
a posterior under a *different prior* while agreeing closely enough at small `d` to look
correct.

The prior should match the generator. A sparse generator paired with a uniform prior is a
misspecification that shows up as over-confidence in dense graphs and is easy to mistake for
an estimator bug later.

---

## 8. The regime split — scoring under a partner's intervention

Everything above is single-agent. Here is what the federation adds.

### The problem

Agent A sees its own private variable and the shared ones. It **cannot see B's private
variable**. That hidden variable can influence two shared variables at once, making them move
together with no edge between them. A cannot tell that apart from a genuine edge — and no
amount of data fixes it, because the two situations are observationally identical.

Formally, A's window is not a DAG at all. It is a **latent projection** of one, which in
general requires a MAG with bidirected edges to represent (Richardson & Spirtes 2002).

### The fix

B **clamps** its private variable. The pathway switches off. For those rows, A's window
genuinely *is* a DAG.

So A's data splits into two blocks:

- **clean** — collected while every variable hidden from A was clamped
- **dirty** — everything else

These are **physically different systems**: in one the hidden variable is active, in the
other it is frozen. The relationship between two shared variables genuinely differs between
them. Pool them and you fit one number to two different truths, matching neither.

**This is the same idea as §6, applied to an intervention you did not perform.** You cannot
drop a local term for the hidden variable — it is not in your window — so instead the split
is expressed as separate scoring blocks.

A learns which block a batch belongs to from the **regime bit**: one bit per round, naming no
variable and carrying no value.

### The four rules (`ma/score_regimes.py`)

| rule | what it does | verdict |
|---|---|---|
| `pooled` | one dataset, one score; ignores the bit | **fails** — no DAG fits a mixture of two regimes |
| `subset` | clean rows only, when any exist | correct but wasteful; discards thousands of good rows for a few hundred clean ones, creating a valley the learner cannot cross |
| `joint` | same structure, **independent parameters** per regime; add the logs | fixes the gradient, but the dirty regime still prefers a structure that mimics the confounding |
| `joint_conf` | `joint`, plus an explicit confounding set | **what we use** |

### `joint_conf` in detail

A hypothesis is a **pair**: a DAG `H` over the window, **and** a set `S` of shared *pairs*
marked as confounded.

```
score(H, S) = [ clean rows scored against the bare H ]
            + [ dirty rows scored against H plus the confounding edges from S ]
```

Two logs, added. `S` applies to the **dirty** regime only, because that is where the
confounding is switched on. Each confounded pair is realised in the dirty block by adding
that edge to the parent sets, oriented to agree with `H`'s own topological order — so
acyclicity is free rather than something to check.

`S` is then **marginalised out**, leaving a posterior over DAGs.

**Why this is tractable at all:** confounding is **confined to the shared set** — every
bidirected edge has both endpoints among the shared variables. That was proved and verified
exhaustively for two agents (`tests/test_projection.py`), and it means `S` ranges over
subsets of shared *pairs* rather than over all latent structures. At `|X| = 3` that is 8
subsets, so the space is `543 × 8 = 4344` and still exact.

> **This confinement result is a TWO-AGENT result.** With more agents and overlapping shared
> sets it must be re-proved, and it may fail. Everything in the scaling plan rests on it.

**The causal claim is `H \ S`, not `H`.** Under `joint_conf` the dirty regime's fitted edges
include the confounding artefacts, so asking about `H` on a confounded episode gives exactly
0.000, always. This has bitten us before.

---

## 9. What the agent is finally judged on

Two different things, and they must not be confused.

**The reward** is what training optimises. **The reported metric** is what evaluation
publishes. They were once different — a documented flaw — and the reward now uses the same
criterion the report does.

**The criterion — the credit set `[U14]`.** An agent's answer is accepted when:

1. every edge touching its **private** variable is **exactly** right, direction included; and
2. the rest is right **up to Markov equivalence**.

Success is the posterior **mass** on that set clearing a threshold (0.7), for **both** agents,
with their answers unioning into an acyclic graph matching the truth.

Clause 1 is strict because those edges are what only that agent can resolve — it is the
federation's whole point. Clause 2 is lenient because §5 says the alternatives are provably
indistinguishable, so demanding more would be demanding a guess.

**The credit set is an ORACLE quantity.** It is defined relative to the true graph, so an
agent can never compute its own credit-set mass. It is available to the evaluator and to the
reward, never to the policy. (It is computed every step anyway, which makes it *free* to hand
over — and free is not the same as legitimate. See `docs/TURN_BUDGET_SPEC.md` §7.)

### The threshold does not scale, and this is unresolved

Measured 21 August (`scripts/sa_criterion_sweep.py`), observational data only, on graphs that
**are** identifiable without acting:

| d | P(true DAG ≥ 0.7 \| singleton class) |
|---|---|
| 3 | 0.936 |
| 4 | 0.724 |
| 5 | **0.400** |
| 6 | **0.400** |

The true graph is still the single best hypothesis 96.7–100% of the time — it is the *mass*
that thins as the posterior spreads over super-exponentially many graphs. At `n_obs = 1000`
the environment is **starved** at `d ≥ 5`: no mass-based criterion is earnable there. The
proposed response is to split GATE 1 into a leak check and a power check, and to raise
`n_obs` with `d`; see `docs/SA_EXPERIMENT_LOG.md`.

---

## 10. Worked example

Window of 4 variables: `Z` (A's private) and `X1, X2, X3` (shared). B's private variable `W`
is invisible to A and points at both `X1` and `X2`.

1. **Observation only.** `X1` and `X2` are strongly correlated. Graphs with an edge between
   them score well; so does the truth plus a confounding pair. A cannot separate them, and
   the mass splits across both. No amount of further observation changes this.
2. **A clamps `Z`.** `Z`'s own local term is dropped for those rows. `Z`'s children now
   respond to a value A chose, so their terms sharpen and edges incident to `Z` get oriented.
   `X1`–`X2` is untouched.
3. **B clamps `W`.** A is told only that the batch is clean. In those rows the `X1`–`X2`
   correlation **disappears**.
4. **Scoring.** Under `joint_conf`, hypotheses with a real `X1 → X2` edge must explain the
   correlation in *both* blocks — and they cannot, because it is absent from the clean one.
   Hypotheses with `X1 ↔ X2` in `S` explain both: present when `W` is free, absent when it is
   frozen. Mass moves to the truth.

Step 3 is the move that costs B a turn and does nothing for B. That is the cooperation the
whole project is trying to elicit.

---

## 11. References

- **Geiger & Heckerman (2002)**, *Parameter priors for directed acyclic graphical models and
  the characterization of several probability distributions.* Annals of Statistics 30(5). —
  BGe.
- **Kuipers, Moffa & Heckerman (2014)**, *Addendum on the scoring of Gaussian directed acyclic
  graphical models.* Annals of Statistics 42(4). — the correction we implement.
- **Chickering (2002)**, *Optimal structure identification with greedy search.* JMLR 3. —
  score equivalence.
- **Verma & Pearl (1990)**, *Equivalence and synthesis of causal models.* UAI. — Markov
  equivalence via skeleton and v-structures.
- **Cooper & Yoo (1999)**, *Causal discovery from a mixture of experimental and observational
  data.* UAI. — the interventional scoring rule.
- **Hauser & Bühlmann (2012)**, *Characterization and greedy learning of interventional Markov
  equivalence classes of DAGs.* JMLR 13. — how interventions shrink the class.
- **Richardson & Spirtes (2002)**, *Ancestral graph Markov models.* Annals of Statistics 30(4).
  — latent projection and MAGs.
- **Robinson (1977)**, *Counting unlabeled acyclic digraphs.* — the sink recurrence behind the
  subset DP.
- **Peters & Bühlmann (2014)**, *Identifiability of Gaussian structural equation models with
  equal error variances.* Biometrika 101(1). — why per-node noise scales are mandatory.
- **MacKay (2003)**, *Information Theory, Inference, and Learning Algorithms*, ch. 28. — the
  automatic-Occam argument for marginal likelihood.

---

## 12. The confusions this document exists to prevent

- **"Score" is not a fit statistic.** It is `P(data | graph)` with parameters integrated out.
- **The regime split is not a bolt-on.** It is §6's rule applied to an intervention performed
  by someone else, and without it a confounded agent cannot be scored at all.
- **The regime bit is not the same as the signalling channel** added on 21 August. The bit
  says *"your window is a DAG right now"*; the signal says *"I acted in this region"*. The
  signal is additive and removable; the bit is load-bearing.
- **MAP accuracy is meaningless here.** Score equivalence makes class members tie exactly, so
  `argmax` reports floating-point ordering.
- **Under `joint_conf` the causal claim is `H \ S`.** Asking about `H` on a confounded episode
  returns 0.000 every time.
- **The credit set cannot be shown to the agent.** It is defined against the truth.
