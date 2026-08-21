# Decision register

Every load-bearing design decision: what was decided, on what basis, and **what would
overturn it**. The last column is the point — a decision whose refutation condition cannot be
stated is a preference, not a decision.

Status: **firm** (measured or proved) · **provisional** (reasoned, not yet tested) ·
**constraint** (imposed by the supervisor or the problem).

---

## Inference

| decision | basis | status | what would overturn it |
|---|---|---|---|
| BGe marginal likelihood, not a fit statistic | Geiger & Heckerman (2002); Kuipers et al. (2014). Measured: with a profile likelihood at `d=3`, the six densest DAGs held **67%** of the mass and the truth ranked 9th of 25 | **firm** | non-Gaussian mechanisms — BGe assumes linear Gaussian |
| Score posterior **mass**, never `argmax` | score equivalence (Chickering 2002) makes class members tie *exactly*, so `argmax` reports floating-point ordering | **firm** | nothing — this is forced |
| Subset DP over enumeration | Robinson sink recurrence; validated to **1e-10** against the v1 enumeration fixture. Extends `d ≤ 6` to `d ≈ 9` | **firm** | needing `d > 9` per window, where the DP itself stops |
| Signed log space in the DP | inclusion–exclusion produces negative intermediates; per-node rescaling provably cannot work | **firm** | nothing |
| Cooper & Yoo (1999) interventional rule | standard; it is the mechanism by which acting beats not acting | **firm** | soft interventions, which we do not use |
| Per-node noise scales | equal error variances make a linear Gaussian SEM identifiable from observation alone (Peters & Bühlmann 2014) — which would delete the problem | **firm** | nothing |
| ER prior, matching the generator | modularity is required for the DP to represent the prior exactly; `for_prior` **refuses** non-modular priors rather than approximating them | **firm** | wanting scale-free graphs, which would need a different inference route |

## The federation

| decision | basis | status | what would overturn it |
|---|---|---|---|
| No central server | supervisor | **constraint** | — |
| No private-variable information crosses the boundary | supervisor | **constraint** | — |
| Independent learners, no CTDE | supervisor preference; de Witt et al. (2020) show IPPO is a strong baseline | **constraint** | — |
| Regime scoring (`joint_conf`) | measured against three alternatives: `pooled` cannot identify a confounded agent at all; `subset` creates a valley the learner cannot cross | **firm** | — |
| Confounding is confined to the shared set | proved and exhaustively verified — `tests/test_projection.py` | **firm for two agents** | **`n > 2` agents with overlapping shared sets. Not yet re-proved, and it gates the whole scaling plan** |
| Causal claim is `H \ S`, not `H` | under `joint_conf` the dirty regime's fitted edges include confounding artefacts | **firm** | — |
| Credit set `[U14]` as the criterion | clause 1 strict because only that agent can resolve its private edges; clause 2 lenient because the alternatives are provably indistinguishable | **firm** | — |
| The credit set is never shown to the agent | it is defined against the true graph — an oracle quantity | **firm** | — |

## Protocol (21 August)

| decision | basis | status | what would overturn it |
|---|---|---|---|
| Turn-taking, one intervention per round | supervisor. Independently supported by Corah & Michael (2019), whose DSGA penalty term quantifies the cost of deciding several plans at once | **constraint**, now also **firm** | — |
| Budget is a **shared pool of rounds** | internalises free-riding: a round A wastes is one B does not get, and the reward is shared. Equivalent to per-agent under round-robin | **firm** | evidence that free-riding is not a real risk, or that shared budgets distort at `n > 2` |
| `step_cost = 0` | measured: at 0.05 passing was *optimal* (EV −0.255 against 0.000) | **firm** | — |
| No voluntary termination | nothing to escape once the step cost is zero. **Load-bearing with the row above** | **firm** | re-introducing a step cost, which would require re-introducing termination |
| A forfeited round generates observational data | user decision; observation cannot break Markov-equivalence ties, so it cannot re-open the leak. Guarded by the pass-only baseline, measured at **0.007** | **firm** | the pass-only baseline climbing above the observationally-identifiable fraction |
| Three-category action-type broadcast | Grimsman et al. (2018) bound greedy's quality by how much of others' decisions each agent sees | **provisional** | supervisor ruling; removable with `disclose_signals=False` |
| Done bit from own posterior concentration, logged only | must not be the credit-set mass, which is an oracle quantity — and is already computed, so it would be *free* to leak | **firm** | — |
| Clamp-only | a **trade**: costs ≤4pp (paired, 8/10 seeds favour both modes, CI `[-0.005, +0.041]`) for a halved action space and a non-degenerate greedy | **provisional** | more seeds resolving the +1.8pp lean as real |

## Baselines

| decision | basis | status | what would overturn it |
|---|---|---|---|
| Report against random, selfish greedy, sequential greedy and joint greedy | selfish greedy conditions on nothing and is a weak opponent; SGA is the literature's decentralised version (Fisher et al. 1978) | **provisional** — sequential and joint greedy are **not yet implemented** | — |
| Do not claim the submodularity **bounds** | the adaptive case needs adaptive submodularity (Golovin & Krause 2011), which expected information gain does not satisfy | **firm** | a proof that our objective is adaptive submodular |
| Greedy is myopic *by design* | that is precisely what makes beating it possible and worth attempting | **firm** | — |

## Positioning

| decision | basis | status |
|---|---|---|
| The contribution is **the federation**, not the method | RL for sequential design is Blau et al. (2022) | **firm** |
| DAD does not apply to our problem | it assumes a continuous design space and a differentiable likelihood; ours is discrete | **firm** |
| Point of departure from FED-CD | it requires a central server, every client observes every variable, and it assumes causal sufficiency — so latent confounding never arises | **firm**, read from the method |
| Cite DAD for permutation invariance | the argument behind our per-node scorer is theirs, not ours | **firm** |
| The percolation framing is **ours** | Chevalley et al. give the `Θ(1/d)` regime; noticing it coincides with `p_c = 1/d` is our observation | **firm** — state as framing, never attribute |

---

## Decisions taken and later reversed

Kept because reversals are the most useful entries in a register like this.

| decision | why reversed |
|---|---|
| Simultaneous actions | student's own suggestion, made without consulting the supervisor. Superseded by turn-taking, but it produced a real finding — greedy agents converge on the same target, and **0 of 74 collisions** had a tie for both agents, so no local convention can separate them |
| Vary as the default mode | rested on a claim we retracted: that a constant intervention cannot identify descendants' dependence. Measured false — clamp recovers 93–98% |
| `all` hidden nodes clamped defines a clean round | unreachable at more than one private node per agent: an agent gets one action and has no authority over its partner's private nodes, so the regime machinery would be silently dead |
| Episode ends when everyone passes | under turn-taking the inactive agent's pass is *forced*, so one agent could end an episode alone. Collapsed 5/10 seeds |
| Topology T3 | removes latent confounding by deleting the boundary the whole design depends on |
