# Disclosure by latent projection — implementation spec

**Status: SPECIFIED, NOT IMPLEMENTED. Scoped 2026-08-23.**

`docs/DISCLOSURE_DESIGN.md` argues *why* this design. This document says *what to build*, in
enough detail that implementation is transcription rather than design. Per the standing rule
(`spec before coding`, 2026-08-21), nothing here gets written until the student signs off.

Supervisor gave the green light on the disclosure category on 2026-08-23 — an existential
confounding claim about shared variables, with the clique-structure leak accepted.

---

## 1. What is being built, in one paragraph

Each agent computes, for every pair of shared nodes, the probability that **its own private
set** contains a common cause of that pair. It broadcasts those numbers every round. Each
receiving agent aggregates the others' claims by noisy-OR and uses the result as a **prior
over confounding assignments** in its existing belief. Nothing else changes: not the subset
dynamic program, not the score, not modularity.

## 2. Decisions taken

Recorded so they are not silently revisited. Both approved by the student, 2026-08-23.

| # | decision | reason |
|---|---|---|
| D1 | Disclosure is **continuous and progressive**, not threshold-gated | A threshold discards information and adds a hyperparameter we cannot afford to tune. "I don't know yet" is representable as `q ≈ base rate`, so no separate unknown state is needed. |
| D2 | Disclosure is **not an action** — it happens every step | It costs no budget, so as an action it is a decision with no trade-off; under full cooperation always-disclose dominates and the policy would burn sample complexity rediscovering that. Keeping it in the environment makes the ablation a config flag rather than a learned behaviour. |
| D3 | The disclosed object is **confounding attributable to the sender's own private set**, not the sender's bidirected edges | The sender's latent projection marginalises out *everything* it cannot see, including the receiver's private nodes. Reporting those would inject a phantom confounder the receiver already observes. |
| D4 | Aggregation is **noisy-OR**, not pooling | Per-agent claims have different subjects (each agent's own private set), so they are logically independent and cannot conflict. See §7. |
| D5 | Only the **bidirected** part is shared. Directed edges among shared nodes are not | Both agents hold data bearing on shared directed edges, so pooling them double-counts. Confounding is safe precisely because the receiver structurally cannot see it. |

**Accepted cost of D1:** the leak becomes time-integrated. A receiver watching the trajectory
of a sender's claim — including how it responds to the receiver's own actions — has a sharper
instrument than any single endpoint. This does not change the privacy *category* (still
existential plus clique structure) but it must be stated in the write-up. Gated disclosure
remains available as a cheap comparison arm if the trajectory leak is challenged.

## 3. The disclosed object

For agent `i` and shared pair `(u, v)`:

    q_i(u, v) = P( exists b in private_i : b -> u AND b -> v | agent i's own data )

Size: `C(|shared|, 2)` floats per agent per round. **Three** at the current topology.
Independent of private-set size, which is the property S_r never had.

Note this is a **joint** query over two edges, not a function of edge marginals —
`P(b->u) * P(b->v)` is not it, and using that would be a silent correctness bug.

## 4. The pipeline, end to end

Steps 2, 3, 4, 5 and 7 are new. Everything else is unchanged.

**1. Actions applied.** `ma/env.py:step()`. Targets resolved, SCM sampled, `self.samples`
extended, `known` and `clean` updated per agent.

**2. Sender computes its own claim.** For each shared pair, a ratio of partition functions
from the existing subset dynamic program:

    q_i(u, v) = Z(b->u and b->v forced) / Z

The forced-parent machinery already exists — the `required` bitmask in
`ma/belief_dp.py:_assignment_weights`. For `|private_i| > 1` the event is a union over
private nodes and needs inclusion–exclusion (§6.3).

**3. Broadcast.** Every agent publishes its `C(|shared|,2)` numbers. No coordinator; each
agent aggregates locally. This is what keeps the design compatible with the no-central-server
constraint.

**4. Receiver aggregates**, over the *other* agents only:

    q_hat(u, v) = 1 - PROD_{i != me} ( 1 - q_i(u, v) )

**5. Prior enters the assignment mixture.** The single insertion point,
`ma/belief_dp.py:joint_conf_marginals`. Today each surviving assignment is weighted by its
partition function alone — a uniform prior over assignments. It becomes:

    log_prior(P) = SUM over shared pairs (u,v):
                       log q_hat(u,v)        if P declares (u,v) confounded
                       log (1 - q_hat(u,v))  otherwise

    mixture weight = log_z(P) + log_prior(P)

**6. Marginals produced.** Unchanged. `_refresh()` calls `edge_marginals`, weights normalise,
`tensordot` combines.

Because the causal claim is `H \ P` (hypothesis graph minus declared confounding pairs),
moving mass onto assignments that name a pair simultaneously *removes* that edge from the
agent's causal graph. The sparsity effect falls out; it is not separately implemented.

**7. Observation extended.** `ma/env.py:observation()` gains the aggregated `q_hat` vector —
`C(|shared|,2)` floats, already in `[0,1]` so no rescaling. Redundant with the marginals in
principle, but it lets the policy react to *being told* directly rather than inferring it from
a shifted posterior.

**8. Policy acts.** Unchanged. Two channels now carry the disclosure: the posterior shift from
step 5 and the raw feature from step 7.

**Ordering.** Round `t`'s disclosure is used when interpreting round `t+1`. This is not a
compromise — an agent can only report what it already knew — and it removes the circularity a
same-round update would create.

## 5. The two traps

Both are silent failures. Both get a regression test (§8).

**TRAP 1 — the sender must compute `q` from its own likelihood only, never from its
disclosure-informed posterior.** Otherwise A's claim feeds B's belief, feeds B's claim, feeds
back to A. That is *data incest* (the term is standard in distributed fusion; see §7), and it
manufactures confidence out of nothing. Computing `q` as a raw likelihood ratio avoids it
structurally, because no prior enters the calculation. **Test: `q_i` must be invariant to what
agent `i` was told.**

**TRAP 2 — add the prior BEFORE the pruning threshold, not after.** `joint_conf_marginals`
drops assignments below `max(log_z) + log(NEGLIGIBLE_WEIGHT)` to skip the expensive marginals
call. An assignment the likelihood alone would discard may be exactly the one disclosure
rescues. Prune on the posterior weight, not the likelihood weight. The existing comment
claiming the threshold is "exact to the precision the arithmetic already has" ceases to be
true the moment a prior is added, and must be updated.

## 6. Open sub-questions

Not blocking, but each needs a decision at implementation time.

**6.1 Tempering.** Raise the disclosed prior to a power `lambda`, bounding how far a wrong
claim can move the posterior. `lambda = 0` recovers no-disclosure and `lambda = 1` full trust,
so the ablation becomes a dial rather than discrete arms. **Default 1.0 until the calibration
measurement (§9) says otherwise.**

**6.2 What to report before any data.** Recommend the topology's measured base rate — 8.8% at
two agents, 16.9% at three (`results/structural_ceiling.json`) — rather than 0.5, so an
uninformative claim is genuinely uninformative rather than a nudge.

**6.3 `|private| > 1`.** `q_i` is a union over private nodes and needs inclusion–exclusion:
three partition calls at two private nodes, seven at three. Required for `T1_2-2-2`, which is
an existing topology. Exact and cheap at these sizes; note the growth and cap it.

## 7. Why noisy-OR and not a pooling rule

Grounded in literature verified 2026-08-23; citations in `docs/BIBLIOGRAPHY.md` §16.

Two objects with opposite evidence-overlap profiles need rules from different families:

| | directed edges among shared nodes | confounding claims |
|---|---|---|
| evidence | overlapping — all agents see it | disjoint — only the owner has it |
| risk | data incest | none |
| right family | conservative fusion / logarithmic pooling | constraint combination |
| our choice | **do not share** (D5) | **noisy-OR** (D4) |

**Constraint combination is the causal-discovery literature's answer** to overlapping variable
sets with latents — Tillman & Spirtes (2011), and Triantafillou & Tsamardinos (2015)'s COmbINE,
which additionally handles interventions. The property that matters is that constraints are
**idempotent**: a fact asserted by two agents is one fact learned twice, not two pieces of
evidence. Posteriors have no such property, which is why pooling them requires overlap
accounting and constraints do not.

COmbINE resolves conflicting constraints by weighting them and solving a maximum-weight
satisfiability problem. **We do not need that machinery.** Each agent's claim is about its own
private set, so the claims have different subjects: "my node confounds `(u,v)`" and "mine does
not" are simultaneously true. There is no conflict to resolve, and the weighting-and-solving
apparatus collapses to the noisy-OR in step 4.

Noisy-OR (Pearl 1988) is the right model because the claims describe **independent sufficient
causes** — different private nodes — rather than competing opinions about one fact. It also
gives the property the student identified independently: one confident voice dominates any
number of quiet ones, because an absent claim is not evidence against.

**And that is exactly the risk.** Noisy-OR is deliberately un-vetoable. One miscalibrated
agent can inject a confounder no number of correct agents can outvote, and exposure grows with
agent count. Full cooperation removes deception; it does not remove honest error.
**Cooperation buys honesty, not accuracy.** This is why §9 runs before the arms.

Ruled out, with reasons: **linear pooling** (Stone 1961) is not externally Bayesian, so
pooling-then-updating differs from updating-then-pooling — disqualifying for a belief updated
every round. **Dempster–Shafer** behaves pathologically under conflicting evidence (Zadeh's
counterexample), and conflict is our interesting case. If we ever do pool, **logarithmic
pooling** is the only defensible choice: Genest (1984) proves it is the *unique* externally
Bayesian operator under unanimity and regularity.

## 8. Test plan

| # | test | why |
|---|---|---|
| T1 | `q_i` from the forced-edge partition ratio matches brute-force enumeration over all DAGs of the window, to 1e-10, at `k=4` | The sender computation is the one genuinely new piece of mathematics |
| T2 | Disclosure disabled reproduces current behaviour **bit-identically** | The arms must differ in exactly one place |
| T3 | `q_i` is invariant to disclosures agent `i` receives | TRAP 1 |
| T4 | An assignment below the likelihood-only prune threshold but above it after the prior survives | TRAP 2 |
| T5 | Noisy-OR degenerates correctly: one partner gives `q_hat == q_other`; all-zero gives `q_hat == 0` | Aggregation arithmetic |
| T6 | Raising `q_hat` on a pair does not decrease posterior mass on assignments naming it | Monotonicity sanity |
| T7 | `q_i` is exactly zero when the sender has no private node parenting both | No phantom claims (D3) |
| T8 | **Metric reachability, per regime** — the identification criterion is earnable under disclosure-on AND disclosure-off | Standing rule: a structurally unearnable metric passed 529 tests once already |

T8 is not optional. `joint_conf_dag_probability` credits an agent only for assignments naming
the true confounded pairs; disclosure should make that *more* reachable, and that must be
demonstrated rather than assumed.

## 9. Measure before building the arms

Two cheap measurements, neither needing training. Both could change the design.

**9.1 Calibration of `q_i` against ground truth, by round.** The whole design rests on senders
being right. If a sender's posterior is sharp and wrong early in an episode, noisy-OR
propagates that error un-vetoably (§7). Load-bearing, and an afternoon's work: replay episodes,
compute `q_i`, compare to `bidirected_pairs` on the true graph, bin by round. If calibration is
poor, §6.1's tempering stops being optional.

**9.2 The interventional ceiling.** `scripts/ma_structural_ceiling.py` measures d-separation on
*observational* data only. Our agents intervene, and clamping severs incoming edges. Working
through the two-intervention case:

- `do(u)` severs `b -> u`; dependence with `v` vanishes. But `v -> u` also vanishes under
  `do(u)`, so one intervention does not discriminate.
- `do(v)`: under `v -> u` the dependence survives; under confounding it vanishes.
- **Both interventions killing the dependence identifies confounding.**

Agents hold authority over shared nodes, so this is available to them unaided at a cost of two
interventions per pair — roughly six against a budget of twenty at the current topology.

If the interventional ceiling is far above the observational 2.3%, **disclosure's value reverts
to saving budget**, which `docs/DISCLOSURE_DESIGN.md` §3 explicitly disclaims. That would make
§3 the part that is wrong, not the finding. Budget is the scarce resource in active discovery
and a protocol that buys it back is a real contribution — but it is a different claim and must
be stated as one.

Extending the existing script to interventional d-separation is a modification, not new
machinery.

## 10. Ablation arms

Run after §9. Each arm differs from the previous in exactly one place.

| arm | disclosure | `PRIVATE_SIGNAL` | what it establishes |
|---|---|---|---|
| `none` | off | off | Floor with no cross-agent information at all |
| `signals` | off | on | **Status quo.** Today's protocol |
| `projection` | on | off | The design |
| `oracle` | ground truth | off | Ceiling — what disclosure is worth if senders are never wrong |

`oracle` is not optional: without it a null result on `projection` is unreadable, because a
flat result cannot be distinguished from a sender-accuracy problem.

Dropping `PRIVATE_SIGNAL` in the `projection` arm is what makes "strictly less disclosure than
the status quo" literally true — it currently announces private-intervention *existence* every
round, which is more leakage than the projection itself. It changes the observation, so it is
part of the arm rather than an assumption.

**Pre-registered predictions**, recorded before any numbers exist:

1. `oracle >= projection >= signals >= none` on identification rate.
2. **The private-clamp rate survives disclosure.** `DISCLOSURE_DESIGN.md` §6.3 argues knowing a
   pair is confounded is not knowing what lies underneath: with both a real edge and confounding
   on one pair, the pair is saturated observationally, so the clamp is what lets an agent
   measure *through* the confounding. If the rate collapses instead, that argument is wrong and
   the altruism result needs restating.
3. `projection` beats `signals` by more on confounded episodes than unconfounded. If it does
   not, the mechanism is not what we think it is.

## 11. Cost

- **Sender:** one extra `log_partition` per (pair × private node) per belief update — 3 at the
  current topology against 25 already made, so roughly +12%.
- **Receiver:** reweighting only. Negligible.
- **Observation:** +`C(|shared|,2)` floats. Three.
- **No change** to the dynamic program, the score, modularity, or the `3^pairs` assignment count.

## 12. Explicitly out of scope

- **The GNN port.** Sequenced *after* this, deliberately: disclosure is per-pair, so its
  natural home in `PerNodeActorCritic` is the edge encoder rather than a bolted-on global
  feature. Building disclosure first freezes the observation and avoids porting twice.
- **Rung 1 (three agents).** `ma/env.py`'s guard refuses any topology hiding more than one node
  from an agent, and that guard is correct. Disclosure does not lift it.
- **Sharing shared directed edges.** D5. This is where all the data-incest risk lives and we
  decline to create the problem.
- **Differential privacy.** Not a federation requirement, and it is structurally incompatible
  with exact inference here — a randomised claim is the non-modular case of `SR_MATH.md` §14.
- **S_r.** Superseded, not deleted. `docs/SR_MATH.md` stays as the fallback and its §14
  modularity argument holds either way.
