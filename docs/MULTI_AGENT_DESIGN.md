# Federated multi-agent active causal discovery — design

Working document, started 2026-08-15, while the single-agent Phase 2 sweep runs. Records
the design discussion so the reasoning is recoverable later, and works out the two pieces
that must exist before any code: **what the agents are measured against**, and **the gate
that says the problem is the problem we think it is**.

Nothing here is implemented. Where a claim is an assumption rather than a measurement, it
says so.

---

## 0. Constraints, from the supervisor

These are boundary conditions, not design choices to be optimised.

**Privacy is about private variables, not about data generally.** Anything concerning an
*exposed* variable may be shared. Anything concerning a *private* variable may not —
including indirectly, so an edge between a private node and an exposed node is itself
private information and cannot be revealed.

**No central server.** Any consolidation, consensus or parameter sharing must be
peer-to-peer. This rules out the obvious "average the models on a coordinator" design.

**No CTDE.** Training must be decentralised too, not merely execution. This rules out a
centralised critic, and therefore rules out the standard MARL answers to credit
assignment (MAPPO, MADDPG and relatives).

**Consequence worth stating plainly.** These three together mean the *only* thing that can
couple the agents is messages that are legal under the privacy rule. Everything the system
achieves beyond independent behaviour has to come through that channel. That is not an
obstacle to the research question — it very nearly *is* the research question. See §5.

---

## 1. What the agents are measured against

The single-agent result means something only because a myopic greedy information-gain
oracle existed to beat. The multi-agent case needs its own reference, and the choice
determines what can be claimed.

### Three reference policies

| reference | information available | coordination | role |
|---|---|---|---|
| **centralised greedy** | the whole graph, all data | perfect, by construction | upper bound |
| **independent greedy** | each agent's local view only | none — zero messages | the floor |
| **random** | — | — | sanity anchor |

*Centralised greedy* is myopic expected information gain computed over the joint graph,
choosing a joint action. It violates every privacy constraint, which is the point: it is
what coordination would be worth if privacy were free.

*Independent greedy* is each agent running the single-agent oracle on its own local belief,
simultaneously, with no communication. This is the honest floor, because it is what you get
for free — it requires no learning, no protocol and no messages.

### The headline metric

```
coordination_gained = (independent − federated) / (independent − centralised)
```

measured on episode cost exactly as `gap_closed` is, with unsolved episodes charged at the
full budget. Read as: **0** means communication bought nothing over acting independently;
**1** means the federated agents recovered everything centralisation would have given.

Two properties make this the right shape:

- It is **honest about the ceiling.** Unlike the single-agent case, where beating the
  myopic oracle was the whole result, federated agents are not expected to beat centralised
  greedy. A metric that treated 1.0 as merely par would misrepresent the achievement.
- It **isolates coordination.** Both endpoints use the same estimator, the same
  interventions and the same budget. The only thing varying is how much the agents can
  work together, which is the quantity under study.

Values above 1 are possible in principle, for the same reason the single-agent agent
exceeded its own oracle: centralised *greedy* is myopic, and sequential planning can beat
it. If that happens it must be reported as beating the myopic centralised reference, never
as beating centralisation.

### Reporting alongside

`coordination_gained` compresses a lot. Report with it:

- absolute episode cost for all three references and the agents, with bootstrap intervals;
- the single-agent-style `gap_closed` per agent against its own local greedy, which
  separates "this agent got better at its own job" from "the agents got better together";
- messages sent per episode (see §5) — a coordination gain bought with unlimited
  communication is a different result from one bought with three bits.

---

## 2. The gates

### GATE-M1 — the task must still require intervening

Direct transplant of single-agent GATE 1, applied to the **consolidated** belief: the
fraction of episodes where the global graph is identified from observational data alone
must match the fraction of joint graphs that are identifiable observationally, computed
exactly from the joint graph space.

Failing high means orientation information is leaking. Failing low means the estimator or
the sample size is too weak. This gate was pinned once at d=3 in the single-agent work and
silently stopped holding from d=5 upward, invalidating a night of runs — hence it is
recorded per run, not checked once.

### GATE-M2 — coordination must be worth something

Direct transplant of GATE 2: **independent greedy must be clearly worse than centralised
greedy, with non-overlapping bootstrap intervals.**

If they tie, no amount of communication can help and there is nothing for the agents to
learn. This is the single most important gate in the document, because a topology where
they tie will still produce plausible-looking training curves and a `coordination_gained`
that hovers around zero for a reason that has nothing to do with the method.

### GATE-M3 — the difficulty must live at the boundary

This one is new, and it exists because of an argument made during the design discussion.

The obvious gate — "can one agent identify the whole graph alone?" — is **vacuous here**.
An agent has a partial view and does not know that some nodes exist at all, so the answer
is trivially no, and a gate whose answer is known in advance measures nothing.

The non-trivial question is *where the remaining ambiguity sits*. If each agent's private
subgraph is separately identifiable and the cross-boundary structure is easy, then
consolidation is decorative: the system would score well while the federated machinery did
nothing.

**Computable target.** Over the enumerated joint graph space, for each graph, determine the
ambiguity that survives after every agent has done everything locally possible. Classify
the residual ambiguity by where it sits:

- entirely within one agent's private subgraph → local problem, no federation needed;
- on **exposed–exposed** edges → the boundary problem proper;
- on **private→exposed** edges → the boundary problem *plus* a privacy conflict, since
  resolving it requires information the privacy rule protects.

The gate: a substantial fraction — a threshold to be fixed **before** measuring, not after
— must fall in the last two categories. If it does not, the topology is wrong and must be
redesigned before anything trains.

The third category deserves emphasis. If most residual ambiguity is on private→exposed
edges, the task may be *unsolvable* under the privacy constraint rather than merely hard.
That would be a finding in itself and is much better discovered by enumeration now than by
a month of training that plateaus for unclear reasons.

---

## 3. Topology

The generative model is the user's "town square": each agent owns a set of **private**
nodes, and there is a set of **exposed** nodes visible to more than one agent. An agent
observes its private nodes and the exposed nodes; it does not know the other agent's
private nodes exist.

Open parameters, all of which GATE-M3 constrains:

- number of private nodes per agent, number of exposed nodes;
- whether private→private edges *across* agents are permitted (probably not — otherwise
  neither agent can ever see them, and they are unidentifiable by anyone);
- whether the exposed set is shared by all agents or pairwise.

Carried over from the earlier scaling brainstorm as still-open: connectivity guarantees,
and a global topological ordering — the latter now looks less like a gap and more like the
key to consolidation (§4).

---

## 4. Consolidation, and the acyclicity trap

Two agents each holding a locally-acyclic belief can consolidate into a graph that
**contains a cycle**. This is not hypothetical: the previous two-agent codebase measured
cycles at roughly 99.5% prevalence in its analytic hypothesis, with a cycle penalty that
turned out to be dead code for the training path.

Naively this seems to demand a central arbiter, which the constraints forbid.

**Candidate resolution — agree on an ordering, not on edges.** A directed graph is acyclic
if and only if it admits a topological ordering. If the agents reach consensus on a
*partial order over the exposed nodes*, then no consolidation of their local beliefs can
produce a cycle through the boundary, because every cross-boundary edge must respect the
agreed order.

Why this is attractive:

- an ordering is a far smaller object than a joint distribution over edges, so it is
  cheap to communicate and plausible to reach consensus on peer-to-peer;
- it concerns **only exposed nodes**, so it is legal under the privacy rule by
  construction;
- it converts consolidation from "merge two distributions" into "agree on an ordering and
  then merge is safe", which is a much better-posed problem.

Open questions, none resolved: how agents reach consensus on an ordering without a
coordinator; whether committing to an ordering early biases the posterior (it is a
constraint on the hypothesis space, so almost certainly yes, and the bias needs
characterising); and whether an order-based prior introduces the well-known non-uniformity
over DAGs that order-modular priors suffer from.

---

## 5. Communication as the object of study

Under the constraints in §0, a purely **local** reward — say, each agent shaped by the
reduction in its own local posterior entropy, potential-based and therefore policy-
invariant per agent — is attractive: fully decentralised, no joint credit assignment, no
centralised critic, and global identification appears only in evaluation.

But it has a hard consequence:

> A purely local objective caps the system at **independent greedy**. The entire
> coordination gap can only be closed by information crossing the boundary.

That is what makes communication the object of study rather than an implementation detail,
and it gives the thesis its sharpest form:

> **What is the minimal boundary-legal message that closes a measurable fraction of the
> independent → centralised gap?**

This is privacy-respecting by construction — the message space is restricted to what the
privacy rule permits — and it produces a *frontier* rather than a binary claim: vary the
communication budget and plot `coordination_gained` against bits exchanged.

Message-space candidates, in increasing order of what they reveal:

1. intended target only ("I will intervene on exposed node 3 this round");
2. intended target plus a scalar confidence;
3. edge marginals restricted to exposed–exposed pairs;
4. a proposed partial ordering over exposed nodes (§4).

Each needs checking against the privacy rule: (3) in particular may leak, since an exposed
node's marginals are shaped by its private parents.

---

## 6. Interventions: hard vs soft

Single-agent, hard interventions were the safe default. Multi-agent introduces failure
modes that do not exist with one actor:

- **Destructive interference.** A hard intervention severs a node from its parents. If
  agent A hard-intervenes on an exposed node that B was relying on to orient an edge, A
  destroys information B needed. This is a real conflict, not merely wasted effort.
- **Mutual blinding.** Simultaneous hard interventions on both endpoints of an edge remove
  the ability to orient it at all.
- **Ill-defined joint actions.** Two agents hard-intervening on the *same* exposed node
  with different values is not a well-defined intervention and needs an arbitration rule —
  which is a coordination mechanism smuggled into the environment.

Soft interventions (shift or scale) avoid all three: they compose additively, preserve the
parent–child structure so no information is destroyed, and keep the interventional
likelihood tractable under simultaneity. The cost is less information per intervention.

This is a *new* argument for soft interventions specific to the multi-agent setting —
distinct from the single-agent question of which is more informative — and it should be
scoped as a design decision rather than inherited.

---

## 7. Credit assignment

Without CTDE there is no centralised critic, so each agent must learn from its own
observations and its own reward. The other agent's learning appears as non-stationarity.

Options, none yet chosen:

- **Local potential-based shaping** (§5) — sidesteps joint credit entirely, at the cost of
  the ceiling described there.
- **Difference rewards computed locally** — an agent estimates its own counterfactual
  contribution from local information. Whether this is possible under the privacy rule is
  an open question.
- **Accept the non-stationarity** and rely on the coordination signal being weak enough
  that independent learners converge. Empirical question; cheap to test first.

---

## 8. Estimator, and why it is coupled to all of this

The single-agent work uses an exact posterior over all enumerated DAGs. This dies twice
over in the federated setting: it does not scale, and it is incoherent for an agent that
does not know how many nodes exist.

**Staging recommendation.** At 2 agents and small `d`, each agent's *local* view is small
enough to enumerate exactly. Keep exact local posteriors for the first federated
experiments so estimator error does not confound federation results, then swap in a
scalable estimator once the federated result is established. Changing two things at once is
what made the previous round unreadable.

**A gate any learned estimator must pass first.** BGe is score-equivalent by construction:
Markov-equivalent DAGs receive identical scores, which is the formal statement that
observational data cannot distinguish them, and it is what makes intervening necessary. A
learned estimator carries no such guarantee. If it breaks score equivalence it will appear
to solve the task while actually leaking orientation information — which is precisely the
defect (`KnownVarianceScore`) that invalidated the previous round of results. **Any
candidate estimator must be tested for score equivalence on observational-only data before
it is used for anything.**

---

## Open questions, collected

1. Threat model: what exactly is protected, from whom, and how is a privacy breach
   *measured* rather than asserted?
2. Does the exposed-node ordering consensus (§4) exist as a decentralised protocol, and
   what does committing to an ordering do to the posterior?
3. Is the task solvable at all when residual ambiguity concentrates on private→exposed
   edges (§2, GATE-M3)?
4. Simultaneous or turn-taking action, and how the interventional likelihood handles joint
   interventions.
5. Does a shared per-node scorer trained on structurally different local neighbourhoods
   (non-IID clients) train stably, and can consensus on weights be reached peer-to-peer?
