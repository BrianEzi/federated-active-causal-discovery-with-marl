# Disclosure design: latent projection at the interface

**Drafted overnight 22/23 August 2026. NOT IMPLEMENTED. Needs a decision, and the decision
is partly the supervisor's.**

Proposed by the student on 2026-08-22, in response to the S_r design in `docs/SR_MATH.md`.
This document works it out properly so the choice between them can be made on evidence.

---

## 1. The proposal in one line

Each agent reports **which pairs of SHARED nodes are confounded from its side** — and
nothing else. No variable names, no counts, no per-round clamp announcements.

That object has a name: the **latent projection** of the agent's window onto the shared set,
i.e. a MAG over shared nodes. Which resolves a question raised earlier and answered badly at
the time — MAGs belong at the **interface**, not inside the inference. "What does my hidden
structure induce on the shared margin" is exactly what a latent projection is for.

## 2. Why it is worth taking seriously

**It dissolves S_r rather than solving it.** No named hidden nodes, no per-round clamp
bitmask, no `strip_r`, no modularity worry (`docs/SR_MATH.md` §14). The receiving agent
conditions on a structural claim about nodes it already observes.

**It scales in the one direction S_r could not.** Disclosure is `O(|shared|^2)` bits and is
**independent of private-set size**. Multi-private topologies and `n >= 3` need no new
machinery, where S_r needed a wider signal for the first and more assignments for the second.

**It is strictly less disclosure than the CURRENT protocol.** Today's `PRIVATE_SIGNAL`
already announces "I intervened on something you cannot see", i.e. existence, every round it
happens. Under this design that category disappears.

## 3. What it is measured to be worth

`scripts/ma_structural_ceiling.py`, 2026-08-23. Of agent-windows that ARE confounded, the
fraction where NO latent-free DAG over the observed nodes reproduces the observed
conditional-independence pattern -- the most any observational-only method could detect at
**infinite** data:

    POOLED across topologies and priors:
        12 detectable of 516 confounded windows = 2.3%, CI [1.3%, 4.0%]

Reported pooled deliberately. Per-configuration numerators are 1-3 windows, so the per-row
intervals are enormous and mutually overlapping -- there is no evidence that topology or
prior moves the ceiling, and quoting a range across rows would be reading noise.

**~98% of confounding is structurally invisible to the receiving agent, at any sample size**
(97.7%, CI [96.0%, 98.7%]).

And it does not improve with scale. At rung 1 the confounding RATE roughly doubles --
16.9% of windows against 8.8% at two agents, CIs [15.3, 18.7] against [7.7, 10.0], non
overlapping -- because each agent's hidden set becomes the union of two others' private
nodes. More confounding, no more of it recoverable. The disclosing agent has none of this difficulty: the confounder is not latent from
ITS side, it reads the answer off its own data.

So the value of disclosure is not "it saves budget". It is:

> disclosure closes a gap that no amount of the receiver's data or intervention budget can
> close on its own, and that gap is ~98% of confounded cases.

## 4. What it leaks, stated honestly

Not "fully private". Three things, in increasing subtlety:

1. **Existentially**: reporting a pair confounded reveals that SOMETHING in the reporter's
   private set is a common cause of two shared nodes. Not how many, not which, not their
   structure among themselves.
2. **Negatively**: reporting NO confounded pairs reveals that nothing private touches the
   shared set that way. Weak, but information.
3. **Clique structure, over time** -- the sharpest one. A single private node confounding
   `X1, X2, X3` produces ALL THREE pairs; two separate nodes produce two separate cliques.
   The clique decomposition of reported pairs therefore constrains the NUMBER and ARRANGEMENT
   of the reporter's private variables, without naming any. The student accepted this
   trade-off on 2026-08-22, conditional on the supervisor.

**Not differentially private**, and DP is not a federation requirement -- a common
conflation. Federated learning means data stays local; vanilla FL carries no formal privacy
guarantee, which is why DP-FedAvg exists as an addition. If DP were wanted here the
mechanism would be randomised response on the reported pairs, and that has a specific
consequence: it makes the received claim uncertain, which is the non-modular case of
`SR_MATH.md` §14. **Differential privacy and exact inference are mutually exclusive here**,
structurally rather than by implementation accident.

## 5. The claim to defend in the thesis

> Each agent discloses the latent projection of its window onto the shared set -- never
> values, never variable identities, never counts. This is the **minimal sufficient
> statistic** for the effect of one agent's private structure on the shared margin: any less
> loses conditional-independence information the partner provably cannot recover
> (~98% of cases, measured), any more reveals private structure the partner does not need.

Precise, defensible, and it survives the obvious follow-up ("what does it leak?") because
§4 answers it exactly.

## 6. The three things to work out before implementing

**6.1 Double counting -- the real technical problem.** A and B both observe the shared
columns. If B reports a posterior formed partly from data A also holds, A cannot simply
multiply it in.

The clean resolution: **B is authoritative on this specific question and A is not.**
Confounding of `(X1, X2)` by `b` is a statement about edges `b -> X1` and `b -> X2`, and A
can NEVER see either (that is what §3 measures). So the natural object is not an opinion to
pool but a **restriction of A's hypothesis space**: A conditions on "the augmented graph
contains `X1 <-> X2`" and continues scoring its own data as before. No pooling, no
double-counting, and it fits the existing machinery -- it is exactly a restriction on which
of the `3^pairs` assignments carry prior mass.

Open sub-question: B's own claim is uncertain early in training. Options are (a) report only
above a confidence threshold, (b) report a probability and have A use it as a PRIOR over
assignments rather than as evidence. (b) is more principled and still avoids double counting,
because a prior is not evidence about the shared data.

**6.2 Chicken-and-egg.** B knows `b` confounds `(X1, X2)` only after identifying its own
structure, which takes interventions. So disclosure is PROGRESSIVE, not available at reset.
This is not fatal -- it makes disclosure a resource that improves through an episode -- but
it means the protocol has to carry "I do not know yet" as distinct from "not confounded".

**6.3 What happens to the altruism result.** The current headline is that agents learn to
clamp their own private node for a partner's benefit (82-91% of clamps, against 25% chance,
against greedy's 19-24%). If B can simply TELL A, does that behaviour lose its purpose?

**Argued no, and this needs testing rather than assuming.** Knowing `(X1,X2)` is confounded
is not knowing the relationship underneath it: with both a real edge `X1 -> X2` AND
confounding on the same pair, the pair is saturated observationally, so the magnitude of the
direct effect is not identified. The clamp is what lets A measure THROUGH the confounding.
Disclosure says the confounding is there; the intervention says what is underneath. That is
arguably a cleaner division of labour than the current story, but it is a claim to measure,
not to assert -- the honest test is whether the private-clamp rate survives disclosure.

## 7. Recommendation

Put this to the supervisor as the **preferred** option rather than as an alternative to S_r.
It is less disclosure than the status quo, much less than S_r, simpler to implement, and it
scales in the direction the ladder needs.

Two questions for him, and they are one question really:

1. Is an **existential confounding claim about SHARED variables** admissible?
2. If yes, is the **clique-structure leak** of §4.3 acceptable?

If (1) is refused outright, then §3's measurement says cross-boundary causal discovery is not
possible at all in this setting -- which is itself a defensible thesis finding, not a
failure, and should be written up as one.
