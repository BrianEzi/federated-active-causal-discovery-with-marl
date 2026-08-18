# Two-agent design -- plan of record, 2026-08-16

> ## AMENDED 2026-08-18 -- read this first
>
> Four things in this document have since been **measured and found wrong**. The design is
> otherwise intact; these are corrections, not a rewrite. Full evidence and the audit trail
> are in `docs/MA_BUILD_LOG.md`.
>
> 1. **Section 3's confounding rates are ~3x overcounts.** `ma/confounding.py` flags any pair
>    with a hidden common *source*, a strict superset of true bidirected edges: where the pair
>    is also ancestrally related the MAG carries a directed edge instead. True rates are
>    **6.3% / 13.4% / 9.0%** at (1,1,2) / (1,1,3) / (2,2,2), not 22.7% / 43.8% / 43.9%. Use
>    `ma/projection.py::bidirected_pairs`.
>
> 2. **Section 9's starting topology (1,1,2) was abandoned.** On the corrected rates only 13
>    of 207 graphs give any agent a bidirected edge, always the same pair -- too rare to learn
>    from. Work is at **(1,1,3)**.
>
> 3. **Section 5's `|X|^2` ancestral-order disclosure is not what enables coordination.**
>    Measured structurally and posterior-weighted, it is worth **~0.005 bits per bit
>    disclosed** -- a correctness guard, not an inference tool. What unlocks coordination is a
>    single **regime bit** per round ("I clamped something you cannot see"), which moves a
>    confounded agent from 0.000 to 0.974. My reframing of section 5 as a pruning device is
>    retracted; the original safety-net reading was right.
>
> 4. **Section 4's claim that the rescue needs "no disclosure at all" is false**, for two
>    compounding reasons. A *randomised* `do()` does not cut confounding -- it swaps one
>    invisible common cause for another -- so interventions now have two modes, `VARY` and
>    `CLAMP`. And even clamping rescues nothing while the agent pools clean and confounded
>    rows, because no DAG fits a mixture. Both had to be fixed before GATE 4 would pass.
>
> **New result this document never anticipated:** confounding is *confined to the shared
> set* -- no bidirected edge can touch a private node (proved, verified exhaustively). That
> is what keeps a per-agent belief a plain DAG plus one flag per shared pair, leaving the
> score decomposable and the subset DP reusable.

Supersedes the topology sections of `docs/MULTI_AGENT_DESIGN.md` (T1/T2/T3). That document
is kept for the reasoning that led here, including the parts now rejected.

Written to disk because the conversation that produced it will be compacted. Everything
below was either **measured** or **argued to a conclusion**; open questions are marked as
such at the end.

---

## 1. The schema

Nodes `V`, one DAG `G`. Partition into:

- `Z_A` — agent A's private variables (any internal structure)
- `Z_B` — agent B's private variables (any internal structure)
- `X` — the shared set (any internal structure)

Agent A observes `O_A = Z_A ∪ X`; agent B observes `O_B = Z_B ∪ X`.

**That is the whole specification. `X = O_A ∩ O_B` — this is two overlapping observation
windows and nothing more.** Sizes `(|Z_A|, |Z_B|, |X|)` and the edge pattern are parameters,
not part of the definition.

This generalises the earlier framing, which split `X` into `X_A ∪ X_B` and enforced
`Z_A — X_A — X_B — Z_B`, forbidding `X_B → Z_A` and `X_A → Z_B`. Under that rule, influence
from A's side to B's side had to traverse **two** shared nodes. The general form allows
`Z_A → x ← Z_B`: a single shared node with parents on both sides. That case is not
recoverable by relabelling, so the loosening is a genuine enlargement.

T1 as specified on 2026-08-15 was **already** the general form (only `Z_A ↔ Z_B` forbidden),
so no code change was needed — but the earlier framing was the wrong description of it.

---

## 2. Which edges can be learned, and by whom

| edge type | A sees both ends | B sees both ends | who can learn it |
|---|---|---|---|
| inside `Z_A` | yes | no | A only |
| `Z_A ↔ X` | yes | no | A only |
| inside `X` | yes | yes | **both** |
| `Z_B ↔ X` | no | yes | B only |
| inside `Z_B` | no | yes | B only |
| **`Z_A ↔ Z_B`** | **no** | **no** | **nobody** |

Five of six categories are covered. The sixth is the blind spot, and the decomposition into
two windows is complete **iff** there are no `Z_A ↔ Z_B` edges.

### The prohibition is forced, not arbitrary

With `|Z_A| = |Z_B| = |X| = 1`, compare `M1: z_A → z_B → x` against
`M2: z_A → x, z_B → x`. A observes `(z_A, x)`; B observes `(z_B, x)`.

| | M1 | M2 |
|---|---|---|
| A: `z_A`, `x` correlated | yes | yes |
| B: `z_B`, `x` correlated | yes | yes |
| A does `do(z_A)` | `x` moves | `x` moves |
| B does `do(z_B)` | `x` moves | `x` moves |
| anyone does `do(x)` | nothing upstream moves | nothing upstream moves |

Identical in every row, and the variances match under a linear-Gaussian parameterisation.
The cross-private edge is perfectly confounded with a direct private→shared edge.

### It is the privacy constraint's shadow

The edge *is* detectable — if A intervenes on `z_A` while B watches `z_B` move. That
requires correlating **A's private actions with B's private observations**, exactly what the
non-disclosure constraint forbids.

So: **federation costs precisely the `Z_A ↔ Z_B` edges.** The restriction is a theorem
about the privacy model, not a modelling convenience. This is a thesis result.

---

## 3. Confounding requires `|X| ≥ 2`

For A, everything in `Z_B` is latent. A latent variable damages A's view only by being a
**common cause of two variables A can see**. A `Z_B` node reaches A's window only through
`X`. Therefore:

- `|X| = 0` — no overlap; two unrelated problems.
- `|X| = 1` — a `Z_B` node touches at most one variable A sees, so it can never be a common
  cause of two. **A's window is confounding-free for any `|Z_A|`, `|Z_B|`.**
- `|X| ≥ 2` — a `Z_B` node can parent two shared nodes; confounding appears.

Measured (`ma/confounding.py`, exhaustive enumeration):

| topology | shared pairs | #DAGs | confounded | ambiguity on shared edges |
|---|---|---|---|---|
| (1,1,1) | 0 | 9 | **0.0%** | 0.0% |
| (1,1,2) | 1 | 207 | 22.7% | 16.2% |
| (1,1,3) | 3 | 11,649 | 43.8% | 28.4% |
| (2,2,2) T1 | 1 | 96,255 | 43.9% | 5.0% |

**`(1,1,1)` is rejected**: zero confounding, and fully decomposable — each agent resolves its
own edge with one intervention on its own private node and never needs `X`. It is T3 by
another route.

**T3 is rejected** (from 2026-08-15): removes confounding by forbidding private→shared
edges, but drops shared-edge ambiguity to 0% — each agent then solves its half alone.

---

## 4. The inference target is a MAG over the agent's OWN set

Objection raised: A does not know whether B has any private nodes, so A cannot place an edge
or say where a confounder sits.

Resolved: a latent projection has nodes drawn **solely from `O_A`**. Hidden variables are
marginalised out and appear as a **bidirected edge between two of A's own variables**,
meaning "these share an unobserved common cause" with no claim about what or where. A never
needs to know B's nodes exist. `x_1 ↔ x_2` is a complete statement in A's own vocabulary.

**Consequence worth building on:** B holds the explanation for A's bidirected edge. A's
irreducible ambiguity is B's ordinary private structure. If B intervenes on the responsible
`z_B`, the association breaks *in A's own data*, with **no disclosure at all**. This is
coordination where the privacy constraint costs literally nothing, and should be a
first-class experiment.

---

## 5. Global acyclicity is NOT implied by local acyclicity + agreement on `X`

**Counterexample.** `X = {x_1, x_2}`, one private node each.

- A infers `x_1 → z_A → x_2` — acyclic, no direct `x_1`–`x_2` edge.
- B infers `x_2 → z_B → x_1` — acyclic, no direct `x_1`–`x_2` edge.
- They agree perfectly on `X` (both: no edge).
- Union: `x_1 → z_A → x_2 → z_B → x_1`. **Cycle.**

Neither agent can detect it.

### The exact condition

Define on `X` only: `x_i ≺_A x_j` iff A's graph has a directed path `x_i ⇝ x_j`, possibly
through `Z_A`. Similarly `≺_B`.

> **Global acyclicity holds iff each local graph is acyclic and `≺_A ∪ ≺_B` is acyclic
> on `X`.**

*Proof sketch.* Any cycle in the union cannot cross directly between `Z_A` and `Z_B`, so it
decomposes into segments each lying wholly in one agent's graph, each entering and leaving
through `X`. A cycle inside one agent's graph is excluded by local acyclicity. So every
cycle projects onto a cycle in `≺_A ∪ ≺_B` restricted to `X`, and conversely. ∎

### Minimum disclosure

For each ordered pair of shared nodes, one bit: *does a directed path exist on my side?*
At most **`|X|²` bits**, naming no private variable and revealing no count or structure.

**It does leak one thing, to be stated not glossed:** reporting `x_i ⇝ x_j` where no direct
shared edge exists reveals that the reporter has **at least one private mediating variable**.
Existence, not identity.

So "how much must they share?" is a **derived quantity**, not a design choice. Second thesis
result, alongside §2.

**This check must be part of the protocol**, or the agents can converge on a jointly
impossible graph and nothing will notice.

---

## 6. Single system vs two systems — SINGLE SYSTEM, decided

I initially argued for two systems (each agent its own data). **Reversed after steelmanning**:

1. **One system is the more common scientific reality** — two groups, one cell line / cohort
   / grid / economy.
2. **Every experiment benefits both agents across their whole windows, not just `X`.** Under
   two systems, A's `do(x_1)` teaches B nothing, so the only transferable content is the
   `X`-subgraph — which forced `|X| = 3` just to have enough to coordinate about. Under one
   system, A's `do(x_1)` also reveals to B how `x_1` affects `z_B`. **Richer coordination
   surface, for free.**
3. **Coordination without communication becomes possible** — B learns from A's *action*, no
   message sent, nothing disclosed.
4. **And its dual: acting is disclosing.** A repeated request pattern lets B infer something
   about A's private structure — an **involuntary side channel**, present with no message.
   Under two systems privacy holds trivially and uninterestingly; under one system it is
   genuinely at risk from the actions themselves, and **quantifying that leak is a real
   contribution** rather than an assumption.

Point 4 is the decisive one. A design where privacy holds by construction says less than one
that measures what acting leaks.

**Cost, stated plainly:** privacy is no longer airtight by construction; the leak must be
measured rather than assumed away.

---

## 7. Budgets and rounds — separate budgets, simultaneous, no collision rule

I briefly proposed a **shared** budget with one experiment per round. **That was an error**
and is withdrawn. It was engineered to dissolve an objection I had raised — that collisions
need an invented arbitration rule — and that objection was itself wrong.

Under separate budgets there is nothing to arbitrate: A runs `do(x_1)`, B runs `do(x_2)`,
and on one shared system both simply happen — the system is experimented on twice that
round. Repeated experiments on one system are ordinary. Nobody is blocked.

Even duplication needs no rule: two batches under `do(x_1)` is not zero value (it halves the
noise) but is sharply diminishing for two budget units. **Duplication is wasteful in exactly
the graded, emergent way wanted, with no invented penalty.**

### Settled mechanics

- **One shared system.** Every experiment is an event both agents observe in their own
  windows.
- **Separate per-agent budgets.** Each agent spends its own.
- **Both act every round, simultaneously.** No turn order imposed, no arbitration, no
  collision penalty.
- **Duplication allowed** — it buys redundant data for two budget units.
- **A round's two experiments are TWO SEPARATE SAMPLE BATCHES**, not one joint
  `do(x_1, x_2)`. A joint intervention is a genuinely different and generally less
  informative experiment; keeping them separate preserves the intervention semantics of the
  already-validated single-agent code.
- **Exchange of shared-variable results permitted; private ones never.**
- Turn-taking is *not* imposed — the agents can invent it through the permitted channel, and
  whether they do is a **finding**. Hard-coding it would hand them the answer.

### The game, named

A **coordination game under simultaneity**: both commit without seeing the other's choice or
outcome; both observe both results afterwards. Neighbours are congestion / allocation games.
Not a bargaining problem — that was the shared-budget version, which would have made results
partly a property of whatever arbitration rule was written.

### Consequence to expect, not be surprised by

Simultaneity removes **within-round** sequencing and keeps only across-round. GATE-M2 measured
that a centralised planner forced to batch two experiments showed **no advantage**
(2.80 vs 2.64), while one that could act-look-choose did (**1.74 vs 2.52**). So the
achievable coordination gain here is smaller than +0.787. This is correct — it makes
*allocation* the thing learned rather than handing over an unearned advantage — but the
expectation should be set in advance.

---

## 8. `coordination_gained` must be rebuilt

Current implementation (`ma/coordination.py`):

```
coordination_gained = mean interventions (independent) − mean interventions (centralised)
```

Both arms: same episodes, same true graphs, cost = interventions used or full budget if
unsolved (same censoring as single-agent). Centralised = one chooser, full posterior, one
target per round, **re-plans**. Independent = each agent scores only nodes it has authority
over, grouping hypotheses by descendant set **restricted to its visible nodes**; both act, so
two interventions per round. Budgets matched in **interventions, not rounds**. Passes when
the centralised CI upper bound < independent CI lower bound. Measured **1.74 vs 2.52,
gained +0.787**.

**What is wrong with it for our purposes:** every intervention from either agent appends to
**one** `samples` array and both arms score **one pooled posterior**. Inference is held
centralised deliberately (so a difference cannot be blamed on worse beliefs). That makes it a
valid gate — *"is there a coordination problem in this topology?"* — but it measures
"limited view for choosing, full information for inference", which is **not** what we now
want.

**Read the +0.787 as: a coordination problem exists here. Nothing more.**

v2 needs per-agent beliefs, the exchange rule, and the comparison
**exchange-allowed vs exchange-forbidden**.

Note the data model (one pooled system) is now *correct* given §6 — what is missing is
per-agent beliefs and the acyclicity check, not a rewrite.

---

## 9. Starting point

**`(1,1,2)`** — `z_A`, `x_1`, `x_2`, `z_B`. 207 graphs, fully enumerable, exact posteriors,
no sampling anywhere. Viable because the single-system decision makes the coordination
surface rich (§6.2); under two systems it would have been too thin and `(1,1,3)` would have
been required.

**`(1,1,3)`** is the scaling step: 11,649 graphs, 3 shared pairs, 43.8% confounded, 28.4% of
ambiguity on shared edges. Then `(2,2,2)`.

`|X|` is the primary sweep axis — it controls confounding and the coordination surface
together.

---

## 10. Deferred and out of scope

- **Non-linear / non-Gaussian mechanisms**: deferred until two-agent works on
  linear-Gaussian, then tested **in the single-agent case first**. Cheap when it comes: the
  subset DP needs only a *decomposable* score and never inspects the likelihood, so a
  discrete BDeu score is a new class next to `BGeScore` and nothing else changes.
- **Exogenous latent confounders**: out of scope. There is already relative latent
  confounding between the agents; adding more would make effects unattributable.
- **Permitting `Z_A ↔ Z_B` edges** (target becomes an equivalence class over a coarser
  object): clearly-scoped further work, not this thesis.

## 11. Terminology

Different variables **and** different samples, shared mechanism, overlapping variable set.
The standard FL taxonomy fits awkwardly — decentralised FL and vertical FL are both
established, their intersection is rare, and what is shared here is a *causal mechanism*
rather than a representation.

**The taxonomy is worth one paragraph of positioning and nothing more.** Every technical
constraint in this document came from the structure (§2, §3, §5), not from the label. User
to discuss the naming with Mirco directly.

---

## 12. Open questions

1. **Protocol for the ancestral-order exchange** — when is it sent, how often, and is it a
   precondition for declaring the graph identified?
2. **How to measure the involuntary leak** (§6.4). Candidate: mutual information between A's
   request sequence and A's private structure, estimated over episodes.
3. **Per-agent belief representation** under confounding — full MAG/PAG machinery, or the
   bidirected-edge summary only?
4. **Definition of "identified"** in the two-agent case: global graph recovered, or each
   agent's local projection recovered, or both plus consistency?
