# Attribution at scale: it factors over components, and the enumerated engine gets WRONG

> ## SUPERSEDED, 31 Aug 22:10 — DO NOT QUOTE THE NUMBERS BELOW
>
> Three claims in this document are wrong. All three were artefacts of defects in my own
> engine, each found by a measurement queued to test the claim above it. A corrected table is
> being produced from a single build and this file will be rewritten when it lands.
>
> **1. "Precision collapses from 98% to 59% as k grows" — FALSE.** Caused by two bugs:
>    * `FactoredAttributedBackend` advertised scope over pairs it had truncated away, so
>      groups it had never been asked about scored WRONG instead of UNSURE. Fixed.
>    * `consistent_with_partner`'s atomicity rule was UNSOUND. Two agents may independently
>      confound the same pair, so a pair can move because agent 2's latent moved while agent
>      1's group containing it did not — and the old test refuted the TRUTH for it. Measured:
>      atomicity alone refuted the true attribution in **27 of 85** oracle messages. Repaired
>      to apply all-or-none only over pairs a group explains EXCLUSIVELY: **27/85 -> 0/85**.
>
>    With both fixed, both engines settle **zero attributions incorrectly** at every size
>    measured. There is no precision collapse.
>
> **2. "The component engine is more precise because it declines to apply rule 1 across
>    components" — FALSE.** The D6 probe found **zero** cross-component messages at k=12, 20
>    and 30. The mechanism I proposed does not occur.
>
> **3. "The component engine gives up 10-25% of decisions" — FALSE.** A one-ulp bug:
>    `frequency_tables` accumulated `+= 1/n`, so a group present in EVERY candidate reached
>    0.9999999999999998 and failed `freq >= 1.0`. It was discarding every claim it was
>    CERTAIN of. Fixed by counting integers and dividing once. The two engines now agree
>    exactly (k=6: 57/57, k=8: 38/38, k=12: 44/44) with the component engine 1.5-2x faster.
>
> **What survives unchanged:** the component factoring is exact (set equality on 240 random
> pair sets); the 11.7x speedup; the reach to k=30; and the finding that rule 1's
> local-disturbance assumption fails measurably often (violations counted per run) while the
> engine degrades to UNSURE rather than to WRONG.
>
> **A process failure worth recording:** the table this file reported mixed rows from
> different builds, because the driving script invoked Python three times and I edited the
> engine between stages. Definitive numbers now come from one process, one build.


31 Aug 2026. Supersedes the "attribution caps at k=12" reading in
`RESUME_PER_PAIR_ATTRIBUTION.md`, and corrects a number I reported from too few episodes.

---

## 1. The headline

Two claims, and the second matters more than the first.

**The attribution candidate set factors exactly over the connected components of the
bidirected graph**, so ownership no longer has to be enumerated jointly and attribution
reaches k=30 at half the cost of the engine it replaces.

**The enumerated-ownership engine does not merely get slow past k=12 -- it gets WRONG.** Its
precision on settled attributions falls 98% -> 95% -> 84% -> 59% -> 63% across k = 6, 8, 12,
20, 30. The component-factored engine holds 100% at k=20 and k=30.

|  k | engine     | right | wrong | unsure | total | precision | recall | scope | viol | s/ep |
|---:|------------|------:|------:|-------:|------:|----------:|-------:|------:|-----:|-----:|
|  6 | enumerated |    58 |     1 |     31 |    90 |     98.3% |  64.4% |  0.82 |   10 | 0.02 |
|  6 | component  |    56 |     1 |     33 |    90 |     98.2% |  62.2% |  0.82 |   10 | 0.03 |
|  8 | enumerated |    42 |     2 |     38 |    82 |     95.5% |  51.2% |  0.79 |   16 | 0.04 |
|  8 | component  |    37 |     2 |     43 |    82 |     94.9% |  45.1% |  0.79 |   16 | 0.04 |
| 12 | enumerated |    49 |     9 |    101 |   159 |     84.5% |  30.8% |  0.79 |   41 | 3.29 |
| 12 | component  |    36 |     5 |    118 |   159 |     87.8% |  22.6% |  0.79 |   41 | 1.13 |
| 20 | enumerated |    37 |    26 |    165 |   228 |     58.7% |  16.2% |  0.64 |   73 |15.68 |
| 20 | component  |    34 |     0 |    194 |   228 |    100.0% |  14.9% |  0.57 |   73 | 7.86 |
| 30 | enumerated |    20 |    12 |    238 |   270 |     62.5% |   7.4% |  0.94 |  122 |15.04 |
| 30 | component  |    18 |     0 |    252 |   270 |    100.0% |   6.7% |  0.60 |  122 | 7.68 |

30 episodes per cell, oracle evidence, scale-free graphs, sigma=0.5, 3 agents at k<=8 and 4
above. A deterministic round-robin sweep of each window drives it, NOT a learned policy --
this measures the BELIEF's reach, and a policy would confound it with how well that policy
happens to probe its partners. `scripts/attr_scale.py`, `results/attr_scale.json`.

Column meanings. **right/wrong/unsure** are the three-outcome verdict per TRUE latent group at
bar 1.0, never summed. **precision** = right / (right + wrong): of the attributions the engine
committed to, how many were correct. **recall** = right / total. **scope** is the mean share
of settled bidirected pairs the belief may speak about at all -- see section 6, and do not
compare `right` across engines without it. **viol** is `assumption_violations`: how many times
a partner message refuted the TRUE attribution, which under oracle evidence can only be rule
1 failing. **s/ep** is wall clock per episode, summed over agents.

---

## 2. Background: what is being recovered, and why it is hard

An attribution is a set of LATENT GROUPS, each `(owner, children)` -- an agent whose private
block holds a hidden variable, and the window nodes it parents. A group with children
`{u,v,w}` accounts for the bidirected edges `uv`, `uw` and `vw` at once, so a latent's
children are always a CLIQUE in the bidirected graph. Correctness is judged up to renaming:
an outsider never learns WHICH of the owner's variables it was, and does not need to, because
the set of pairs it explains is its identity from outside. That is also the privacy claim.

Identifiability is what makes this a federated problem rather than an inference one. One
latent parenting `{u,v,w}` and three latents parenting each pair induce IDENTICAL
observational data. An intervention separates them -- disturb the single latent and all three
associations move together -- and the only agent who can perform it is the one who owns the
variable. Recovering the grouping therefore requires a partner to experiment on your behalf.

The belief is a version space over attributions, enumerated by giving each confounded pair a
non-empty set of owners and reducing each owner's pairs to their maximal cliques. That is
`(2^n - 1)^P` assignments in the partner count `n` and the confounded-pair count `P`:
5 / 35 / 482 / 8.4e10 / 8.9e15 hypotheses at k = 4 / 8 / 12 / 20 / 30.

---

## 3. The two pruning rules, and which one is sound

A partner acts on a private node. The agent is told WHICH partner acted -- never which node --
and observes which of its confounded pairs moved. The message is `(actor, moved)`.

**Rule 2, ATOMICITY. Sound unconditionally.** A latent moves as a unit, so a candidate that
assigns a clique to one latent and then sees only PART of it move is refuted. This is also
where the coordination story lives: an action that moves everything separates nothing, a
PARTIAL response refutes a hypothesis outright, so an agent needs its partner to probe its
private variables one at a time -- exactly the experiment a partner has no selfish reason to
run.

**Rule 1, LOCAL DISTURBANCE. An explicit modelling assumption, and false in general.** At
least one moved pair must be covered by a group the candidate attributes to the actor.
`responds_to` marks a group as responding when the intervened node is an ANCESTOR of that
group's latent, and an actor's private node can sit above a THIRD agent's latent through the
shared block -- so the actor genuinely causes movement in pairs it does not own and the
message mixes owners. Measured at 3 agents: `moved` carried a foreign owner's pairs in 10 of
115 signals, and the true attribution was refuted by its own evidence in 9 of them.

**Rule 1 is kept anyway, and that is a knowing trade.** Deleting it and generalising atomicity
to every owner IS sound, and it was tried: `right` collapsed from 72 to 0 over 162 groups.
Rule 1 carries the entire discriminative power of the channel. `local_disturbance=False`
switches it off and is the sensitivity analysis this assumption must be reported with.

---

## 4. Why per-pair factoring fails, and what does factor

Per-pair factoring -- the plan of record until 31 Aug -- does not work. Atomicity refutes a
PARTIAL clique response, which requires knowing which pairs share a latent: the clique
structure, which is precisely the joint fact a per-pair belief cannot represent. Rule 1 is a
statement about the groups an owner holds. A per-pair split loses both, and losing rule 1
loses the chapter.

What factors is the CONNECTED COMPONENT of the bidirected graph:

    attributions_for(pairs, owners) == PRODUCT over components of attributions_for(c, owners)

exactly. A group's children form a clique and a clique never spans components, so every group
lives inside one component; owner assignment is per pair; the maximal-clique reduction is
within an owner's pair set; the coverage check is per pair. Verified as SET EQUALITY on 240
random pair sets at 2-4 owners and pinned in
`tests/crosscheck/test_component_attribution.py`. Cost falls from `(2^n - 1)^P` to a SUM of
`(2^n - 1)^Pc` over components.

On the evidence side the split is just as clean. **Atomicity is UNARY at group granularity** --
the test on each group names no other group -- so a global candidate passes iff each
component's part passes. It decomposes exactly. **Rule 1 is the only joint constraint**, and it
is a disjunction: "some group owned by the actor intersects `moved`", where `moved` can name
pairs in several components at once.

That clause is applied by UNIT PROPAGATION to a fixpoint. With `C_j` the components still
holding a candidate that satisfies message `j` locally:

    |C_j| == 1   only one component can satisfy it, so it must -- filter it. EXACT.
    |C_j| == 0   nothing satisfies it -- drop the message and count it, exactly as the
                 enumerated backend drops a message that would empty the candidate set.
    |C_j| >= 2   skip, and re-test after every other prune, because pruning elsewhere can
                 make it unit later.

Those skips are the entire approximation. Everything else is exact.

**Soundness.** The represented belief is the product, filtered by all of atomicity and some of
rule 1, so it is a SUPERSET of the enumerated belief. At bar 1.0 the verdict asks only two
questions -- does EVERY candidate name this group (right), does NO candidate name it (wrong) --
and both survive passing to a superset. So this engine is LESS DECIDED, never differently
decided.

---

## 5. Why the component engine is more precise, and it is not luck

The rule it under-applies is the rule that is unsound. A cross-component rule-1 clause is one
the component engine declines to apply; the enumerated engine applies all of them, including
the ones that refute the truth. So as the bidirected graph fragments -- 1.0 components at
k=12, 1.2 at k=20 and k=30 -- the enumerated engine makes more bad prunes and the component
engine makes fewer.

That is visible in the table as a REGIME CHANGE rather than a smooth trade:

* k <= 12: a genuine trade. At k=12 the component engine buys +3.3 points of precision for
  13 lost decisions.
* k >= 20: not a trade. At k=20, 34 right against 37, zero misattributions against 26, at half
  the wall clock.

---

## 6. Caveats, stated rather than buried

**Scope is not matched at k=30.** 0.60 for the component engine against 0.94 for the
enumerated one, because dense components exceed the per-component pair budget and are
truncated. Part of that 100% precision is bought with coverage, and the k=30 row should not be
read as a clean win. At k=20 the scopes are close (0.57 against 0.64) and the result stands on
its own. Raising `max_component_candidates` trades some precision back for scope and has not
been swept.

**A number I previously reported was wrong.** "k=12, 0 wrong at cap 5" came from 8 episodes.
At 30 episodes k=12 shows 9 wrong for the enumerated engine and 5 for the component one. It
was a small-sample artefact reported as a property of the backend.

**`wrong` is not an engine bug.** Under oracle evidence the truth cannot leave a sound
candidate set, so a confident misattribution can only mean a message refuted the TRUE
attribution -- rule 1's assumption failing. The `viol` column counts those directly so the two
cannot be confused. 122 violations at k=30 produce 12 wrong verdicts for one engine and 0 for
the other.

**Recall falls with k for both engines**, from ~64% at k=6 to ~7% at k=30. Attribution gets
harder with window size for reasons that have nothing to do with the belief representation:
more pairs to settle, the same budget, and a partner that must spend rounds on experiments
that do nothing for itself.

---

## 7. Two defects found and fixed in this work

**The partner channel was dead for every backend but one.**
`ma/env.py::_disclose_partner_responses` gated on `!= ATTRIBUTED`, so a run using
`factored_attributed` -- the backend that exists precisely to carry attribution past k=5 --
received NO partner messages and its attribution could never be settled by evidence. It
survived because every factored-attribution number to date came from a driver calling
`observe_partner` directly (`tests/crosscheck/`, `scripts/attr_scale.py`), never through the
env. Any env-path attribution number from before 31 Aug is measuring a dead channel.

**A cache that would have been unsound.** The first optimisation carried each component's
PRUNED candidate list across rebuilds. Concretely the failure: pair `uv` sits alone in
component A, a message arrives whose only support is A, the clause is unit and A is filtered;
later `xy` settles in component B which could ALSO have supported that message, so the clause
was never unit and A's prune was never licensed. The cached list keeps a prune the enlarged
scope has withdrawn. What is cached instead is, per (component, message), the index sets
"survives atomicity" and "survives atomicity and rule 1" -- both functions of that component's
own candidates and the message alone, so scope-independent. k=12 fell from 2.64 to 1.13
s/episode with identical verdicts.

---

## 8. Reproducing

```bash
.venv/bin/python -m pytest tests/crosscheck/test_component_attribution.py -q -s
.venv/bin/python scripts/attr_scale.py --episodes 30 --out results/attr_scale.json
```

The eval pass over trained checkpoints selects the engine explicitly:

```bash
PYTHONPATH=. .venv/bin/python scripts/attr_score.py --backend component_attributed \
    --n_agents 4 --private_size 6 --n_shared 6 --budget 60 --episodes 150 \
    --policy results/sweep/oracle/k12s50n04b200_s0_best.pt --out results/attr/k12_s0.json
```
