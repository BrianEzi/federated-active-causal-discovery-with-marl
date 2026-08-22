# Per-block confounding subsets (S_r) — full derivation

Working document. Every equation here maps to a specific place in `ma/belief_dp.py`;
those pointers are given at the end so the derivation can be checked against code once
it's implemented.

---

## 1. Setup and notation

An agent's window has `k` observed nodes, indexed `1..k`. Some subset of them,
`shared_positions`, are visible to other agents too (window-local indices — this is what
`ma/belief_dp.py`'s `WindowBeliefDP` already uses). Write `pairs` for all unordered pairs
of shared nodes, `(u,v) ∈ pairs`.

Outside the window sits the agent's **hidden set** `S = {h_1, ..., h_m}`, `m = |S|` —
nodes private to OTHER agents, causally upstream, never observed by this agent. `m=1` is
the case already built and validated to 1e-10 against an independent reimplementation.
`m ≥ 2` is what breaks, and what this document fixes.

Data arrives in **rounds** `r = 1, 2, ..., R`, each contributing `n_int` rows. Each round
has a **clamped set** `C_r ⊆ S`: the specific hidden nodes that were held at a fixed value
(zero variance) during that round's data collection. This is the environment's own
ground truth — not the agent's — tracked exactly via `targets` in `ma/env.py`. It is not
uncertain; it is a fact about how the data was collected, exactly analogous to knowing an
intervention occurred at all (Cooper & Yoo 1999's rule already assumes this kind of
knowledge for the node the agent itself intervenes on).

---

## 2. What "confounding" means in this model

We are not modelling the hidden nodes explicitly — the belief is only ever a distribution
over DAGs `H` on the `k` window nodes. A hidden common cause of `u` and `v` is represented
as a **proxy edge** directly between them: `u → v` or `v → u`, required to be present in
`H`. This is a computational device (lets the existing BGe/DP machinery be reused
unchanged) rather than a claim that the augmented edge IS the hidden mechanism.

A **hypothesis** is a pair `(H, P)`: a DAG `H` over the window, and a set `P` of such
proxy edges, `P`'s edges required to be present in `H`. The **local score decomposition**
(already built, `ma/belief_dp.py::_assignment_weights`) is:

```
(1)   local(v, π) = clean(v, π \ P_into_v) + dirty(v, π)         [π must contain P_into_v]
```

where `π` is v's candidate parent set, `P_into_v` is the subset of `π` that `P` forces via
some proxy edge into `v`, and `clean`/`dirty` are local BGe scores computed on two
disjoint row-subsets.

---

## 3. Why local scores of disjoint row-groups can be summed

This is the property everything else rests on, so it's worth stating precisely even
though it isn't new. For a **fixed** parent set `π`, BGe's log-marginal-likelihood is a
sum of per-row sequential-predictive terms:

```
(2)   log P(X | π) = Σ_i log P(x_i | x_{<i}, π)
```

Under the conjugate Normal-Wishart family this is invariant to how the rows are ordered
or partitioned, provided `π` is held fixed throughout. So for **any** partition of the
rows into disjoint groups `G_1, ..., G_J`:

```
(3)   log P(X | π) = Σ_j log P(X_{G_j} | π)
```

This is exactly why `clean_table + dirty_table` in equation (1) is valid, and it is the
property the generalisation below leans on: split rows into more than two groups, score
each group once, add.

---

## 4. Why m=1 is exact (recap, not new)

With one hidden node, define `z_r = 𝟙[h ∈ C_r]` — was the (only) hidden node clamped in
round `r`. This is unambiguous: `z_r=1` means the round's data provably cannot exhibit
`h`-driven correlation (zero variance in `h` that round); `z_r=0` means it might.

```
(4)   contribution_r(v, π | P) = ℓ(v, π \ P_into_v | rows_r)   if z_r = 1
                                = ℓ(v, π            | rows_r)   if z_r = 0
```

Pooling all `z_r=1` rounds into one group and all `z_r=0` rounds into another (equation 3
says this is lossless) gives exactly `clean_table` and `dirty_table`. Two tables, one sum.

---

## 5. Why m ≥ 2 breaks it — the fraction attempt

Naive generalisation: track `f_r = |C_r| / m` (fraction of hidden nodes clamped this
round), mix `clean` and `dirty` with weight `f_r`. **Wrong**: `_assignment_weights` under
this scheme receives only the scalar `f_r`, so it cannot distinguish `C_r = {h_1}` from
`C_r = {h_2}` when `m=2` — both give `f_r = 0.5` — even though a hypothesis about `h_1`
specifically should be scored very differently in the two cases. Reverted; guard restored
in `ma/env.py`.

---

## 6. Why "know the exact clamped set" isn't enough either

Second attempt: since `C_r` is exactly known (not uncertain), strip a proxy edge `u→v`
for round `r` iff **every** hidden node that could structurally cause it was in `C_r`.
Define, from the topology mask alone (no data needed):

```
(5)   Conf(u,v) = { h ∈ S : allowed_edges[h,u] ∧ allowed_edges[h,v] }
```

the set of hidden nodes that could **structurally** be a common cause of `(u,v)`. Strip
`u→v` for round `r` iff `Conf(u,v) ⊆ C_r`.

**Checked directly against the rung-1 topology mask** (3 agents, 1 private node each):
for agent 0, `Conf(u,v) = {h_1, h_2}` — **both** hidden nodes, for **every** shared pair
`(u,v)`. There is no structural discrimination between pairs to exploit here; every hidden
node threatens every pair equally. So equation (5)'s condition degenerates to "every
hidden node in `S` was clamped this round" — the same requirement as the naive whole-block
rule, and under round-robin turn order (one agent's action per round), unreachable in a
**single** round the moment `m ≥ 2`, because achieving `C_r = S` needs multiple agents to
each clamp their own node in the same round.

This is the real obstacle: it is not an implementation gap, it is that `(u,v)`'s
confounding status can never be fully resolved by any single round's batch under
round-robin at n ≥ 3. Some rounds rule out `h_1`; other rounds rule out `h_2`; no round
rules out both.

---

## 7. The fix: put "which hidden node" INTO the hypothesis

If a hypothesis about `(u,v)`'s confounding names a **specific** candidate `h`, then
per-round scoring is deterministic again (same mechanism as §4, applied per candidate).
Extend the per-pair assignment space from 3 choices to:

```
(6)   A(u,v) ∈ {∅} ∪ {(u→v, h) : h ∈ Conf(u,v)} ∪ {(v→u, h) : h ∈ Conf(u,v)}
```

(`Conf(u,v)` from equation 5 — at the current topology this is all of `S` for every pair,
but the formula is general and would discriminate at a topology where it doesn't
degenerate.) An overall assignment `𝒜 = (A(u,v))_{(u,v) ∈ pairs}` still picks independently
per pair, exactly as today (the existing model doesn't coordinate identity across pairs
either — this is a consistent extension, not a new simplification).

For node `v`, the **required-parent set** under `𝒜` is as before —
`need(v,𝒜) = {u : A(u,v) forces u→v}` — but now each bit `u` in `need(v,𝒜)` carries its
own cited hidden node, call it `h(u,v)`.

**The per-round strip set** (this replaces equation 1's single `P_into_v`):

```
(7)   strip_r(v,𝒜) = { u ∈ need(v,𝒜) : h(u,v) ∈ C_r }
```

Read: a forced-parent bit is stripped for round `r` exactly when **its own cited hidden
node** was clamped that round. Bits in the same `need(v,𝒜)` citing a different hidden node
are unaffected and stay required for that same round. That per-bit independence is what
§6 could not express.

Note the two constraints are separate and act at different levels:

- **hypothesis-space constraint** (all rounds): `π ⊇ need(v,𝒜)` — a declared proxy edge
  must be present in `H`. Unchanged from equation (1).
- **per-round scoring**: strip only `strip_r(v,𝒜)` from `π` before scoring that round's
  rows. Varies round to round; `need(v,𝒜)` itself does not.

**Per-round contribution and total** (generalises equation 4):

```
(8)   contribution_r(v, π | 𝒜) = ℓ(v, π \ strip_r(v,𝒜) | rows_r)

(9)   L(v, π | 𝒜) = Σ_r contribution_r(v, π | 𝒜)
```

Worked example. Node `v` has two forced parents: `u` citing `h_1`, and `w` citing `h_2`,
so `need(v,𝒜) = {u,w}` for every round. Then:

```
      C_r = {}          strip_r = {}      score ℓ(v, π)
      C_r = {h_1}       strip_r = {u}     score ℓ(v, π\{u})
      C_r = {h_2}       strip_r = {w}     score ℓ(v, π\{w})
      C_r = {h_1,h_2}   strip_r = {u,w}   score ℓ(v, π\{u,w})
```

The two middle rows are the point. Both have `|C_r| = 1`, so the fraction scheme of §5
scored them identically, and the whole-block rule of §6 called both "not fully clean" and
scored them identically too. Here they differ, and they differ in the right direction:
whichever confounding hypothesis that round's clamp actually silenced is the one that
stops being charged for it.

By equation (3), rounds with the **same** clamped-set `C` can be pooled into one table
before summing — group rounds by the exact bitmask `C_r`, not by `|C_r|`:

```
(10)  L(v, π | 𝒜) = Σ_{C ∈ observed clamped-sets} ℓ( v, π \ strip_C(v,𝒜) | rows with C_r = C )
```

where `strip_C(v,𝒜)` is equation (7) with `C_r` replaced by the fixed value `C`. This is
the direct generalisation of `clean_table`/`dirty_table`: instead of 2 tables, there are
as many tables as **distinct clamped-bitmasks actually appear in the episode's data** —
not `2^m` in the worst case, just whatever the turn sequence happened to produce.

**Check: does this reduce to the validated m=1 case?** `Conf(u,v) = {h}` (only one
candidate), so equation (6) gives `{∅, (u→v,h), (v→u,h)}` — 3 choices, identical to
today's `{absent, u→v, v→u}`. Equation (7) with a single candidate `h` collapses to
`strip_r = {u} ⟺ h ∈ C_r`, i.e. exactly `z_r` from §4: clamped round → strip the proxy
parent (the "clean" table), unclamped round → keep it (the "dirty" table). Equations (9)-(10) reduce exactly to
`clean_table + dirty_table`. The new machinery is a strict generalisation, not a
different model at m=1.

---

## 8. Assembling the marginal

Node-level weight over surviving parent sets (existing DP machinery, unchanged mechanism,
just using `L` from equation 10 instead of the old two-table sum):

```
(11)  log w(v | 𝒜) = log Σ_{π ⊇ need(v,𝒜)} exp( L(v,π|𝒜) + log_prior_DAG(π) )
```

Assignment weight (existing):

```
(12)  log w(𝒜) = Σ_v log w(v | 𝒜)
```

Posterior that a specific pair is confounded at all, marginalising over which candidate
`h` (and orientation) is responsible:

```
(13)  P((u,v) confounded | data) = [ Σ_{𝒜 : A(u,v) ≠ ∅} w(𝒜)·prior(𝒜) ]
                                  / [ Σ_{all 𝒜} w(𝒜)·prior(𝒜) ]
```

---

## 9. The open parameter: prior over "which h"

`prior(𝒜) = Π_{(u,v)} prior(A(u,v))`, independent per pair as today. Per pair, with
`q` = total prior mass on "confounded by something":

```
(14)  prior(A(u,v) = ∅)            = 1 - q
      prior(A(u,v) = (u→v, h))     = q / (2m)     for each of the m candidates
      prior(A(u,v) = (v→u, h))     = q / (2m)
```

Two natural choices for `q`, both consistent with §7's reduction check:

- **`q = 2/3`, matching today's implicit `{absent, u→v, v→u}` uniform-thirds prior
  exactly at m=1**, and keeping "P(confounded by SOMETHING)" constant as `m` grows rather
  than growing with the number of candidates.
- **`q = 1/2`** — a flat 50/50 prior on "confounded vs not," splitting the confounded half
  evenly over `2m` specific hypotheses. Also reduces correctly at m=1
  (`prior(u→v)=prior(v→u)=1/4` each — NOT the same as today's implicit thirds, so this
  does NOT reduce to the exact validated m=1 numbers).

**Recommend `q = 2/3`** on the strength of the reduction check alone — it's the only one
of the two that reproduces today's validated m=1 posterior bit-for-bit, which is the gate
this whole design has to clear before rung1 is trustworthy.

---

## 10. Cost

Per pair: `1 + 2m` choices. Before cyclicity pruning (unchanged — cyclicity only depends
on the `u→v`/`v→u` direction per pair, not on `h`, so `_forces_a_cycle` needs no change):

```
(15)  |assignments| = (1 + 2m)^{|pairs|}
```

| shape | m | pairs | assignments |
|---|---|---|---|
| current (n=2) | 1 | 3 | 27 (validated) |
| rung1 | 2 | 3 | **125** |
| rung3 | 4 | 10 | **~3.5 × 10⁹** |

Separately, the **table** cost (equation 10) is bounded by the number of distinct
clamped-bitmasks `C` that actually occur across an episode's rounds — bounded above by
`2^m` but in practice much smaller, since it only grows with how many genuinely different
clamp patterns the turn sequence produces, not with the full power set.

**Scope for this pass: rung1 only** (125 assignments, no truncation needed). Rung2/rung3
need sparsity truncation on top of this — enumerate only assignments with at most `M`
simultaneously-confounded pairs — which is a separate task with its own error-budget
validation, not attempted here.

---

## 11. Mapping onto the code, for implementation

| equation | code location |
|---|---|
| (5) `Conf(u,v)` | new, precomputed once per `WindowBeliefDP.__init__` from `topology.allowed_edges()` |
| (6) extended `self.assignments` | `WindowBeliefDP.__init__`, replaces the `product(*per_pair)` line |
| `C_r` per round | `ma/env.py`: replace `self.clean[agent]` (currently a float fraction) with a per-row **bitmask over `hidden`'s positions**, computed from `targets` exactly as `hidden_clamped` already is today, just not collapsed to a scalar |
| (10) grouped tables | `WindowBeliefDP.regime_tables`, group by `np.unique(clean_bits, axis=0)` instead of by float value |
| (7)-(9) per-bit stripping | `WindowBeliefDP._assignment_weights`, per forced-parent bit `u` look up its cited `h(u,v)` and check membership in the round-group's `C` |
| guard in `ma/env.py.__init__` | removed once (11)-(13) pass the m=1 reduction test AND a rung1 acceptance test with a real, independent check on the VALUE (not shape/range) |

---

## 12. What still needs a value-level check before this ships

Everything above is a derivation, not yet a proof-by-test. Before the guard comes off:

1. **Bit-for-bit at m=1**: run the new code path with `m=1` and confirm it reproduces the
   existing `1e-10` oracle agreement exactly (§7's reduction check, executed not just
   argued).
2. **A hand-checkable m=2 case**: small `k`, small data, compute `P((u,v) confounded)`
   both via the new DP path and via brute-force enumeration over the full augmented-graph
   hypothesis space (feasible at `k ≤ 4`), confirm agreement.
3. **The counter-example from the earlier review**: the swap-which-partner-clamps test
   that showed non-identical (but confounded, uninterpretable) beliefs under the old
   fraction scheme — rerun it here and show the difference is now attributable to the
   *correct* mechanism (equation 7's per-bit stripping), not an artifact.

---

## 13. Does naming `h` break the privacy constraint?

Asked directly, and the answer needs two things kept apart.

**Structural metadata** — that `S = {h_1, ..., h_m}` exists, its size `m`, and which agent
owns which element. Every agent already constructs its window from
`topology.hidden_from(agent)`; this is schema, not values. Knowing a column exists is not
knowing what is in it, and equation (6) uses `h` only as an INDEX. Nothing new.

**Per-round `C_r`** — which hidden nodes were clamped. This is the load-bearing one, and it
is **already broadcast by the existing protocol**. `ma/env.py::_record_signals` sends each
agent's action as one of `{NO_INTERVENTION, SHARED_SIGNAL, PRIVATE_SIGNAL}`, per partner.
At rung1 (one private node per agent):

```
      PRIVATE_SIGNAL from agent j   <=>   h_j ∈ C_r
```

so `C_r` is recoverable from the signal vector with no new channel. The agent never sees
`h`'s values; it sees a three-category action label its partner already sends, and scores a
structural hypothesis against its OWN columns.

**Two conditions this rests on, both must hold:**

1. **Clamp-only.** The signal does not encode mode, so a private VARY also emits
   `PRIVATE_SIGNAL` while cleaning nothing. Under `action_modes=(CLAMP,)` — the default
   since 2026-08-22 — intervened and clamped coincide and the equivalence above is exact.
   Under both modes it is not, and would need a fourth signal category.
2. **One private node per agent.** At multi-private topologies (rung 2, rung 4), "agent j
   clamped SOME private node" does not say WHICH, so `C_r` is underdetermined and genuinely
   would require disclosure beyond today's protocol. Those shapes stay blocked regardless
   of this document.

**Coupled decision, flagged.** `disclose_signals` is provisional pending the supervisor. If
the signalling channel is ruled inadmissible, S_r at `n >= 3` falls with it: the only
fallback is to treat `C_r` as unknown and marginalise it, and

```
      P(rows_r | 𝒜) = Σ_C P(C) · Π_v ℓ(v, π \ strip_C(v,𝒜) | rows_r)
```

puts `Σ_C` OUTSIDE `Π_v`, which is exactly the modularity break described in
`ma/belief_dp.py`'s module docstring — no per-(node, parent-set) table can express it, and
the subset DP cannot compute it. So "is the signalling channel admissible" and "can we do
S_r at three agents" are one question, not two.
