# n-agent refactor — spec for review

**Drafted overnight 21/22 August. NOT IMPLEMENTED. Needs sign-off before any code.**

Goal: reach **5 agents and ~30 nodes**, scaling **incrementally in both axes**. This document
says what changes, in what order, what breaks, and what each step must prove before the next
one starts.

---

## 1. The size of the problem, measured

The two-agent assumption is **structural, not configurational**:

| module | `AGENTS` / `"A"` / `"B"` references |
|---|---|
| `ma/env.py` | 30 |
| `ma/evaluate.py` | 19 |
| `ma/policy.py` | 16 |
| `ma/coordination.py` | 6 |
| `ma/topology.py` | 3 |
| `ma/confounding.py` | 2 |

And `Topology` carries the partition as **fields** — `a_private`, `b_private`, `exposed` —
so it is a two-agent data structure, not a two-agent setting.

---

## 2. What the overnight confinement result forces

Measured 22 August (`scripts/ma_confinement_n_agents.py`): confounding stays confined to the
shared set for 2, 3 and 4 agents — **but only under the right edge rule**.

The current rule, "no edge between two nodes private to different agents", is the two-agent
special case. Under partial overlap it is **too permissive**: a node visible to agents
`{0, 2}` is private to nobody, so the rule permits it to parent a node private to agent 1 —
an edge **no agent can see**. That single edge breaks confinement, and a counterexample was
found immediately.

**The rule the refactor must encode:**

> An edge may exist only if **some agent observes both of its endpoints.**

Under a disjoint partition the two rules coincide exactly, so this changes nothing today and
is required the moment visibility overlaps. It is also the more principled statement: an edge
no one can observe is not learnable by anyone, so admitting it to the hypothesis space only
adds structure no data can bear on.

---

## 3. Target API

```python
@dataclass(frozen=True)
class Topology:
    name: str
    private: Tuple[Tuple[int, ...], ...]   # private[i] = agent i's private nodes
    exposed: Tuple[int, ...]               # visible to all
    # OPTIONAL, later: explicit per-node visibility, superseding private/exposed
    visibility: Optional[Tuple[FrozenSet[int], ...]] = None

    @property
    def n_agents(self) -> int: ...
    def observed_by(self, agent: int) -> Tuple[int, ...]: ...
    def hidden_from(self, agent: int) -> Tuple[int, ...]: ...
    def may_intervene_on(self, agent: int) -> Tuple[int, ...]: ...
    def allowed_edges(self) -> np.ndarray:  # the JOINTLY-VISIBLE rule
```

**Agents become integers `0..n-1`, not the strings `"A"`/`"B"`.** `AGENTS` becomes
`env.agents`, derived from the topology. Strings were fine for two and become noise at five.

**A compatibility shim is deliberately NOT provided.** A shim would let a stale caller keep
working against the old semantics, and this project has already been bitten twice by exactly
that — a budget that silently changed meaning, and a clean-rule that silently could not fire.
Every call site should break loudly and be fixed.

---

## 4. What each component needs

**`ma/topology.py`** — the above, plus the jointly-visible mask. Smallest change, do first.

**`ma/env.py`** — mechanical: every `for name in AGENTS` becomes `for agent in self.agents`.
Three places need real thought:

- **turn order** — round-robin generalises directly; random already does. With `n` agents an
  agent acts one round in `n`, so the round budget must scale (see §6).
- **the clean rule** — already `any(hidden clamped)`. At `n` agents `hidden_from(i)` is the
  union of everyone else's private nodes, so a single clamp cleans only *part* of it. **This
  is the multi-private case the environment currently refuses**, and it becomes unavoidable
  at `n > 2` even with one private node each. **Per-block confounding subsets are therefore a
  BLOCKER for `n ≥ 3`, not a later nicety.**
- **signalling** — `n−1` partner signals instead of one. Fixed-width one-hot per partner
  keeps the observation shape stable; it grows as `O(n)`.

**`ma/score_regimes.py`** — `joint_conf` marginalises a confounding set `S` over shared
pairs. With per-block subsets it becomes `Σ_r log Σ_{S_r ⊆ S} P(S_r)·BGe(block_r | H + S_r)`,
costing `R · 2^|S|` because the per-block log-scores **add**. Plus sparsity truncation:
enumerate only assignments with at most `m` simultaneously-confounded pairs, which takes
`|X|=6` from 1.4×10⁷ to **451**, with a stated error bound rather than a hope.

**`ma/policy.py`** — one net per agent already; the dict just gets longer. Parameter sharing
across agents is an option worth testing but is **not** required, and adding it at the same
time would confound the scaling result with an architecture change.

**`ma/evaluate.py`** — the union-acyclicity and credit-set logic generalises, but the `[U14]`
criterion needs restating for `n` agents: *every* agent clears its credit set, and the union
of all `n` answers is acyclic and globally equivalent. Note this gets **harder as `n` grows**
purely combinatorially, so cross-`n` success rates are not directly comparable and must not
be plotted as if they were.

---

## 5. Ladder — each rung must pass before the next

| rung | what it proves | gate |
|---|---|---|
| **0** | 2 agents on the refactored code reproduces today's numbers | learned ≈ 0.55, private-clamp ≈ 82%, within seed noise. **If this fails, the refactor is wrong** — nothing downstream is meaningful |
| **1** | 3 agents, 1 private each, 3 shared (`d=6`) | needs per-block `S_r`. Confinement verified |
| **2** | 3 agents, more nodes (`d≈9`) | window still within the DP's reach |
| **3** | 5 agents, 1 private each, 5 shared (`d=10`) | the coordination question at target agent count |
| **4** | 5 agents, 5 private each, 5 shared (`d=30`, window 10) | the headline scaling claim |

**Rung 0 is not optional.** A refactor that changes the numbers has broken something, and the
only way to know is to check before adding agents.

**Report where it stops, and treat that as a result.** If rung 3 holds and rung 4 does not,
"five agents at ten nodes works, thirty nodes does not, and here is the binding constraint" is
a legitimate and more useful finding than an unqualified claim.

---

## 6. Parameters that must move with `n` or `d`

| parameter | rule | basis |
|---|---|---|
| `budget` (rounds) | scale with `n` — each agent needs a comparable number of turns, so `rounds ≈ turns_per_agent × n` | at 10 rounds and 5 agents each agent acts twice, which is certainly too few |
| `prior_p` | **`2 ln(d)/d`** | measured overnight: 92–99% connected across `d=5..30`, mean degree 2.6–6.5, inside the ER-2..ER-6 band. ER-2 gives **1% connected at `d=30`** and is unusable |
| `n_obs` | rises steeply with `d` | measured: the criterion needs ~8000 rows at `d=5` and ~16000 at `d=6`, roughly doubling per node |
| `m` (sparsity truncation) | swept, with a stated error budget | untested |

**The `n_obs` requirement is the most likely thing to make rung 4 unaffordable**, and it is
worth checking the cost of `d=10` windows *before* building toward them.

---

## 7. Risks

1. **Per-block `S_r` is a blocker for `n ≥ 3`**, not an optimisation. It should be built and
   tested at two agents — where the answer is already known — before any agent is added.
2. **`n_obs` scaling may price out the target.** Measure early.
3. **Free-riding should worsen with `n`.** At two agents the index was 0.85–0.88 with a
   shared round budget. With four partners to lean on, the temptation grows and the shared
   pool spreads thinner. This is a *measurement*, not a thing to design against in advance.
4. **Cross-`n` comparability.** `[U14]` demands every agent succeed, so success falls with `n`
   for combinatorial reasons alone. Report per-agent rates alongside the joint criterion.

---

## 8. What I would do first

`ma/topology.py` alone: the new API, integer agents, and the jointly-visible edge rule, with
the two-agent topology expressed in the new form and **every existing test still passing**.
That is a contained change with a hard gate, and it is the foundation everything else sits on.

Then **rung 0** — two agents on the refactored code, reproducing today's numbers — before a
third agent exists anywhere in the codebase.
