# Two-Agent Federated Active Causal Discovery — Problem Statement

**Status**: authoritative as of 2026-08-19. Supersedes `docs/MA_DESIGN.md` wherever the two
conflict. Written from scratch after the accreted design was rejected; every clause below
traces either to a user ruling (marked **[U]**) or to a measurement (marked **[M]**).
Anything I decided myself is marked **[MY CALL]** and is open to reversal.

---

## 1. The system

There is **one** structural causal model **[U1]**. Not two systems, not two datasets from
related systems — one linear-Gaussian SCM over `d` variables, generated once per episode.

The variables are partitioned into three sets:

| set | notation | who sees it |
|---|---|---|
| A's private | `Z_A` | agent A only |
| B's private | `Z_B` | agent B only |
| shared / exposed | `X` | both agents |

This is a **vertical partition with overlap** **[U4]** — the agents hold different *columns*
of the same rows, and the `X` columns are held by both. No agent holds a subset of rows.

**Edges between `Z_A` and `Z_B` are forbidden in both the generator and the prior** **[U3]**.
Neither agent ever observes both endpoints, so no data anyone could collect bears on such an
edge; permitting it would make the global graph unrecoverable by construction. Generator and
prior carry the *same* mask — a mismatch would surface later as systematic overconfidence
that looks exactly like an estimator bug.

Everything else is permitted: within `Z_A`, within `Z_B`, within `X`, and both directions
between a private set and `X`.

**Starting size** is the minimum that exhibits the phenomenon **[U2]**, and we scale from
there. That is `(|Z_A|, |Z_B|, |X|) = (1, 1, 3)`, giving `d = 5` and a per-agent window of
`k = 4`. Not `(1,1,2)`: the confounded-pair rate there is 6.3% (13 of 207 graphs, always the
same pair) — too rare to learn from **[M]**.

**Agents do not know the global topology.** Agent count is known only implicitly **[U5]** —
an agent knows its own window and that some variables it cannot see may exist.

## 2. What each agent observes and does

### Data

Each episode begins with `n_obs = 100` observational samples **[U6]**, restricted to the
agent's own columns. `n_obs` is a swept parameter, not a constant of the design.

`n_obs = 100` is a deliberate choice, not an inherited default. In the single-agent case,
`n_obs = 20000` left a median of 2 interventions before identification and made 77% of
episodes solvable in ≤2 moves; dropping toward zero at `d = 7` raised that to a median of 4
with 92% of episodes needing ≥3 **[M]**. A small `n_obs` is what creates a sequential problem
at all.

### Interventions

**Every intervention is hard** — the target's incoming edges are cut. There are two modes,
differing only in the value assigned:

- **VARY** — assign `X_t ~ N(0, σ²)`, redrawn per sample.
- **CLAMP** — assign `X_t = c`, a constant, held across all samples of that round.

Both are hard interventions. Neither is a soft / mechanism-shift intervention; soft
interventions are out of scope **[U7]**.

**Why both modes exist, and why this is not a free parameter** **[M]**. The two modes serve
opposite purposes and cannot be collapsed:

- *For learning your own structure*, VARY is far better. Injecting known variance gives you
  covariance to measure against. Measured at `d=5`: with ample observational data, greedy
  cost 1.90 (VARY) vs 3.53 (constant); with almost none, **2.79 / 96.3% solved vs
  12.44 / 47.5% solved**. Estimability was never the issue — *information* was.
- *For your partner*, only CLAMP works. A randomised value on your private node swaps one
  invisible variance source for another; the partner still sees a latent common cause.
  Rescue rate is 0.000 at `intervene_scale` 2.0 and 1.0, and rises only as the scale goes
  to zero **[M]**.

So **clamping is a genuine sacrifice**: it removes a confounder for your partner at the cost
of a much weaker experiment for yourself. That trade-off is the coordination problem, and it
is measured rather than assumed. *(This reverses my own earlier proposal to delete the mode
split; I was wrong and the measurement above is why.)*

### Authority

An agent may intervene on its own private nodes **and on the shared nodes `X`**. Shared
authority over `X` is deliberate — it is the surface on which coordination and contention
actually occur, and removing it removes the problem.

### Turn structure

**Simultaneous** **[U8]**. Both agents choose an action; both are applied to the same system;
one batch of `n_int` samples is drawn under the joint regime. If both target the same node,
the more restrictive assignment wins (clamp beats vary).

**Budgets are separate** **[U9]** — each agent has its own intervention budget, so one agent
cannot consume the other's capacity. `PASS` is always available.

### Information exchange

**After acting**, each agent learns which **shared** nodes were targeted, and is told nothing
about the other's private nodes **[U10]**. This enters the observation vector. Because it
arrives *after* the action, it cannot condition the current move — only future ones.

Separately, one **regime bit** per round crosses the boundary: *"I clamped something you
cannot see."* **[U13]**. This is the minimum communication that lets an agent tell a clean
batch from a confounded one without revealing which node, how many, or any value. Measured:
pooled rows give 0.000 identification under confounding; regime-separated gives 0.162;
regime plus the agent's own interventions inside the clean regime gives 1.000 **[M]**.

**Open with the supervisor.** Whether the regime bit is admissible under Mirco's federation
constraint ("no information regarding private variables") is his call. It reveals *that* a
hidden variable was fixed, never *which* or *what value*. It requires no central server and
no shared parameters. Recorded here as the minimal viable disclosure and flagged as the
single point in the design most needing his input.

## 3. Belief

**Exact enumeration over DAGs is abandoned** **[U11]**. It dies at `k ≈ 6`, and the current
two-agent code enumerates 543 graphs per window.

**Each agent maintains its belief with the subset DP** **[U11, ruled]** — the order-space
Robinson sink recurrence in signed log space, already built and verified for the single-agent
case. It gives *exact* edge marginals and the exact partition function in `O(k·2^k)`, scaling
to windows of `k ≈ 15–20` with no approximation. Coarser approximations (independent-edge
belief, MCMC, variational) are held in reserve for beyond that, not used now.

Note the scaling axis this fixes: **the window, not the world**. Federation keeps `k` small
even as `d` grows, which is the structural reason this is the right tool here.

**The policy observes edge marginals**, `k(k−1)` numbers, plus its remaining budget and the
disclosure bits. It does *not* observe a posterior vector over graphs. This was already true;
the exact-enumeration problem was in the *inference*, not the observation.

### Confounding

Because `Z_B` is hidden from A but causally active, A's window is **not** a DAG — it is a
latent projection with possible bidirected edges.

**Proved and exhaustively verified**: bidirected edges can only ever appear between two
**shared** nodes. Never private–private, never private–shared. Verified exhaustively at
`(1,1,2)` and `(1,1,3)` and by sampling at `(2,2,2)`; zero violations. This is what makes
explicit modelling tractable at all.

Consequently the hypothesis space is `(DAG on the window) × (subset S of confounded shared
pairs)`, with at most `|X|(|X|−1)/2` candidate bidirected edges. Cost is
`DP(k) × 2^(|X| choose 2)` — **exponential in the shared-set size, not the window size**.
At `|X| = 3` that is 8 DP passes; at `|X| = 4`, 64. Since `X` is the federation boundary and
is small by design, this is the trade we accept **[MY CALL]**.

### The valley, re-measured at `n_obs = 100` — my hypothesis was falsified **[M]**

I predicted the *valley* that motivates the explicit-confounding rule would disappear at low
`n_obs`, on the reasoning that it came from discarding thousands of observational rows. **It
does not. It gets substantially worse.** 300 episodes, `n_obs=100`, `n_int=100`, identification
on unconfounded episodes as the partner's clamp probability rises:

| rule | p=0.00 | 0.25 | 0.50 | 1.00 | verdict | confounded payoff |
|---|---|---|---|---|---|---|
| pooled | 0.790 | 0.775 | 0.786 | 0.815 | no valley | +0.138 |
| subset | 0.790 | **0.358** | 0.624 | 0.900 | **VALLEY (−0.432)** | **+0.828** |
| joint | 0.790 | 0.834 | 0.849 | 0.849 | no valley | +0.103 |
| joint_conf | 0.221 | 0.723 | 0.875 | 0.945 | no valley | +0.621 |

The valley at `n_obs=2000` was −0.094; at `n_obs=100` it is **−0.432**. My causal story was
wrong. The valley is not caused by *discarding* many rows — it is caused by the clean subset
being *small and noisy* at low clamp probability. Lowering `n_obs` makes every rule
data-poorer, so the clean subset gets relatively worse, not better.

**Therefore the explicit-confounding rule (`JOINT_CONF`) is retained.** It is the only rule
that is both valley-free and pays off under confounding. Its one cost is deliberate and
documented: it starts *lower* than the others at p=0 (0.221 vs 0.790), because without any
clean data an agent genuinely cannot tell a confounded pair from a directed edge, and the
rule refuses to pretend otherwise.

## 4. Objective

**Reward is a shared scalar** **[U15]**: `+1` on the team target, minus a per-agent step cost.
Not CTDE — no observations, parameters, gradients, or critics are shared. A selfish agent has
no reason to clamp for its partner, so a per-agent reward makes the target behaviour strictly
dominated; the shared scalar is the minimum that makes cooperation rational.

**Training is independent PPO.** No centralised critic **[supervisor constraint]**.

### Definition of success **[U14]**

Three conditions, all required:

1. **Private recovery** — each agent recovers its own private substructure as a DAG.
2. **Shared recovery** — each agent recovers the shared structure to **CPDAG** resolution
   (orientation within an equivalence class is not required where it is not identifiable).
3. **Global consistency** — the union of the two agents' recovered structures resolves to
   the true global graph.

Condition 3 is a real constraint, not a formality: two locally-correct agents that agree
edge-for-edge on `X` can still union into a **cyclic** graph. Acyclicity of the union is
therefore an explicit check, not an afterthought **[MY CALL]**.

Condition 3 is also *the test of whether communication worked*. Without the regime bit, the
private structures stay correct but the shared orientations cannot be disambiguated under
confounding, and the union fails.

## 5. Baselines **[U16]**

- **Random** — the minimum bar. Must include a random policy that *clamps*, not only one that
  varies, or the comparison is rigged.
- **Greedy / myopic oracle** — one-step expected information gain per agent.
- **No-intervention control**.

Stated in advance: the greedy oracle is a single-step optimum, not a sequential one, and
expected information gain is not adaptively submodular, so Golovin & Krause's `(1−1/e)`
guarantee does not apply. That is precisely why beating it is possible.

⚠️ **Carried-over warning** **[M]**: performance belongs to the `(policy, belief-rule)`
**pair**. A `joint_conf`-trained policy scored under `subset` collapses to 0.000 on
confounded episodes and ~0.02 overall — below random's 0.370. Greedy drops 0.542 → 0.190 on
the same switch. Every reported comparison must hold the belief rule fixed.

## 6. Deliberately dropped from the old design

- The `|X|²` **ancestral-order** disclosure of `MA_DESIGN.md` §5. Measured at ~0.005 bits per
  bit — a correctness guard, not an enabler **[M]**.
- **Exact posterior enumeration** over the window **[U11]**.
- **Topology T3** (exposed nodes with no private parents) — it removes latent confounding by
  deleting the boundary the entire design depends on.
- The **§3 confounding rates** in `MA_DESIGN.md`, which overcount by ~3× (they count any
  hidden common *source*; the excess pairs are all ancestrally related) **[M]**.
- **Clamp selectivity** as a near-term goal. Three attempts failed; pricing clamping only
  taxes it (0.05) or kills all action (0.15). An agent cannot observe whether its partner is
  confounded, so "clamp only when needed" may be unattainable in this formulation **[M]**.

## 7. Known open problems

1. **Seed instability** — 1 in 10 seeds collapses into passing immediately; sd 0.154 on a
   median of 0.312. This is the biggest threat to any two-agent claim and is the first thing
   to fix.
2. **The MH sampler is under-mixed** — acceptance 5.8%, effective support ~172 graphs. The
   current `burn_in=50_000, thin=50` is an explicit **stopgap**. A principled fix (partition
   MCMC, or exact DP sampling) needs academic verification before we invest in it. The subset
   DP removes the need for it entirely up to `k ≈ 15`, which is the real answer here.
3. **Edge marginals hide joint error** — ~10% of posterior mass can sit on a wrong skeleton
   while every marginal looks correct. The §4 success criteria must be evaluated on the joint
   object, not the marginals.
4. **The `n_int` sweep** has not been run **[U6]**.
5. **`JOINT_CONF` costs a factor of `2^(|X| choose 2)`** on top of the DP. Fine at `|X| ≤ 4`;
   it is the binding constraint on growing the shared boundary.
