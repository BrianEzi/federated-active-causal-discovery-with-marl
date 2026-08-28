# Action plan to freeze — evening of 28 Aug to Monday 31 Aug

Written against [`OBJECTIVE.md`](OBJECTIVE.md). Every item names what it produces, what it
costs, and what would make me abandon it. Items are ordered so that anything cut from the
bottom leaves a coherent thesis.

**Division:** the second agent owns ladder consolidation, seeds, return normalisation and
figures. This plan is the other track. Shared files are flagged per item.

---

## P1 — The SHD-vs-budget curve *(tonight, ~2h, eval only)*

**Why it is first.** It is the headline figure of the thesis story and **it does not exist** —
`grep` finds no SHD anywhere in the repo. It also fixes three separate problems at once:

- **The metric floor.** All-or-nothing identification reads 0.000 for every arm under sampled
  evidence, random included. A metric that cannot separate random from a trained policy is not
  evidence. SHD has a gradient everywhere.
- **External validity.** SHD is what the causal discovery literature reports. Nothing in this
  project has ever been expressed in a metric an outside reader already trusts.
- **The penalty framing.** Budget on the x-axis is the "how many experiments did you need"
  question, which is the one the application actually asks.

**What it is.** Per pair, the belief's mark against the true mark, over `{NONE, FWD, BACK, BI}`
(`cb/versionspace.py:42`). An unsettled pair counts as an error, so the curve starts high and
falls as experiments accumulate. This is close to the complement of `credit_fraction`, so most
of the machinery exists — it needs to be a **count against budget**, recorded per round, not a
single end-of-episode fraction.

**Deliverable.** `ma/shd.py` plus a curve: SHD (y) against experiments consumed (x), one line
per arm, on identical episodes. Both tracks use it.

**Abandon if:** it cannot separate random from greedy. That would mean the metric is as
degenerate as the one it replaces.

**Shared files:** new module only. No edits to `ma/env.py`, `ma/policy.py`, `ma/baselines.py`.

## P2 — Debug the attribution scorer *(tonight, ~1h)*

**The failed control.** Under ORACLE evidence `score_groups` reports `wrong` at 0.075–0.113,
where its own docstring guarantees `wrong` cannot occur ("the truth never leaves the candidate
set"). The control fired, so no attribution-under-noise number is trustworthy yet.

**Leading hypothesis, to be confirmed or killed:** `group_frequency.get(group, 0.0)` returns 0
for a true group **absent from the belief's enumeration**, and at `bar=1.0` that scores
`1 - 0 >= 1.0` → WRONG. If so, "wrong" currently means *"no candidate matches this true
group"* — a canonicalisation mismatch between `groups_from_dag` and the belief's own grouping —
not a false attribution. Supporting evidence: the rate is **identical across greedy, probe and
random**, which a policy-driven error would not be.

**Deliverable.** Either a fix, or a documented statement that `wrong` means something other
than false attribution — in which case every attribution number in the repo needs its
interpretation restated.

**Abandon if:** the cause is deeper than canonicalisation. Then attribution stays
formulation-level and is written up as such, which is an acceptable outcome.

## P3 — Heterogeneous private sets *(Friday, ~2h)*

**The hole.** `federated_topology(n_agents, private_size, n_shared)` gives every agent the
**same** private_size — the name is derived as `T_4agent_2each_3shared`. So every result in
this repo is on homogeneous windows. In a real consortium that is the unrealistic assumption,
not a detail.

**Why it should already work.** `PortableRoleActorCritic` has no learned width that depends on
k or n: the node encoder maps one node's features, the edge encoder one pair, and the action
head is `Linear(hidden, 1)` applied **per node** — so the number of logits varies while the
parameters do not. Deep-Sets pooling over partners does the same for agent count. The
architecture supports unequal windows; the topology constructor simply cannot express them.

**Why it matters beyond realism.** Identical parameter shapes across sites of different sizes
is exactly what makes **gradient averaging well-defined**. A fixed-width action head would make
FedAvg across heterogeneous sites impossible without padding hacks. So this is the prerequisite
for the federated claim, not a side quest.

**Deliverable.** `federated_topology` accepting a per-agent sequence of private sizes, tests,
and one transfer measurement: a policy trained on homogeneous windows, evaluated on
heterogeneous ones.

**Abandon if:** the belief backends assume equal window sizes somewhere structural.

**Shared files:** `ma/topology.py`. Flag to the other agent before editing.

## P4 — False attribution under noise *(Saturday, ~3h, requires P2)*

The number nothing in the project has, and the one a consortium would be judged on: how often
is a latent attributed to the **wrong** site. Under oracle it is impossible by construction;
under sampled evidence a Type I error in `estimated_moved` can prune the truth, so it becomes
reachable. Requires P2 to be trustworthy.

**Abandon if:** P2 shows `wrong` is not measuring false attribution.

## P5 — The convergence figure *(Saturday, if P1-P4 land)*

The second agent's noise dial already has three points (+0.053 / +0.100 / +0.123 at
n_int 100 / 1,000 / 4,000) showing the margin over greedy **growing** with data quality. Adding
the oracle point makes the argument the thesis wants: the deterministic case is the **limit**,
not a toy, and the sampled setting converges to it.

---

## Cut, explicitly

FedAvg, GRPO in any form, the alpha-blend, factored attribution, Erdos-Renyi, cost-heterogeneous
experiments. Each is defensible; none is reachable **and validatable** in the time.

## Standing rules for every item

- **MI gate before any learned number is quoted.** It is a floor, not a quality measure, and it
  must never be compared across arms with different objectives.
- **Matched pairs on identical episodes**, and state the seed count in the table.
- **State the evidence regime** in every result file. The attribution `*_scored.json` files do
  not record `vs_evidence`, which is why nobody could tell they were all oracle.
- **A control that cannot fail is not a control.** Phase 0's omniscient benchmark and its
  censoring-dominated metric are the example to avoid repeating.
