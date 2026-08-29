# The rung plan — from a defensible thesis to a publication-ready result

Written 29 Aug 2026. Four rungs, each a COMPLETE story on its own. Cut from the bottom and
what remains is still a thesis; that is the property the ordering is chosen for.

The organising fact: **k ≤ 5 is the only regime where every claim is checkable against an
exact answer.** `ma/baselines.py::MAX_ENUMERATED_K` caps the exact-DP greedy oracle at 5,
`cb/versionspace.py` is usable to 6, and `scripts/vs_evaluate.py` returns an exact optimum
there rather than the bound it degrades to on the factored path. So the base rung is not a
toy — it is the only place the headline numbers can be *verified* rather than compared.

---

## Rung 1 — EXACT. Everything is checkable.

**Config.** 3 agents, 2 private each + 3 shared → k = 5, d = 9. Oracle evidence,
round-robin, scale-free, `episode_mix=confounded`, `normalise_returns=True`.

**Why these numbers.** k=5 keeps the enumerated greedy oracle and the exact optimum.
2 private per agent is the minimum that makes the privacy claim non-empty — at one private
node, naming the agent names the variable (`cb/attribution.py`). 3 agents keeps the
attribution enumeration at 3 owner-sets rather than 7.

**What it delivers**

| thesis result | how Rung 1 evidences it |
|---|---|
| 1. Identification reduces to a forced set cover | learned vs greedy vs **exact optimum**, regret in rounds, not a bound |
| 3. Cooperation gap / decentralisation gap | `gnn_portable` vs `gnn_solo` vs `gnn_hybrid`, all against the same exact optimum |
| **Novel: federated latent attribution** | the `attributed` backend runs here and nowhere above k=6 |

**Backends available:** all five. **Arms:** learned (×3 archs), greedy, greedy_partitioned,
probe_then_work, random_vary, random_clamp, pass, exact optimum.

**Missing before it can run: nothing.** This is why it goes first.

**Grade framing:** on its own this is a complete, verified, honest thesis. Everything above
is upside.

## Rung 2 — SCALE. One axis, out to where nothing else reaches.

**Config.** The window ladder as built: k = 4, 8, 12, 20, 30, factored backend, oracle
evidence, 3 seeds each. Already run; re-scored under the corrected metrics.

**What it adds.** The scaling claim, and the forced-cover characterisation used
*predictively*: required cover measured at 0.757k (k=4) to 0.542k (k=30), so the budget a
window needs is not fitted after the fact.

**What it costs.** Regret becomes a **bound**, not an optimum. Attribution is unavailable —
the enumeration wall is ~5 confounded pairs at any agent count.

**Missing**
- **A5, the oracle-cover planner.** Above k=5 there is no optimal arm at all, so every
  comparison is learned vs heuristic. A5 intervenes on exactly the forced set and restores
  "vs optimal". Eval-only, no training.
- **B8, factored attribution.** The only way attribution reaches this rung. Costed at
  ~4–6 h build; compute is free (3,045 numbers at k=30 against 282M candidates enumerated).
- **Re-score everything.** Every SHD number predates the argmax + de-duplication defaults,
  and the w08 flip (greedy winning → learner winning) shows this changes verdicts.

## Rung 3 — REALISM. The idealisation removed.

**Config.** Rungs 1 and 2 repeated with `vs_evidence=sampled`, retrained rather than
transferred.

**What it adds.** Thesis result 2 — RL beats greedy at fixed budget under sampled inference,
converging as n → ∞. **This is the only result with a large compute bill.**

**Why a retrain and not a transfer.** Measured: an oracle-trained policy evaluated under
sampled evidence LOSES to greedy on soft SHD and on the error component. The mechanism is in
the repeat rate — greedy 0.247/0.331 against the learner's 0.110/0.138. Under oracle a
repeat is wasted; under sampling it is how you buy power. The learner correctly learned a
rule that is wrong in the new regime, so the transfer test cannot answer the question.

**Missing:** the retrains. w04 × 3 seeds done. w08/w12 in flight. w20/w30 need the cluster.

## Rung 4 — GENERAL. Publication-ready.

Each item is independent; take them in any order as time allows.

| item | why it matters | cost |
|---|---|---|
| **E5 Erdős–Rényi** | every result is scale-free; zero ER runs exist. One rung bounds the graph-model dependence | config + retrain |
| **C1 heterogeneous private sets** | `federated_topology` cannot express unequal sites. Real consortia are unequal, and identical parameter shapes across UNEQUAL sites is what makes the portable head non-trivial | topology edit + retrain |
| **C3 solo with n× episodes** | **promoted from the extension list.** At 2 agents the decentralisation gap is −0.017 — it vanishes. That is the 1/n data signature, so C3 decides whether decentralisation costs DATA or CAPABILITY. It answers Mirco directly | 4 h compute |
| **solo + `normalise_returns` at a06/a08** | solo entropies there are ~1.89 against a max of 1.946 — those nets **never trained**, so the gap at 6 and 8 agents is currently unquotable | cheap retrain |
| E4 FCI/PC comparison | no comparison to a standard discovery algorithm anywhere | 3 h |

---

## Working backwards: what to build, in order

**Now — nothing blocks Rung 1. Run it.**
The only pre-flight is `scripts/preflight_metrics.py`, which exists and is green.

**Tonight's builds, in dependency order**

1. **A5 oracle-cover planner** — unblocks Rung 2's "vs optimal", and is eval-only so it also
   retro-fits every checkpoint already trained. Highest value per hour on the list.
2. **Re-score under corrected defaults** — eval-only, and it changes verdicts.
3. **B8 factored attribution** — the only route for attribution above k=6, and the strongest
   remaining novelty. Gated on B1, which is now complete.
4. **solo + `normalise_returns`** — one config field, makes the decentralisation gap quotable.

**Explicitly not tonight:** C2 FedAvg and D5 GRPO (cut), and the Rung 4 items, which are
independent and can be taken whenever compute frees up.

## The rule that keeps this honest

Every rung is reported with the MI gate beside it, the evidence mode named, and the
evaluation policy named. Three fields, always. Every wrong claim on this project so far has
come from one of the three being left implicit.
