# Backends, engines and the flow of an experiment

Written 29 Aug 2026. Reference map, not a plan. Every count below is a census of
`results/` at that date, not an aspiration.

---

## 1. Belief backends — what an agent's window belief actually IS

Declared in [`ma/env.py`](../ma/env.py) as `BACKENDS`. The environment talks to all of them
through one call, `edge_marginals`, so they are interchangeable at the env boundary. They
differ in what identification MEANS and in how far they scale.

| backend | module | belief representation | statistics? | k limit | runs |
|---|---|---|---|---|---|
| `exact` | `crosscheck/posterior.py` | full Bayesian posterior over all DAGs, enumerated | yes | d ≤ 4 (543 DAGs) | 0 |
| `constraint` | `cb/backend.py` | bootstrap over CI-test skeleton + orientation | **yes — real CI tests** | ~6 | 12 |
| `version_space` | `cb/versionspace.py` | SET of whole-window MAGs still consistent | no | ≤ 6 (3^edges) | 36 |
| `attributed` | `cb/attribution.py` | version space over (structure, **who owns each latent**) | no | ≤ 6 | 6 |
| `factored` | `cb/factored.py` | one small version space **per pair**, O(k²) | no | 30+ | **86** |

`CLAIM_BACKENDS = (constraint, version_space, attributed, factored)` — these four expose
bootstrap-shaped frequency matrices (`adjacency`, `directed`, `bidirected`), which is what
`cb/claims.py`, the greedy baseline and `scripts/shd.py` all read.

**What each one is FOR.**
- `version_space` — the deterministic idealisation. Answers "can agents learn to divide
  experiments" with a computable optimum, in milliseconds per episode. Not a finite-data claim.
- `attributed` — the same, plus the only channel by which one agent can help another:
  whose hidden variable explains a confounded pair. **This is the novel contribution.**
- `factored` — scale. Gives up the joint constraints (ancestrality, maximality), so it is
  CONSERVATIVE: it stays unsure where enumeration would have settled, and **never settles
  wrongly**. That property is what makes oracle-evidence SHD a pure ambiguity count.
- `constraint` — the realistic path. Real CI tests on real samples. Barely used (12 runs).

## 2. Evidence mode — `vs_evidence`, an axis ORTHOGONAL to the backend

| mode | how a pair is pruned | runs |
|---|---|---|
| `oracle` | by the **true ancestry** — the data is bypassed entirely | **96** |
| `sampled` | by `estimated_reveal` on the actual samples, with a power gate on the negative direction | 8 |

**The scope limit that matters, and it is easy to miss.** `FactoredBackend.reset_marks`
seeds every pair from `self.truth`: a true non-edge starts as `frozenset({NONE})`. So the
**skeleton is oracle in both modes** — only ORIENTATION is ever estimated. Sampled evidence
therefore admits orientation errors but never adjacency errors.

**96 of 104 runs are oracle.** That is the single biggest gap in the evidence base.

## 3. Policy architectures — `policy_arch`

Dispatched in [`ma/policy.py:536`](../ma/policy.py). All four GNN arms share
`PortableRoleActorCritic`: it scores ONE variable at a time from features that never
reference how many variables exist, so the same weights are meaningful at any k or n.

| arch | networks | gradients cross agents? | what it isolates |
|---|---|---|---|
| `mlp` | one flat net | — | the pre-GNN baseline |
| `gnn` | one shared net, non-portable | yes | — |
| `gnn_portable` | **one net shared by all** | yes | parameter sharing, decentralised execution. **The ladder default** |
| `gnn_solo` | one complete net PER agent | **no** | portability from ARCHITECTURE, not weight sharing — fully decentralised |
| `gnn_hybrid` | shared trunk, per-agent heads | trunk only | how to READ a belief (universal) vs what to DO (per-role) |

`gnn_portable` vs `gnn_solo` is the decentralisation spectrum. There is **no FedAvg arm** —
the missing middle, and Mirco's objection.

## 4. Baselines — [`ma/baselines.py`](../ma/baselines.py)

| name | rule |
|---|---|
| `pass` | never acts. What does observational data alone reach? |
| `random_vary` | uniform over targets, VARY only |
| `random_clamp` | uniform over (target, mode). The primary floor |
| `greedy` (`UncertaintyGreedyAgent`) | argmax over nodes of "unsure claims touching me". **Truth-free** |
| `greedy_partitioned` | the same, with the shared surface split by positional convention + least-touched tie-break |
| `GreedyAgent` | reads the exact DP score tables. Enumerates, so k ≤ 5 |
| `probe_then_work` | probes private nodes on a fixed schedule (attributed backend only) |
| `forced_clamp` | always clamps its own private node |

**Not built:** the oracle-cover planner (slate A5) — an arm that intervenes on exactly the
forced set, which would turn every comparison into learned vs heuristic vs **optimal**.

## 5. Reference engines — `crosscheck/`

Frozen, never production. Exists to check `ma/` and `cb/` against an exact computation:
`posterior.py` (enumerated Bayesian posterior), `dp.py` (the same without enumerating),
`belief_dp.py` (subset DP per window), `score*.py` (linear-Gaussian structure scores).

## 6. The flow of one experiment

```
  ma/topology.py        federated_topology(n_agents, private_size, n_shared)
       |                fixed equal blocks + a shared set; edge mask = "some agent
       |                sees both endpoints", so cross-private edges cannot exist
       v
  Topology.sample_dag   scale-free (preferential attachment, m=2) or Erdos-Renyi
       |
       v
  ma/scm.py             linear-Gaussian SCM; n_obs observational rows per episode
       |
       v
  ma/env.py             per agent: window = own private block + shared set
   TwoAgentEnv          each round every agent picks (node, vary|clamp) within its
       |                AUTHORITY = own private + all shared. n_int rows per round.
       |                Samples are SHARED; beliefs are not.
       v
  belief backend        edge_marginals(...) -> frequency matrices
   (section 1)          under vs_evidence=oracle this reads the TRUE ancestry
       |
       v
  cb/claims.py          score_window -> right / unsure / wrong per claim
       |                identification = every required claim right, ZERO tolerance
       v
  reward               joint or per-agent; optional difference_reward, normalise_returns
       |
       v
  ma/policy.py          IndependentPPO, one of the five archs
   scripts/ma_train.py  -> results/<dir>/<name>_s<seed>.json  + .pt checkpoint
```

**Then, all evaluation-only, all rebuilding the env from the run's OWN config block:**

| script | question |
|---|---|
| `scripts/rescore_from_config.py` | re-score against greedy at the bar the task GRADES on |
| `scripts/shd.py` | soft/hard structural Hamming distance per window |
| `scripts/shd_diagnose.py` | WHY the SHD says what it says — 6 checks, incl. `--evidence` override |
| `scripts/mi_gate.py` | **the gate**: did this policy train at all? I(S;A)/H |
| `scripts/budget_curve.py` | success against budget / required cover |
| `scripts/required_cover.py` | the forced set size, measured |
| `scripts/vs_evaluate.py` | exact optimum and regret (enumerable backends only) |
| `scripts/transfer_eval.py` | train on one regime, evaluate on another |

**Two rules that exist because they were violated.** Rebuild the env from the result's own
`config` block, never from flags retyped by hand. And run the MI gate before quoting any
learned number — a rung that never trained is not a negative result.

## 7. Census — where the evidence actually is

- **86 of 104 runs are `factored`**, and **96 of 104 are `oracle`**. The scaling ladder is
  one backend in one evidence mode.
- `constraint` — the only backend that touches real CI tests — has **12 runs**.
- `attributed` — the novel contribution — has **6 runs**, and its scorer is unverified
  (slate B1).
- Graph model is scale-free everywhere. **Zero Erdos-Renyi runs.**
