# Parameters — measured, derived, or asserted

Every parameter that can change a headline number, with **how its value was arrived at**.

This document exists to make a viva question land softly. "Why 0.7?" has three possible
honest answers — *we measured it*, *it follows from something*, or *we picked it and never
checked* — and the third is only damaging if we don't know which one it is.

- **MEASURED** — a sweep or experiment chose the value; the evidence is named
- **DERIVED** — follows from a definition, a proof, or a cited result
- **ASSERTED** — chosen by judgement and never swept. **These are the exposure.**

Defaults below are read from the code as of 21 August 2026.

---

## Two-agent environment (`ma/env.py`, `MAConfig`)

| parameter | default | status | basis |
|---|---|---|---|
| `n_obs` | 1000 | **MEASURED** | at 100 the posterior *never* reaches the 0.7 threshold — best of 150 episodes was 0.579, so the environment was harder than its own success criterion allowed. 1000 leaves ~94% of episodes needing interventions |
| `n_int` | 100 | **ASSERTED** | never justified, never swept. Interacts with the round budget: it sets how much evidence one clamp carries, which is the open question "can an agent locate where confounding was removed" |
| `budget` | 10 rounds | **ASSERTED**, semantics DERIVED | a shared pool of rounds follows from internalising free-riding (`TURN_BUDGET_SPEC` §2); the *value* 10 is a judgement call |
| `turn_order` | `simultaneous` | **DERIVED** | default kept only so pre-21-August commands reproduce. Turn-taking is the supervisor's directive and is opted into explicitly |
| `action_modes` | both | **MEASURED**, and the default is stale | clamp-only costs ≤4pp (paired, CI `[-0.005, +0.041]`) and halves the action space. Adopted as a **trade** — the default should follow |
| `identify_threshold` | 0.7 | **ASSERTED, and known not to scale** | inherited from the single-agent case. Measured 21 August: among graphs identifiable without acting, the true DAG clears 0.7 only **40%** of the time at `d=5,6`. **The most exposed parameter in this table** |
| `prior_p` | 0.5 | **ASSERTED, and known to be wrong at scale** | at `d=30` gives expected degree **14.5** against a literature norm of 2–4. Must become `f(d)`. Note the two thresholds differ: percolation is `1/d`, but *connectivity* — which is what we want — needs `ln(d)/d` |
| `intervene_scale` | 2.0 | **ASSERTED** | only affects `vary`, which clamp-only removes |
| `score_rule` | `joint_conf` | **MEASURED** | the four rules were compared; `pooled` cannot identify a confounded agent at all, `subset` creates a valley the learner cannot cross |
| `disclose_regime` | `False` | **ASSERTED**, and every reported number sets it `True` | the no-bit arm was the baseline. **The ablation has not been run**, so we cannot yet claim the bit earns its place |
| `disclose_shared_targets` | `True` | **DERIVED** | shared columns are visible to both agents, so a partner could infer the target from the data anyway |
| `disclose_signals` | `True` | **ASSERTED, provisional** | pending supervisor confirmation; removable with one flag |
| `step_cost` | 0.0 | **MEASURED** | at 0.05 a random-level policy has expected value **−0.255** against 0.000 for passing, so passing was optimal and every "collapse" was correct behaviour. **Coupled to the absence of voluntary termination** — changing one alone re-opens the collapse |
| `reward_criterion` | `u14` | **DERIVED** | forced by score equivalence: exact-DAG accuracy demands a guess between provably indistinguishable graphs |

## PPO (`ma/policy.py`, `PPOConfig`)

| parameter | default | status |
|---|---|---|
| `hidden` 128, `lr` 3e-4, `clip` 0.2, `gae_lambda` 0.95, `value_coef` 0.5, `epochs` 4 | — | **ASSERTED** — standard PPO values, never swept in this project |
| `gamma` | 0.99 | **DERIVED** | with a terminal +1 and no step cost, discounting is the *only* speed pressure. An explicit turn penalty was considered and rejected as near-equivalent under a fixed budget |
| `entropy_coef` | 0.01 | **ASSERTED** | exploration matters here — collapse was a live failure mode — but the value was never swept |
| `episodes_per_update` | 16 | **ASSERTED** | |
| `total_episodes` | 4000 | **ASSERTED**; reported runs use 2000 | no convergence study; 2000 was chosen for wall-clock |
| `potential_shaping` | 0.0 | **DERIVED** | off. Potential-based shaping is policy-invariant (Ng, Harada & Russell 1999), so it could only affect speed, and the reward is already a clean shortest-path objective |

## Topology

| parameter | value | status |
|---|---|---|
| `T1_1_1_3` — 1 private each, 3 shared | in use | **DERIVED** — `\|X\| ≥ 2` is required for confounding to be possible at all; cross-private edges are forbidden because no agent could ever observe one |
| multi-private topologies | **refused** | **DERIVED** — a single clamp leaves a block *partially* clean, which the regime rules would score as confounding-free. The environment raises rather than scoring wrong data |

---

## The honest summary

**The three parameters most likely to be challenged, in order:**

1. **`identify_threshold = 0.7`** — asserted, inherited, and *measured not to scale*. We know
   the failure mode and have a proposed fix (split GATE 1 into a leak check and a power
   check, and raise `n_obs` with `d`). Not yet implemented.
2. **`prior_p = 0.5`** — asserted, and demonstrably wrong at `d = 30`. Blocking for any
   scaling claim.
3. **`n_int = 100`** — asserted, never swept, and it controls how much evidence a single
   de-confounding clamp delivers. That is close to the centre of the thesis.

The PPO block is uniformly asserted. That is defensible — they are standard values and the
contribution is not the optimiser — but it should be *said*, not discovered.
