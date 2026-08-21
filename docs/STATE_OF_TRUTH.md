# State of truth — 21 August 2026

**Read this before citing any number from this repository.**

The logs in `docs/logs/` are chronological and contain claims that were later withdrawn. This
file is the supersession layer: it exists so that a retracted claim cannot be resurrected by
someone reading the logs forwards. Every entry carries an evidence pointer.

Experiment freeze: **31 August 2026**. Dissertation due **8 September 2026**.

---

## Established

Claims we are willing to defend, with where the evidence lives.

### Machinery

| claim | evidence |
|---|---|
| The subset DP reproduces the enumerated posterior to **1e-10** | `tests/ma/test_belief_dp.py` against `tests/fixtures/ma_reference_posteriors.npz`, generated independently by `legacy/ma_v1/env.py` |
| Exact inference extends from `d ≤ 6` (enumeration) to `d ≈ 9` (DP) | `sa/dp.py`; the arithmetic **must** be signed log space, per-node rescaling provably cannot work |
| BGe is score-equivalent, so Markov-equivalent DAGs tie **exactly** | Chickering (2002); `tests/test_score.py`. **Consequence: MAP accuracy is a meaningless metric here** — `argmax` reports floating-point ordering |
| Covered-edge-reversal closure reproduces the enumerated equivalence class exactly | max size error **0** at `d=3,4`, `scripts/sa_criterion_sweep.py` |
| Confounding is confined to the shared set — every bidirected edge has both endpoints exposed | proved and exhaustively verified, `tests/test_projection.py`. **TWO-AGENT RESULT ONLY** (see Open) |

### The two-agent result (turn-taking, 21 August)

Round-robin, shared round budget 10, `step_cost = 0`, regime bit on, 10 seeds, 150 eval
episodes. Source: `results/ma_fixed/tb_*.json`, `docs/logs/MA_BUILD_LOG.md`.

| claim | number |
|---|---|
| Learned beats random on **every** seed | 0.563 vs 0.210 (both modes), 0.553 vs 0.380 (clamp-only), 10/10 |
| Learned beats greedy by ~3.5x | 0.55 vs 0.147–0.173 |
| **The collapse is gone** | 0/10 seeds, against 5/10 under the previous rules; seed sd 0.222 → **0.039** |
| **Agents learn the altruistic move** | learned spends **82–91%** of clamps on its own private node; chance is **25%**; greedy is **19–24%** |
| Learned acts **less** and scores more | 6.1 interventions vs 9.2 for random and greedy |
| Clamping beats varying **per move** | a random policy restricted to clamps goes 0.210 → **0.380** |
| The observational-leak guard holds | pass-only baseline scores **0.007** |
| Connected graphs are harder | 0.49 vs 0.70; always reported split, never pooled |
| No free-riding at two agents | free-rider index 0.85–0.88 |

**Why the altruism claim survives scrutiny.** A private clamp is not *purely* altruistic — it
also yields interventional information about that node's children. What rules out the selfish
reading is greedy: an agent optimising exactly that self-information targets the node at
chance or below, and scores a third as well.

### Single-agent criterion work

| claim | evidence |
|---|---|
| GATE 1 fails at `d ≥ 5` because the environment is **starved**, not mis-scored | among singleton-MEC graphs the true DAG clears 0.7 only **40%** of the time at `d=5,6`; `scripts/sa_criterion_sweep.py` |
| The arithmetic closes exactly | predicted rate `0.081 × 0.400 = 0.030` against measured `0.025 [0.005, 0.050]` |
| The **MEC-mass criterion is disqualified** | satisfiable **without acting** in 46–96% of episodes — a criterion you can meet by sitting still cannot score a task premised on acting |
| Clamping recovers 93–98% of varying, for one's own structure | `docs/logs/SA_EXPERIMENT_LOG.md`, 2026-08-20. **Single-agent, pooled scoring** |
| Varying de-confounds nothing for a partner | rescue 0.000 at scale 2.0 and 1.0 |

---

## Retracted

**These claims appear in the logs and are wrong. Do not cite them.**

| retracted claim | why | what replaced it |
|---|---|---|
| Oracle agreement of **99.4–100%** | the metric was **93–98% vacuous** — equal-variance shortcut | MAP accuracy is meaningless for the Bayes estimator |
| "The greedy oracle never clamps" | greedy is **provably indifferent** between clamp and vary — both cut the target's incoming edges and induce the same partition, so its clamp fraction is a **coin flip** | greedy has no term for the partner at all |
| "A constant intervention cannot identify descendants' dependence" (our own docstring) | measured false | clamp recovers 93–98%; the mechanism is **pooling**, not collinearity |
| **Every two-agent number before 2026-08-21** | simultaneous action, per-agent intervention budget, `step_cost = 0.05` | the turn-taking table above |
| "1-in-10 seeds collapse, sd 0.154" (2026-08-19) | did not reproduce | 0/10 collapse, sd 0.039 |
| "Clamp-only is proven equivalent to both modes" | paired, both-modes leads on **8/10** seeds, mean +0.018, CI **[-0.005, +0.041]** | clamp-only is a **trade** costing ≤4pp for a halved action space |
| "`n_obs=100` explains the `d=6` GATE 1 failure" | it fails at `n_obs=1000` too | the criterion is unreachable at `d ≥ 5` at this sample size |
| "Turn-taking drains the clean regime of value" | wrong division of labour — orientation comes from the dirty regime, disambiguation from the clean one, and both score the same structure | nothing is lost structurally |
| "Clean rows halve under turn-taking" | assumed a shared budget pool; with per-agent budgets B clamps as often as before | no reduction |
| GATE 2: "greedy fails because agents collide" | collisions cannot occur under turn-taking, yet greedy still loses to random | greedy has no term for the partner |

---

## Open

| question | why it matters | what would settle it |
|---|---|---|
| **Does confinement hold for `n > 2` agents?** | **Everything in the scaling plan rests on it.** With overlapping shared sets, a third agent's private node may confound a pair visible to a fourth. If it fails, the belief needs full MAG machinery and the score stops decomposing | a structural enumeration, like the original — cheap |
| Does the **regime bit** earn its place? | every current number has disclosure ON; without the ablation we cannot claim the federation channel matters | the no-bit arm, ~30 min |
| Is the learned policy better than a **proper** decentralised baseline? | our greedy conditions on nothing; SGA is the literature's version and turn-taking makes it implementable | implement sequential greedy and joint greedy |
| What `n_obs` does `d ≥ 5` need? | blocks any single-agent scaling claim | same script, sweep `n_obs` |
| `prior_p = f(d)` — which threshold? | at `d=30`, fixed `p=0.5` gives expected degree 14.5 against a literature norm of 2–4. **Connectivity needs `ln(d)/d`, percolation only `1/d`** — and we want connected graphs | sweep, and state which threshold we chose and why |
| Per-block confounding subsets `S_r` | needed the moment an agent has more than one private node; the environment currently **refuses** such topologies rather than scoring them wrong | cost is `R × 2^{\|S\|}` — the per-block log-scores add |
| Is the signalling channel admissible? | provisional; removable with `disclose_signals=False` | supervisor confirmation |
| The `d=6` "beats greedy" result | `gap_closed` 1.23 at budget 2, 15/15 non-degenerate — but scored on the mis-specified criterion | **needs a full retrain**: no checkpoints were saved and only aggregates were logged. 25.7 h of cluster compute is unrecoverable for that reason |

---

## Standing lessons

Each was paid for.

1. **Log the raw quantity, never the verdict.** Storing `identified: false` instead of the
   posterior mass turned a criterion change into a 25.7-hour retrain.
2. **Save the policies.** Ten two-agent seeds were once evaluated and discarded.
3. **Test that a metric can be EARNED, per regime.** 529 tests passed on a metric that was
   structurally unearnable on confounded episodes.
4. **Verify on representative data.** Subset-DP passed 29 tests on `rng.normal`, then returned
   `Z=0` on the first real episode.
5. **Verify directly, not through a consumer.** A broken sampler looked like a mixing problem
   for three rounds because it was measured through the oracle.
6. **Write the spec before the code.** Four bugs on 20–21 August were all silent design
   decisions made mid-implementation, each caught only by an expensive grid.
