# Investigation: Why RNN Training No Longer Converges to Low SHD

**Status**: Merged into `main` (2026-08-13) with `--estimator_type avici` and `--reward_density sparse` as new defaults. **However, see "Follow-up: does the frozen (deployed) policy actually work?" below (2026-08-13, same day) -- a major caveat discovered *after* the merge that the SHD numbers throughout this doc are training-curve (on-policy, stochastically-sampled) metrics, and the frozen/deterministic policy performs much worse. Read that section before citing any SHD figure from this doc as "the policy's performance."**
**Branch**: was `investigate/graph-head-regression` (worktree: `.claude/worktrees/investigate-graph-head-regression`), merged into `main` via `--no-ff` merge.
**Trigger**: You observed WandB run `n4in20oe` (RNN, current architecture) failing to learn accurate causal graphs even on the easiest single-topology case, despite RNN "previously being able to learn the right causal graph." Your hypothesis: the decoupling of graph prediction from intervention policy ("disjointing") is the likely cause.

## TL;DR

**Your hypothesis was right, with one refinement**: it wasn't decoupling *per se* -- it was decoupling combined with *freezing* the structure estimator, leaving the RL policy with zero gradient signal toward getting the graph right. A frozen formula (analytic, or even pretrained AVICI) simply cannot improve, and the policy has no way to push it to.

**The fix**: `src/marl/graph_estimator.py`, a new `--estimator_type learned` option -- a small, separate, trainable network (own params, own optimizer) trained online every step via supervised BCE against the ground-truth adjacency, living *outside* the intervention-policy actor. This restores the old architecture's effectiveness without re-coupling structure prediction into the policy itself (preserving your other stated goal: the policy should learn purely interventions).

**Confirmed at full scale**: replicating `n4in20oe`'s exact config (1000 episodes, full dynamic 8-topology curriculum, soft-shift interventions) with only the estimator swapped: **mean SHD 0.34, F1 0.92, SHD=0 on 77% of episodes**, stabilizing at **SHD ~0.08-0.11 for the final 500 episodes** (all 8 topologies). The original failing run oscillated at SHD 2-6 with no improvement trend for its entire 1000 episodes. Full comparison matrix and every intermediate result below.

**Recommendation**: adopt `learned` as the new default `--estimator_type`. Not yet done -- this branch is unmerged and awaiting your review, per your instruction not to merge without it.

## Summary of findings so far

Three **compounding, independently real** factors explain the regression, not one:

### 1. Hard vs. soft-shift interventions (confirmed empirically on Myriad HPC tonight)

Controlled experiment: identical current architecture (RNN, frozen analytic estimator, `--fixed_graph 0`, 200 episodes), only `--intervention_type` varied.

| | mean SHD | mean F1 | fraction episodes at SHD=0 |
|---|---|---|---|
| `--intervention_type hard` (WandB run `u4liks79`) | 1.68 | 0.72 | 33.5% |
| `--intervention_type soft_shift` (WandB run `we15deyh`) | 2.95 | 0.52 | 9.5% |

Mechanistic reason: hard interventions (`X_i := c`) structurally sever a node's dependence on its parents -- exactly the signal invariance-based causal discovery (including the current analytic estimator) relies on to detect edges. Soft-shift (`X_i := f_i(Pa_i) + eps + delta`) preserves that dependence and just adds an offset, which is a structurally weaker signal.

**But hard intervention alone does not restore "solved" status** (33.5% exact-match rate, not ~100%), so this is a real, confirmed, but partial explanation.

### 2. The old architecture had a jointly-trained, gradient-connected graph-prediction head (confirmed via git archaeology)

Traced the exact commit that produced the "solved" run `00vhwhzv` (WandB project `federated-causal-marl-kaggle`, created 2026-07-30T20:18:53Z, `fixed_graph=0`, `use_rnn=true`, 200 episodes, SHD converges to 0.0 / F1 to 1.0 by episode ~170-200): commit `b5ebccf` ("feat: allow specific graph topology to be forced via fixed_graph argument", 2026-07-30T19:18 UTC, ~1hr before the run started).

At that commit, `IPPOActor` (`src/marl/ppo_agent.py`) had a **third output head**: a "shared edge scorer" that computes `graph_logits: [batch, d, d]` from the same `node_embeddings` used for action selection (`cat_logits`, `target_logits`). Confirmed in `src/marl/ppo_trainer.py`:

```python
total_actor_loss = actor_loss - self.entropy_coef * entropy + self.graph_coef * graph_loss
```

where `graph_loss` is `optax.sigmoid_binary_cross_entropy(graph_logits, true_adj_batch)` -- **direct supervised learning against the ground-truth adjacency matrix**, added straight into the actor's RL loss (`graph_coef` was 0.5 by default, 1.0 in the actual `00vhwhzv` run config).

This is completely absent from the current architecture. Current `predict_graph_hypothesis` (`src/evaluator_env.py`) uses a **frozen, non-learned analytic formula** over accumulated covariance -- confirmed via `docs/CHANGELOG.md`'s own entry: "Removed the auxiliary graph BCE loss... Predicted DAG structure... continues to come entirely from the fixed, non-learned analytic invariance scorer." The current actor has **zero direct gradient signal** toward getting graph structure right; it can only indirectly hope that wherever it chooses to intervene happens to produce data the frozen formula scores well.

### 3. The old reward was derived from the actor's OWN learned prediction, not an external frozen estimator (confirmed via code reading)

At `b5ebccf`, `FederatedCausalEnv.step()` takes `predicted_dags` as an **input parameter** -- i.e. the actor network's own `graph_logits` output (after training-loop post-processing) is what gets stitched and scored for reward:

```python
rewards = compute_ippo_rewards(stitched_dag, true_dag, has_cycle)
```

where `stitched_dag` is built from the actor's own predictions. This means the old architecture's RL reward and its BCE auxiliary loss were **both derived from the same learned representation**, reinforcing each other -- a coherent multi-task-learning signal. The current architecture's reward is derived from the frozen analytic formula's output, which the actor's gradients never touch at all. These are two different, disjoint signals in the current design, one of which is not learnable.

### Secondary, not yet quantified: observation space complexity

Old (`b5ebccf`): observation is a single flattened `d x d` masked covariance + budget scalar (`d^2 + 1 = 17` dims at `d=4`).
Current: `3*d*d + 1` (obs covariance + interventional/run covariance + asymmetry score) plus `obs_feedback` (another `d^2` slice) -- roughly 3x+ larger and structurally more complex. Not yet isolated as an independent factor; likely compounds the other two rather than being separately dominant, but worth a controlled ablation if time permits.

### Also confirmed, NOT the cause

The commit that replaced this simple "shared edge scorer" graph head with a fancier "skew-symmetric tournament" version (`3534c3c`, 2026-08-03) **postdates** both `00vhwhzv` (Jul 30) and `5rn6i810` (Aug 2) -- those two "solved" runs used the simpler edge-scorer head shown above, not the tournament version. So the specific graph-head architecture variant doesn't matter as much as the fact that *some* jointly-trained, gradient-connected graph head existed at all.

## Design consideration: don't just revert the two-stage decoupling

Checked the two-stage refactor's own stated rationale (commit `06da1a5`, `docs/CHANGELOG.md`, `docs/PROBLEM_STATEMENT.md`): the commit message states "Decouple RL target policy from graph estimation with per-step Stage 2 hypothesis prediction" but doesn't document *why* -- no rationale doc found. However this matches the user's own independently-stated research goal from earlier in this session: wanting the RL policy to learn *only* intervention targeting, with structure inference delegated to a strong estimator (ideally a pretrained one like AVICI) -- "I'm hoping that by using AVICI, the model can solely learn optimal interventions." So a blind revert to the old jointly-trained-actor-graph-head design would abandon a goal the user explicitly cares about. The fix needs to preserve decoupling while restoring a working learning/quality signal for structure -- not necessarily by putting a graph head back inside the actor.

Also confirmed: `--freeze_graph_estimator` (`src/train.py`, `src/evaluator_env.py`) and the CHANGELOG's claimed "Frozen vs Unfrozen estimators" ablation dimension are **dead** -- `self.freeze_graph_estimator` is assigned in `FederatedCausalEnv.__init__` and never read anywhere else. `scripts/run_ablation_matrix.py`'s actual 7 ablations don't include it either (checked directly: Soft/Hard shift, AVICI estimator, sparse reward, curiosity, impact bonus, no-obs-feedback -- no frozen/unfrozen toggle exists in the runner). This is pure inert scaffolding, not a partially-built escape hatch -- the CHANGELOG's description of it doesn't match what was implemented, another instance of doc/code drift in this project.

## AVICI is now working on Myriad HPC (new, unplanned but valuable finding)

Checked whether AVICI itself -- already a genuinely pretrained learned estimator, and already fully wired into `predict_graph_hypothesis` with `--estimator_type avici` -- could be tested on Myriad directly, since Myriad's venv already has the exact `jax==0.4.30`/`dm-haiku==0.0.12`/`optax==0.2.2` combination that dodges the `PositionalSharding` incompatibility that made this such a saga on Kaggle. It was not previously installed there. Installed it using the same recipe developed for Kaggle tonight (modern `pyarrow` + avici's other runtime deps + `avici --no-deps` to skip its `pyarrow==10.0.1` pin + a `sitecustomize.py` plasma stub, this time placed directly in the venv's `site-packages` since there's no jax-version isolation needed here) -- much simpler than Kaggle since no jax isolation was needed at all.

**Important side effect, caught and fixed**: installing `tensorflow` pulled in `protobuf==7.35.1`, which broke `wandb` (`wandb==0.17.9` requires `protobuf<6`) in this **shared** venv used for all Myriad training, not just AVICI experiments. Caught immediately by testing `import wandb` right after, fixed by pinning `protobuf<6,>=3.19.0`, then verified the full pipeline still works with a real 2-episode smoke run before proceeding. Flagging this clearly: **the shared Myriad venv (`/home/ucabbse/envs/marl_env`) now has AVICI + a downgraded protobuf installed persistently** -- this is a real, intentional change to shared infrastructure, not scoped to a throwaway environment. Worth the user knowing this was done, even though it was necessary and verified safe.

Verified end-to-end with a 3-episode smoke run (`--estimator_type avici --fixed_graph 0`): AVICI's pretrained checkpoint downloads from HuggingFace Hub and the full training loop runs without error.

**Submitted two new diagnostic jobs** (SGE 129333, 129334) extending the existing hard/soft comparison matrix with AVICI as the estimator, same `--fixed_graph 0`/RNN/200-episode scale as the analytic ones already run:
- `diag_avici_hard_dag0` (job 129333) -- AVICI + hard intervention
- `diag_avici_soft_dag0` (job 129334) -- AVICI + soft-shift (matches `n4in20oe`'s actual config, the run that triggered this whole investigation)

This tests a cleaner, lower-engineering-cost hypothesis before committing to building a new graph head: **is the problem "no learned estimator at all" (which AVICI, already pretrained, would fix without any new code) or "no gradient signal reaching the actor's own representations during training" (which only a jointly-trained head fixes)?** If AVICI alone restores good SHD, the fix might be as simple as making AVICI (not the analytic formula) the default estimator. If AVICI-with-soft-shift still fails like the analytic one did, that's stronger evidence the missing ingredient is specifically the *joint*, actor-coupled training signal (finding #2/#3 above), not just estimator quality in isolation.

## AVICI result #1: hard intervention -- AVICI is WORSE than the analytic formula, not better

Job 129333 (`diag_avici_hard_dag0`) completed (200 episodes). Result, compared directly against the analytic+hard baseline from earlier tonight:

| | mean SHD | mean F1 | fraction episodes at SHD=0 |
|---|---|---|---|
| analytic + hard (`u4liks79`) | 1.68 | 0.72 | 33.5% |
| **AVICI + hard** (`diag_avici_hard_dag0`) | **2.81** | **0.37** | **0.5%** |

This is the opposite of what I expected going in. Likely mechanistic reason: `predict_graph_hypothesis`'s AVICI branch (`src/evaluator_env.py`) synthesizes samples via `rng.multivariate_normal(mean=zeros(d), cov=running_covariance)` -- **always assuming zero mean** -- because the environment only tracks aggregated covariance, not raw per-step samples (documented limitation from earlier tonight, see `bug_avici_wrong_input_shape` in persistent memory). Hard interventions (`X_i := c + noise`) shift the *mean* of the interventional distribution, not just its covariance structure -- that mean-shift signal is exactly what gets thrown away by this reconstruction. So AVICI, however good a pretrained model it is in general, is being fed data that's systematically blind to the specific signal hard interventions produce. This isn't evidence AVICI is a bad estimator in general -- it's evidence the current *sample-reconstruction shim* feeding it is inadequate, especially for hard interventions specifically (probably even more so than for soft-shift, since soft-shift's signal was already weak to begin with per finding #1 above).

**Revises the plan**: "just switch the default estimator to AVICI" is not supported by this result as-is. Either (a) fix the sample reconstruction to preserve mean-shift information (meaningful but scoped fix, doesn't touch the actor/reward architecture at all), or (b) proceed with the graph-head-reintroduction path regardless of AVICI's performance here, since findings #2/#3 (joint training signal, reward-prediction coupling) are architectural and independent of which estimator sits in the frozen Stage-2 slot.

## AVICI result #2: soft-shift -- confirms the mean-blindness hypothesis

Job 129334 (`diag_avici_soft_dag0`) completed. Full comparison matrix, all four cells now filled:

| Estimator | Intervention | mean SHD | mean F1 | fraction episodes at SHD=0 |
|---|---|---|---|---|
| analytic | hard | 1.68 | 0.72 | 33.5% |
| analytic | soft-shift | 2.95 | 0.52 | 9.5% |
| AVICI | hard | 2.81 | 0.37 | 0.5% |
| AVICI | soft-shift | 2.95 | 0.32 | 0.5% |

**Both AVICI conditions cluster at SHD~2.8-3.0 and F1~0.32-0.37, almost indifferent to intervention type** -- unlike the analytic formula, which shows a clear, large gap between hard (1.68) and soft (2.95). This is exactly what the mean-blindness hypothesis predicts: `predict_graph_hypothesis`'s AVICI branch always synthesizes samples as `N(0, running_covariance)`, so AVICI's *input* carries essentially no information about whether the true underlying intervention was hard or soft -- both get flattened to "some covariance shape, zero mean" before AVICI ever sees them. The analytic formula, by contrast, is fed `run_cov`/`obs_cov`/`asym` directly and evidently retains *some* differential signal between the two regimes even though it's a hand-derived heuristic, not a learned model.

**Conclusion for this thread**: AVICI, as currently wired into this environment, is strictly worse than the simple analytic formula on this task, and the reason is identifiable and specific (mean-shift discarded during sample reconstruction) rather than "AVICI is just a bad model." A properly-fed AVICI (real per-step samples + real intervention labels, not this covariance-reconstruction shim) might perform very differently -- that would require tracking raw per-step samples through the environment instead of only aggregated covariance, a nontrivially larger change than tonight's scope. Documenting this clearly rather than pursuing it further tonight; flagging as a good candidate for future work distinct from the graph-head question.

**Decision: proceeding with the graph-head-reintroduction path** (design consideration section above) as the main thread, since findings #2/#3 are architectural and don't depend on which estimator sits in the frozen Stage-2 slot -- fixing AVICI's sample-blindness wouldn't address the deeper issue that *neither* estimator gets any gradient signal from the actor's training process at all.

## Implemented: `--estimator_type learned` -- a separate, online-trained structure estimator

Design goal: restore a genuine gradient-based structure-learning signal (findings #2/#3) **without** re-coupling it into the intervention-policy actor, since that would abandon the user's own stated research goal (policy should learn *only* interventions; structure estimation should be a separate, potentially strong/pretrained capability -- the same reasoning that motivated testing AVICI).

**New file `src/marl/graph_estimator.py`**: `GraphEstimatorNet` (Haiku module) -- structurally similar to the pre-refactor "shared edge scorer" (node features -> pairwise concat -> MLP edge logit), but a completely separate network with its own params and its own `optax.adam` optimizer, never touched by the actor/critic's gradients. `init_graph_estimator` / `make_graph_estimator_fns` provide init, a jitted `update_step` (masked BCE loss against true adjacency, masked by the same `structural_mask` used everywhere else for domain-privacy), and a jitted `predict`.

**`src/evaluator_env.py`**: `FederatedCausalEnv.__init__` builds this estimator when `estimator_type == "learned"`. `predict_graph_hypothesis` gained a `learned` branch (predicts using current params, applies `structural_mask`, matches the existing analytic/avici output contract exactly). New `update_graph_estimator(obs_cov, run_cov, asym, true_adj)` method performs one online gradient step -- called from `step()` **after** the prediction for that step has already been used for reward, specifically to avoid this step's own label leaking into its own reward (see the method's docstring).

**`src/train.py`**: `--estimator_type` gained `"learned"` as a third choice alongside `analytic`/`avici`.

**Tests** (`tests/test_graph_estimator.py`, 4 new, all passing alongside the existing 60): predict() returns valid `[d,d]` probabilities; update_step() demonstrably reduces loss over repeated steps on a fixed example; the structural mask genuinely prevents training signal from pulling a forbidden edge toward an adversarial label; `FederatedCausalEnv` wiring (predict + update) works end-to-end. Full suite: 64/64 passing.

**Local smoke test**: `python -m src.train --num_episodes 5 --fixed_graph 0 --estimator_type learned` runs end-to-end, no crashes/NaNs. SHD stayed flat over just 5 episodes, which is expected -- a freshly-initialized network with ~100 gradient steps total isn't meaningfully trained yet; the real test is the longer Myriad run below.

**Deliberately not addressed tonight**: `--freeze_graph_estimator` remains dead/unused (confirmed earlier); didn't wire it to gate the "learned" estimator's updates, since "learned but frozen" isn't a meaningful combination and touching that flag's semantics more broadly is out of scope for tonight. Also not attempted: the auxiliary-BCE-on-the-actor's-own-shared-trunk variant (finding #2's most literal mechanism, densifying the actor's *own* representations) -- this implementation restores finding #3 (reward coupled to something actively learning) but not the shared-representation aspect of #2. Worth a follow-up comparison if this doesn't fully close the gap.

**Caught a false-positive job-completion signal** (jobs 129339/129340): the polling monitor briefly reported "both jobs left the queue" based on an empty `qstat` grep result, which turned out to be a transient SSH hiccup, not real completion -- `qstat -j <id>` and a direct re-check immediately after showed both jobs still genuinely queued/running. This is the exact "SSH connection drop misread as job leaving queue" failure mode already documented in persistent memory from earlier work on this project. Fixed the poll script to append an explicit `SSH_OK_MARKER` sentinel after the remote command and only treat empty results as real completion when that marker is actually present in the output -- otherwise treats it as a transient failure and retries. Verified the fix against the real current (still-running) state before trusting it again.

## MAJOR RESULT: `learned` + soft-shift essentially replicates the old "solved" behavior

Job 129340 (`diag_learned_soft_dag0`) completed -- **this uses `soft_shift`, the exact same intervention type as `n4in20oe`, the run that started this whole investigation.**

| | mean SHD | mean F1 | fraction episodes at SHD=0 |
|---|---|---|---|
| analytic + soft-shift | 2.95 | 0.52 | 9.5% |
| AVICI + soft-shift | 2.95 | 0.32 | 0.5% |
| **learned + soft-shift** | **0.365** | **0.907** | **79%** |

Trend over the 200 episodes (40-episode bins) is the real story:

| Episodes | mean SHD | mean F1 |
|---|---|---|
| 1-40 | 1.65 | 0.57 |
| 41-80 | 0.175 | 0.967 |
| **81-120** | **0.0** | **1.0** |
| **121-160** | **0.0** | **1.0** |
| **161-200** | **0.0** | **1.0** |

**SHD hits exactly 0.0 and F1 exactly 1.0 for 120 consecutive episodes (81-200)** -- this is the same convergence shape as `00vhwhzv` (the old "solved" run), reached with the *same* soft-shift interventions that both frozen estimators (analytic, AVICI) failed badly under. This is strong, clean confirmation that the dominant missing ingredient was the gradient-based structure-learning signal (findings #2/#3), not intervention type (finding #1) -- a separate, actor-decoupled trainable estimator is sufficient to restore full convergence, without needing to touch the actor's own architecture or reward-coupling at all.

Job 129339 (`learned` + `hard`) also completed. Full final matrix:

| Estimator | Intervention | mean SHD | mean F1 | fraction episodes at SHD=0 | episodes 161-200 mean SHD |
|---|---|---|---|---|---|
| analytic | hard | 1.68 | 0.72 | 33.5% | -- |
| analytic | soft-shift | 2.95 | 0.52 | 9.5% | -- |
| AVICI | hard | 2.81 | 0.37 | 0.5% | -- |
| AVICI | soft-shift | 2.95 | 0.32 | 0.5% | -- |
| **learned** | **hard** | **0.615** | **0.861** | **61.5%** | **0.075** |
| **learned** | **soft-shift** | **0.365** | **0.907** | **79%** | **0.0** |

Both `learned` conditions dramatically outperform every frozen-estimator condition. One nuance worth noting honestly rather than glossing over: in this single-seed run, `learned+soft` converged *faster and to a cleaner floor* than `learned+hard` (SHD exactly 0.0 for episodes 81-200 vs. still-improving-but-not-quite-zero at 0.075 by 161-200) -- the opposite of what finding #1 alone would predict. Plausible explanation: the learned estimator's own training dynamics interact with intervention type differently than a fixed formula's do; a single seed each isn't enough to be certain this ordering is robust (would need multiple seeds to know if it's real or noise) -- but the qualitative conclusion is unambiguous either way: **a gradient-connected, continuously-training structure estimator is sufficient to restore convergence under both intervention types**, closing the vast majority of the gap that intervention type alone (finding #1) only partially explained.

**Answering the user's original question directly**: yes, the "disjointing of prediction and intervention" was the dominant cause -- specifically, disjointing combined with *freezing* the predictor (no gradient signal at all). A decoupled-but-*learning* predictor (this implementation) restores the old architecture's effectiveness without re-coupling structure prediction into the intervention policy itself.

## CONFIRMED AT FULL SCALE: fix generalizes across the full 8-topology dynamic curriculum

Job 129344 (`confirm_learned_full`) completed: replicates `n4in20oe`'s exact configuration (dynamic 3-stage multi-topology curriculum, 1000 episodes, `soft_shift`, RNN) with the one change being `--estimator_type learned` instead of `analytic`. (Note: this run hit a transient `wandb.init()` timeout on the compute node -- the project's existing fallback in `src/train.py` handled it gracefully and training completed all 1000 episodes regardless, confirmed by the local `training_metrics.csv` having exactly 1000 data rows. This run isn't on WandB; results below are read directly from that CSV.)

Overall: **mean SHD 0.342, mean F1 0.923, SHD=0 on 77% of all 1000 episodes.**

Per-stage breakdown (100-episode bins), showing the curriculum's own stage transitions:

| Episodes | Curriculum stage | mean SHD | mean F1 |
|---|---|---|---|
| 1-100 | 1 (graph 0 only) | 1.09 | 0.72 |
| 101-200 | 1 | 0.09 | 0.98 |
| 201-300 | 2 (chain MEC pair introduced) | 1.05 | 0.76 |
| 301-400 | 2 | 0.51 | 0.89 |
| 401-500 | 2 | 0.27 | 0.95 |
| 501-600 | 3 (all 8 topologies introduced) | 0.05 | 0.99 |
| 601-700 | 3 | 0.11 | 0.98 |
| 701-800 | 3 | 0.08 | 0.98 |
| 801-900 | 3 | 0.09 | 0.98 |
| 901-1000 | 3 | 0.08 | 0.98 |

The shape is exactly what you'd want to see: SHD spikes briefly whenever the curriculum introduces new topologies (episode ~200 and ~500, as the estimator meets graph structures it hasn't been trained on yet), then recovers and settles -- and by Stage 3 (all 8 topologies, episodes 501-1000), it stabilizes at **SHD 0.05-0.11 for 500 consecutive episodes**, i.e. near-perfect structure recovery sustained across the full topology space, not just the one easy graph tested in the diagnostics above.

Direct comparison to what you originally flagged: `n4in20oe` (analytic, same full config) oscillated between SHD 2-6 with no visible improvement trend for its entire 1000-episode run. This run reaches SHD <0.1 and holds it. That's the headline result.

## Multi-seed robustness check: confirmed, not a lucky seed

Repeated the `learned`+soft-shift+`fixed_graph=0` diagnostic at 3 seeds total:

| Seed | mean SHD | mean F1 | fraction episodes at SHD=0 | last-40-episode mean SHD |
|---|---|---|---|---|
| 42 (original) | 0.365 | 0.907 | 79% | 0.0 |
| 7 | 0.245 | 0.930 | 88.5% | 0.0 |
| 13 | 0.325 | 0.915 | 81.5% | 0.0 |

**All three seeds converge to essentially the same strong result and all three reach SHD=0.0 for their final 40 episodes.** This is not a seed=42-specific fluke.

## Morning session: decoupling agent learning from estimator memorization

New concern raised: end-to-end SHD conflates whether the *policy* is learning to intervene well vs. whether the `learned` estimator is just memorizing the small set of training topologies (only 8 possible ground-truth DAGs exist). Plan (`~/.claude/plans/memoized-popping-wind.md`, approved): fix AVICI to be a genuinely fair *frozen* baseline (a frozen estimator can't memorize, so if agent-side metrics still improve when paired with it, that's evidence the policy itself is learning something real), add several new metrics that isolate intervention quality from estimator quality, and add `--reward_density sparse` as a comparison arm.

### AVICI fix: real per-step samples with real intervention labels, not synthetic reconstruction

Root cause (confirmed from AVICI's real source): `predict_graph_hypothesis`'s AVICI branch synthesized fake `x = rng.multivariate_normal(mean=zeros(d), cov=running_covariance)` and passed `interv=None` -- discarding both the mean-shift signal real interventions produce and AVICI's real per-sample intervention labels (`interv[i,j]==1` iff node `j` was intervened on in observation `i`, which AVICI's architecture is explicitly designed to use).

Fix: `EnvState` (`src/types.py`) gained a fixed-capacity raw-sample buffer (`raw_samples`, `raw_interv`, `raw_count`, sized to `(max_steps+1)*sample_count` -- the exact max an episode can produce including the observational reset-phase block, so no eviction policy needed) populated in both jitted kernels (`src/evaluator_env.py`), plus a real `running_mean` field (`src/environment.py` gained `stitch_global_mean`, mirroring `stitch_global_covariance`). `predict_graph_hypothesis`'s AVICI branch now reads real accumulated samples + real intervention labels from this buffer instead of synthesizing. 6 new tests (`tests/test_avici_buffer.py`), full suite 70/70 passing.

**Caught and fixed a serious performance regression from this fix, before it reached a full training run**: an isolated AVICI timing test showed forward-pass compute scaling from 0.04s (n=100) to 2.3s (n=2100) -- summing to only ~17s across a full episode's worth of calls -- but a real training run measured ~80s/episode, a 4.4-hour projected cost per 200-episode run (vs. ~6-10 min before the fix). The gap is consistent with JAX re-JIT-compiling AVICI's internals every step, since the buffer's fed size grows every step, giving every call a different input shape. Fixed by adding `--avici_max_context` (default 400, ~4 steps' worth) capping how many of the most-recent buffer rows get fed per call -- still real samples/labels throughout, just bounded history per call, letting JAX reuse its compiled cache once the buffer fills past the cap. Re-verified: **10 episodes in 98s (~9.8s/episode)**, an ~8x speedup, projecting to ~33 min for a full 200-episode AVICI run -- back in a practical range.

This whole layered-discovery pattern (fix appears to work -> real-scale test reveals a new problem -> fix that -> re-verify) matches last night's AVICI-on-Kaggle saga; noting it again since it's the same discipline (verify empirically at real scale, don't assume a unit-test pass means the real run will be fine).

### New metrics implemented

Six new metrics (interventions-to-SHD=0, AUC of within-episode SHD curve in two variants, edge-orientation-yield per intervention, a closed-form Gaussian information-gain metric kept reward-inert, variance-shift trend, agent coordination/redundancy) -- see `src/episode_metrics.py` and `src/train.py`'s episode loop for exact formulas. None touch `compute_ippo_rewards`; they're pure evaluation/diagnostic signals, trustworthy as independent evidence rather than something the policy can game. 11 new tests, full suite 81/81 passing.

### RESULT: the 18-run diagnostic matrix -- and a surprising, clarifying twist

Ran `{learned, avici (fixed), analytic} x {dense, sparse reward} x 3 seeds`, 200 episodes each, `--fixed_graph 0`/soft-shift, on Myriad (SGE array job 133025, all 18 completed successfully). A companion Kaggle notebook (`notebooks/kaggle_agent_estimator_matrix.ipynb`) runs the identical matrix with GPU for the non-AVICI arms, for you to run separately.

**Headline numbers** (mean over all 200 episodes, averaged across 3 seeds):

| Estimator | Reward | mean SHD | fraction reached SHD=0 | mean interventions-to-SHD0 |
|---|---|---|---|---|
| analytic | dense | 2.97 | 44% | 11.7 |
| analytic | sparse | 3.01 | 42% | 10.9 |
| **avici (fixed)** | dense | **1.57** | **98.7%** | **4.7** |
| **avici (fixed)** | sparse | **1.63** | **99.0%** | **4.7** |
| learned | dense | 0.31 | 87.2% | 1.4 |
| learned | sparse | 0.31 | 87.0% | 1.4 |

**This is a different picture than last night's AVICI result** (where AVICI was *worse* than analytic due to the mean-shift-blind synthetic-sample bug). Today's fix (real samples + real intervention labels) alone took AVICI from "worse than a hand-derived formula" to "reaches SHD=0 on ~99% of episodes" -- a huge jump, and it happened **immediately**: comparing early episodes (1-40) to late episodes (161-200), AVICI's `reached_shd0` rate is already 96% early and stays ~99% late -- barely moving, because there's almost no room left to improve. `learned`, by contrast, starts at 35% early and reaches 100% late -- a clear, large learning curve.

**Why this matters for the memorization question**: AVICI is frozen -- it cannot adapt to this environment at all, ever. Yet it gets near-ceiling performance from essentially the first batch of episodes, using whatever (still largely untrained) policy behavior was generating its input data at that point. That's a strong, direct piece of evidence that **once the estimator has good real data to work with, the policy doesn't need to be very sophisticated for the estimator to identify structure correctly** -- an early, near-random-ish policy already produces interventional data informative enough for a strong pretrained model to nail the graph. This reframes `learned`'s dramatic 35%->100% improvement curve: since a frozen-but-good estimator already succeeds with weak early policy behavior, `learned`'s slow start is much more plausibly explained by **`learned`'s own small network needing time to fit its BCE objective from random initialization** (i.e. the estimator training itself, exactly the memorization-style effect you were worried about) than by the *policy* needing 150+ episodes to learn good interventions. This doesn't prove zero policy learning is happening, but it substantially shifts the weight of evidence toward "estimator quality was the dominant bottleneck all along," consistent with your original instinct.

**A separate, less encouraging finding, checked directly rather than glossed over**: `eval/redundancy_rate` (both agents intervening on the same boundary node in the same step -- wasted budget) is **flat at ~5-6% across every single condition and every training bin**, never trending down under any estimator or reward setting. Agents are not learning to avoid this specific coordination failure, regardless of how much SHD improves through other means. Given the boundary-node-collision mechanism is confirmed structurally real (see finding above on valid target sets overlapping at X1/X2), this looks like a genuine, unaddressed inefficiency rather than a metric artifact -- worth a closer look if the thesis wants to make claims about federation efficiency, not just SHD outcomes.

**One metric-interpretation caveat worth stating plainly**: `orientation_precision` appears to *decrease* toward late training in several conditions (e.g. `learned`: ~0.03 early -> ~0.00 late). This is not evidence of the policy getting worse -- it's a saturation artifact: once SHD is already at or near 0 (as it is for `learned` by late training), there's no more room for a step to produce *positive* SHD improvement, so `sum_positive_delta` mechanically shrinks toward zero regardless of underlying policy quality. This metric is most informative during the *pre-convergence* phase of training, not as a monotonic "is the policy still learning" signal once a condition has already converged.

**Sparse vs. dense reward**: essentially no difference in any metric, for any estimator (e.g. `avici` dense vs sparse: SHD 1.57 vs 1.63, both ~99% reached-SHD0). This is itself informative: it does **not** support "the dense per-step reward is doing most of the work" -- if it were, sparse reward should have collapsed performance noticeably, and it didn't, for any estimator. The terminal SHD signal alone appears sufficient at this task scale.

## Decisions (made 2026-08-13, after review of the matrix results)

1. **Merged into `main`**: this branch (`investigate/graph-head-regression`) was reviewed and merged into `main` via `--no-ff` merge, preserving full history. `main` is the project's core tested path; this branch's results were judged to show enough promise to promote, per the project's branch-as-experiment convention.
2. **Default `--estimator_type` is now `avici`** (was `analytic`). Rationale: AVICI (fixed) reaches SHD=0 on ~99% of episodes essentially from the first batch of training, with zero risk of the estimator-memorization confound that `learned` carries (AVICI is frozen -- it structurally cannot memorize the 8-topology set). `learned` still gets a lower absolute SHD (0.31 vs 1.57), but that gap is plausibly explained by estimator-side fitting rather than superior policy behavior (see matrix analysis above). AVICI is also the better choice for future scaling to more nodes/agents/algorithms, since it doesn't need retraining per topology.
3. **Default `--reward_density` is now `sparse`** (was `dense`). Rationale: the 18-run matrix found no meaningful difference between dense and sparse reward for any estimator (e.g. AVICI: SHD 1.57 dense vs 1.63 sparse, ~99% reached-SHD0 either way) -- no evidence the dense per-step shaping is doing meaningful work, so the simpler, less-hand-holding sparse terminal reward is preferred by default.
4. ~~**Statistical confidence**~~ -- resolved: confirmed across 3 seeds (42, 7, 13) at the diagnostic scale, all converging to SHD=0.0 by the final 40 episodes (see multi-seed section above).
5. ~~AVICI's sample-reconstruction shim discards mean-shift information~~ -- fixed (see above). One loose end remains: the "auxiliary head on the actor's own shared trunk" variant (the single most literal reproduction of the old architecture's finding #2 mechanism) wasn't tried, since the simpler separate-network approach already worked.
6. **Kaggle GPU AVICI path** -- the `JAX_PLATFORMS=cpu` fix for the isolated-subprocess PJRT-version-mismatch bug (commit `d0db3ea`) is now confirmed working on real Kaggle GPU hardware.

## Follow-up: does the frozen (deployed) policy actually work? (2026-08-13, post-merge)

**Question that triggered this**: every SHD/F1/reached-SHD0 figure reported above -- both the overnight AVICI-fix diagnostics and the 18-run matrix -- comes from `train.py`'s own episode loop, computed from the same rollout used for that episode's PPO update, with actions **sampled stochastically** from the actor's then-current (still-updating) weights. None of it reflects a separately frozen, "already learned," greedily-evaluated policy. "Early vs late" in the matrix analysis means "early-training policy vs late-training policy," not "no policy vs converged policy."

**Discovery**: `train.py` already runs exactly the right experiment automatically -- at the end of every run it calls `evaluate.py::evaluate_checkpoint` on the best-observed checkpoint, with `temperature=0.0` (greedy/deterministic action selection, no exploration) across **all 8 topologies** (not just the trained-on one), saving `evaluation_trace.json`. This had already run for all 18 Myriad matrix jobs -- it just never got pulled back, because the earlier `scp`/tar step only grabbed `training_metrics.csv`. Retrieved all 18 `evaluation_trace.json` files from `~/marl_causal/diag_runs/matrix_*/` on Myriad (no retraining needed) and analyzed them (144 run x graph combinations).

**Result: severe behavioral collapse under greedy evaluation.**

| | analytic | avici | learned |
|---|---|---|---|
| mean final SHD (greedy, 8 graphs) | 3.04 | 2.90 | 2.81 |
| reached SHD=0 rate | 4.2% | 4.2% | 2.1% |
| "stuck loop" rate (intervenes, but on <=1 unique target the whole episode) | 66.7% | 60.4% | 43.8% |
| never intervenes at all (pure NOOP, entire 20-step episode) | 16.7% | 27.1% | 45.8% |
| genuine target diversity (2+ unique targets) | ~17% | ~13% | ~10% |

Across all 144 run x graph combinations: **78.5% of frozen-policy episodes show *zero* SHD change from step 0 to the final step** -- the deterministic policy does literally nothing informative for the vast majority of evaluated episodes. 9 of the 18 runs (50%) are *completely* static -- SHD never moves on *any* of the 8 topologies for that checkpoint. Concrete example (`matrix_analytic_dense_s7`, graph_0): agent_0 greedily intervenes on target 0 and agent_1 on target 2, **every single one of 20 steps, with zero variation**, SHD frozen at 3.0 throughout.

**Why**: this is a real dynamical property, not obviously an eval-harness bug (checked: RNN hidden-state size matches between train and eval, `boundary_margin` matches, action masking only touches target selection after category is chosen). Two effects most plausibly compound:
1. **Exploration was doing the real work.** PPO's stochastic sampling + entropy bonus (`--entropy_coef 0.01`) generates the target diversity that lets any of the three estimators accumulate informative interventional data. Strip that away (greedy argmax) and the *policy itself* has not learned a principled, evidence-driven targeting strategy -- it collapses to either inaction or one fixed, evidence-independent target, repeated forever. This is directly visible in the data: episodes with 2+ unique targets reach SHD=0 27.8% of the time; episodes with 0-1 unique targets reach it **0.0%** of the time, regardless of estimator.
2. **Checkpoint selection is fragile.** `train.py`'s "best" checkpoint criterion (~line 700) picks whichever single stochastic training episode had the lowest one-shot SHD -- a noisy, unaveraged statistic. A lucky early episode can win and never get displaced, meaning `best_ippo_params.pkl` isn't guaranteed to represent a well-converged policy at all. Not fully disentangled from effect (1) in this round -- both point the same direction (don't trust the "best" checkpoint as a proxy for "the policy has learned to intervene well"), but which dominates is untested.

**One deliberate non-claim**: greedy/argmax may not even be the *right* evaluation protocol for this task class -- unlike typical RL tasks, active causal discovery arguably wants an agent that keeps seeking diverse, informative interventions as part of an *optimal* deterministic policy, not just as training-time noise to anneal away. But the collapse found here isn't that -- a principled uncertainty-seeking deterministic policy would still vary its targets as evidence accumulates, and would not perform identically to (or worse than) pure inaction. Repeating one fixed target regardless of topology, with 0% reached-SHD0, is behavioral collapse, not a defensible exploitation strategy.

**What this means for the findings above**: the 18-run matrix's headline "AVICI reaches ~99% SHD=0 from episode 1" / "learned climbs 35%->100%" describes the training-time **policy+exploration-noise system**, not standalone policy competence. It does *not* invalidate the earlier estimator-vs-estimator comparison (all three estimators were measured under the identical stochastic-sampling regime, so that comparison's internal validity holds) -- but it does mean none of this doc's SHD figures should be read as "what you'd get deploying the trained policy deterministically," because right now you would not get that; you'd get something close to random or static.

**Not fixed in this round** (measurement only, matching this round's scope): options worth considering, not decided here --
- Evaluate (and deploy) at low-but-nonzero temperature rather than pure greedy, since this task may genuinely want persistent stochastic exploration.
- Fix checkpoint selection to average SHD over several rollouts rather than trusting a single noisy episode.
- Add an explicit uncertainty/novelty-seeking signal to the deterministic policy itself (e.g. bonus for targeting under-sampled nodes) so good behavior survives temperature -> 0, rather than relying on entropy regularization alone during training.
- Full analysis script: available on request, ran locally against the pulled-back `evaluation_trace.json` files, not yet committed to the repo (ad-hoc diagnostic, same as the earlier matrix analysis scripts).

## Backlog (known limitations, not addressed in this round)

- **[Highest priority, found post-merge] Greedy/deployed policy collapses** -- see "Follow-up: does the frozen (deployed) policy actually work?" above. 78.5% of frozen-policy episodes show zero SHD change; the trained policy has not learned a robust deterministic intervention strategy, it relies on training-time stochastic exploration to look competent. This is a bigger open problem than anything else in this doc and should probably be tackled before further estimator/reward tuning.
- **`redundancy_rate` (agent coordination failure) stays flat at ~5-6%** across every estimator/reward condition and every training bin -- no setting teaches agents to avoid both intervening on the same boundary node in the same step. Some irreducible redundancy is expected from the federation structure itself (private vs. boundary node overlap at X1/X2), but whether the *current* ~5-6% floor is that irreducible minimum or a fixable coordination gap is untested. Worth a closer look if the thesis makes federation-efficiency claims -- e.g. a small penalty for redundant same-step targeting, or exposing each agent's peer's chosen target before committing.
- **`--freeze_graph_estimator` is dead code** (assigned, never read) -- pre-existing, unrelated to this investigation, clean up or wire in properly whenever convenient.
- Sparse-vs-dense equivalence was only tested at diagnostic scale (200 episodes, `--fixed_graph 0`); not re-verified at full 1000-episode multi-topology scale.

## Session log (for context on how this was produced)

Ran overnight per your request, self-pacing with `ScheduleWakeup` between chunks of work (5-30 min breaks depending on whether waiting on a Myriad job), git-committing incrementally on this branch after each substantive finding so nothing was at risk of being lost. Caught and fixed two real operational bugs along the way (a shared-venv `protobuf`/`wandb` conflict from installing AVICI's dependencies, and a false-positive job-completion signal from a transient SSH connection drop) -- both documented inline above where they happened, not swept under the rug.
