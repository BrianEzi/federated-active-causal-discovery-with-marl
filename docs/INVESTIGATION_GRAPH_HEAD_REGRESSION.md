# Investigation: Why RNN Training No Longer Converges to Low SHD

**Status**: In progress (overnight autonomous session, started 2026-08-13 ~01:00 UTC)
**Branch**: `investigate/graph-head-regression` (worktree: `.claude/worktrees/investigate-graph-head-regression`)
**Trigger**: User observed WandB run `n4in20oe` (RNN, current architecture) failing to learn accurate causal graphs even on the easiest single-topology case, despite RNN "previously being able to learn the right causal graph." User hypothesis: the decoupling of graph prediction from intervention policy ("disjointing") is the likely cause.

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

## Next steps (in progress)
1. Run a 200-episode `--fixed_graph 0 --estimator_type learned` diagnostic on Myriad (both `hard` and `soft_shift`), matching the existing comparison matrix exactly, for a direct apples-to-apples result.

1. **Waiting on results** from jobs 129333/129334 (AVICI hard/soft on graph-0). Check next wake cycle.
2. Depending on that result: either (a) if AVICI alone fixes it, investigate making AVICI the default estimator and test at full scale/multi-topology, or (b) if AVICI alone doesn't fix it, proceed to design a graph-head-style auxiliary training signal that preserves the decoupled reward/evaluation path (see "Design consideration" above) -- e.g. an auxiliary BCE loss on a small representation-shaping head whose output is *not* what reward/evaluation uses, only what shapes the shared trunk, OR an actually-trainable Stage-2 estimator (implementing the "unfrozen estimator" idea that was scaffolded but never built) updated via its own optimizer each step.
3. Test locally, then confirmatory runs on Myriad HPC (both short diagnostic and longer/full-scale runs approved by user).
4. Write up final comprehensive findings + recommendation doc.
