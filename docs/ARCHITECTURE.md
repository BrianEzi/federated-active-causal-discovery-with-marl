# System Architecture: End-to-End Episode Walkthrough

This doc replaces an earlier, now-stale version (it described a 3-way action space with a "Peer Request" category and a learned graph-prediction head inside the actor -- both removed; see `docs/CHANGELOG.md`). It describes the system **as it currently exists on `main`**: what happens at the start of an episode, during one step, and at the end of an episode, including every model, input, output, reward term, and observation channel involved. For the mathematical Dec-POMDP formulation, see `docs/PROBLEM_STATEMENT.md`; for the investigation history behind several of the design choices below, see `docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md`.

## 0. The setup, in one paragraph

Two agents (`K=2`) jointly try to discover a hidden 4-node (`d=4`) causal DAG by choosing where to intervene. The graph is always one of the same 4 nodes: `Z1` (idx 0, Agent 0's private node), `X1` (idx 1, Agent 0's boundary node), `X2` (idx 2, Agent 1's boundary node), `Z2` (idx 3, Agent 1's private node). Agent 0 owns/observes `{Z1, X1, X2}`; Agent 1 owns/observes `{X1, X2, Z2}` -- `X1`/`X2` are the only nodes both agents can see and intervene on. Each agent runs its own actor/critic (Independent PPO -- "Disjoint IPPO": no shared parameters, no communication channel other than the shared boundary nodes' statistics). A separate "Stage 2" component (one of three interchangeable estimator types) turns accumulated covariance data into a predicted edge-probability matrix every step; the two agents' predictions get stitched into one global DAG and scored against the true adjacency for reward.

## 1. Start of an episode: `FederatedCausalEnv.reset()`

1. **Topology sampling** (`src/generators.py::generate_4node_topologies`): picks one of **8 hand-authored ground-truth DAGs** (Chain, Reversed Chain, Collider, Reversed Collider, Fork, Reversed Fork, Fork+Collider, Reversed Fork+Collider -- all structurally valid arrangements of `Z1-X1-X2-Z2`), plus a topological order for that DAG. `--fixed_graph N` forces one specific topology (used throughout this investigation for fast diagnostics); omitting it samples uniformly (or from an `allowed_topologies` curriculum subset -- see `--curriculum_stage2_ratio`) every episode.
2. **SCM parameterization** (`generate_scm_params`): for each true edge, samples a random linear weight `beta ~ U([-2,-0.5] U [0.5,2])` (uniform in the two-sided range, i.e. never near-zero) into weight matrix `W`. If `--mechanism_type NONLINEAR_ANM`, also initializes a small per-node MLP (Glorot-uniform, 16 hidden units) instead of using `W` directly. Default is `LINEAR`.
3. **`EnvState` initialization** (`src/environment.py::init_env`, `src/types.py::EnvState`): a JAX pytree holding the true adjacency, topological order, SCM params, per-agent budgets (`initial_budget` each, default 20.0), and all running statistics, all zeroed:
   - `running_covariance` / `running_mean` `[d,d]` / `[d]` -- cumulative sample-count-weighted averages, updated every step (see 2.4 below).
   - `obs_covariance` `[d,d]` -- a **fixed baseline**, set once here at reset from the initial observational (non-interventional) sample batch, and never updated again for the rest of the episode. Serves as the "no intervention" reference point for invariance testing.
   - `int_covariance` `[d,d,d]` -- slice `[k,:,:]` holds the covariance measured the *last* time node `k` was intervened on (replaced, not averaged, each time node `k` is targeted again).
   - `int_mask` `[d]` -- 1.0 once a node has ever been intervened on, else 0.0 (monotonic for the episode).
   - `raw_samples` / `raw_interv` `[capacity,d]` -- a growing buffer of every individual sample drawn all episode, with per-sample intervention labels, feeding AVICI (capacity = `(max_steps+1) * sample_count`, sized to the exact max an episode can ever produce).
   - `budgets` `[K]`, `step_count` (0).
4. **Initial observational sampling** (`jitted_initial_obs_kernel`): draws `sample_count` (default 100) samples from the SCM with **no intervention active**, via `src/scm.py::_sample_scm_jitted` -- for each sample, walks the topological order, computes each node's value as `W[node,:] . parents + noise` (noise ~ Gaussian by default, scale `--noise_scale`), writes results into `obs_covariance`, `running_covariance`/`running_mean`, and the raw-sample buffer.
5. **First graph prediction** (`predict_graph_hypothesis`, see section 3): produces `last_predicted_dag`, a `[d,d]` probability-ish matrix (0.5 everywhere off-diagonal before any real signal exists, then the real estimator output).
6. **First observation** (`_get_obs_dict`, see section 4) is built and returned to the caller (`train.py`) along with `{"true_adjacency": ...}`.

## 2. One step: `FederatedCausalEnv.step(joint_actions, predicted_dags, key)`

### 2.1 Action selection (in `train.py`, happens *before* `step()` is called)

For each agent `k`, its actor network (`src/marl/ppo_agent.py`) does a forward pass on that agent's current observation:

- **Models**: `IPPOActor`/`IPPOCritic` (feedforward MLP) or `IPPORNNActor`/`IPPORNNCritic` (GRU-based, hidden dim 64) -- **RNN is the default** (`--use_rnn` defaults to `True`; `--no_rnn` switches to feedforward). Both output `(cat_logits [2], target_logits [d])` from the actor and a scalar value estimate from the critic. `Inductive*` variants also exist (`InductiveIPPOActor`/`InductiveIPPORNNActor`) but aren't the default path.
- **Category** (`ActionCategory`, a 2-way enum: `INTERVENE=0`, `NOOP=1`) is sampled from `cat_logits` (`sample_actions_jitted`, categorical sampling from softmax at training time; `evaluate.py`'s frozen-checkpoint eval instead uses `temperature`-controlled sampling, `0.0` = greedy argmax).
- **Target**: `target_logits` gets masked (`mask_invalid_targets`) to `-1e9` everywhere the agent isn't allowed to intervene (outside its own local+boundary nodes) before sampling -- so an agent can only ever choose a target it's structurally permitted to touch.
- Each agent's `(obs, cat_action, target_action, value, log_prob)` gets appended to that agent's own `RolloutBuffer` (`src/marl/ppo_trainer.py`) for the later PPO update -- there is no shared buffer or gradient path between the two agents.

### 2.2 Intervention application

`step()` builds an `InterventionSpec` (`mask [d]`, `type [d]`, `value [d]`) from both agents' chosen actions: for each agent, if it chose `INTERVENE`, has enough budget (`budgets[k] >= action_cost`, default cost 1.0), and the target is within its own domain or the shared boundary (`X1`/`X2`), that node's `mask` entry is set to 1.0 and `value` to `shift_val` (default 2.0). Both agents can legally target the same boundary node in the same step -- if they do, both pay the cost but the SCM only applies one effective intervention (a confirmed, real source of the `redundancy_rate` finding in the investigation doc).

Two intervention mechanisms exist (`--intervention_type`, **default `hard`** as of 2026-08-13):
- **`hard`**: `X_i := c + noise` -- fully replaces the node's value, severing its dependence on parents. The classical "perfect intervention."
- **`soft_shift`**: `X_i := f_i(Pa_i) + noise + delta` -- preserves the parent-dependence, just adds an offset. A weaker, more realistic "imperfect intervention" -- empirically a much harder structure-learning signal (see the investigation doc's finding #1), which is why `hard` is now the default.

Budgets are then deducted (`new_budgets = budgets - costs`) -- **note**: at current defaults (`initial_budget=20`, `action_cost=1`, `max_steps=20`), an agent can afford to intervene *every single step* without ever exhausting its budget, so this constraint does not currently bind (flagged in the investigation doc's backlog).

### 2.3 SCM sampling under the chosen intervention

`_sample_scm_jitted` draws a fresh batch of `sample_count` samples under the just-built `InterventionSpec` (same topological-order walk as reset, but now respecting `mask`/`type`/`value` per node).

### 2.4 State updates (`jitted_env_step_kernel`, all in JAX, one call per step)

From this step's `samples [sample_count, d]`:
- Per-agent local covariance/mean, masked to that agent's observable nodes, then **stitched** into one global `stitched_cov`/`stitched_mean` (`stitch_global_covariance`/`stitch_global_mean` -- a sample-count-weighted average across the two agents' overlapping views).
- **`info_gains`**: Frobenius norm of `(stitched_cov - old running_covariance)`, projected onto each agent's observable mask, normalized by node count -- this is the `intrinsic_coef`-weighted exploration bonus (off by default, `intrinsic_coef=0.0`), *not* the same thing as the evaluation-only Gaussian-entropy metric used elsewhere in the investigation doc.
- **`running_covariance`/`running_mean` update**: `updated = (old * n_old + stitched * sample_count) / (n_old + sample_count)` -- a cumulative weighted average with an ever-growing denominator. **This is the mechanism identified as the root cause of the greedy-policy-collapse finding**: each new step's marginal weight shrinks over the episode (~33% at step 2, <5% by step 15 at defaults), so the observation converges toward a near-fixed-point regardless of ongoing agent behavior. See `docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md`'s "Follow-up" section for the full analysis and a state-representation redesign plan.
- **`int_covariance`/`int_mask` update**: for whichever node(s) were just intervened on, `int_covariance[node]` is **replaced** (not averaged) with this step's `stitched_cov`; `int_mask[node]` set to 1.0 permanently.
- **Raw-sample buffer**: this step's `samples` and a broadcast intervention-label row are written at the current write cursor (`jax.lax.dynamic_update_slice`, since the offset is a runtime value).

### 2.5 Graph (re-)prediction -- Stage 2 estimator

See section 3 for the three estimator types. Whichever is active, `predict_graph_hypothesis` is called with `(obs_covariance, running_covariance, asymmetry)` and returns a fresh `[d,d]` edge-probability matrix, masked by `structural_mask` (the union of both agents' edge-authority masks -- structurally impossible edges like `Z1<->X2` or `Z1<->Z2` are always zeroed). The two agents' local slices of this get **stitched** (`src/stitching.py::stitch_predicted_dags`) into one global binary DAG via competitive differential thresholding: edge `i->j` activates iff `P(i->j) > 0.5` **and** `P(i->j) - P(j->i) > boundary_margin` (default 0.10) -- this is what structurally prevents the stitched output from ever containing a bidirectional 2-cycle. Higher-order cycles (3+ nodes) are separately checked via DFS (`detect_cycle`).

### 2.6 Reward computation (`src/rewards.py::compute_ippo_rewards`)

`diff = |stitched_dag - true_dag|`. Each agent's error is its **own local edges plus the shared boundary edges** (Agent 0: edges touching `Z1`, plus `X1<->X2`; Agent 1: edges touching `Z2`, plus `X1<->X2` -- so both agents are penalized for boundary mistakes, incentivizing implicit coordination there without any direct communication channel). Two reward regimes (`--reward_density`, **default `sparse`** as of the 18-run matrix result showing no measurable difference from dense):
- **`sparse`**: `0.0` every non-terminal step; `-error * edge_penalty` only on the final step of the episode.
- **`dense`**: `(prev_shd - current_shd) * edge_penalty` every step (reward for *improving* structure, not just being correct) -- always `-error` on the very first step (no `prev_shd` yet).

Both regimes: `-cycle_penalty` (default 10.0) added if the stitched DAG has a cycle; reward divided by `max_steps` (reward normalization); then optionally `+ intrinsic_coef * info_gain` and `+ impact_coef * impact_score` (both `0.0`/off by default). If `estimator_type == "learned"`, `update_graph_estimator` runs **after** this step's prediction/reward are already fixed (supervised BCE step against the true adjacency, own optimizer, gradients never touch the actor).

### 2.7 New observation (`_get_obs_dict`, see section 4) and termination check

Episode ends (`terminated=True`) when `step_count >= max_steps` (default 20) **or** both agents' budgets are exhausted (rarely triggers first, given 2.2's note). `step()` returns `(obs_dict, rewards, terminated, info)`, where `info` includes `true_adjacency`, `info_gains`, `impact_scores`, `shd`, `predicted_dag`, `shd_delta` (per-agent, for attributing structural improvement to whichever agent(s) intervened), and `asym_matrix`.

## 3. The three Stage-2 graph estimators (`predict_graph_hypothesis`)

All three consume the same `(obs_covariance, running_covariance, asymmetry)` inputs and produce a `[d,d]` probability matrix; they differ in how.

| `--estimator_type` | How it works | Trainable? | Default status |
|---|---|---|---|
| `avici` | Pretrained AVICI (`scm-v0` checkpoint) -- a permutation-invariant transformer, fed the real `raw_samples`/`raw_interv` buffer (capped to the most recent `avici_max_context` rows, default 400, to avoid per-step JIT recompilation) | No (frozen) | **Current default.** Reaches ~99% SHD=0 from early training in the diagnostic matrix, no memorization risk since it can't adapt, size-flexible for future scaling. |
| `analytic` | Pure formula: `S = 0.5*(|running_cov| + |obs_cov|)`, `O = 2*asymmetry`, `prob = sigmoid(S+O)` -- an invariance-testing heuristic, no learned parameters at all | No (frozen, and structurally shape-generic -- no retraining needed at a different `d`) | Original default; now the fallback if AVICI fails to load. |
| `learned` | `GraphEstimatorNet` (`src/marl/graph_estimator.py`) -- small separate Haiku network, own Adam optimizer, trained online every step via supervised BCE against the true adjacency (masked by `structural_mask`) | Yes, but its weight shapes are fixed to `d` at init -- does not generalize to a different node count without retraining | Gets the lowest raw SHD in testing, but the investigation's frozen-policy follow-up found its gains lean disproportionately on the estimator fitting itself rather than the policy learning -- see the doc's memorization discussion. |

If AVICI fails to import/load, the env silently falls back to `analytic` with a printed notice (`estimator_type` reassigned internally).

## 4. Observation composition (`_get_obs_dict`)

Per agent `k`, four parts concatenated into one flat vector (`obs_dim = 3*d*d + 1 [+ d*d if obs_feedback]` = 65 dims at `d=4` with feedback on, the current default):

1. `m_obs_cov` (`d*d`) -- the fixed reset-time baseline covariance, masked to agent `k`'s observable nodes.
2. `m_run_cov` (`d*d`) -- the cumulative running-average covariance, same mask (the channel identified as saturating -- see 2.4).
3. `m_asym` (`d*d`) -- the invariance-asymmetry matrix (`compute_invariance_asymmetry_matrix`: `A[i,j] = |1 - Var(X_j|do(X_i))/Var_obs(X_j)| - |1 - Var(X_i|do(X_j))/Var_obs(X_i)|`, gated by `int_mask` so unintervened nodes contribute nothing), same mask.
4. *(if `--obs_feedback true`, the current default)* `m_pred_dag` (`d*d`) -- agent `k`'s masked slice of the previous step's stitched global predicted DAG, fed back as an extra input. Whether this actually helps is untested as a controlled ablation (queued as follow-up work).
5. `budget` (`1`) -- agent `k`'s remaining budget scalar.

## 5. End of episode (in `train.py`, after `terminated=True`)

1. **GAE + PPO update, per agent independently** (`src/marl/ppo_trainer.py`): each agent's `RolloutBuffer` is padded to a static `max_steps` shape (avoids XLA recompilation across episodes of different lengths) and passed through `compute_gae` (`gamma=0.99`, `lambda=0.95`, raw/unnormalized advantages and returns). `IPPOTrainer.update_step` then computes the clipped PPO actor loss (`clip_eps=0.2`), an MSE critic loss, and an entropy bonus (`entropy_coef=0.01`), applies gradient-clipped Adam updates (warmup+cosine-decay LR schedule) to that agent's own actor and critic parameters, then resets its buffer. **No parameters or gradients are ever shared between the two agents.**
2. **Checkpointing**: if this episode's final stitched-DAG SHD is the best seen so far (strictly lower, or tied with higher F1), both agents' current actor/critic params are pickled to `best_ippo_params.pkl`. Note: this is a **single noisy stochastic episode's** SHD, not an average over several -- flagged in the investigation doc as a real fragility (a lucky early episode can "win" and never get displaced).
3. **Metrics logging**: `train/*` (episode reward, curriculum stage, per-agent info-gain) and `eval/*` (SHD, F1, plus the newer agent-vs-estimator-learning diagnostics: interventions-to-SHD0, SHD-curve AUC, orientation precision, redundancy rate, node/boundary coverage, entropy gain) get appended to `training_metrics.csv` (and WandB, if enabled) -- **all computed from this same still-training, stochastically-sampled episode**, not a frozen evaluation.
4. **After all episodes finish**: `evaluate.py::evaluate_checkpoint` runs automatically on the best-saved checkpoint, greedily/deterministically (or at `--eval_temperature`, default `0.0`) across **all 8 topologies** (not just the trained-on one), saving `evaluation_trace.json`. This is the frozen-policy evaluation used in the investigation's collapse-finding follow-up -- it is a genuinely separate measurement from every `eval/*` number logged during training.
