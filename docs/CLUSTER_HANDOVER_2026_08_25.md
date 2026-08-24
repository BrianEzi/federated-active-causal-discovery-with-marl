# Handover — overnight cluster runs, 25 August 2026

**You are an agent with cluster access, starting cold. Read this file completely, then
`docs/logs/SA_EXPERIMENT_LOG.md` entries dated 2026-08-24, before launching anything.**

Everything is pushed on `explore/constraint-based` — clone HEAD; the own-counts observation fix (`fed869f`) and this doc are the last entries.

**Deadlines: experiment freeze 31 August. Dissertation due 8 September.**

---

## 1. Setup

```bash
git clone https://github.com/BrianEzi/federated-active-causal-discovery-with-marl
cd federated-active-causal-discovery-with-marl
git checkout explore/constraint-based
pip install -r requirements.txt          # numpy, scipy, torch, pytest
python -m pytest tests/ -q               # expect ~285 passed, 0 xfailed, ~100 s
```

Run tests from the repo root (fixture paths are cwd-relative).
**DO NOT MERGE THIS BRANCH INTO MAIN** — it deletes `sa/` and `legacy/`; cherry-pick if
the exploration is adopted. **Do not run the test suite alongside a training job.**

## 2. The project, in three sentences

MSc thesis: federated active causal discovery with multi-agent RL. Agents each see part
of one causal system (private + shared variables), spend a shared budget on
interventions, and update structural beliefs; the questions are whether agents can LEARN
to choose experiments (better than greedy) and whether privacy-preserving cooperation
helps. Confounding — a hidden node of one agent driving pairs another agent sees — is
the thesis's central quantity.

## 3. What changed on 2026-08-24 (read the log for the numbers)

- **Constraint-based belief engine is live** behind `MAConfig.belief_backend="constraint"`
  (`cb/`): bootstrap PC + interventional orientation. The exact Bayesian engine
  (`crosscheck/`, frozen) is UNSOUND for topologies hiding >1 node from an agent and is
  refused there by a capability check; the constraint engine is what makes 3-agent runs
  possible at all.
- **Vary-mode interventions adopted** for constraint arms (`action_modes=(VARY,)`):
  randomised values give first-order detection power; clamp-to-0 hides everything in
  variances.
- **Role-aware GNN policy** (`policy_arch="gnn"`): per-node scorer with
  is_shared/has_authority role features; equivariant within roles.
- **"claims" reward criterion** (`reward_criterion="claims"`, `cb/claims.py`): every
  edge-question is scored settled-right / unsure / settled-wrong at a 0.7 confidence
  bar; dense reward = per-step change in (right − wrong)/total; identified = all
  REQUIRED claims right and nothing anywhere confidently wrong. This replaced an
  all-or-nothing criterion whose conjunction turned 95% per-claim accuracy into 36%
  success and a luck-dominated training signal.
- **Episode mix control** (`episode_mix="confounded" | "unconfounded" | "any"`):
  training runs on confounded episodes (the thesis's subject, only ~15% of free draws);
  unconfounded episodes are the standing SANITY arm.
- **Nine bugs found and fixed** since the engine landed, every one by direct
  measurement, none by a downstream metric. Log entries 2026-08-23/24 carry each.

## 4. Where the numbers stand (laptop, k=4 topology, 100-episode paired evals)

| policy | confounded eps | unconfounded eps |
|---|---|---|
| scripted near-oracle (pair completion) | 0.250 | 0.590 |
| greedy_uncertainty (myopic baseline) | 0.140 | 0.580 |
| random | 0.160 | 0.440 |
| learned, 1500 episodes | 0.12 — **indistinguishable from random** (see below) | — |

- Sanity gate (must hold on every eval): **zero settled-wrong confounding claims on
  unconfounded episodes.** Measured 0/120 windows at handover.
- VERIFICATION (120 paired episodes, 2026-08-25): the prelim eval's learned 0.17 vs
  random 0.07 was BATCH NOISE — on the verification batch both score exactly 14/120,
  and the checkpoint's behaviour matches random (pair-completion 93 vs 99 of 120;
  entropy still 1.40 of 1.61). **There is no evidence yet that training separates from
  random; that is the open question the overnight runs answer, not a confirmation
  run.** Also measured: pair-completion is NOT the differentiator (random completes
  82% of pairs automatically at this budget) — the scripted policy's 2x margin over
  random comes from BALANCED coverage: each authority node exactly one block, own
  private node first. Greedy completes pairs only 50% yet edges the field (0.14).
  The learnable behaviour is balance + ordering, not raw coverage.
- **LATE ADDITION (2026-08-25, after the verification): the unlearnability was
  diagnosed and fixed.** The observation lacked the agent's own intervention history,
  so "touch each node once" could not be learned. Per-node own-counts are now in the
  observation (commit `fed869f`); an obs-only rule using them scores 0.30 vs random
  0.07 -- at the scripted ceiling. Overnight runs pick this up automatically; old
  checkpoints are refused by the obs_size check. The overnight expectation IMPROVES:
  training should now separate from random early; if it does not, suspect per-step
  reward noise and report, don't retune.
- Timing note: a full episode costs ~1.0 s in isolation (either n_obs); 2.3 s/ep was
  observed during the prelim train, unexplained, plausibly laptop contention. Measure
  on-cluster before sizing arrays.

## 5. The overnight job

`scripts/ma_train.py` is one seed per invocation, built for array jobs. Common flags:

```
COMMON="--backend constraint --policy_arch gnn --vary_only --disclose_regime \
  --turn_order round_robin --reward_criterion claims --episode_mix confounded \
  --n_obs 1000 --n_int 250 --cb_n_boot 12 --entropy_coef 0.003 --orthogonal_init \
  --eval_episodes 200"
```

Priority order (drop from the bottom if capacity binds):

1. **Rung 0 long runs** — the "learned catches greedy/scripted" question:
   `--seed {0..4} --arm cb_claims_rung0 --budget 8 --train_episodes 16000 \
    --out results/cb_gnn/claims_rung0_s{seed}.json`
2. **Rung 1 — three agents** (the never-before-runnable configuration):
   add `--three_agents --budget 9`, arm `cb_claims_rung1`, seeds 0–2.
3. **B=25 arm**, seed 0 only, `--cb_n_boot 25`, arm `cb_claims_rung0_b25`: tests whether
   the confounding claim's 51%-at-bar rate is bootstrap granularity (frequencies move in
   steps of 1/12 at B=12) or structural. Costs ~2× per episode.

**Measure one episode's wall-clock on the cluster before sizing the array** (standing
practice — laptop reference: ~2.3 s/episode at these settings). Do not run seeds of the
same arm concurrently on one node.

## 6. Gates and canaries — check these, in this order, on every completed run

1. `collapsed: false` and `first_success_episode` small (reward is being sampled).
2. Entropy falls (start 1.61; the 1500-ep run reached 1.40; expect <1.2 by 16k).
   Flat-at-max entropy = the policy never differentiated.
3. `learned` vs paired baselines: at handover, learned (1500 eps) is INDISTINGUISHABLE
   from random. The overnight question is whether 16k episodes separates it — first
   from random, then toward greedy_uncertainty (~0.14) and the scripted ceiling (0.25).
   Learned ≈ random after 16k is a REAL FINDING about this reward/architecture at this
   scale — report it as such with the paired numbers; do not quietly retune and rerun.
4. Sanity: any settled-wrong confounding on an unconfounded eval arm → STOP EVERYTHING.
   That failure mode (inventing confounders) is the one the thesis cannot survive.
5. Same seed twice must reproduce bit-for-bit (determinism test exists in the suite).

## 7. Traps that have already cost time

- **Numbers are only comparable within a criterion, a backend, an architecture, and an
  episode mix.** Checkpoints record all four and refuse mismatched loads. Never quote
  claims-criterion success against u14 numbers.
- All comparisons are PAIRED: `run_arm` gives every arm the same episode seeds. Never
  compare success rates across different seed batches (random drifted 0.16 → 0.07
  between batches; within a batch the pairing absorbs it).
- `results/` is tracked; `ma_train.py` refuses to overwrite a result produced under a
  different config unless `--force`. Respect it — the guard exists because of a real
  silent-clobber incident.
- `crosscheck/` is frozen reference code; `ma/nets.py` is verbatim-frozen
  (`tests/test_depth.py`). The GNN wrapper lives in `ma/policy.py`.
- The `greedy` (exact-DP) baseline raises on the constraint backend by design;
  `greedy_uncertainty` is the constraint-side baseline and is auto-included.
- Log to `docs/logs/SA_EXPERIMENT_LOG.md` as results land — `[MEASURED]` / `[DECIDED]` /
  `[CORRECTED]`, nulls and self-corrections included. Push after each block of work.

## 8. Known-open items (do not silently "fix" these overnight)

- Confounding claims reach the 0.7 bar in only 51% of truly-confounded pairs (19% are
  structural misses — no per-episode power). The B=25 arm probes the granularity half.
  Anything deeper is a design decision for the student.
- The per-replicate u14 translation is superseded but still reported (`mass_credit`
  diagnostics); leave it as a diagnostic.
- JCI-style regime-indicator engine: designed, deliberately NOT built (deadline). The
  pooling it would fix was measured net-positive as-is.
- Scale ladder (k=7–9) and the disclosure track wait on tonight's results.

## 9. Reading order

1. This file, then `docs/logs/SA_EXPERIMENT_LOG.md` from "2026-08-24" onward.
2. `docs/MORNING_2026_08_24.md` — the first overnight's decisions and results.
3. `cb/claims.py` and `cb/orient.py` docstrings — the criterion and the engine's logic.
4. `docs/HANDOVER_2026_08_23.md` — the wider project, still accurate for everything
   not superseded above.
