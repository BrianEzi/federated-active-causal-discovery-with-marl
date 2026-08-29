# Cluster work package — sampled-evidence ladder, w20 and w30

For the agent with Myriad access. Written 29 Aug 2026. **Read §1 before running anything.**

Split: this laptop is running **w04, w08, w12 × 3 seeds** locally (14.7 core-hours, ~3h).
The cluster half is **w20 and w30 × 3 seeds** — 92 core-hours, which is what a cluster is for.

---

## 1. Why this exists, and what it is testing

Every ladder result in the repo is `vs_evidence=oracle` — **96 of 104 runs** (see
`docs/ENGINES_AND_FLOW.md` §7). Thesis result 2 is *"under sampled inference the RL policy
beats greedy at fixed budget, converging as n → ∞"*, and **there is no ladder data for it.**

I evaluated the existing oracle-trained checkpoints under sampled evidence — a TRANSFER
test — and it fails:

| rung | learned (argmax) soft SHD | greedy | paired |
|---|---|---|---|
| w08 | 0.1428 | **0.1082** | +0.0346 ± 0.0055 **SIG** |
| w12 | 0.1170 | **0.0878** | +0.0292 ± 0.0042 **SIG** |

Greedy also wins the ERROR component (+0.0046 ± 0.0023 and +0.0082 ± 0.0021, both SIG) and
per-window solve. Identification is 0.000 for every arm.

**The mechanism, and why the transfer test cannot answer result 2.** Look at the repeat rate:
greedy **0.247 / 0.331**, learner **0.110 / 0.138**. Under ORACLE evidence a repeat is
strictly wasted — the ancestry is already revealed — so the learner correctly learned not to
repeat. Under SAMPLED evidence a repeat is exactly how you buy statistical power. The trained
policy encodes a rule that is *actively wrong* in the new regime. Only a retrain settles it.

**Do not report the transfer numbers as a result about RL.** They are a result about transfer.

## 2. The runs

Configs are copied from each rung's own result file, verified 29 Aug. **Do not retype them
from memory** — that error has been made twice on this project.

```bash
# w20: 4 agents, 10 private each, 10 shared, budget 20
# w30: 3 agents, 15 private each, 15 shared, budget 15
for seed in 0 1 2; do
  python scripts/ma_train.py --arm w20samp --seed $seed \
    --n_agents 4 --private_size 10 --n_shared 10 --budget 20 \
    --n_obs 60 --n_int 20 --turn_order round_robin --backend factored \
    --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 \
    --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
    --episode_mix confounded --cb_n_boot 12 \
    --vs_evidence sampled --vs_evidence_alpha 0.001 \
    --train_episodes 4000 --eval_episodes 200 --no_wandb --force \
    --out results/sampled/w20_s${seed}.json

  python scripts/ma_train.py --arm w30samp --seed $seed \
    --n_agents 3 --private_size 15 --n_shared 15 --budget 15 \
    ... same flags ... --out results/sampled/w30_s${seed}.json
done
```

`scripts/run_sampled_ladder.sh` has the exact invocation as a function — copy `run()` from it
rather than the block above, and call `run w20 4 10 10 20 $seed` / `run w30 3 15 15 15 $seed`.

## 3. Costing, and the caveat on it

Measured on this laptop: same 160 episodes at w08 took **10.6 s oracle, 89.4 s sampled** —
a **8.4× slowdown**. Applied to each rung's recorded oracle `train_seconds`:

| rung | oracle | sampled est. | × 3 seeds |
|---|---|---|---|
| w20 | 96 min | 13.4 h | 40.3 h |
| w30 | 124 min | 17.3 h | 51.9 h |

**The 8.4× was measured at k=8 and sampled cost scales with CI tests ≈ k².** Treat both rows
as LOWER BOUNDS — w30 could plausibly be 25–30 h per seed. Request wall-clock accordingly,
and prefer three separate single-seed jobs over one array job that dies at the limit.

## 4. What to run when they land

```bash
python scripts/shd_diagnose.py results/sampled/w20_s0.json --check decompose
python scripts/mi_gate.py results/sampled/w20_s0.json      # THE GATE
```

**The MI gate is mandatory, not optional.** A rung that never trained is not a negative
result. If I(S;A)/H is at the floor, that row's score is void — say so rather than reporting
it. `shd_diagnose.py --check decompose` now splits soft SHD into WRONG / UNSETTLED / RESIDUAL,
which under sampled evidence is the split that matters: at w04 error was only 6% of the
metric and the rest was still ambiguity.

**Two known traps in the metric, both documented in `docs/FINDINGS_SHD_2026_08_29.md`.**
1. `scripts/shd.py` evaluates the learned arm with `deterministic=False`. Use argmax, or at
   least report both — it was worth half the gap at w08 and w12 under oracle.
2. `scripts/shd.py` averages over WINDOWS, so a shared pair is counted once per agent — n
   times. Use `--check global` for the de-duplicated version. Under oracle that alone
   accounted for greedy's entire win at w04/w12/w20 and reversed it at w08.

## 5. Blockers on this end

- **This branch is 4+ commits ahead of origin and push is keychain-blocked**, so you cannot
  see `scripts/shd_diagnose.py`, `docs/FINDINGS_SHD_2026_08_29.md` or
  `docs/ENGINES_AND_FLOW.md` until Brian pushes interactively.
- `ssh myriad` does not resolve from this laptop — the `~/.ssh/config` block in
  `docs/MYRIAD_HPC_GUIDE.md` is not installed here. Cluster access is yours only.
