#!/bin/bash
# JOB 5 of docs/HANDOVER_2026_08_27.md section 9: isolate the GENERATOR from the episode
# count. The 4-agent headline (learned 0.937 vs greedy 0.405) was Erdos-Renyi at 20,000
# episodes; a later scale-free run at 3,000 had learned LOSING to greedy (0.442 vs 0.583).
# Two things changed at once, so the result is unattributable. Holding the episode count at
# 20,000 for BOTH generators is what makes the comparison mean something.
#
# Config is `results/vs_strict/reference_4a_b8.json`'s exactly -- 4 agents, 1 private,
# 3 shared, budget 8 -- so these runs are comparable to the ceiling (1.0) and the optimal
# round count (5.1) already measured there.
#
# Priced first, 320 episodes: 38 s scale-free, 36 s Erdos-Renyi, so ~40 min per run. Six
# runs concurrently on a 16-core box leaves ten cores free, which is the point -- the
# earlier lesson was that six UNTHROTTLED jobs put a machine's load average at 79 and
# starved each other. Every job here is pinned to one BLAS thread.
set -u
cd "$(dirname "$0")/.."
PY="C:/Workspace/MSc Project/.venv/Scripts/python.exe"
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=results/vs_generator
mkdir -p "$OUTDIR" logs

EPISODES=${EPISODES:-20000}
COMMON="--n_agents 4 --private_size 1 --n_shared 3 --budget 8 --backend version_space \
  --claim_bar 1.0 --reward_criterion claims --per_agent_reward --policy_arch gnn \
  --turn_order round_robin --episode_mix confounded --disclose_regime \
  --n_obs 60 --n_int 20 --observe_belief_channels --observe_partner_counts --vary_only"

for MODEL in sf er; do
  for SEED in 0 1 2; do
    OUT="$OUTDIR/gen_${MODEL}_s${SEED}.json"
    if [ -f "$OUT" ]; then
      echo "$OUT exists -- skipping"
      continue
    fi
    echo "launching $MODEL seed $SEED -> $OUT"
    "$PY" -m scripts.ma_train $COMMON --graph_model "$MODEL" \
      --train_episodes "$EPISODES" --eval_episodes 150 --seed "$SEED" \
      --no_wandb --arm "gen_${MODEL}" --out "$OUT" \
      > "logs/gen_${MODEL}_s${SEED}.log" 2>&1 &
  done
done
wait
echo "JOB5-ALL-DONE"
