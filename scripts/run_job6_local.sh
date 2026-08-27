#!/bin/bash
# JOB 6 of docs/HANDOVER_2026_08_27.md section 9: the noise dial, CONVERGED. The earlier
# attempt (results/vs_dial/) stopped at 3,000 episodes and the addendum says explicitly not
# to read anything into it. This runs the same three noise levels to 20,000.
#
# Config copied from results/vs_dial/n1000_s0.json's own config block, NOT from the
# attribution jobs -- the dial runs use --n_obs 1000 and --vs_evidence_alpha 0.001, and a
# first pricing pass at --n_obs 60 understated the cost by about a third AND would have
# produced numbers that could not be set against the runs they are meant to converge.
#
# Priced at the real config, 160 episodes: 243 min at n_int 100, 283 at 1000, 462 at 4000.
# Six runs concurrently, each pinned to one BLAS thread; the n_int 4000 pair sets the wall
# clock at roughly eight hours.
set -u
cd "$(dirname "$0")/.."
PY="C:/Workspace/MSc Project/.venv/Scripts/python.exe"
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=results/vs_dial_converged
mkdir -p "$OUTDIR" logs

EPISODES=${EPISODES:-20000}
COMMON="--n_agents 4 --private_size 1 --n_shared 3 --budget 8 --backend version_space \
  --claim_bar 1.0 --reward_criterion claims --per_agent_reward --policy_arch gnn \
  --turn_order round_robin --episode_mix confounded --disclose_regime \
  --n_obs 1000 --observe_belief_channels --observe_partner_counts --vary_only \
  --graph_model sf --vs_evidence sampled --vs_evidence_alpha 0.001"

for LEVEL in 100 1000 4000; do
  for SEED in 0 1; do
    OUT="$OUTDIR/dial_n${LEVEL}_s${SEED}.json"
    if [ -f "$OUT" ]; then
      echo "$OUT exists -- skipping"
      continue
    fi
    echo "launching n_int=$LEVEL seed $SEED -> $OUT"
    "$PY" -m scripts.ma_train $COMMON --n_int "$LEVEL" \
      --train_episodes "$EPISODES" --eval_episodes 150 --seed "$SEED" \
      --no_wandb --arm "dial_n${LEVEL}" --out "$OUT" \
      > "logs/dial_n${LEVEL}_s${SEED}.log" 2>&1 &
  done
done
wait
echo "JOB6-ALL-DONE"
