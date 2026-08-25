#!/bin/bash -l
# TRAIN the scale ladder. Evaluation is a separate array (submit_ladder_eval.sh) because
# the four arms are independent and, at the top rungs, cost more wall-clock than the
# training itself -- running them inline forces eval_episodes down until the confidence
# intervals stop resolving anything.
#
# 9 rungs x 5 seeds = 45 tasks.
#
#$ -N ma_ladder_train
#$ -cwd
#$ -t 1-45
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -e
mkdir -p logs results/ladder
source ~/envs/sa_env/bin/activate
cd ~/ma_tb
export PYTHONPATH=.
export TMPDIR=~/.tmp
mkdir -p ~/.tmp
# ONE THREAD, deliberately. The subset DP is numpy-heavy since the 2026-08-25 vectorisation
# and BLAS would otherwise grab every core on the node -- 45 array tasks each trying to use
# the whole machine is slower than 45 tasks each using one.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source cluster/ladder_rungs.sh
TASK=$((SGE_TASK_ID - 1))
rung_config $((TASK / 5))
SEED=$((TASK % 5))

OUT="results/ladder/${ARM}_s${SEED}.json"
if [ -f "$OUT" ]; then echo "$OUT exists -- skipping"; exit 0; fi

echo "=== TRAIN $ARM seed $SEED (d=$D budget=$BUDGET train=$TRAIN) : $(date) ==="
python -m scripts.ma_train \
  --seed "$SEED" --arm "$ARM" --skip_eval \
  --n_agents "$AGENTS" --n_private "$PRIV" --n_shared "$SHARED" \
  --n_obs 1000 --n_int 100 --budget "$BUDGET" --train_episodes "$TRAIN" \
  --turn_order round_robin --clamp_only \
  --out "$OUT"
echo "=== done $(date) ==="
