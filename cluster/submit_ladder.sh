#!/bin/bash -l
# THE SCALE LADDER: 2 agents / 5 nodes up to 5 agents / 30 nodes.
#
# One axis at a time, so a rung that fails says WHICH axis broke it:
#   rungs 0-2  agents   2 -> 3 -> 5, holding 1 private and 3 shared
#   rungs 3-4  shared   3 -> 4 -> 5, holding 5 agents and 1 private
#   rungs 5-8  private  1 -> 2 -> 3 -> 4 -> 5, holding 5 agents and 5 shared
#
# Cost grows on two axes that MULTIPLY: the window k = private + shared drives the O(3^k)
# subset DP, and C(shared,2) drives the confounding assignments. Screening caps the second
# at `screen_keep`, so past rung 4 the growth is k alone.
#
# PER-RUNG BUDGETS, not one setting for all. Episode wall-clock at rung 8 is orders above
# rung 0, and parallelism buys throughput ACROSS runs but never latency WITHIN one -- so a
# rung is made schedulable by asking for fewer, longer episodes, not more cores. The
# numbers come from scripts/ma_rung_timing.py; rerun it before changing them.
#
# Five seeds per rung. One episode is worth 1/eval_episodes, so a rung reported on fewer
# than ~150 eval episodes cannot resolve the differences this ladder is looking for.
#
#$ -N ma_ladder
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
# ONE THREAD, deliberately. The DP is now numpy-heavy and BLAS would grab every core on the
# node, so N single-core array tasks would each try to use the whole machine and thrash.
# Throughput here comes from the array, not from threads inside a task.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TASK=$((SGE_TASK_ID - 1))
RUNG=$((TASK / 5))
SEED=$((TASK % 5))

case $RUNG in
  0) AGENTS=2; PRIV=1; SHARED=3; BUDGET=10; TRAIN=2000; EVAL=200 ;;
  1) AGENTS=3; PRIV=1; SHARED=3; BUDGET=12; TRAIN=2000; EVAL=200 ;;
  2) AGENTS=5; PRIV=1; SHARED=3; BUDGET=15; TRAIN=2000; EVAL=200 ;;
  3) AGENTS=5; PRIV=1; SHARED=4; BUDGET=15; TRAIN=1500; EVAL=200 ;;
  4) AGENTS=5; PRIV=1; SHARED=5; BUDGET=15; TRAIN=1200; EVAL=150 ;;
  5) AGENTS=5; PRIV=2; SHARED=5; BUDGET=15; TRAIN=1000; EVAL=150 ;;
  6) AGENTS=5; PRIV=3; SHARED=5; BUDGET=20; TRAIN=800;  EVAL=150 ;;
  7) AGENTS=5; PRIV=4; SHARED=5; BUDGET=20; TRAIN=600;  EVAL=150 ;;
  8) AGENTS=5; PRIV=5; SHARED=5; BUDGET=20; TRAIN=400;  EVAL=150 ;;
esac

ARM="rung${RUNG}_${AGENTS}a_${PRIV}p_${SHARED}x"
OUT="results/ladder/${ARM}_s${SEED}.json"
if [ -f "$OUT" ]; then
  echo "$OUT already exists -- skipping"
  exit 0
fi

echo "=== $ARM seed $SEED : $(date) ==="
python -m scripts.ma_train \
  --seed "$SEED" --arm "$ARM" \
  --n_agents "$AGENTS" --n_private "$PRIV" --n_shared "$SHARED" \
  --n_obs 1000 --n_int 100 --budget "$BUDGET" \
  --train_episodes "$TRAIN" --eval_episodes "$EVAL" \
  --turn_order round_robin --clamp_only \
  --out "$OUT"
echo "=== done $(date) ==="
