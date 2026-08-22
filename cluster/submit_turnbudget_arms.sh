#!/bin/bash -l
# Four two-agent arms x 10 seeds = 40 independent tasks, as an SGE array.
#
# Why the cluster and not the laptop: on the night of 21/22 August these same four arms were
# run locally, the machine went to sleep around 01:00, and NOTHING completed. A cluster task
# is immune to that. One task per (arm, seed) so a partial failure is legible and a rerun is
# cheap.
#
#   arms 1-10   nobit_clamp     regime bit OFF -- the ablation. Every number we have has the
#                               bit ON, so we cannot yet claim the federation channel earns
#                               its place. Highest value of the four.
#   arms 11-20  randturn_clamp  random turn order instead of round-robin -- raised by the
#                               supervisor, never swept.
#   arms 21-30  tb_clamp        seeds 10-19, to resolve the +1.8pp clamp-only lean.
#   arms 31-40  tb_both         seeds 10-19, its pair (the comparison is PAIRED, so both arms
#                               need the same seeds).
#
#$ -N ma_turnbudget
#$ -cwd
#$ -t 1-40
#$ -l h_rt=04:00:00
#$ -l mem=4G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -e
mkdir -p logs results/ma_fixed
source ~/envs/sa_env/bin/activate
cd ~/ma_tb
export PYTHONPATH=.
export TMPDIR=~/.tmp
mkdir -p ~/.tmp
# One thread per task: 40 single-threaded tasks schedule far better here than a few
# multi-threaded ones, and torch otherwise grabs every core on the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TASK=$((SGE_TASK_ID - 1))
ARM_INDEX=$((TASK / 10))
OFFSET=$((TASK % 10))

COMMON="--n_obs 1000 --n_int 100 --budget 10 --train_episodes 2000 --eval_episodes 150"

case $ARM_INDEX in
  0) ARM=nobit_clamp;     SEED=$OFFSET;         EXTRA="--turn_order round_robin --clamp_only" ;;
  1) ARM=randturn_clamp;  SEED=$OFFSET;         EXTRA="--turn_order random --clamp_only --disclose_regime" ;;
  2) ARM=tb_clamp;        SEED=$((OFFSET + 10)); EXTRA="--turn_order round_robin --clamp_only --disclose_regime" ;;
  3) ARM=tb_both;         SEED=$((OFFSET + 10)); EXTRA="--turn_order round_robin --disclose_regime" ;;
esac

OUT="results/ma_fixed/${ARM}_s${SEED}.json"
if [ -f "$OUT" ]; then
  echo "$OUT already exists -- skipping"
  exit 0
fi

echo "=== $ARM seed $SEED : $(date) ==="
echo "cmd: python -m scripts.ma_train --seed $SEED --arm $ARM $COMMON $EXTRA --out $OUT"
python -m scripts.ma_train --seed "$SEED" --arm "$ARM" $COMMON $EXTRA --out "$OUT"
echo "=== done $(date) ==="
