#!/bin/bash -l
# RUNG 0 WITH THE REGIME BIT -- the validated two-agent configuration, as an anchor.
#
# The main ladder runs with `disclose_regime` OFF, because the guard refuses it from three
# agents onward: the bit is a clean FRACTION (`n_clamped / len(hidden)`), which is an
# identity only when an agent has exactly one hidden node. Two agents with one private node
# each is the ONLY rung where it is both legal and meaningful.
#
# Measured at 250 episodes, budget 10, prior 0.5:
#
#   disclose_regime   random  greedy   pass
#              True    0.356   0.212  0.008     <- reproduces results/ma_fixed/ (0.387/0.240)
#             False    0.080     --   0.010
#
# So the bit is worth ~4.5x on the success rate, and without it the whole ladder runs in a
# regime where the baselines sit near the floor. This arm exists so the ladder has a rung
# whose baselines are known-good, which makes the no-bit rungs interpretable as a
# CONTRAST rather than as an unexplained collapse.
#
# PINNED to --prior_p 0.5, matching results/ma_fixed/ exactly, so this is a reproduction
# rather than a new measurement.
#
#$ -N ma_rung0_bit
#$ -cwd
#$ -t 1-5
#$ -l h_rt=8:00:00
#$ -l mem=8G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -e
mkdir -p logs results/rung0_bit results/rung0_bit_eval
source ~/envs/sa_env/bin/activate
cd ~/ma_tb
export PYTHONPATH=.
export TMPDIR=~/.tmp
mkdir -p ~/.tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SEED=$((SGE_TASK_ID - 1))
ARM="rung0bit_2a_1p_3x_d5"
OUT="results/rung0_bit/${ARM}_s${SEED}.json"

if [ ! -f "$OUT" ]; then
  echo "=== TRAIN $ARM seed $SEED : $(date) ==="
  python -m scripts.ma_train \
    --seed "$SEED" --arm "$ARM" --skip_eval --disclose_regime --prior_p 0.5 \
    --n_agents 2 --n_private 1 --n_shared 3 \
    --n_obs 1000 --n_int 100 --budget 10 --train_episodes 3000 \
    --turn_order round_robin --clamp_only --out "$OUT"
fi

for EVAL_ARM in learned random_clamp greedy pass; do
  EOUT="results/rung0_bit_eval/${ARM}_s${SEED}_${EVAL_ARM}.json"
  [ -f "$EOUT" ] && continue
  python -m scripts.ma_eval_arm --run "$OUT" --arm "$EVAL_ARM" --episodes 250 --out "$EOUT"
done
echo "=== done $(date) ==="
