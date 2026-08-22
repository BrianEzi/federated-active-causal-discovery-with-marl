#!/bin/bash -l
# Does porting sa/policy.py's init + entropy_coef fix ma's collapse-prone defaults?
#
# sa/ measured, independently of anything in ma/: (1) default PyTorch init can leave the
# actor near-deterministic at episode 0 with nothing to explore with, (2) entropy_coef=0.01
# plateaus policy entropy at 1.09/1.386 rather than decaying properly. ma/ currently runs
# at exactly that entropy_coef, with default init. ma/'s OWN "1-in-10 seed collapse" was
# fixed via the turn-budget mechanism instead, so it is NOT confirmed this bites ma too --
# hence a measured A/B, not a silent default change.
#
# SAME SEEDS (0-9), SAME SETTINGS as results/ma_fixed/tb_clamp_s0-9.json, so this is a
# direct paired comparison against an existing, already-analysed baseline.
#
#   arms 1-10   baseline     entropy_coef=0.01 (default), no orthogonal init -- reproduces
#                            tb_clamp exactly, run fresh so torch's own RNG stream matches
#                            (env seed alone does not pin torch's init/sampling).
#   arms 11-20  sa_derived   entropy_coef=0.003, orthogonal_init -- the sa/-ported values.
#
#$ -N ma_sa_derived
#$ -cwd
#$ -t 1-20
#$ -l h_rt=04:00:00
#$ -l mem=4G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -e
mkdir -p logs results/sa_derived
source ~/envs/sa_env/bin/activate
cd ~/ma_tb
export PYTHONPATH=.
export TMPDIR=~/.tmp
mkdir -p ~/.tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TASK=$((SGE_TASK_ID - 1))
ARM_INDEX=$((TASK / 10))
SEED=$((TASK % 10))

COMMON="--n_obs 1000 --n_int 100 --budget 10 --train_episodes 2000 --eval_episodes 150 \
        --turn_order round_robin --clamp_only --disclose_regime"

case $ARM_INDEX in
  0) ARM=sa_derived_baseline; EXTRA="--entropy_coef 0.01" ;;
  1) ARM=sa_derived_ported;   EXTRA="--entropy_coef 0.003 --orthogonal_init" ;;
esac

OUT="results/sa_derived/${ARM}_s${SEED}.json"
if [ -f "$OUT" ]; then
  echo "$OUT already exists -- skipping"
  exit 0
fi

echo "=== $ARM seed $SEED : $(date) ==="
echo "cmd: python -m scripts.ma_train --seed $SEED --arm $ARM $COMMON $EXTRA --out $OUT"
python -m scripts.ma_train --seed "$SEED" --arm "$ARM" $COMMON $EXTRA --out "$OUT"
echo "=== done $(date) ==="
