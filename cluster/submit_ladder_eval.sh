#!/bin/bash -l
# EVALUATE the trained ladder. 9 rungs x 5 seeds x 4 arms = 180 tasks, all independent.
#
# Submit with a hold on the training array so an arm cannot start before its checkpoint
# exists:  qsub -hold_jid ma_ladder_train cluster/submit_ladder_eval.sh
#
# Arms: learned against the three references the thesis compares to. `random_vary` is
# absent because these runs are clamp-only, where it has no legal action -- its absence is
# a property of the arm list, not an oversight.
#
#$ -N ma_ladder_eval
#$ -cwd
#$ -t 1-180
#$ -l h_rt=24:00:00
#$ -l mem=8G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -e
mkdir -p logs results/ladder_eval
source ~/envs/sa_env/bin/activate
cd ~/ma_tb
export PYTHONPATH=.
export TMPDIR=~/.tmp
mkdir -p ~/.tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source cluster/ladder_rungs.sh
ARMS=(learned random_clamp greedy pass)
TASK=$((SGE_TASK_ID - 1))
rung_config $((TASK / 20))
SEED=$(((TASK % 20) / 4))
EVAL_ARM=${ARMS[$((TASK % 4))]}

# 150 everywhere. One episode is worth 0.7 percentage points at 150, against 2.0 at 50 --
# and the differences this ladder exists to detect live below 2 points. Holding it constant
# also keeps rungs comparable to each other, which a per-rung episode count would not.
EPISODES=150

RUN="results/ladder/${ARM}_s${SEED}.json"
OUT="results/ladder_eval/${ARM}_s${SEED}_${EVAL_ARM}.json"
if [ ! -f "$RUN" ]; then echo "no trained run at $RUN -- skipping"; exit 0; fi
if [ -f "$OUT" ]; then echo "$OUT exists -- skipping"; exit 0; fi

echo "=== EVAL $ARM seed $SEED arm $EVAL_ARM : $(date) ==="
python -m scripts.ma_eval_arm --run "$RUN" --arm "$EVAL_ARM" \
  --episodes "$EPISODES" --out "$OUT"
echo "=== done $(date) ==="
