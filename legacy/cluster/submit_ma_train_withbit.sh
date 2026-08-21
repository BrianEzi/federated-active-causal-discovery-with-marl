#!/bin/bash -l
# Two-agent training: independent PPO, no CTDE.
#
# ARM ORDER IS DELIBERATE [U, 2026-08-19]. Tasks 1-20 are the NO-REGIME-BIT baseline;
# tasks 21-40 are the with-bit arm. The baseline is the simpler system -- it needs no
# disclosure protocol at all -- so it fails in fewer ways and a bug in it is attributable.
# The with-bit arm is then a measured delta against a clean reference rather than an
# assertion, and the baseline can run before the supervisor rules on whether the bit is
# admissible.
#
# BUDGET 8 [U, 2026-08-19]. The two gates want opposite budgets: GATE 2 (discrimination)
# peaks at 2-3 and is dead by 16, while GATE 3 (coordination) registers nothing below 5,
# because a confounded episode needs an agent to spend moves clamping for its partner AND
# moves experimenting on itself. Training follows GATE 3 -- coordination is the thesis.
#
# 20 SEEDS, NOT 3. The known failure mode is a 1-in-10 collapse into passing immediately,
# sd 0.154 on a median of 0.312. Three seeds cannot separate that from a real effect, and
# quoting a 3-seed median was a mistake made once already.
#
# WALLTIME. Measured locally at ~0.44 s/episode for joint_conf at budget 5; budget 8 runs
# longer episodes, so call it ~0.6 s/episode -> 4000 episodes is ~40 min, plus evaluation.
# 4 h is a wide margin on a projection, and projections here have been wrong both ways.
#
#   qsub submit_ma_train.sh

#$ -N ma_wb
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-20
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/ma_train
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast
export WANDB_MODE=offline WANDB_SILENT=true

# WITH-BIT ONLY. The combined array (176251) runs the no-bit baseline as tasks 1-20 and
# with-bit as 21-40, and Myriad is scheduling only a couple of tasks at a time -- so the
# with-bit arm, which is the one actually asked for, would not start for many hours. This
# job runs it directly. The combined array keeps producing the no-bit control alongside.
SEED=$((SGE_TASK_ID - 1))
ARM="withbit"
REGIME_FLAG="--disclose_regime"

echo "=== ma_train ${ARM} seed ${SEED} (task ${SGE_TASK_ID}) ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.ma_train2 \
  --seed "${SEED}" --arm "${ARM}" ${REGIME_FLAG} \
  --n_obs 1000 --n_int 100 --budget 8 \
  --train_episodes 4000 --eval_episodes 200 \
  --out "results/ma_train/${ARM}_s${SEED}.json"

echo "=== done $(date) ==="
