#!/bin/bash -l
# THE CONFOUND-BREAKER. Both arms with step_cost = 0.
#
# The overnight result was: with the regime bit 0/20 seeds collapse, without it 20/20 do.
# That reads as "the bit makes the problem learnable". But the expected-value arithmetic
# says something else. Using the MEASURED random-policy numbers:
#
#     with bit : 0.250 - 0.05*7.40 = -0.120   vs pass = 0.000  -> PASS is better
#     no bit   : 0.133 - 0.05*7.76 = -0.255   vs pass = 0.000  -> PASS is better
#
# Passing is optimal in BOTH arms at random-policy level. The step cost is a barrier the
# policy has to climb over, and with the bit there exists a policy that clears it
# (0.660 - 0.05*5.44 = +0.388) while without the bit there may not be one. If that is the
# whole story then the headline is about REWARD DESIGN, not about the disclosure.
#
# Removing the step cost separates the two. Expected outcomes and what each would mean:
#   no-bit stops collapsing        -> the collapse was the step cost; the bit's role is
#                                     smaller than claimed and the headline must be restated
#   no-bit still collapses         -> the bit really is what makes the task learnable
#   with-bit improves a lot too    -> the step cost was limiting BOTH arms, and every
#                                     number reported so far understates what is achievable
#
# Tasks 1-10 no-bit, 11-20 with-bit. Ten seeds each, which is enough to tell a 0/10 from a
# 10/10 collapse rate -- the effect being tested is that large or it is not there.
#
#   qsub submit_ma_stepcost.sh

#$ -N ma_cost0
#$ -cwd
#$ -l h_rt=03:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-20
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/ma_stepcost
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast
export WANDB_MODE=offline WANDB_SILENT=true

if [ "${SGE_TASK_ID}" -le 10 ]; then
  SEED=$((SGE_TASK_ID - 1)); ARM="nobit_cost0"; REGIME_FLAG=""
else
  SEED=$((SGE_TASK_ID - 11)); ARM="withbit_cost0"; REGIME_FLAG="--disclose_regime"
fi

echo "=== ${ARM} seed ${SEED} (task ${SGE_TASK_ID}) ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.ma_train2 \
  --seed "${SEED}" --arm "${ARM}" ${REGIME_FLAG} \
  --n_obs 1000 --n_int 100 --budget 8 --step_cost 0.0 \
  --train_episodes 3000 --eval_episodes 200 \
  --out "results/ma_stepcost/${ARM}_s${SEED}.json"

echo "=== done $(date) ==="
