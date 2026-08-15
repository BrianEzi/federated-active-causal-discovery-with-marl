#!/bin/bash -l
# d=6, stage 2 of 2: train and evaluate one seed per array task.
#
# MUST be submitted with -hold_jid on the references job, or every task will find no cache,
# compute its own references, and burn an hour each doing identical work:
#
#   REFS=$(qsub -terse submit_sa_d6_refs.sh)
#   qsub -hold_jid ${REFS} submit_sa_d6_seeds.sh
#
# One seed per task rather than all three in one, because at d=6 a seed costs ~3.5 hours
# and three of them serially would not fit a sensible walltime.
#
# Only edge_marginals is run here. The exact-posterior observation is 3,781,504 numbers
# wide at d=6 -- a 484M-parameter first layer -- so condition A is structurally out of
# reach at this size. That is not a limitation of the experiment; it IS the finding the
# A-vs-B comparison at d=4 and d=5 exists to quantify.

#$ -N sa_d6
#$ -cwd
#$ -l h_rt=10:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -t 1-3
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/d6
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa

export TMPDIR=~/.tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEED=$((SGE_TASK_ID - 1))

echo "=== d=6 seed ${SEED} ==="
echo "host $(hostname)  started $(date)"

# Runs the per-node architecture, the only configuration with evidence behind it: a
# supervised probe puts it at 0.814 accuracy predicting the oracle's choice against the flat
# network's 0.528 (chance 0.279). Its parameter count is also independent of d, so the model
# form used at d=4 and d=5 carries here unchanged -- which is what makes a d=6 run a
# continuation of the same experiment rather than a separate one.
python -u -m scripts.run_experiment \
  --d 6 --observation edge_marginals \
  --arch pernode --include_counts \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --train_episodes 4000 --eval_episodes 150 --budget 20 \
  --seeds "${SEED}" --tag "d6_s${SEED}" \
  --ref_cache ~/sa_runs/d6/refs_d6.pkl \
  --out ~/sa_runs/d6/d6_s${SEED}.json

echo "=== done seed ${SEED} $(date) ==="
