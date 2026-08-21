#!/bin/bash -l
# E3: does depth help in RL, not just in the supervised probe?
#
# CONDITIONAL. Only run if scripts/analyse_depth.py reports RULE FIRES. If it does not,
# this job is not submitted and the arm is recorded as skipped -- a pre-registered rule
# that gets run anyway "just to see" is not a rule.
#
# Set DEPTH to the depth the rule selected before submitting:
#   qsub -v DEPTH=2 submit_sa_e3_depth.sh
#
# Two arms x 5 seeds. Five rather than three because this is a head-to-head between two
# configurations whose difference is expected to be small: the supervised pilot showed
# 0.014 between depths on identical data, and G4's instability limit is 0.5 gap closed.
# Three seeds cannot resolve a difference of that size.

#$ -N sa_e3_depth
#$ -cwd
#$ -l h_rt=10:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -t 1-2
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/e3
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=offline WANDB_SILENT=true

if [ -z "${DEPTH}" ]; then
  echo "ERROR: DEPTH not set. Submit with: qsub -v DEPTH=<n> submit_sa_e3_depth.sh" >&2
  echo "The depth must be the one scripts/analyse_depth.py selected." >&2
  exit 1
fi

# Task 1 is the control at depth 1; task 2 is the depth the rule chose.
if [ "${SGE_TASK_ID}" = "1" ]; then LAYERS=1; else LAYERS="${DEPTH}"; fi

echo "=== E3 layers=${LAYERS} (rule selected depth ${DEPTH}) ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d 5 --observation edge_marginals --arch pernode --include_counts \
  --n_obs 5000 --train_episodes 6000 --eval_episodes 300 \
  --seeds 0 1 2 3 4 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --layers "${LAYERS}" \
  --gate1_episodes 200 --tag "e3_layers${LAYERS}" \
  --wandb_project sa-phase2 \
  --out "results/e3/e3_layers${LAYERS}.json"

echo "=== done $(date) ==="
