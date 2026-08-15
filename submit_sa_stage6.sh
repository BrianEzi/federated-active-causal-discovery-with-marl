#!/bin/bash -l
# The overnight single-agent lever sweep: 34 configurations, 110 (config, seed) runs.
#
# The matrix lives in scripts.sweep_stage6.py, not here -- this script only asks it what
# task N should run. That keeps the definition of the experiment in one place, readable by
# both the submitter and the analysis.
#
# One array task per CONFIGURATION (all its seeds inside), because the four reference
# policies cost ~6 minutes at d=5 and depend only on the environment levers. Computing
# them per seed instead would waste several hours across the matrix.
#
#   qsub submit_sa_stage6.sh
#   qstat -u $USER
#   results in ~/sa_runs/stage6/<tag>.json
#
# Longest expected task is ~100 min (train_episodes_12000, 3 seeds); 8h is deliberate
# headroom, since an array task killed at the walltime loses every seed it was holding.

#$ -N sa_stage6
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-4
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/stage6
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa

export TMPDIR=~/.tmp
# One BLAS thread per task: the array provides the parallelism, and letting each task grab
# many threads oversubscribes the node and slows the whole matrix down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ARGS=$(python -m scripts.sweep_stage6 --cli "${SGE_TASK_ID}")
TAG=$(python -c "
from scripts.sweep_stage6 import build_matrix
print(build_matrix()[${SGE_TASK_ID} - 1]['tag'])
")

echo "=== task ${SGE_TASK_ID}: ${TAG} ==="
echo "host $(hostname)  started $(date)"
echo "args ${ARGS}"

# -u so progress is visible in the log while the job runs rather than only at the end.
python -u -m scripts.run_experiment ${ARGS} --out ~/sa_runs/stage6/${TAG}.json

echo "=== done ${TAG} $(date) ==="
