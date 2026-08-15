#!/bin/bash -l
# SGE array job for the single-agent experiment matrix.
#
# One array task per (d, observation, seed) combination. Seeds are fully independent, so
# this is near-linear speedup -- the local machine was running them serially on one core.
#
# Uses ~/envs/sa_env, a venv built specifically for `sa/`. Deliberately NOT the shared
# marl_env: installing extra dependencies into that environment previously caused a
# protobuf/wandb conflict, and `sa/` is meant to be isolated from the old project.
#
# Edit the CONFIGS array to change the matrix, and set -t to match its length.
# Check with:  qstat -u $USER
# Results in:  ~/sa_runs/<name>/result.json  and  logs/

#$ -N sa_matrix
#$ -cwd
#$ -l h_rt=06:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-15
#$ -o logs/
#$ -e logs/

mkdir -p logs
mkdir -p ~/.tmp
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa

export TMPDIR=~/.tmp
# One thread per task: the array already provides the parallelism, and letting each task
# grab many BLAS threads oversubscribes the node and slows everything down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# (d, observation, seed) -- 5 seeds x {d=5 edge_marginals, d=5 posterior, d=4 edge_marginals}
CONFIGS=(
  "5 edge_marginals 0" "5 edge_marginals 1" "5 edge_marginals 2" "5 edge_marginals 3" "5 edge_marginals 4"
  "5 posterior 0"      "5 posterior 1"      "5 posterior 2"      "5 posterior 3"      "5 posterior 4"
  "4 edge_marginals 0" "4 edge_marginals 1" "4 edge_marginals 2" "4 edge_marginals 3" "4 edge_marginals 4"
)

CFG=${CONFIGS[$((SGE_TASK_ID - 1))]}
read -r D OBS SEED <<< "$CFG"

NAME="d${D}_${OBS}_s${SEED}"
OUTDIR=~/sa_runs/${NAME}
mkdir -p "${OUTDIR}"

echo "=== ${NAME} ==="
echo "task ${SGE_TASK_ID}  d=${D}  observation=${OBS}  seed=${SEED}"
echo "host $(hostname)  started $(date)"

# -u so output is unbuffered and progress is visible in the log while the job runs, rather
# than appearing only at the end.
python -u -m scripts.run_experiment \
  --d "${D}" \
  --observation "${OBS}" \
  --seeds "${SEED}" \
  --train_episodes 6000 \
  --eval_episodes 300 \
  --out "${OUTDIR}/result.json"

echo "=== done ${NAME} $(date) ==="
