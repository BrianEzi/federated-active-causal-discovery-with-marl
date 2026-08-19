#!/bin/bash -l
# STAGE 2 of 2 -- the GNN (per-node) agent across intervention budgets.
#
# 14 configs x 3 seeds = 42 tasks. Task -> config is integer division by 3, so seeds of one
# config land on consecutive task ids and a partial failure is easy to read off the logs.
#
# MUST be submitted with a hold on the refs job, or every task recomputes the greedy oracle
# and three tasks race to write the same cache file:
#
#   qsub -hold_jid <refs-job-id> submit_sa_gnn_budget.sh
#
# HYPERPARAMETERS ARE THE E4 / d=7 ARM, UNCHANGED. lr 1e-3, hidden 256,
# episodes_per_update 16, include_counts, arch pernode. This is not tuning -- holding them
# fixed is what makes the budget axis interpretable. The Phase 2 sweep measured the largest
# effect of ANY of 13 levers on this architecture at 0.288, so the budget result cannot be
# explained away as a tuning artefact.
#
# WALLTIME. Locally, 3 seeds at d=5/budget=2 took 329 s INCLUDING first-time references.
# Training alone is ~0.12 s/episode at d=5 and ~0.16 s at d=7, so 4000 episodes is ~8-11
# minutes per seed. With references cached, an hour per task is a wide margin. 4 h is
# deliberately wider still: the local estimate is a projection, and projections in this
# project have been wrong in both directions before.

#$ -N gnn_budget
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-42
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/gnn_budget results/gnn_budget
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast
export WANDB_MODE=offline WANDB_SILENT=true

CONFIGS=(
  "5 100 2"   "5 100 3"   "5 100 5"   "5 100 8"
  "7 100 3"   "7 100 5"   "7 100 16"
  "5 1000 2"  "5 1000 3"  "5 1000 5"  "5 1000 8"
  "7 1000 3"  "7 1000 5"  "7 1000 16"
)

INDEX=$(( (SGE_TASK_ID - 1) / 3 ))
SEED=$(( (SGE_TASK_ID - 1) % 3 ))
read -r D NOBS BUDGET <<< "${CONFIGS[$INDEX]}"
TAG="d${D}_nobs${NOBS}_b${BUDGET}"

echo "=== ${TAG} seed ${SEED} (task ${SGE_TASK_ID}) ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d "${D}" --observation edge_marginals --arch pernode --include_counts \
  --n_obs "${NOBS}" --budget "${BUDGET}" \
  --train_episodes 4000 --eval_episodes 150 --gate1_episodes 200 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --oracle_draws 4000 \
  --seeds "${SEED}" --tag "${TAG}_s${SEED}" \
  --ref_cache ~/sa_runs/gnn_budget/refs_"${TAG}".pkl \
  --out "results/gnn_budget/${TAG}_s${SEED}.json"

echo "=== done $(date) ==="
