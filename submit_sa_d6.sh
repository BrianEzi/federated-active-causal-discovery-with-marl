#!/bin/bash -l
# d=6: locating where the learned agent stops beating the myopic oracle.
#
# Measured so far (gap_closed: 0 = matches random, 1 = matches greedy, >1 = beats greedy):
#     d=5, n_obs=1000, budget 2   +1.214   beats greedy 3/3
#     d=5, n_obs=1000, budget 3   +1.081   beats greedy 3/3
#     d=7, n_obs=100,  budget 3   +0.529   beats greedy 0/3
#     d=7, n_obs=1000, budget 3   +1.193   beats greedy 2/3
#
# So the agent clearly wins at d=5 and clearly does not at d=7 with scarce data. d=6 is
# the missing rung, and the interesting question is whether the advantage decays smoothly
# with dimension or falls off a cliff -- the first is a scaling curve worth reporting, the
# second suggests something specific breaks between 5 and 7.
#
# Budgets 2 and 3 only: that is where discrimination peaks and where the d=5 advantage was
# largest, so it is where a difference would show up first. 5 seeds rather than 3, because
# the d=7 spread (one seed at -0.94) showed 3 is not enough to separate a trend from a
# collapse.
#
#   qsub submit_sa_d6.sh

#$ -N sa_d6
#$ -cwd
#$ -l h_rt=03:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-20
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/d6_exact results/d6_exact
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast
export WANDB_MODE=offline WANDB_SILENT=true

CONFIGS=("6 100 2" "6 100 3" "6 1000 2" "6 1000 3")
INDEX=$(( (SGE_TASK_ID - 1) / 5 ))
SEED=$(( (SGE_TASK_ID - 1) % 5 ))
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
  --ref_cache ~/sa_runs/d6_exact/refs_"${TAG}".pkl \
  --out "results/d6_exact/${TAG}_s${SEED}.json"

echo "=== done $(date) ==="
