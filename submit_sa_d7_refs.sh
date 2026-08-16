#!/bin/bash -l
# d=7 stage 1 of 2: the shared reference traces.
#
# Separate job, exactly as E4 did at d=6, because three seeds sharing `--ref_cache` would
# otherwise all find it missing at the same moment and each compute it -- triple the cost
# and a race on the file.
#
# The references are the expensive part at d=7, not the training. A belief update is 40 ms,
# but every scored action costs one sampled-oracle call at 1.04 s (4000 MH draws), and both
# greedy baselines plus the random baseline are scored on every step of 300 episodes.
#
#   qsub submit_sa_d7_refs.sh

#$ -N sa_d7_refs
#$ -cwd
#$ -l h_rt=06:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/d7 results/d7
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast
export WANDB_MODE=offline WANDB_SILENT=true

# 8G, against 24G for the enumerated d=6 arm. The DP path holds a [7, 128] score table and
# never materialises a DAG list -- the memory saving IS the method.

echo "=== d=7 references: n_obs=20000 ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d 7 --observation edge_marginals --arch pernode --include_counts \
  --n_obs 20000 --train_episodes 6000 --eval_episodes 300 --budget 20 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --oracle_draws 4000 \
  --seeds 0 --tag d7_refs --refs_only \
  --ref_cache ~/sa_runs/d7/refs_d7_n20000.pkl \
  --gate1_episodes 200 \
  --out "results/d7/d7_refs.json"

echo "=== done $(date) ==="
