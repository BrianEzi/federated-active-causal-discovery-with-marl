#!/bin/bash -l
# E4 stage 1 of 2: reference policies for d=6 at n_obs=20000, computed once and cached.
#
# Measured by the timing probe (job 146526): the four references cost 648s together at this
# size -- random 131s, greedy_oracle 80s, edge_marginal_greedy 414s, no_intervention 23s.
# Modest on its own, but every seed needs the SAME references because they define the
# gap-closed scale, and sharing one cache guarantees all three seeds are measured against a
# numerically identical opponent rather than three that merely should agree.
#
#   REFS=$(qsub -terse submit_sa_e4_refs.sh)
#   qsub -hold_jid ${REFS} submit_sa_e4_seeds.sh

#$ -N sa_e4_refs
#$ -cwd
#$ -l h_rt=02:00:00
#$ -l mem=24G
#$ -pe smp 2
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/e4 results/e4
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== E4 references: d=6, n_obs=20000 ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d 6 --observation edge_marginals --arch pernode --include_counts \
  --n_obs 20000 --train_episodes 6000 --eval_episodes 300 --budget 20 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --seeds 0 --tag e4_refs \
  --ref_cache ~/sa_runs/e4/refs_d6_n20000.pkl --refs_only

echo "=== done $(date) ==="
