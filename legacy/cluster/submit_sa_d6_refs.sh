#!/bin/bash -l
# d=6, stage 1 of 2: compute the reference policies once and cache them.
#
# Split out from the training stage because at d=6 a single posterior update costs ~0.7s,
# which puts the four reference policies at roughly an hour. Every seed needs the SAME
# references -- they define the gap-closed scale -- so computing them once and sharing the
# cache saves hours and, more importantly, guarantees all three seeds are measured against
# a numerically identical opponent.
#
#   qsub submit_sa_d6_refs.sh          # then submit_sa_d6_seeds.sh with -hold_jid

#$ -N sa_d6_refs
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/d6
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa

export TMPDIR=~/.tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== d=6 references ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d 6 --observation edge_marginals \
  --train_episodes 4000 --eval_episodes 150 --budget 20 \
  --seeds 0 --tag d6_refs \
  --ref_cache ~/sa_runs/d6/refs_d6.pkl --refs_only

echo "=== done $(date) ==="
