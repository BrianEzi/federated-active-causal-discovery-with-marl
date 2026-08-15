#!/bin/bash -l
# E4 timing probe: how long does d=6 at n_obs=20000 actually cost on Myriad?
#
# Sizing a d=6 job from a smaller measurement has now gone wrong twice -- once predicting
# 3.5 h against an actual 4.7 h, once predicting walltime kills that never came. The
# hot-path optimisation projects ~2.7 h/seed, but that extrapolates a laptop benchmark to a
# cluster node, so it is a prediction to be checked rather than a number to plan on.
#
# ONE seed, a short training run, and the real n_obs. Cost per environment step is what is
# being measured; the training length only has to be long enough to average over.
# Deliberately not producing a result anyone should cite -- 400 episodes is far too few.

#$ -N sa_d6_timing
#$ -cwd
#$ -l h_rt=03:00:00
#$ -l mem=16G
#$ -pe smp 2
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/timing
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=offline WANDB_SILENT=true

echo "=== d=6 timing probe, n_obs=20000 ==="
echo "host $(hostname)  started $(date)"
START=$(date +%s)

python -u -m scripts.run_experiment \
  --d 6 --observation edge_marginals --arch pernode --include_counts \
  --n_obs 20000 --train_episodes 400 --eval_episodes 60 --seeds 0 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --gate1_episodes 100 --tag d6_timing_probe \
  --out results/timing/d6_timing.json

END=$(date +%s)
echo "=== done $(date), elapsed $((END - START))s ==="
echo "Scale by 6000/400 = 15x for a full training run, then add references and evaluation."
