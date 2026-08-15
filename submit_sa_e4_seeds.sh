#!/bin/bash -l
# E4 stage 2 of 2: gate-valid d=6, one seed per array task.
#
# WALLTIME IS MEASURED, NOT PROJECTED. Timing probe (job 146526) on a real node:
# 400 training episodes took 993s, so 6000 episodes is 993 * 15 = 14,895s = 4.14 h. Plus
# evaluation at 300 episodes twice. 8 h per seed leaves comfortable margin.
#
# This corrects a projection made earlier the same day. The hot-path optimisation was
# benchmarked on a laptop, where d=6 at n_obs=20000 went from 1850.6 to 845.7 ms/step, and
# I scaled the old Myriad figure of 4.7 h/seed by that laptop RATIO to get ~2.7 h/seed.
# Mixing a Myriad baseline with a laptop ratio is not a valid extrapolation: the node is
# slower per core. What the optimisation did achieve is the thing that mattered -- 4.14 h
# at TWENTY TIMES the sample count, against 4.7 h before. Without it this arm would have
# been roughly 9 h/seed and would not have fitted.
#
# One seed per task rather than three in sequence, which the earlier draft had. Three
# sequential seeds is ~13 h against a 12 h walltime -- it would have been killed partway
# through the third seed, losing it entirely.
#
#   qsub -hold_jid <refs-job-id> submit_sa_e4_seeds.sh

#$ -N sa_e4_seeds
#$ -cwd
#$ -l h_rt=08:00:00
#$ -l mem=24G
#$ -pe smp 2
#$ -t 1-3
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/e4 results/e4
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=offline WANDB_SILENT=true

# Memory at 24G: the engine holds the 3.78M-DAG array (136 MB), the parent-set ids (91 MB)
# and the flat gather index added by the optimisation (182 MB), plus the posterior and its
# per-step temporaries.

SEED=$((SGE_TASK_ID - 1))
echo "=== E4 seed ${SEED}: d=6, n_obs=20000 ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d 6 --observation edge_marginals --arch pernode --include_counts \
  --n_obs 20000 --train_episodes 6000 --eval_episodes 300 --budget 20 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --seeds "${SEED}" --tag "e4_d6_s${SEED}" \
  --ref_cache ~/sa_runs/e4/refs_d6_n20000.pkl \
  --gate1_episodes 200 \
  --wandb_project sa-phase2 \
  --out "results/e4/e4_d6_s${SEED}.json"

echo "=== done $(date) ==="
