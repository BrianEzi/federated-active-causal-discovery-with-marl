#!/bin/bash -l
# E4: gate-valid d=6. The first d=6 result on an environment that actually requires
# intervening -- GATE 1 needs n_obs >= 20000 at d=6, and the earlier d=6 runs used 1000.
#
# WALLTIME IS A PLACEHOLDER until job 146526 (the timing probe) reports. Set h_rt from that
# measurement before submitting; do not submit on the projection alone. d=6 runtime has now
# been mis-predicted twice -- 3.5 h against an actual 4.7 h, and a walltime kill that never
# came -- so the rule for this arm is measure, then size.
#
# Projection to check the measurement against: the hot-path work took d=6 at n_obs=20000
# from 1850.6 to 845.7 ms per step on a laptop, which scales the old 4.7 h/seed to ~2.7 h.
# Three seeds run sequentially inside one task, so ~8 h plus references.

#$ -N sa_e4_d6
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l mem=24G
#$ -pe smp 2
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/e4
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=offline WANDB_SILENT=true

# Memory is 24G rather than 12G: at d=6 the engine holds the 3.78M-DAG array (136 MB), the
# parent-set ids (91 MB) and the flat gather index (182 MB) added by the optimisation, plus
# the posterior and its temporaries on every step.

echo "=== E4: d=6, n_obs=20000, 3 seeds ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d 6 --observation edge_marginals --arch pernode --include_counts \
  --n_obs 20000 --train_episodes 6000 --eval_episodes 300 \
  --seeds 0 1 2 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --gate1_episodes 200 --tag e4_d6_n20000 \
  --wandb_project sa-phase2 \
  --out results/e4/e4_d6_n20000.json

echo "=== done $(date) ==="
