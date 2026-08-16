#!/bin/bash -l
# d=7 stage 2 of 2: one seed per array task, on the enumeration-free path.
#
# THE ENVIRONMENT IS ALREADY VALIDATED. Job 148129 task 2 ran GATE 1 and GATE 2 at d=7 on
# 2026-08-16 at 02:28 and both passed: observational-only rate 0.0717 (CI 0.0517-0.0933)
# against a predicted singleton target of 0.0756 (CI 0.0737-0.0777), and greedy at 2.00
# interventions against random at 4.39 with disjoint intervals. Task 1 of that job was the
# d=6 control, where the DP path reproduces the enumerated answer.
#
# WALLTIME. Training is cheap: 40 ms per belief update at n_obs=20000, so 6000 episodes is
# well under an hour. The cost is the sampled oracle -- 1.04 s per call at 4000 draws --
# which scores every action of both evaluation passes over 300 episodes. Estimated ~1.5 h
# per seed; 6 h is a wide margin, and this is a projection rather than a measurement, so
# the margin is deliberate.
#
#   qsub -hold_jid <refs-job-id> submit_sa_d7_seeds.sh

#$ -N sa_d7_seeds
#$ -cwd
#$ -l h_rt=06:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-3
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/d7 results/d7
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast
export WANDB_MODE=offline WANDB_SILENT=true

SEED=$((SGE_TASK_ID - 1))
echo "=== d=7 seed ${SEED}: n_obs=20000, subset-DP path ==="
echo "host $(hostname)  started $(date)"

# Config matches the d=6 E4 arm exactly -- same lr, hidden, batch, budget, n_obs, episode
# counts and observation. The ONLY difference is d and the backend, which is what makes the
# +1.31 / +1.19 / +1.09 / ? trend a trend rather than four unrelated numbers.
python -u -m scripts.run_experiment \
  --d 7 --observation edge_marginals --arch pernode --include_counts \
  --n_obs 20000 --train_episodes 6000 --eval_episodes 300 --budget 20 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --oracle_draws 4000 \
  --seeds "${SEED}" --tag "d7_s${SEED}" \
  --ref_cache ~/sa_runs/d7/refs_d7_n20000.pkl \
  --gate1_episodes 200 \
  --wandb_project sa-phase2 \
  --out "results/d7/d7_s${SEED}.json"

echo "=== done $(date) ==="
