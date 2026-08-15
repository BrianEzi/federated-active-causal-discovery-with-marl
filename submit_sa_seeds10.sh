#!/bin/bash -l
# Ten seeds on the two cheap gate-valid headline configurations.
#
# The central claim currently rests on 3 seeds per size, which is the first thing an
# examiner presses on. After the 2026-08-15 optimisation work a seed at d=4 costs ~45 s and
# at d=5 ~96 s of training, so ten of each is under half an hour in total -- the cheapest
# available strengthening of the number the whole thesis rests on.
#
# Runs from a SEPARATE checkout at the post-optimisation commit, leaving ~/marl_sa on the
# code Phase 2 started with. The optimisations are bit-identical, so results are directly
# comparable; the separate checkout is about not changing code under a running experiment.

#$ -N sa_seeds10
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -t 1-2
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/seeds10
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=offline WANDB_SILENT=true

if [ "${SGE_TASK_ID}" = "1" ]; then D=4; else D=5; fi

echo "=== 10 seeds at d=${D}, n_obs=5000 ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d "${D}" --observation edge_marginals --arch pernode --include_counts \
  --n_obs 5000 --train_episodes 6000 --eval_episodes 300 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --gate1_episodes 200 --tag "seeds10_d${D}" \
  --wandb_project sa-phase2 \
  --out "results/seeds10/seeds10_d${D}.json"

echo "=== done $(date) ==="
