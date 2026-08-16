#!/bin/bash -l
# GATE 1 and GATE 2 at d=6 and d=7, on the enumeration-free path.
#
# WHY THIS RUNS BEFORE ANY d=7 TRAINING. The d=6 results from 2026-08-14 had to be thrown
# away because GATE 1 failed on that environment -- the task did not require intervening,
# so "the agent beats greedy" measured nothing. Training at d=7 before validating the d=7
# environment would repeat that mistake exactly one size up, and burn far more compute
# doing it.
#
# Task 1 is d=6, where BOTH gates can also be computed by enumeration. It is the control:
# if the DP path disagrees with the enumerated answer at d=6, the d=7 number in task 2 is
# not to be believed regardless of what it says.
#
# WALLTIME, from measurement on this laptop (Myriad is 1.2-2x faster per core on this
# workload, so this is conservative):
#   - one sampled-oracle call at d=7, 4000 draws: 0.77 s
#   - GATE 2 greedy: 300 episodes x up to 10 steps = 3000 calls = ~40 min
#   - GATE 2 random: 300 episodes x 10 steps x 37 ms belief update = ~2 min
#   - GATE 1: 600 resets at n_obs=20000 = ~5 min, plus the singleton estimate (32 chains
#     x 2000 draws) = ~3 min
# So ~55 min per task. 3 h leaves a wide margin for a slow node.
#
#   qsub submit_sa_gates_dp.sh

#$ -N sa_gates_dp
#$ -cwd
#$ -l h_rt=03:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-2
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/gates_dp
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast

# 8G is ample: the DP path holds a [d, 2^d] table -- 7 x 128 doubles at d=7 -- and never
# materialises a DAG list at all. This is the whole point of it.

D=$((SGE_TASK_ID + 5))          # task 1 -> d=6, task 2 -> d=7

echo "=== gates on the DP path: d=${D} ==="
echo "host $(hostname)  started $(date)"

python -u scripts/gates_dp.py \
  --d "${D}" --n_obs 20000 --n_int 100 --budget 10 \
  --prior_p 0.5 --threshold 0.7 \
  --episodes 300 --gate1_episodes 600 --oracle_draws 4000 \
  --seed 0 \
  --out "results/gates_dp/gates_d${D}.json"

echo "=== done $(date) ==="
