#!/bin/bash -l
# STAGE 1 of 2 -- reference policies for the GNN budget sweep, one config per array task.
#
# WHY THIS IS A SEPARATE JOB. The greedy oracle costs ~8.5 s per episode at d=7 (measured
# locally, 2026-08-19: 200 episodes took ~28 minutes). References are identical across
# seeds, so computing them inside every seed task would repeat the single most expensive
# part of the run three times over. `--ref_cache` writes them once; stage 2 reads them.
# Running the seeds without this hold would also RACE -- three tasks writing one cache file.
#
# WHAT THE SWEEP IS FOR. Two questions the baseline budget sweep raised and could not
# answer, because neither baseline can adapt:
#   Q1  Budget 2-3 is peak greedy-vs-random discrimination. Does the learned agent hold its
#       advantage there, or is its gain an artefact of slack budgets where all arms converge?
#   Q2  At d=7 the curves CROSS: greedy plateaus at 0.905 while random climbs to 0.960, so
#       ~9% of episodes the myopic oracle never solves at any budget. Does the learned agent
#       capture that headroom, or inherit greedy's blind spot?
#
# BOTH n_obs SETTINGS ARE RUN, and this is deliberate rather than thorough-for-its-own-sake.
# Measured 2026-08-19 at d=5: observational-only identification is 0.000 at n_obs=100 (best
# episode of 150 reached 0.579 mass, against a 0.7 threshold) rising to 0.060 at n_obs=1000,
# against an asymptotic singleton-MEC target of 0.0892. So n_obs=100 FAILS GATE 1 on the low
# side -- the environment is harder than its own success criterion allows, and any number
# from it describes a different task. n_obs=1000 is the honest comparison; n_obs=100 is kept
# only so the gate failure is measured rather than assumed.
#
#   qsub submit_sa_gnn_budget_refs.sh

#$ -N gnn_bud_refs
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-14
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/gnn_budget results/gnn_budget
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast
export WANDB_MODE=offline WANDB_SILENT=true

# d n_obs budget. Budgets chosen from results/budget/budget_sweep.json:
#   d=5: 2 and 3 straddle peak discrimination, 5 is the new default, 8 is slack.
#   d=7: 3 is peak discrimination, 5 the default, 16 is where greedy's plateau is visible
#        and random has already overtaken it.
CONFIGS=(
  "5 100 2"   "5 100 3"   "5 100 5"   "5 100 8"
  "7 100 3"   "7 100 5"   "7 100 16"
  "5 1000 2"  "5 1000 3"  "5 1000 5"  "5 1000 8"
  "7 1000 3"  "7 1000 5"  "7 1000 16"
)

read -r D NOBS BUDGET <<< "${CONFIGS[$((SGE_TASK_ID - 1))]}"
TAG="d${D}_nobs${NOBS}_b${BUDGET}"
echo "=== refs ${TAG} ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment \
  --d "${D}" --observation edge_marginals --arch pernode --include_counts \
  --n_obs "${NOBS}" --budget "${BUDGET}" \
  --eval_episodes 150 --gate1_episodes 200 \
  --oracle_draws 4000 \
  --refs_only \
  --ref_cache ~/sa_runs/gnn_budget/refs_"${TAG}".pkl \
  --tag "refs_${TAG}"

echo "=== done $(date) ==="
