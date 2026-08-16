#!/bin/bash -l
# d=7: does the advantage over greedy return at a longer horizon?
#
# THE QUESTION. Three seeds at d=7, n_obs=20000 landed at gap_closed +0.994 / +1.001 /
# +1.017 -- parity with the myopic greedy oracle, completing a monotone decay from +1.31
# (d=4), +1.19 (d=5), +1.09 (d=6). All gates pass, no canaries fire, so the result stands.
#
# THE HYPOTHESIS. Greedy's ABSOLUTE cost fell as d grew: 2.05 interventions at d=5 but only
# 1.94 at d=7, because n_obs was held at 20000 while the graph grew, so observation alone
# pins down more. You cannot plan ahead over a two-step horizon. If that is the mechanism,
# lowering n_obs lengthens the horizon and the advantage should return.
#
# THE TENSION THIS TESTS. GATE 1 needs ENOUGH observational data to identify the graphs that
# are identifiable in principle -- at d=7, n_obs=2000 fails it (0.0167 against a target of
# 0.0779, under-powered). So there is a window: enough data to pass GATE 1, few enough to
# leave a horizon worth planning over. This sweep finds out whether that window is still
# open at d=7, or whether it has closed -- which would itself be a clean result about where
# the method stops paying.
#
# Either outcome is reportable. The prediction is registered here, before the numbers.
#
#   qsub -hold_jid <refs-job-id> submit_sa_d7_nobs.sh
# Each task computes its own references, since they depend on n_obs.

#$ -N sa_d7_nobs
#$ -cwd
#$ -l h_rt=08:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -t 1-9
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/d7nobs results/d7nobs
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa_fast
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ucabbse/marl_sa_fast
export WANDB_MODE=offline WANDB_SILENT=true

# 9 tasks = 3 n_obs settings x 3 seeds. n_obs varies fastest so a partial grid still gives
# complete seed sets at the smaller settings, which are the ones the hypothesis is about.
IDX=$((SGE_TASK_ID - 1))
NOBS_LIST=(5000 10000 20000)
NOBS=${NOBS_LIST[$((IDX % 3))]}
SEED=$((IDX / 3))

echo "=== d=7 n_obs=${NOBS} seed=${SEED} ==="
echo "host $(hostname)  started $(date)"

# n_obs=20000 seeds duplicate the completed runs deliberately: same config, fresh
# references, as an internal replication. If they do not reproduce +0.99 to +1.02 then the
# comparison across n_obs is not trustworthy either.
python -u -m scripts.run_experiment \
  --d 7 --observation edge_marginals --arch pernode --include_counts \
  --n_obs "${NOBS}" --train_episodes 6000 --eval_episodes 300 --budget 20 \
  --lr 1e-3 --hidden 256 --episodes_per_update 16 \
  --oracle_draws 4000 \
  --seeds "${SEED}" --tag "d7_nobs${NOBS}_s${SEED}" \
  --ref_cache ~/sa_runs/d7nobs/refs_d7_n${NOBS}.pkl \
  --gate1_episodes 400 \
  --wandb_project sa-phase2 \
  --out "results/d7nobs/d7_n${NOBS}_s${SEED}.json"

echo "=== done $(date) ==="
