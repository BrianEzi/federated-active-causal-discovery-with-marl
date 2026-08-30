#!/bin/bash -l
# THE HEAVY CELL of the oracle sweep, k30s50n04b150 -- carries the SCALING headline.
# docs/HANDOVER_LAPTOP2_2026_08_30.md assigned this to the cluster; nothing had actually
# been submitted. Local laptops cannot run it at all: the GNN's neighbour-gather step at
# k=30 with this sweep's budget (98, far above anything tested before this sweep) tries to
# allocate a single ~2.8GB tensor, confirmed by two reproducible RuntimeErrors on a
# 13.85GB-RAM machine. Requesting generous per-slot memory here so the cluster node does
# not hit the same wall.
#
# Commands taken VERBATIM from `scripts/sweep.py --emit jobs --tier heavy` on
# explore/constraint-based -- never hand-retype a sweep command, see
# docs/HANDOVER_LAPTOP2_2026_08_30.md section 3.
#
#$ -N oracle_heavy
#$ -cwd
#$ -t 1-3
#$ -l h_rt=16:00:00
#$ -l mem=16G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -e
mkdir -p logs results/sweep/oracle
source ~/envs/sa_env/bin/activate
cd ~/ma_tb
export PYTHONPATH=.
export TMPDIR=~/.tmp
mkdir -p ~/.tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEED=$((SGE_TASK_ID - 1))
OUT="results/sweep/oracle/k30s50n04b150_s${SEED}.json"
if [ -f "$OUT" ]; then
  echo "$OUT already exists -- skipping"
  exit 0
fi

echo "=== k30s50n04b150 seed $SEED : $(date) ==="
python scripts/ma_train.py --arm k30s50n04b150 --seed "$SEED" \
  --n_agents 4 --private_size 15 --n_shared 15 --budget 98 --n_obs 60 --n_int 20 \
  --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only \
  --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
  --episode_mix confounded --normalise_returns --vs_evidence oracle \
  --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out "$OUT"
echo "=== done $(date) ==="
