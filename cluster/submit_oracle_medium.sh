#!/bin/bash -l
# THE MEDIUM TIER of the oracle sweep, moved off a local laptop mid-run.
#
# WHY THIS EXISTS. docs/HANDOVER_LAPTOP2_2026_08_30.md assigned the medium tier to a local
# laptop with 6 parallel workers. That laptop has 13.85GB RAM; running 4-6 of these jobs
# concurrently was slowing the machine and risked the same out-of-memory failure confirmed
# on the heavy cell (cluster/submit_oracle_heavy.sh). Moving the tier here lets each task
# get its own node/slot instead of competing for one machine's RAM.
#
# The 18 commands below are pasted VERBATIM from
# `scripts/sweep.py --emit jobs --tier medium` on explore/constraint-based, one per array
# task -- never hand-retype a sweep command (docs/HANDOVER_LAPTOP2_2026_08_30.md section 3).
# `[ -f out ] ||` is dropped since SGE_TASK_ID already gives each task exactly one job; the
# guard is redundant here but the RESULT FILE remains the resume state -- a re-submitted
# task that already has output just overwrites it (--force), matching --tier medium's own
# job list, which contains no duplicate output paths.
#
#$ -N oracle_medium
#$ -cwd
#$ -t 1-18
#$ -l h_rt=06:00:00
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

CMD=""
case $SGE_TASK_ID in
1) OUT=results/sweep/oracle/k20s50n04b150_s0.json
   CMD="python scripts/ma_train.py --arm k20s50n04b150 --seed 0 --n_agents 4 --private_size 10 --n_shared 10 --budget 75 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
2) OUT=results/sweep/oracle/k20s50n04b150_s1.json
   CMD="python scripts/ma_train.py --arm k20s50n04b150 --seed 1 --n_agents 4 --private_size 10 --n_shared 10 --budget 75 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
3) OUT=results/sweep/oracle/k20s50n04b150_s2.json
   CMD="python scripts/ma_train.py --arm k20s50n04b150 --seed 2 --n_agents 4 --private_size 10 --n_shared 10 --budget 75 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
4) OUT=results/sweep/oracle/k12s75n08b150_s0.json
   CMD="python scripts/ma_train.py --arm k12s75n08b150 --seed 0 --n_agents 8 --private_size 3 --n_shared 9 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
5) OUT=results/sweep/oracle/k12s75n08b150_s1.json
   CMD="python scripts/ma_train.py --arm k12s75n08b150 --seed 1 --n_agents 8 --private_size 3 --n_shared 9 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
6) OUT=results/sweep/oracle/k12s75n08b150_s2.json
   CMD="python scripts/ma_train.py --arm k12s75n08b150 --seed 2 --n_agents 8 --private_size 3 --n_shared 9 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
7) OUT=results/sweep/oracle/k12s25n08b150_s0.json
   CMD="python scripts/ma_train.py --arm k12s25n08b150 --seed 0 --n_agents 8 --private_size 9 --n_shared 3 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
8) OUT=results/sweep/oracle/k12s25n08b150_s1.json
   CMD="python scripts/ma_train.py --arm k12s25n08b150 --seed 1 --n_agents 8 --private_size 9 --n_shared 3 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
9) OUT=results/sweep/oracle/k12s25n08b150_s2.json
   CMD="python scripts/ma_train.py --arm k12s25n08b150 --seed 2 --n_agents 8 --private_size 9 --n_shared 3 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
10) OUT=results/sweep/oracle/k12s50n10b150_s0.json
   CMD="python scripts/ma_train.py --arm k12s50n10b150 --seed 0 --n_agents 10 --private_size 6 --n_shared 6 --budget 125 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
11) OUT=results/sweep/oracle/k12s50n10b150_s1.json
   CMD="python scripts/ma_train.py --arm k12s50n10b150 --seed 1 --n_agents 10 --private_size 6 --n_shared 6 --budget 125 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
12) OUT=results/sweep/oracle/k12s50n10b150_s2.json
   CMD="python scripts/ma_train.py --arm k12s50n10b150 --seed 2 --n_agents 10 --private_size 6 --n_shared 6 --budget 125 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
13) OUT=results/sweep/oracle/k12s50n08b150_s0.json
   CMD="python scripts/ma_train.py --arm k12s50n08b150 --seed 0 --n_agents 8 --private_size 6 --n_shared 6 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
14) OUT=results/sweep/oracle/k12s50n08b150_s1.json
   CMD="python scripts/ma_train.py --arm k12s50n08b150 --seed 1 --n_agents 8 --private_size 6 --n_shared 6 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
15) OUT=results/sweep/oracle/k12s50n08b150_s2.json
   CMD="python scripts/ma_train.py --arm k12s50n08b150 --seed 2 --n_agents 8 --private_size 6 --n_shared 6 --budget 100 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
16) OUT=results/sweep/oracle/k12s50n04b500_s0.json
   CMD="python scripts/ma_train.py --arm k12s50n04b500 --seed 0 --n_agents 4 --private_size 6 --n_shared 6 --budget 166 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
17) OUT=results/sweep/oracle/k12s50n04b500_s1.json
   CMD="python scripts/ma_train.py --arm k12s50n04b500 --seed 1 --n_agents 4 --private_size 6 --n_shared 6 --budget 166 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
18) OUT=results/sweep/oracle/k12s50n04b500_s2.json
   CMD="python scripts/ma_train.py --arm k12s50n04b500 --seed 2 --n_agents 4 --private_size 6 --n_shared 6 --budget 166 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out $OUT" ;;
esac

if [ -f "$OUT" ]; then
  echo "$OUT already exists -- skipping"
  exit 0
fi

echo "=== task $SGE_TASK_ID : $(date) ==="
echo "cmd: $CMD"
$CMD
echo "=== done $(date) ==="
