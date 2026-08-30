#!/usr/bin/env bash
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /Users/brianezinwoke/Workspace/federated-active-causal-discovery-with-marl
[ -f "results/sweep/oracle/k12s50n03b150_s1.json" ] || .venv/bin/python scripts/ma_train.py --arm k12s50n03b150 --seed 1 --n_agents 3 --private_size 6 --n_shared 6 --budget 38 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000 --eval_episodes 200 --no_wandb --force --out results/sweep/oracle/k12s50n03b150_s1.json
