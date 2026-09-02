#!/usr/bin/env bash
# GENERATOR CONTROL, re-run under the current engine.
# sec:meth_ladder promises an Erdos-Renyi control against the scale-free generator that
# every reported result uses. The only existing data (results/vs_generator/gen_er_*) has
# belief_backend=version_space -- a superseded representation -- so it cannot sit beside
# a current number. Found by agent C, 2 Sep 23:0x.
# Only the ER arm is trained: the scale-free arm at this cell and budget already exists
# as results/sweep12k/k12s50n04b150_s{0,1,2}.json, and every flag below is copied from
# that job with --graph_model changed. Nothing else differs.
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /Users/brianezinwoke/Workspace/federated-active-causal-discovery-with-marl
[ -f "results/generator12k/er_s0.json" ] || .venv/bin/python -u scripts/ma_train.py --arm er12k --seed 0 --graph_model er --n_agents 4 --private_size 6 --n_shared 6 --budget 50 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 12000 --eval_episodes 200 --no_wandb --force --turn_aware_credit --local_epochs 4 --out results/generator12k/er_s0.json
[ -f "results/generator12k/er_s1.json" ] || .venv/bin/python -u scripts/ma_train.py --arm er12k --seed 1 --graph_model er --n_agents 4 --private_size 6 --n_shared 6 --budget 50 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 12000 --eval_episodes 200 --no_wandb --force --turn_aware_credit --local_epochs 4 --out results/generator12k/er_s1.json
[ -f "results/generator12k/er_s2.json" ] || .venv/bin/python -u scripts/ma_train.py --arm er12k --seed 2 --graph_model er --n_agents 4 --private_size 6 --n_shared 6 --budget 50 --n_obs 60 --n_int 20 --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only --claim_bar 1.0 --reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 12000 --eval_episodes 200 --no_wandb --force --turn_aware_credit --local_epochs 4 --out results/generator12k/er_s2.json
echo GENERATOR12K_DONE
