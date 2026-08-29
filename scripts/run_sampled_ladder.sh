#!/usr/bin/env bash
# Sampled-evidence retrain of the window ladder.
#
# WHY. Every ladder result is vs_evidence=oracle (96 of 104 runs). Evaluating an
# oracle-trained policy under sampled evidence is a TRANSFER test and it fails: greedy beats
# the learner on soft SHD (+0.035 w08, +0.029 w12, both SIG) and on the error component
# itself. The mechanism is visible in the repeat rate -- greedy 0.247/0.331 against the
# learner's 0.110/0.138. Under ORACLE evidence a repeat is strictly wasted, so the learner
# correctly learned not to repeat; under SAMPLED evidence a repeat is how you buy power.
# Its trained rule is actively wrong for the regime. Only a retrain answers result 2.
#
# Configs are copied from each rung's own result file, verified 29 Aug. Do not retype them.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # 9 jobs on 10 cores; no thrash

run () {  # name agents private shared budget seed
  local name=$1 agents=$2 priv=$3 shared=$4 budget=$5 seed=$6
  local out="results/sampled/${name}_s${seed}.json"
  [ -f "$out" ] && { echo "skip $out"; return; }
  .venv/bin/python scripts/ma_train.py \
    --arm "${name}samp" --seed "$seed" \
    --n_agents "$agents" --private_size "$priv" --n_shared "$shared" --budget "$budget" \
    --n_obs 60 --n_int 20 --turn_order round_robin --backend factored \
    --policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 \
    --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
    --episode_mix confounded --cb_n_boot 12 \
    --vs_evidence sampled --vs_evidence_alpha 0.001 \
    --train_episodes 4000 --eval_episodes 200 --no_wandb --force \
    --out "$out" > "results/sampled/${name}_s${seed}.log" 2>&1
  echo "done ${name}_s${seed}"
}

for seed in 0 1 2; do
  run w04 4 1  3  8  "$seed" &
  run w08 4 4  4  12 "$seed" &
  run w12 4 6  6  16 "$seed" &
done
wait
echo "ALL LOCAL SAMPLED RUNS COMPLETE"
