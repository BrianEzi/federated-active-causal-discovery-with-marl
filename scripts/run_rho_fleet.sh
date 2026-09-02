#!/bin/bash
# The ANSWER-RATE (rho) transfer curve: 7 rates x 3 seeds at the winning k=8 configuration.
#
# Only `--evidence_power` varies; everything else is the configuration that produced the
# transfer result in docs/FINDINGS_TRANSFER_2026_09_02.md, held fixed. rho=1.00 is the plain
# oracle control that must LOSE at transfer if the dose-response claim is real.
#
# Named rho / "answer rate" per docs/AGENT_C_METHODOLOGY_BRIEF.md Phase 0b -- "power" collides
# with statistical power, which is the quantity the sampled regime is actually about. The CLI
# flag is still `--evidence_power`; only the reporting vocabulary changes.
#
# WORKERS=5 is the measured saturation point for this machine (4 P-cores + 6 E-cores); the
# profile in results/machines/laptop-b.json shows efficiency already down to 52% at 6.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/power/rho logs/power/rho

WORKERS=${WORKERS:-5}
RATES=${RATES:-"1.00 0.95 0.90 0.85 0.80 0.70 0.50"}
SEEDS=${SEEDS:-"0 1 2"}

jobs_file=$(mktemp)
for rho in $RATES; do
  for s in $SEEDS; do
    tag="rho${rho}_s${s}"
    out="results/power/rho/${tag}.json"
    [ -f "$out" ] && continue          # resumable: a finished cell is skipped
    echo "$rho $s $tag $out" >> "$jobs_file"
  done
done

total=$(wc -l < "$jobs_file")
echo "$(date +%H:%M:%S)  launching $total cells, $WORKERS workers"

run_one() {
  read -r rho s tag out <<< "$1"
  .venv/bin/python scripts/ma_train.py --arm "$tag" --seed "$s" --budget 70 \
    --evidence_power "$rho" --train_episodes 8000 \
    --n_agents 4 --private_size 4 --n_shared 4 --n_obs 60 --n_int 20 \
    --turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only \
    --graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
    --episode_mix confounded --normalise_returns --vs_evidence oracle \
    --observe_belief_channels --observe_reprobe_signal \
    --turn_aware_credit --local_epochs 4 --eval_episodes 100 --no_wandb --force \
    --out "$out" > "logs/power/rho/${tag}.log" 2>&1
  echo "$(date +%H:%M:%S)  done $tag"
}
export -f run_one

# xargs -P is the queue: it keeps exactly WORKERS jobs in flight and starts the next as one
# finishes, which is what a fixed-size worker pool needs. A plain `&` loop would launch all 21.
cat "$jobs_file" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs_file"
echo "$(date +%H:%M:%S)  FLEET COMPLETE"
