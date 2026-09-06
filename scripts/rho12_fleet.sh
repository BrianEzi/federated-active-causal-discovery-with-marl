#!/bin/bash
# RQ2 at the PRINCIPAL CELL: seven answer rates x three seeds, k_v=12.
#
# WHY. Every partial-oracle result in the thesis sits at k_v=8 while RQ1's headline sits at
# k_v=12, and "sampled evaluation is expensive" explains that but does not justify it. Agent A
# measured k=12 sampled evaluation at 8.5 s/episode against k=8's 5.6 -- 1.5x, not the 2-3x we
# assumed -- so the grid is affordable at the principal cell.
#
# SETTINGS ARE THE SWEEP CELL'S, DELIBERATELY, AND MUST NOT BE "FIXED" TO MATCH THE k=8 GRID.
# The k=8 rho fleet differs from `k12s50n04b150` in four fields: budget 70 vs 50, 8,000 vs
# 12,000 episodes, and observe_belief_channels / observe_reprobe_signal on vs off. The point of
# this grid is comparability with RQ1, so it takes RQ1's cell. Consequence, to be stated
# wherever it is reported: this is a REPLICATION of the finding at the principal cell, not a
# matched pair with the k=8 grid. Both stay in the thesis.
#
# THE FLAG IS `--evidence_power`. `vs_evidence_power` is the CONFIG FIELD, not the flag; passing
# the field name would be accepted-and-ignored by argparse prefix matching in the worst case and
# silently drop the dial. `verify_rho12_flag.py` re-reads every finished run and asserts the
# value landed, per the 5 Sep silent-drop lesson where `env_from_config` dropped
# `skeleton_source` and a whole evaluation measured the wrong thing while stamping the right
# label.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=results/rho12
mkdir -p "$OUT" logs/rho12

WORKERS=${WORKERS:-6}
RATES=${RATES:-"1.00 0.95 0.90 0.85 0.80 0.70 0.50"}
SEEDS=${SEEDS:-"0 1 2"}

COMMON="--n_agents 4 --private_size 6 --n_shared 6 --budget 50 --n_obs 60 --n_int 20 \
--turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only \
--graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
--episode_mix confounded --normalise_returns --vs_evidence oracle \
--train_episodes 12000 --eval_episodes 200 --no_wandb --force \
--turn_aware_credit --local_epochs 4"

jobs=$(mktemp)
# Lowest rates first. They are the ones the k=8 grid says carry the effect, so if the fleet is
# cut short for any reason the cells that decide the finding are the ones already on disk.
for rho in $RATES; do
  for s in $SEEDS; do
    out="$OUT/rho${rho}_s${s}.json"
    [ -f "$out" ] && continue
    echo "$rho $s $out" >> "$jobs"
  done
done
sort -k1,1n "$jobs" -o "$jobs"
echo "$(date +%H:%M:%S)  rho12 fleet: $(wc -l < "$jobs") runs, $WORKERS workers"

run_one() {
  read -r rho s out <<< "$1"
  .venv/bin/python -u scripts/ma_train.py --arm "k12rho${rho}" --seed "$s" \
    $COMMON --evidence_power "$rho" --out "$out" \
    > "logs/rho12/rho${rho}_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  done rho=$rho s=$s"
}
export -f run_one
export COMMON
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  RHO12 FLEET COMPLETE"
