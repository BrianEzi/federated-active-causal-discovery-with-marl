#!/bin/bash
# Local insurance run of the low rungs while Myriad works. Same code path AND the same rung
# table as the cluster (cluster/ladder_rungs.sh), so a local number and a cluster number at
# the same rung are comparable rather than two pipelines that happen to agree.
set -e
export PYTHONPATH=.
source cluster/ladder_rungs.sh
mkdir -p results/ladder_local results/ladder_local_eval
run () {
  RUNG=$1; SEED=$2; TRAIN_OVERRIDE=$3; EVAL_OVERRIDE=$4
  rung_config "$RUNG"
  [ -n "$TRAIN_OVERRIDE" ] && TRAIN=$TRAIN_OVERRIDE
  [ -n "$EVAL_OVERRIDE" ] && EVAL=$EVAL_OVERRIDE
  OUT="results/ladder_local/${ARM}_s${SEED}.json"
  if [ ! -f "$OUT" ]; then
    echo "=== TRAIN $ARM seed $SEED budget=$BUDGET train=$TRAIN ($(date +%H:%M)) ==="
    python -m scripts.ma_train --seed "$SEED" --arm "$ARM" --skip_eval \
      --n_agents "$AGENTS" --n_private "$PRIV" --n_shared "$SHARED" \
      --n_obs 1000 --n_int 100 --budget "$BUDGET" --train_episodes "$TRAIN" \
      --turn_order round_robin --clamp_only --out "$OUT" 2>&1 | tail -1
  fi
  for ARMNAME in learned greedy random_clamp pass; do
    EOUT="results/ladder_local_eval/${ARM}_s${SEED}_${ARMNAME}.json"
    [ -f "$EOUT" ] && continue
    python -m scripts.ma_eval_arm --run "$OUT" --arm "$ARMNAME" \
      --episodes "$EVAL" --out "$EOUT" 2>&1 | tail -1
  done
}
# Reduced episode counts: this is insurance against a queued cluster, not the real grid.
run 0 0 1500 150
run 1 0 1500 150
run 0 1 1500 150
run 1 1 1500 150
run 2 0 1200 100
echo "LOCAL LADDER DONE $(date +%H:%M)"
