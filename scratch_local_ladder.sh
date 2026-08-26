#!/bin/bash
# Local insurance run of the low rungs while Myriad queues. Same code path as the cluster.
set -e
export PYTHONPATH=.
mkdir -p results/ladder_local results/ladder_local_eval
run () {
  A=$1; P=$2; X=$3; SEED=$4; TRAIN=$5; EVAL=$6
  B=$((3*A)); D=$((A*P+X)); ARM="rung_${A}a_${P}p_${X}x_d${D}"
  OUT="results/ladder_local/${ARM}_s${SEED}.json"
  if [ ! -f "$OUT" ]; then
    echo "=== TRAIN $ARM seed $SEED ($(date +%H:%M)) ==="
    python -m scripts.ma_train --seed "$SEED" --arm "$ARM" --skip_eval \
      --n_agents "$A" --n_private "$P" --n_shared "$X" \
      --n_obs 1000 --n_int 100 --budget "$B" --train_episodes "$TRAIN" \
      --turn_order round_robin --clamp_only --out "$OUT" 2>&1 | tail -2
  fi
  for ARMNAME in learned greedy random_clamp pass; do
    EOUT="results/ladder_local_eval/${ARM}_s${SEED}_${ARMNAME}.json"
    [ -f "$EOUT" ] && continue
    python -m scripts.ma_eval_arm --run "$OUT" --arm "$ARMNAME" \
      --episodes "$EVAL" --out "$EOUT" 2>&1 | tail -1
  done
}
run 2 1 3 0 800 100
run 2 1 3 1 800 100
run 3 1 3 0 800 100
run 3 1 3 1 800 100
run 5 1 3 0 800 100
echo "ALL LOCAL RUNGS DONE $(date +%H:%M)"
