#!/bin/bash
# The federation ladder at beta=0.7: a SECOND cell for fig:ladder and a meaningful equivalence
# bound for 4.3.1.
#
# WHY A SECOND CELL. At beta=1.5 both arms are pinned at the floor -- my six new seeds gave
# A-E of -0.000167 +/- 0.000281 (best) and +0.000164 +/- 0.000110 (final), with four of six
# cells at EXACTLY 0.000000 in each convention, and the two conventions disagreeing in sign.
# An equivalence bound there compares two arms with no room to be wrong. Agent A's three
# criteria for choosing 0.7 over its neighbours:
#
#   1. the margin IS the myopic arm's error: 0.00403 at beta=0.7 against 0.00077 at 1.5, 5.2x
#   2. both arms stay competent -- recovery 0.913, window rate 0.939, clear of the 0.70 gate,
#      where beta=0.6 and 0.5 fall to 0.760 and 0.602 and the learned arm is visibly struggling
#   3. it sits BELOW the coordination sign flip that C10 pins at beta=0.9, so it tests for a
#      federation cost where coordination is known to be load-bearing
#
# beta=0.8 also satisfies (2) and (3) but its margin is a quarter smaller and its learned SHD
# inverts against 0.7's in a way agent A has not explained.
#
# Budget 24 = ceil(0.7 * cover(12) * 12 * 4). Everything else is the ladder's own configuration,
# copied from `results/central12k/run_ladder12k.sh` -- the literal original invocations, not a
# reconstruction. Arm A is `--local_epochs 4`; arm E is the POOLED path, `--local_epochs 0`
# plus the two observation flags, and E=1 is NOT pooled (ma/policy.py:856).
#
# RUNS LOCALLY BY DESIGN. Agent A put this third behind rho12on and rho12b and gated it on the
# queue clearing. The MYRIAD queue has not cleared -- both arrays are live there -- but this
# machine is idle and does not compete with them.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=results/ladder_b070
mkdir -p "$OUT" logs/ladder_b070
WORKERS=${WORKERS:-6}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}

COMMON="--n_agents 4 --private_size 6 --n_shared 6 --budget 24 --n_obs 60 --n_int 20 \
--turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only \
--graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
--episode_mix confounded --normalise_returns --vs_evidence oracle --eval_episodes 200 \
--no_wandb --force --turn_aware_credit --train_episodes 12000"

jobs=$(mktemp)
for s in $SEEDS; do
  for arm in A E; do
    out="$OUT/b070_${arm}_s${s}.json"
    [ -f "$out" ] && continue
    echo "$arm $s $out" >> "$jobs"
  done
done
echo "$(date +%H:%M:%S)  beta=0.7 ladder: $(wc -l < "$jobs") runs, $WORKERS workers"

run_one() {
  read -r arm s out <<< "$1"
  if [ "$arm" = "A" ]; then extra="--local_epochs 4"
  else extra="--local_epochs 0 --observe_belief_channels --observe_partner_counts"; fi
  .venv/bin/python -u scripts/ma_train.py --arm "b070_${arm}" --seed "$s" \
    $COMMON $extra --out "$out" > "logs/ladder_b070/${arm}_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  done arm=$arm seed=$s"
}
export -f run_one
export COMMON
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  BETA07 LADDER COMPLETE"
