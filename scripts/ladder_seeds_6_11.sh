#!/bin/bash
# Federation ladder at k12, arms A (federated) and E (pooled), seeds 6..11.
#
# WHY. 4.3.1 reports "federating costs nothing measurable" from three seeds at k_v=20 on a
# metric that has saturated, which is an absence of evidence rather than a bound. Doubling the
# seed count at k12 is what lets agent A replace it with a TOST equivalence bound against a
# stated margin plus a power statement.
#
# FLAGS ARE NOT RECONSTRUCTED. Agent A rebuilt them from the stored configs and asked me to
# sanity-check, noting "the original commands are not in any log". They are:
# `results/central12k/run_ladder12k.sh` carries the literal original invocations for seeds 0..5.
# Every flag below is copied from that file, and agent A's reconstruction matches it exactly --
# checked field by field, including the two that distinguish the arms.
#
#   arm A (federated):  --local_epochs 4
#   arm E (pooled):     --local_epochs 0 --observe_belief_channels --observe_partner_counts
#
# `--local_epochs 0` is the pooled path and E=1 is NOT pooled (ma/policy.py:856), which is the
# one flag in this script that would silently measure the wrong arm if it were wrong.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=results/central12k
mkdir -p "$OUT" logs/ladder

WORKERS=${WORKERS:-6}
SEEDS=${SEEDS:-"6 7 8 9 10 11"}

COMMON="--n_agents 4 --private_size 6 --n_shared 6 --budget 50 --n_obs 60 --n_int 20 \
--turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only \
--graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
--episode_mix confounded --normalise_returns --vs_evidence oracle --eval_episodes 200 \
--no_wandb --force --turn_aware_credit --train_episodes 12000"

jobs=$(mktemp)
for s in $SEEDS; do
  for arm in A E; do
    out="$OUT/v2_k12_${arm}_s${s}.json"
    [ -f "$out" ] && continue
    echo "$arm $s $out" >> "$jobs"
  done
done
echo "$(date +%H:%M:%S)  ladder seeds: $(wc -l < "$jobs") runs, $WORKERS workers"

run_one() {
  read -r arm s out <<< "$1"
  if [ "$arm" = "A" ]; then extra="--local_epochs 4"
  else extra="--local_epochs 0 --observe_belief_channels --observe_partner_counts"; fi
  .venv/bin/python scripts/ma_train.py --arm "v2_k12_${arm}" --seed "$s" \
    $COMMON $extra --out "$out" > "logs/ladder/v2_k12_${arm}_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  done arm=$arm seed=$s"
}
export -f run_one
export COMMON
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  LADDER SEEDS COMPLETE"
