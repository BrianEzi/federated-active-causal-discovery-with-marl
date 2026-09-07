#!/bin/bash
# JOB 2 (agent A's queue): does the skeleton-generosity gain survive a FINITE BUDGET?
#
# A missed adjacency is fatal to this belief -- `cb/factored.py` skips a pair closed to NONE in
# every later update, so interventional proof of a link cannot reopen it -- while a spurious one
# is merely expensive. The measured CEILING at the principal cell on 60 rows runs 61.0% at
# alpha=0.01 to 71.3% at alpha=0.7, so generosity is worth ten points WITH EVERY NODE
# INTERVENED ON. Generosity also creates more ambiguous pairs, and resolving those costs
# interventions. This asks whether the gain is still there when the budget is the real one.
#
# budget 50 is achieved, budget 400 is the ceiling in the same units. Commands are agent A's
# verbatim -- the split says I train and they evaluate EXCEPT where they hand me an exact
# command, because a copied command cannot drift from their protocol and an improvised one can.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/skel_alpha logs/skel_alpha
WORKERS=${WORKERS:-5}

jobs=$(mktemp)
for A in 0.01 0.1 0.3 0.5 0.7; do
  for B in 50 400; do
    out="results/skel_alpha/a${A}_b${B}.json"
    [ -f "$out" ] && continue
    echo "$A $B $out" >> "$jobs"
  done
done
echo "$(date +%H:%M:%S)  job2 skeleton-alpha: $(wc -l < "$jobs") evaluations, $WORKERS workers"

run_one() {
  read -r A B out <<< "$1"
  .venv/bin/python scripts/global_shd_paired.py \
    results/sweep12k/k12s50n04b150_s0.json results/sweep12k/k12s50n04b150_s1.json \
    results/sweep12k/k12s50n04b150_s2.json \
    --episodes 100 --checkpoint best --sample \
    --override_skeleton estimated --override_n_obs 60 --override_skeleton_alpha "$A" \
    --override_budget "$B" --out "$out" > "logs/skel_alpha/a${A}_b${B}.log" 2>&1
  echo "$(date +%H:%M:%S)  done alpha=$A budget=$B"
}
export -f run_one
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  JOB2 COMPLETE"
