#!/bin/bash
# JOB 1 (agent A's queue): score the channels-OFF fleet for the ON/OFF ablation figure.
#
# Agent A's command, verbatim except for the arms choice explained below. The protocol MUST
# match the ON fleet's or the two cannot be drawn on one axis: 100 paired episodes, `best`
# checkpoint, sampled evidence, and `--override_budget 50` so the myopic and random arms do not
# move with the rate. Without that flag the OFF fleet's budgets run 53..100 and every rate would
# be scored against a different baseline.
#
# ARMS: `--arms all` at every cell, deliberately. The OFF fleet has NO rho=1.00 cell, so agent
# A's suggested `--baseline_from` chain has no root, and borrowing the ON fleet's baselines is
# ruled out because the configs differ. With `--override_budget 50` the baselines SHOULD be
# identical across rates and the 3x saving would be sound -- but "should be" is the reasoning
# that produced the mispairing this project already has a guard for, and at 100 episodes the
# full computation is cheap. Computing them is the answer to agent A's "say which you did".
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/rho12b/xfer logs/rho12b_xfer
WORKERS=${WORKERS:-6}

jobs=$(mktemp)
for R in 0.50 0.70 0.80 0.85 0.90 0.95; do
  for S in 0 1 2; do
    src="results/rho12b_myriad/rho${R}_s${S}.json"
    out="results/rho12b/xfer/rho${R}_s${S}.json"
    [ -f "$out" ] && continue
    [ -f "$src" ] || { echo "MISSING $src" >&2; continue; }
    echo "$R $S $src $out" >> "$jobs"
  done
done
echo "$(date +%H:%M:%S)  job1 OFF-fleet scoring: $(wc -l < "$jobs") evaluations, $WORKERS workers"

run_one() {
  read -r R S src out <<< "$1"
  .venv/bin/python scripts/global_shd_paired.py "$src" \
    --episodes 100 --checkpoint best --sample --override_evidence sampled \
    --override_budget 50 --out "$out" > "logs/rho12b_xfer/rho${R}_s${S}.log" 2>&1
  echo "$(date +%H:%M:%S)  scored rho=$R s=$S"
}
export -f run_one
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  JOB1 COMPLETE"
