#!/bin/bash
# Rebuild the full 21-cell answer-rate transfer grid under the DETERMINISTIC evaluation path.
#
# WHY THE WHOLE GRID AND NOT JUST THE MISSING ROWS. Two defects are being fixed at once and the
# second forces the scope:
#
#   1. `global_shd_paired.py` stored no per-episode rows for `--arms learned`, so 18 of 21
#      cells shipped a mean and a paired SE with nothing underneath them. Mine.
#   2. `play()` did not seed the torch RNG until 2026-09-02 21:15 (agent A). A learned arm
#      evaluated with `--sample` drew its actions from the global generator, so a re-run of the
#      SAME checkpoint returned different numbers. Measured over 24 re-runs: greedy and random
#      reproduced exactly, the learned arm moved 0.10-2.22 paired SE, median ~0.4.
#
# Defect 2 means the three rho=1.00 baselines are affected too -- their LEARNED arm is redrawn
# even though the greedy/random vectors every other cell pairs against are not. Re-running only
# the 18 would leave the grid internally inconsistent: 18 deterministic cells paired against 3
# baselines whose learned arm came from the old path.
#
# The published numbers are NOT wrong. Agent A's measurement puts the variation inside the
# reported intervals. They are, however, not reproducible, which is a separate and unacceptable
# property for results shipping with their checkpoints. Expect the rebuilt deltas to differ
# from the published ones by roughly one standard error; that is the known cost, not a finding.
#
# Phase 1 is serial: every learned-only cell pairs against a baseline, so a corrupt or partial
# baseline would silently poison a whole seed column.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=results/power/rho/deterministic
mkdir -p "$OUT" logs/power/rho

EPISODES=${EPISODES:-200}
WORKERS=${WORKERS:-4}
RATES=${RATES:-"0.95 0.90 0.85 0.80 0.70 0.50"}
SEEDS=${SEEDS:-"0 1 2"}

echo "$(date +%H:%M:%S)  phase 1: three rho=1.00 baselines (3 arms each), serial"
for s in $SEEDS; do
  out="$OUT/xfer_rho1.00_s${s}.json"
  [ -f "$out" ] && { echo "  s$s exists, skip"; continue; }
  .venv/bin/python scripts/global_shd_paired.py "results/power/rho/rho1.00_s${s}.json" \
    --episodes "$EPISODES" --sample --override_evidence sampled --out "$out" \
    > "logs/power/rho/det_rho1.00_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  baseline s$s done"
done

jobs=$(mktemp)
for rho in $RATES; do
  for s in $SEEDS; do
    src="results/power/rho/rho${rho}_s${s}.json"
    out="$OUT/xfer_rho${rho}_s${s}.json"
    base="$OUT/xfer_rho1.00_s${s}.json"
    [ -f "$out" ] && continue
    [ -f "$src" ] && [ -f "$base" ] || continue
    echo "$rho $s $src $out $base" >> "$jobs"
  done
done
echo "$(date +%H:%M:%S)  phase 2: $(wc -l < "$jobs") learned-only cells, $WORKERS workers"

run_one() {
  read -r rho s src out base <<< "$1"
  .venv/bin/python scripts/global_shd_paired.py "$src" \
    --episodes "${EPISODES}" --sample --override_evidence sampled \
    --arms learned --baseline_from "$base" --out "$out" \
    > "logs/power/rho/det_rho${rho}_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  done rho=$rho s=$s"
}
export -f run_one
export EPISODES
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  GRID REBUILD COMPLETE"
