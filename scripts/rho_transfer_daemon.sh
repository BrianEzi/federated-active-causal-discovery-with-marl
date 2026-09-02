#!/bin/bash
# Keep the transfer sweep saturated without supervision: every few minutes, evaluate any cell
# that has finished training and whose seed already has a baseline, then stop when all 21 are
# done or nothing is left to do.
#
# WHY A DAEMON RATHER THAN ONE PASS. Training and transfer evaluation proceed at different
# rates, so a single pass of run_rho_transfer.sh only picks up whatever happened to be trained
# at the moment it ran, and anything finishing a minute later waits for a human. This polls,
# which means the expensive half overlaps the cheap half automatically and the machine is
# never idle while there is evaluable work.
#
# ORDER MATTERS AND IS DELIBERATE. Rates are evaluated 0.50, 0.85, 0.80, 0.90, 0.70, 0.95 --
# endpoints and the calibrated optimum FIRST, resolution last. The dose-response shape needs
# rho=0.50 (far end) and rho=0.85 (optimum) far more than it needs 0.95 and 0.70, so if this is
# interrupted again the partial curve still has its structure rather than a cluster of points
# near 1.0. The training fleet runs the opposite order, which is exactly the risk this hedges.
#
# Concurrency is deliberately low (2). Training is still running and the two must share; the
# machine profile puts useful parallelism at ~3x, and starving the fleet to speed up transfer
# just moves the bottleneck.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

EPISODES=${EPISODES:-200}
WORKERS=${WORKERS:-2}
POLL=${POLL:-180}
RATES=${RATES:-"0.50 0.85 0.80 0.90 0.70 0.95"}
SEEDS=${SEEDS:-"0 1 2"}

run_xfer() {
  read -r rho s src out base <<< "$1"
  .venv/bin/python scripts/global_shd_paired.py "$src" \
    --episodes "${EPISODES}" --sample --override_evidence sampled \
    --arms learned --baseline_from "$base" \
    --out "$out" > "logs/power/rho/xfer_rho${rho}_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  xfer done rho=$rho s=$s"
}
export -f run_xfer
export EPISODES

while true; do
  jobs_file=$(mktemp)
  for rho in $RATES; do
    for s in $SEEDS; do
      src="results/power/rho/rho${rho}_s${s}.json"
      out="results/power/rho/xfer_rho${rho}_s${s}.json"
      base="results/power/rho/xfer_rho1.00_s${s}.json"
      [ -f "$out" ] && continue        # already evaluated
      [ -f "$src" ] || continue        # not trained yet
      [ -f "$base" ] || continue       # seed has no baseline yet
      echo "$rho $s $src $out $base" >> "$jobs_file"
    done
  done

  n=$(wc -l < "$jobs_file")
  if [ "$n" -gt 0 ]; then
    echo "$(date +%H:%M:%S)  evaluating $n cell(s)"
    cat "$jobs_file" | xargs -P "$WORKERS" -I{} bash -c 'run_xfer "$@"' _ {}
    # Refresh the curve after every batch so a partial result is always readable on disk
    # rather than only existing once everything finishes.
    .venv/bin/python scripts/rho_curve_report.py --dir results/power/rho \
      > logs/power/rho/CURVE_latest.txt 2>&1
    echo "$(date +%H:%M:%S)  curve refreshed"
  fi
  rm -f "$jobs_file"

  done_n=$(ls results/power/rho/xfer_*.json 2>/dev/null | wc -l)
  [ "$done_n" -ge 21 ] && { echo "$(date +%H:%M:%S)  ALL 21 TRANSFER CELLS DONE"; break; }
  sleep "$POLL"
done
