#!/bin/bash
# Transfer evaluation for the answer-rate fleet, exploiting the 3x baseline saving.
#
# THE SAVING. greedy and random_vary do not read the trained policy, so for a fixed
# (cell, seed, episodes, evidence) they replay identical episodes whatever sits in the learned
# arm -- measured, greedy scored 0.06649 hard SHD in both the p10 and p07 transfer tests,
# which differ only in the training answer rate. So per seed the baselines are computed ONCE
# (rho=1.00, three arms) and every other rate is evaluated learned-only against them.
#
# Under sampled evidence one arm is 6-9 s/episode, so this turns 21 three-arm evaluations into
# 3 three-arm plus 18 one-arm: roughly 24 core-hours down to 9.
#
# Resumable: any cell whose xfer_ output already exists is skipped. Safe to re-run.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/power/rho logs/power/rho

EPISODES=${EPISODES:-200}
WORKERS=${WORKERS:-3}
RATES=${RATES:-"0.95 0.90 0.85 0.80 0.70 0.50"}    # 1.00 is the baseline rate, done first
SEEDS=${SEEDS:-"0 1 2"}

# -- phase 1: one three-arm baseline per seed, at rho=1.00 -----------------------------------
# Serial across seeds is deliberate. These three runs produce the per-episode greedy/random
# vectors every later run pairs against; a failure here invalidates that seed's whole column,
# so it is worth not having them contend with each other.
for s in $SEEDS; do
  base="results/power/rho/xfer_rho1.00_s${s}.json"
  [ -f "$base" ] && { echo "$(date +%H:%M:%S)  baseline s$s exists, skipping"; continue; }
  src="results/power/rho/rho1.00_s${s}.json"
  [ -f "$src" ] || { echo "$(date +%H:%M:%S)  !! $src not trained yet, cannot build baseline"; continue; }
  echo "$(date +%H:%M:%S)  baseline s$s (3 arms, $EPISODES episodes)"
  .venv/bin/python scripts/global_shd_paired.py "$src" \
    --episodes "$EPISODES" --sample --override_evidence sampled \
    --out "$base" > "logs/power/rho/xfer_rho1.00_s${s}.log" 2>&1
done

# -- phase 2: learned-only for every other rate, paired against its seed's baseline ----------
jobs_file=$(mktemp)
for rho in $RATES; do
  for s in $SEEDS; do
    src="results/power/rho/rho${rho}_s${s}.json"
    out="results/power/rho/xfer_rho${rho}_s${s}.json"
    base="results/power/rho/xfer_rho1.00_s${s}.json"
    [ -f "$out" ] && continue
    [ -f "$src" ] || continue          # not trained yet; a later re-run picks it up
    [ -f "$base" ] || continue         # no baseline for this seed yet
    echo "$rho $s $src $out $base" >> "$jobs_file"
  done
done

n=$(wc -l < "$jobs_file")
echo "$(date +%H:%M:%S)  phase 2: $n learned-only cells, $WORKERS workers"

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

[ "$n" -gt 0 ] && cat "$jobs_file" | xargs -P "$WORKERS" -I{} bash -c 'run_xfer "$@"' _ {}
rm -f "$jobs_file"
echo "$(date +%H:%M:%S)  TRANSFER SWEEP PASS COMPLETE"
