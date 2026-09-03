#!/bin/bash
# The answer-rate grid scored by ARGMAX instead of sampling, all 21 cells, deterministic path.
#
# WHY, AND WHY ALL 21 RATHER THAN THE 15 THAT ARE MISSING. Action selection at evaluation is
# one of the two conventions every number in this thesis depends on, and the existing control
# in `results/power/rho/argmax/` covers two rates -- so the curve's SHAPE under argmax is
# unmeasured, and monotonicity and saturation are currently sampled-evaluation claims only.
# Those two existing cells were also scored before `global_shd_paired.play` seeded the torch
# RNG on 2 Sep 21:15, so they do not reproduce. Re-running all seven rates puts the whole
# argmax curve on one deterministic footing instead of splicing two conventions together.
#
# NO BASELINE PHASE. Argmax changes how the LEARNED arm picks actions and nothing else. The
# myopic and random arms carry their own seeded generators and are byte-identical to the ones
# already computed in `results/power/rho/deterministic/`, which is why `--baseline_from` can
# pair against them directly. That check is not taken on trust: the sampled rebuild confirmed
# the myopic vectors reproduce exactly at all 21 cells.
#
# Cost at six workers, measured on the sampled rebuild: ~42 min per learned-only cell, four
# waves, so roughly 2.5 hours.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=results/power/rho/argmax_det
DET=results/power/rho/deterministic
mkdir -p "$OUT" logs/power/rho

EPISODES=${EPISODES:-200}
WORKERS=${WORKERS:-6}
RATES=${RATES:-"1.00 0.95 0.90 0.85 0.80 0.70 0.50"}
SEEDS=${SEEDS:-"0 1 2"}

for s in $SEEDS; do
  [ -f "$DET/xfer_rho1.00_s${s}.json" ] || {
    echo "ABORT: $DET/xfer_rho1.00_s${s}.json missing; run rebuild_grid_deterministic.sh first"
    exit 1; }
done

jobs=$(mktemp)
for rho in $RATES; do
  for s in $SEEDS; do
    src="results/power/rho/rho${rho}_s${s}.json"
    out="$OUT/argmax_rho${rho}_s${s}.json"
    base="$DET/xfer_rho1.00_s${s}.json"
    [ -f "$out" ] && continue
    [ -f "$src" ] || continue
    echo "$rho $s $src $out $base" >> "$jobs"
  done
done
echo "$(date +%H:%M:%S)  argmax grid: $(wc -l < "$jobs") cells, $WORKERS workers"

run_one() {
  read -r rho s src out base <<< "$1"
  # No --sample: this is the argmax arm. Everything else matches the sampled grid exactly.
  .venv/bin/python scripts/global_shd_paired.py "$src" \
    --episodes "${EPISODES}" --override_evidence sampled \
    --arms learned --baseline_from "$base" --out "$out" \
    > "logs/power/rho/argmaxdet_rho${rho}_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  done rho=$rho s=$s"
}
export -f run_one
export EPISODES
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  ARGMAX GRID COMPLETE"
