#!/bin/bash
# Re-emit the 18 learned-only transfer cells WITH their per-episode rows.
#
# WHY. `global_shd_paired.py` gated row storage on `--arms all`, so the three rho=1.00
# baselines carry 200 per-episode values per arm and the other eighteen carry none. The
# aggregate in CURVE.json is enough to redraw the curve and not enough to check it: a reader
# cannot recompute the paired SE, inspect the resolved fraction, or confirm that greedy really
# is the identical vector across rates. Fixed in the script; this re-emits the affected cells.
#
# The deltas MUST come back identical -- same checkpoint, same seed, same 200 episode seeds,
# same baseline. `verify_rerun.py` checks exactly that, so this doubles as a reproducibility
# test rather than only a data-completeness fix. A mismatch would mean something in the
# evaluation path is not deterministic, which would matter far more than the missing rows.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/power/rho/rerun logs/power/rho

WORKERS=${WORKERS:-4}
RATES=${RATES:-"0.95 0.90 0.85 0.80 0.70 0.50"}
SEEDS=${SEEDS:-"0 1 2"}

jobs=$(mktemp)
for rho in $RATES; do
  for s in $SEEDS; do
    src="results/power/rho/rho${rho}_s${s}.json"
    out="results/power/rho/rerun/xfer_rho${rho}_s${s}.json"
    base="results/power/rho/xfer_rho1.00_s${s}.json"
    [ -f "$out" ] && continue
    [ -f "$src" ] && [ -f "$base" ] || continue
    echo "$rho $s $src $out $base" >> "$jobs"
  done
done
echo "$(date +%H:%M:%S)  re-emitting $(wc -l < "$jobs") cells with rows, $WORKERS workers"

run_one() {
  read -r rho s src out base <<< "$1"
  .venv/bin/python scripts/global_shd_paired.py "$src" \
    --episodes 200 --sample --override_evidence sampled \
    --arms learned --baseline_from "$base" --out "$out" \
    > "logs/power/rho/rerun_rho${rho}_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  done rho=$rho s=$s"
}
export -f run_one
cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  RERUN COMPLETE"
