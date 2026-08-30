#!/usr/bin/env bash
# THE LAUNCHER. Restart-safe, sleep-proof, gated.
#
#   scripts/launch.sh oracle  8
#   scripts/launch.sh sampled 8
#
# Restart-safe: every job line begins `[ -f "$out" ] ||`, so re-running this after a crash,
# a reboot or a Ctrl-C picks up exactly the runs that have no result file yet. There is no
# separate resume script and no state to keep in sync -- the result files ARE the state.
#
# Sleep-proof: `caffeinate -i` holds off idle sleep for as long as the sweep runs. A laptop
# that suspends mid-sweep does not corrupt anything (see restart-safe), but it does quietly
# turn a 20-hour plan into a 40-hour one, and the first anyone notices is the morning.
#
# Gated: the two preflights run first and a failure stops the launch. Both are fast, and
# both have caught defects that would have invalidated every run -- most recently a
# baseline registry that would have evaluated the whole sweep with no ceiling arm.
set -uo pipefail
cd "$(dirname "$0")/.."

EVIDENCE=${1:-oracle}
WORKERS=${2:-8}
TIER=${3:-}                       # cheap | medium | heavy | empty for the whole sweep
SEEDS=${SEEDS:-3}
SEED_LIST=${SEED_LIST:-}          # e.g. "0" to take only seed 0 of a tier
EPISODES=${EPISODES:-4000}
OUT_DIR=${OUT_DIR:-results/sweep/$EVIDENCE}
CALIBRATION=${CALIBRATION:-results/sweep/calibration_$EVIDENCE.json}

TIER_ARGS=""
[ -n "$TIER" ] && TIER_ARGS="--tier $TIER"
SEED_ARGS="--seeds $SEEDS"
[ -n "$SEED_LIST" ] && SEED_ARGS="--seed_list $SEED_LIST"

mkdir -p "$OUT_DIR/logs"

echo "=== gates ==="
.venv/bin/python scripts/preflight_metrics.py || { echo "METRIC PREFLIGHT FAILED"; exit 1; }
# Gate only the cells THIS launch will run. Checking all twenty costs ten minutes, most
# of it on cells another machine is running, and a gate nobody waits for is a gate nobody
# runs. `--only` keeps it under a minute for the cheap tier.
GATE_CELLS=$(.venv/bin/python scripts/sweep.py --emit json $TIER_ARGS \
  --calibration "$CALIBRATION" 2>/dev/null \
  | .venv/bin/python -c 'import json,sys; print(",".join(c["name"] for c in json.load(sys.stdin)))')
.venv/bin/python scripts/preflight_runs.py feasibility --episodes 20 --only "$GATE_CELLS" \
  || { echo "FEASIBILITY GATE FAILED -- beta is mis-normalised somewhere"; exit 1; }

echo
echo "=== launching: $EVIDENCE ${TIER:-all-tiers}, $EPISODES episodes, $WORKERS workers ==="
JOBS=$(mktemp)
.venv/bin/python scripts/sweep.py --emit jobs $SEED_ARGS --episodes "$EPISODES" \
  --evidence "$EVIDENCE" --out_dir "$OUT_DIR" --calibration "$CALIBRATION" $TIER_ARGS \
  | grep -v '^#' > "$JOBS"
echo "$(wc -l < "$JOBS") jobs, longest first"

export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# `caffeinate -i` only holds off IDLE sleep; a closed lid still suspends. -s would also
# hold off system sleep on AC power, which is what an overnight run wants.
caffeinate -i -s xargs -P "$WORKERS" -I CMD sh -c 'CMD' < "$JOBS"
status=$?
rm -f "$JOBS"

echo
echo "=== done (xargs exit $status) ==="
DONE=$(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l)
echo "$DONE result files in $OUT_DIR"
echo "Re-run this script to pick up anything that failed; finished runs are skipped."
exit $status
