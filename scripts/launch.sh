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
# HEALTH FIRST, because it is the one that invalidates the others. A sweep launched into a
# thrashing machine runs 2-5x slow and misreports its own cost, which is how an entire run
# plan got built on inflated numbers on 30 Aug. Set MAX_SWAPINS high to override.
.venv/bin/python scripts/preflight_runs.py health --max_swapins "${MAX_SWAPINS:-200}" \
  || { echo "MACHINE HEALTH GATE FAILED -- fix the memory pressure before launching"; exit 1; }
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

# ONE SCRIPT PER JOB, then xargs over the PATHS. This looks indirect and is not:
# `xargs -I` caps the replacement string at 255 bytes on macOS, and a job line here is
# about 600 characters, so the obvious `xargs -P N -I CMD sh -c CMD` dies instantly with
# "command line cannot be assembled, too long" -- after the gates have passed, which is
# the worst possible place to fail. Paths are short, so this has no such limit, and each
# script keeps its own `[ -f "$out" ] ||` guard so restart-safety is unchanged.
JOB_DIR="$OUT_DIR/jobs"
rm -rf "$JOB_DIR"; mkdir -p "$JOB_DIR"
n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  n=$((n + 1))
  printf '#!/usr/bin/env bash\nexport PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\ncd %s\n%s\n' \
    "$(pwd)" "$line" > "$JOB_DIR/$(printf '%03d' $n).sh"
done < "$JOBS"
chmod +x "$JOB_DIR"/*.sh
echo "wrote $n job scripts to $JOB_DIR"

# `caffeinate -i -s` holds off idle AND system sleep, which is what an overnight run wants.
# It is macOS-only, so fall back rather than failing on Linux -- this script runs on three
# machines and a hard dependency on one platform's power tool would strand two of them.
RUNNER="xargs -P $WORKERS -n 1 bash"
if command -v caffeinate > /dev/null 2>&1; then
  ls "$JOB_DIR"/*.sh | caffeinate -i -s $RUNNER
else
  echo "(no caffeinate -- not macOS. Disable sleep yourself if this is a laptop.)"
  ls "$JOB_DIR"/*.sh | $RUNNER
fi
status=$?
rm -f "$JOBS"

echo
echo "=== done (xargs exit $status) ==="
DONE=$(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l)
echo "$DONE result files in $OUT_DIR"
echo "Re-run this script to pick up anything that failed; finished runs are skipped."
exit $status
