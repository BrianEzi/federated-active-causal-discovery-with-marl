#!/usr/bin/env bash
# Run one sweep cell, picking up wherever the last attempt stopped.
#
#   scripts/resume_or_start.sh results/sweep/oracle/k30s50n04b150_s0.json \
#       .venv/bin/python scripts/ma_train.py --arm k30s50n04b150 ... --out <same path>
#
# WHY THIS EXISTS. k=30 is 12.3 hours per run, measured. No single Myriad job gets that
# much walltime, so the run has to survive being killed and resubmitted -- which is only
# possible because `CheckpointWriter` writes resume state every 50 updates and, since
# 30 Aug 2026, something finally reads it.
#
# Three cases, in order:
#   the result file exists          -> finished. Do nothing. This is what makes blind
#                                      resubmission safe, and it is the same guard the
#                                      sweep's job lines use.
#   a *_resume_uNNNN.pt exists      -> continue from the HIGHEST one. The writer keeps the
#                                      last two, so a checkpoint torn by a kill mid-write
#                                      still leaves an intact predecessor.
#   neither                         -> start from scratch.
#
# Submit the same array job as many times as it takes; each pass advances every unfinished
# task and skips every finished one.
set -uo pipefail

OUT=$1; shift
[ $# -gt 0 ] || { echo "usage: $0 <out.json> <command...>" >&2; exit 2; }

if [ -f "$OUT" ]; then
  echo "skip     $(basename "$OUT") -- already complete"
  exit 0
fi

# `_resume_uNNNN.pt` sorts correctly as text because the writer zero-pads to four digits.
STEM=${OUT%.json}
LATEST=$(ls -1 "${STEM}"_resume_u*.pt 2>/dev/null | sort | tail -1)

if [ -n "$LATEST" ]; then
  echo "resume   $(basename "$OUT") from $(basename "$LATEST")"
  exec "$@" --resume_from "$LATEST"
fi

echo "start    $(basename "$OUT") from scratch"
exec "$@"
