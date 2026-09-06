#!/bin/bash
# Paired scoring for the k12 ladder, arms A (federated) and E (pooled), all seeds present.
#
# Both checkpoint conventions, per the standing rule: `best` is the mi-gate selection and
# `final` is the last update, and at 12,000 episodes each has a tail the other does not
# (docs/FINDINGS_CHECKPOINT_TAIL_2026_09_02.md). Reporting one without the other has already
# produced one retraction on this project.
#
# `--arms all` rather than `--baseline_from`: arms A and E build their environments from
# DIFFERENT configs -- E carries observe_belief_channels and observe_partner_counts -- and the
# 3x saving is only sound when the myopic arm provably replays identical episodes. It does for
# a rate sweep, where only one field moves and it is not an observation flag. Here it is an
# observation flag, so the baselines are recomputed rather than assumed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=results/central12k/scored
mkdir -p "$OUT" logs/ladder
WORKERS=${WORKERS:-4}

jobs=$(mktemp)
for f in results/central12k/v2_k12_?_s*.json; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .json)
  for ck in best final; do
    out="$OUT/${b}_${ck}.json"
    [ -f "$out" ] && continue
    echo "$f $ck $out" >> "$jobs"
  done
done
echo "$(date +%H:%M:%S)  ladder scoring: $(wc -l < "$jobs") evaluations, $WORKERS workers"

run_one() {
  read -r src ck out <<< "$1"
  .venv/bin/python scripts/global_shd_paired.py "$src" --episodes 200 --sample \
    --checkpoint "$ck" --out "$out" > "logs/ladder/score_$(basename "$out" .json).log" 2>&1
  echo "$(date +%H:%M:%S)  scored $(basename "$out")"
}
export -f run_one
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  LADDER SCORING COMPLETE"
