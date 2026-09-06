#!/bin/bash
# RQ2 at the principal cell with a RATE-COMPENSATED budget, so effective beta is 1.5 everywhere.
#
# WHY THIS REPLACES `rho12_fleet.sh`. That fleet held the budget fixed at 50, the cell's derived
# beta=1.5 value under a FULL oracle. At answer rate rho an agent gets an answer to only rho of
# its interventions, so a fixed budget means the effective training pressure falls with the dial:
#
#     k=12 budget 50:  rho=1.00 -> beta 1.51   0.90 -> 1.36   0.70 -> 1.06   0.50 -> 0.75
#
# The low rates were being trained at HALF the standard pressure, and they failed the competence
# floor accordingly (window rate 0.173 at rho=0.50, 0.412 at rho=0.70, against 0.994 at rho=1.00).
# That measured budget starvation and called it answer-rate intolerance. Brian's call, and the
# methodologically sound one: hold beta at 1.5 and let the dial be the only thing that moves.
#
#     rho     0.95  0.90  0.85  0.80  0.70  0.50
#     budget    53    56    59    63    72   100     -> effective beta 1.51-1.52 throughout
#
# The same fault runs the other way in the k=8 grid, which held budget at 70 and therefore
# trained at beta 3.02 at rho=1.00 down to 1.51 at rho=0.50. Both sweeps confounded the answer
# rate with training pressure; this one does not.
#
# rho=1.00 IS ABSENT ON PURPOSE. Budget 50 at rho=1.00 already is beta 1.5, and the ladder's
# twelve arm-A runs are exactly that cell, already scored under both checkpoint conventions.
# Training three more would give a worse-powered copy of a control we hold.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/rho12b logs/rho12b
WORKERS=${WORKERS:-6}

jobs=$(mktemp)
n=0
while IFS= read -r cmd; do
  n=$((n+1))
  out=$(echo "$cmd" | grep -o -- '--out [^ ]*' | cut -d' ' -f2)
  [ -f "$out" ] && continue
  echo "$n|$cmd" >> "$jobs"
done < cluster/jobs/rho12b.txt
echo "$(date +%H:%M:%S)  rho12b fleet: $(wc -l < "$jobs") runs, $WORKERS workers"

run_one() {
  n=${1%%|*}; cmd=${1#*|}
  eval ".venv/bin/${cmd}" > "logs/rho12b/job${n}.log" 2>&1
  echo "$(date +%H:%M:%S)  done job $n"
}
export -f run_one
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  RHO12B FLEET COMPLETE"
