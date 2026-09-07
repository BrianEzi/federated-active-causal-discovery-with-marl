#!/bin/bash
# Pull the Myriad array's rho12 output into a SEPARATE directory, never on top of the local run.
#
# WHY SEPARATE. Both fleets write `results/<fleet>/rho{RATE}_s{SEED}.json` -- the same paths on
# two machines. Copying the cluster's straight into the repo would overwrite whichever local
# cells had finished, silently, and the two are not guaranteed identical: training is stochastic
# and seeded, but identical seeds do not give identical floating-point trajectories across
# different BLAS builds and CPUs. Once overwritten there is no way to tell which machine
# produced a given number.
#
# So: fetch to `results/<fleet>_myriad/`, run `scripts/diff_dual_path.py`, and only then decide
# which path each cell comes from -- and say which in the write-up. Never average them.
set -eu
cd "$(dirname "$0")/.."
# PARAMETERISED because the fleet moved. The uncompensated grid was `rho12`; the
# rate-compensated one is `rho12b`, and a hardcoded path here would have fetched nothing at
# 01:00 while reporting success -- the loop simply finds no files and the caller sees an
# empty landing directory rather than an error.
REMOTE=${REMOTE:-rho12b}
DEST=${DEST:-results/${REMOTE}_myriad}
mkdir -p "$DEST"

# rsync is NOT present in this Git-for-Windows shell -- the first version of this script used
# it and failed on the first fetch. scp over an explicit remote listing is the portable path,
# and it has a second virtue: `ma_train.py` writes its JSON once at the end, so a name that
# appears in the listing is a FINISHED run and a half-written file cannot be picked up.
# Nothing is ever deleted here; this is a landing zone and must not remove local evidence.
files=$(ssh myriad "ls ma_tb/results/$REMOTE/rho*_s?.json 2>/dev/null" | tr -d '\r')
[ -n "$files" ] || { echo "no completed cells on myriad yet"; exit 0; }
for f in $files; do
  b=$(basename "$f")
  # The JSON guard must NOT gate the checkpoints. It used to: an early fetch that caught the
  # JSON before its `_best.pt` was written would skip the whole cell forever afterwards, and
  # `rho12on/rho0.80_s2` reached agent A as a result with no checkpoint to score. Each artefact
  # is now checked for its own existence.
  [ -f "$DEST/$b" ] || { scp -q "myriad:$f" "$DEST/$b" && echo "  fetched $b"; }
  for ck in _best.pt .pt; do
    [ -f "$DEST/${b%.json}${ck}" ] && continue
    scp -q "myriad:ma_tb/results/$REMOTE/${b%.json}${ck}" "$DEST/${b%.json}${ck}" 2>/dev/null       && echo "  fetched ${b%.json}${ck}"
  done
done

echo
echo "landed in $DEST:"
ls -1 "$DEST"/*.json 2>/dev/null | wc -l
echo "now run:  PYTHONPATH=. .venv/bin/python scripts/diff_dual_path.py --other $DEST"
