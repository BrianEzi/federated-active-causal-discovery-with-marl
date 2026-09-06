#!/bin/bash
# Pull the Myriad array's rho12 output into a SEPARATE directory, never on top of the local run.
#
# WHY SEPARATE. Both fleets write `results/rho12/rho{RATE}_s{SEED}.json` -- the same 21 paths on
# two machines. Copying the cluster's straight into the repo would overwrite whichever local
# cells had finished, silently, and the two are not guaranteed identical: training is stochastic
# and seeded, but identical seeds do not give identical floating-point trajectories across
# different BLAS builds and CPUs. Once overwritten there is no way to tell which machine
# produced a given number.
#
# So: fetch to `results/rho12_myriad/`, run `scripts/diff_dual_path.py`, and only then decide
# which path each cell comes from -- and say which in the write-up. Never average them.
set -eu
cd "$(dirname "$0")/.."
DEST=${DEST:-results/rho12_myriad}
mkdir -p "$DEST"

# rsync is NOT present in this Git-for-Windows shell -- the first version of this script used
# it and failed on the first fetch. scp over an explicit remote listing is the portable path,
# and it has a second virtue: `ma_train.py` writes its JSON once at the end, so a name that
# appears in the listing is a FINISHED run and a half-written file cannot be picked up.
# Nothing is ever deleted here; this is a landing zone and must not remove local evidence.
files=$(ssh myriad "ls ma_tb/results/rho12/rho*_s?.json 2>/dev/null" | tr -d '\r')
[ -n "$files" ] || { echo "no completed cells on myriad yet"; exit 0; }
for f in $files; do
  b=$(basename "$f")
  [ -f "$DEST/$b" ] && continue          # already fetched; a finished run never changes
  scp -q "myriad:$f" "$DEST/$b" && echo "  fetched $b"
  for ck in _best.pt .pt; do
    scp -q "myriad:ma_tb/results/rho12/${b%.json}${ck}" "$DEST/${b%.json}${ck}" 2>/dev/null || true
  done
done

echo
echo "landed in $DEST:"
ls -1 "$DEST"/*.json 2>/dev/null | wc -l
echo "now run:  PYTHONPATH=. .venv/bin/python scripts/diff_dual_path.py --other $DEST"
