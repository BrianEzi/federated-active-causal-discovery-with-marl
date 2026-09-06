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

# -a preserves mtimes so `--check`-style hashing downstream stays meaningful; no --delete, since
# this directory is a landing zone and nothing here should ever remove local evidence.
rsync -a --info=stats1 \
  --include='rho*_s?.json' --include='rho*_s?_best.pt' --include='rho*_s?.pt' --exclude='*' \
  myriad:ma_tb/results/rho12/ "$DEST"/

echo
echo "landed in $DEST:"
ls -1 "$DEST"/*.json 2>/dev/null | wc -l
echo "now run:  PYTHONPATH=. .venv/bin/python scripts/diff_dual_path.py --other $DEST"
