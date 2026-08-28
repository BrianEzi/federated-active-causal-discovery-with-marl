#!/usr/bin/env bash
# The seeds the window ladder is missing: w20 has two, w30 has one, the rest have three.
#
# SEQUENTIAL ON PURPOSE. Run three at once on this machine and each takes ~210 s per update
# against the original runs' 23 s -- 16 cores, torch defaulting to 8 threads per process,
# so three trainings plus an evaluation oversubscribe by 2x and thrash. One at a time
# finishes the set sooner in wall-clock than three in parallel do.
#
# Flags come from each rung's own result file via train_from_config.py, which verifies the
# round trip through ma_train's own code path before it spends anything.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/seeds

run () {                                  # run <source> <seed> <name>
  echo "=== $3 starting $(date +%H:%M:%S) ==="
  python -u -m scripts.train_from_config "$1" --seed "$2" \
      --out "results/ladder/$3.json" --run > "logs/seeds/$3.log" 2>&1
  echo "=== $3 finished $(date +%H:%M:%S) rc=$? ==="
}

run results/ladder/w20_s0.json 2 w20_s2
run results/ladder/w30_s0.json 1 w30_s1
run results/ladder/w30_s0.json 2 w30_s2
echo "ALL DONE $(date +%H:%M:%S)"
