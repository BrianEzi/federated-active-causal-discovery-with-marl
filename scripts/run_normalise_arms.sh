#!/usr/bin/env bash
# Item 4 of the roadmap: does RETURN NORMALISATION reproduce the reward-scale gain without
# the hand-picked constant?
#
# THE COMPARISON, and it needs both halves.
#   8 agents: plain reward scores 0.100 (a08_s0, entropy 1.814 -- it never trained).
#             reward_scale 0.214 scores 0.620 (scale21_s0, entropy 1.340).
#             normalise_returns must land near the second to confirm the mechanism.
#   3 agents: plain reward already trains fine (a03_s0, 0.833, entropy 0.699), and the
#             DIFFERENCE reward cost 0.293 here. A fix for the broken rungs that wrecks the
#             working ones is a trade, not a fix, so the 3-agent control is not optional.
#
# Flags come from each arm's own result file. scale21_s0 is deliberately NOT the source for
# the 8-agent arm -- it carries reward_scale 0.214, and the point is to run without it.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/norm

run () {                                  # run <source> <seed> <name>
  echo "=== $3 starting $(date +%H:%M:%S) ==="
  python -u -m scripts.train_from_config "$1" --seed "$2" \
      --out "results/ladder/$3.json" --run --extra --normalise_returns \
      > "logs/norm/$3.log" 2>&1
  echo "=== $3 finished $(date +%H:%M:%S) rc=$? ==="
}

run results/ladder/a08_s0.json 0 a08norm_s0
run results/ladder/a03_s0.json 0 a03norm_s0
run results/ladder/a08_s0.json 1 a08norm_s1
run results/ladder/a06_s0.json 0 a06norm_s0
echo "ALL DONE $(date +%H:%M:%S)"
