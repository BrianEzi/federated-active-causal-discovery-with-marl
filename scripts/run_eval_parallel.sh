#!/usr/bin/env bash
# Eval-only work that needs no training, to run beside the two training queues.
#
# ONE process, sequential. Two trainings already hold 16 threads on 16 cores; a third
# concurrent job is affordable, a fourth was measured to thrash.
#
# 1. THE AGENT LADDER'S REQUIRED COVER. The window ladder turned out to be budget-starved at
#    its top end. Nobody has asked the same question of the AGENT axis, where the collapse at
#    6 and 8 agents is currently attributed to credit assignment and reward scale. If those
#    rungs are also below ratio 1.0 then part of that collapse is budget too, and the
#    difference-reward and reward-scale results were measured on a starved arm.
#
# 2. THREE SEEDS ON THE BUDGET-NORMALISED CURVE. w04/w06/w08/w12 already have s1 and s2
#    checkpoints. The curve costs evaluation only, so the headline figure can carry three
#    seeds today rather than after the retrains.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/eval results/cover

echo "=== agent-ladder required cover $(date +%H:%M:%S) ==="
for rung in a02_s0 a03_s0 a06_s0 a08_s0; do
  python -u -m scripts.required_cover "results/ladder/${rung}.json" \
      --episodes 40 --closed_form_only --out "results/cover/${rung}.json" \
      > "logs/eval/cover_${rung}.log" 2>&1
  echo "  ${rung} rc=$?"
done

echo "=== budget curve, seeds 1 and 2 $(date +%H:%M:%S) ==="
for seed in 1 2; do
  python -u -m scripts.budget_curve \
      results/ladder/w04_s${seed}.json results/ladder/w06_s${seed}.json \
      results/ladder/w08_s${seed}.json results/ladder/w12_s${seed}.json \
      --episodes 120 --out "results/cover/budget_curve_s${seed}.json" \
      > "logs/eval/curve_s${seed}.log" 2>&1
  echo "  seed ${seed} rc=$?"
done
echo "EVAL DONE $(date +%H:%M:%S)"
