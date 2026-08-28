#!/usr/bin/env bash
# Second batch of eval-only work, beside the two training queues.
#
# 1. MI GATE ON THE AGENT LADDER. docs/ROADMAP_AGENT_B says the six-agent coordination row
#    "is not yet fair -- it uses a06_s0, which failed the MI gate". That verdict came from
#    mi_check2.py, which no longer exists, so it cannot currently be checked. Now it can.
#    scale21_s0 is included because it is the arm the return-normalisation result will be
#    compared against, and a gate reading on it is needed before that comparison is made.
#
# 2. THE AGENT AXIS AGAINST ITS OWN RATIO. The window axis turned out to be a budget story.
#    The agent axis is NOT starved, but it has never been plotted against budget/required
#    either, and the per-episode variation is there for free.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/eval results/cover

echo "=== MI gate, agent ladder $(date +%H:%M:%S) ==="
python -u -m scripts.mi_gate \
    results/ladder/a02_s0.json results/ladder/a03_s0.json \
    results/ladder/a06_s0.json results/ladder/a06_s1.json results/ladder/a06_s2.json \
    results/ladder/a08_s0.json results/ladder/scale21_s0.json results/ladder/scale21_s1.json \
    --episodes 15 --out results/cover/mi_gate_agents.json \
    > logs/eval/mi_agents.log 2>&1
echo "  rc=$?"

echo "=== budget curve, agent ladder $(date +%H:%M:%S) ==="
python -u -m scripts.budget_curve \
    results/ladder/a02_s0.json results/ladder/a03_s0.json \
    results/ladder/a06_s0.json results/ladder/a08_s0.json \
    --episodes 120 --out results/cover/budget_curve_agents.json \
    > logs/eval/curve_agents.log 2>&1
echo "  rc=$?"
echo "EVAL DONE $(date +%H:%M:%S)"
