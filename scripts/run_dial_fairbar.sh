#!/bin/bash
# Re-score the converged noise dial against a greedy configured at the graded bar.
#
# The dial's own numbers come from scripts/ma_train.py's evaluation, which builds
# UncertaintyGreedyAgent at its bar=0.7 default while these runs grade at claim_bar=1.0.
# That handicap was worth +0.233 to greedy at four agents and INVERTED the attribution
# headline, so no learned-vs-greedy number from the dial can be quoted until it is redone.
#
# Evaluation only, from the saved checkpoints; the 7-hour training runs are not repeated.
set -u
cd "$(dirname "$0")/.."
PY="C:/Workspace/MSc Project/.venv/Scripts/python.exe"
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

mkdir -p results/dial_fairbar logs
for f in dial_n100_s0 dial_n100_s1 dial_n1000_s0 dial_n1000_s1 dial_n4000_s0 dial_n4000_s1; do
  src="results/vs_dial_converged/$f.json"
  out="results/dial_fairbar/$f.json"
  [ -f "$src" ] || { echo "$src not written yet -- skipping"; continue; }
  [ -f "$out" ] && { echo "$out exists -- skipping"; continue; }
  "$PY" -u -m scripts.rescore_from_config "$src" --episodes 150 --out "$out" \
      > "logs/fairbar_$f.log" 2>&1 &
done
wait
echo "DIAL-FAIRBAR-DONE"
