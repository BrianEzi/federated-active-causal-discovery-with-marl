#!/bin/bash
# Re-score every attribution checkpoint against greedy at the bar the task is GRADED on.
#
# Evaluation only -- `UncertaintyGreedyAgent` is a rule, not a trained policy, so fixing its
# bar costs no training. Run locally because the cluster released this array one task at a
# time behind a concurrency limit; seven concurrent here beats a serialised queue.
#
# Writes to results/attr_bar1/. The cluster copy of the same array writes to its OWN
# filesystem and is NOT synced into this directory -- two evaluation processes writing one
# file has already cost this project a result set.
set -u
cd "$(dirname "$0")/.."
PY="C:/Workspace/MSc Project/.venv/Scripts/python.exe"
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p results/attr_bar1 logs
EPISODES=${EPISODES:-150}
MAXJOBS=${MAXJOBS:-7}          # job 6 holds six cores; leave the box some headroom

# arm | n_agents | private | budget | graph | reward | seed | max_edges
SPECS=(
  "attr4a 4 2 16 sf --per_agent_reward 0 7"
  "attr4a 4 2 16 sf --per_agent_reward 1 7"
  "attr3a_guarded 3 2 12 sf --per_agent_reward 0 7"
  "attr3a_guarded 3 2 12 sf --per_agent_reward 1 7"
  "attr3a_peragent 3 2 12 sf --per_agent_reward 0 -"
  "attr3a_peragent 3 2 12 sf --per_agent_reward 1 -"
  "attr3a_peragent 3 2 12 sf --per_agent_reward 2 -"
  "attr3a_shared 3 2 12 sf - 0 -"
  "attr3a_shared 3 2 12 sf - 1 -"
  "attr3a_shared 3 2 12 sf - 2 -"
  "attr3a_p3 3 3 12 sf --per_agent_reward 0 -"
  "attr3a_p3 3 3 12 sf --per_agent_reward 1 -"
  "attr3a_er 3 2 12 er --per_agent_reward 0 -"
  "attr3a_er 3 2 12 er --per_agent_reward 1 -"
)

for SPEC in "${SPECS[@]}"; do
  read -r ARM N_AGENTS PRIVATE BUDGET GRAPH REWARD SEED MAX_EDGES <<< "$SPEC"
  [ "$REWARD" = "-" ] && REWARD=""
  GUARD=""
  [ "$MAX_EDGES" != "-" ] && GUARD="--max_edges $MAX_EDGES"
  SCORE_REWARD="--per_agent_reward"
  [ -z "$REWARD" ] && SCORE_REWARD="--shared_reward"

  CKPT="results/attr_scale/${ARM}_s${SEED}.pt"
  OUT="results/attr_bar1/${ARM}_s${SEED}_scored.json"
  [ -f "$CKPT" ] || { echo "no checkpoint $CKPT -- skipping"; continue; }
  [ -f "$OUT" ] && { echo "$OUT exists -- skipping"; continue; }

  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "re-scoring $ARM seed $SEED"
  "$PY" -m scripts.attr_score --n_agents "$N_AGENTS" --private_size "$PRIVATE" \
    --n_shared 3 --budget "$BUDGET" --graph_model "$GRAPH" $SCORE_REWARD $GUARD \
    --greedy_bar 1.0 --policy "$CKPT" \
    --episodes "$EPISODES" --seed "$SEED" --out "$OUT" \
    > "logs/bar1_${ARM}_s${SEED}.log" 2>&1 &
done
wait
echo "RESCORE-ALL-DONE"
