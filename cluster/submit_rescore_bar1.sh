#!/bin/bash -l
# RE-SCORE every attribution checkpoint against a greedy baseline configured at the bar the
# task is actually graded on.
#
# WHY THIS IS CHEAP. `UncertaintyGreedyAgent` is a rule, not a trained policy -- fixing its
# `bar` changes nothing about the learned checkpoints, which are all saved beside their
# results. So this is EVALUATION ONLY: no retraining, and the 20,000-episode training runs
# behind these numbers are not repeated.
#
# WHY IT IS NEEDED. Every construction in the repository built greedy at bar=0.7 while these
# backends grade at claim_bar=1.0, so the baseline stopped scoring claims the task still
# counted open. Measured 2026-08-27: worth +0.233 to greedy at 4 agents on scale-free, and
# on the attributed 3-agent task it REVERSES the headline -- learned minus greedy goes from
# +0.142 +/- 0.024 to -0.091 +/- 0.025 on identical episodes.
#
# Results go to results/attr_bar1/ so the bar-0.7 numbers in results/attr_scale/ survive
# untouched and the two can be compared.
#
#$ -N attr_bar1
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l mem=6G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/
set -u

source ~/envs/sa_env/bin/activate
cd ~/ma_attr
mkdir -p logs results/attr_bar1 ~/.tmp
export PYTHONPATH=. TMPDIR=~/.tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

EVAL_EPISODES=${EVAL_EPISODES:-150}

# arm | n_agents | private_size | budget | graph_model | reward_flag | seed | max_edges
# Identical to cluster/submit_attr_scale.sh's table, minus the tasks that were never run.
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

SPEC=${SPECS[$((SGE_TASK_ID - 1))]}
read -r ARM N_AGENTS PRIVATE BUDGET GRAPH REWARD SEED MAX_EDGES <<< "$SPEC"
[ "$REWARD" = "-" ] && REWARD=""
GUARD=""
[ "$MAX_EDGES" != "-" ] && GUARD="--max_edges $MAX_EDGES"

SCORE_REWARD="--per_agent_reward"
[ -z "$REWARD" ] && SCORE_REWARD="--shared_reward"

CKPT="results/attr_scale/${ARM}_s${SEED}.pt"
OUT="results/attr_bar1/${ARM}_s${SEED}_scored.json"

if [ ! -f "$CKPT" ]; then
  echo "no checkpoint at $CKPT -- nothing to re-score"
  exit 0
fi
if [ -f "$OUT" ]; then
  echo "$OUT exists -- skipping"
  exit 0
fi

echo "=== re-scoring $ARM seed $SEED at greedy_bar 1.0 : $(date) ==="
python -m scripts.attr_score --n_agents "$N_AGENTS" --private_size "$PRIVATE" \
  --n_shared 3 --budget "$BUDGET" --graph_model "$GRAPH" $SCORE_REWARD $GUARD \
  --greedy_bar 1.0 --policy "$CKPT" \
  --episodes "$EVAL_EPISODES" --seed "$SEED" --out "$OUT"
echo "=== done $(date) ==="
