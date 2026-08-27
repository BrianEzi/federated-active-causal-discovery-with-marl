#!/bin/bash -l
# The ATTRIBUTION axis, jobs 1-4 of docs/HANDOVER_2026_08_27.md section 4: does the
# attribution result hold up as it SCALES, under a different REWARD, and under a different
# GENERATOR. Every run here uses flags that already exist; nothing in ma/, cb/ or
# crosscheck/ is touched, because the other session is editing those files.
#
# Writes to results/attr_scale/ -- a NEW directory. Two evaluation processes writing the
# same file has already cost one result set.
#
# Each task trains one (arm, seed) and then scores it with scripts/attr_score.py against
# probe_then_work, greedy_uncertainty and random_vary on IDENTICAL episodes, so the
# reported margin is a per-episode paired difference rather than a difference of two means.
#
#$ -N attr_scale
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l mem=6G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/
set -u

source ~/envs/sa_env/bin/activate
cd ~/ma_attr
mkdir -p logs results/attr_scale ~/.tmp
export PYTHONPATH=. TMPDIR=~/.tmp
# Cap the BLAS threads. Six unthrottled jobs put a machine's load average at 79 and
# starved each other; on a 1-slot SGE task an unthrottled BLAS is simply theft.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TRAIN_EPISODES=${TRAIN_EPISODES:-4000}
EVAL_EPISODES=${EVAL_EPISODES:-150}

# arm | n_agents | private_size | budget | graph_model | reward_flag | seed | max_edges
# Budget is 4 rounds per agent throughout, which is what the 3-agent result was measured
# at; `scripts/vs_evaluate.py --optimal_rounds` gives the exact optimum for a new size and
# should be consulted before any of these margins is trusted at a new one.
#
# `max_edges` is the density guard (ma/density_guard.py), "-" for unguarded. MEASURED
# 2026-08-27: unguarded, four agents is heavy-tailed -- median reset under a second, one
# draw at 48 s, one that did not finish in twenty minutes. At `max_edges 7` the same
# configuration is median 0.73 s, max 3.22 s, and it rejects ~26% of confounded draws.
#
# SIX AGENTS IS NOT IN THIS TABLE and that is a result, not an omission. Guarded at 7
# edges a single 6-agent reset did not complete in the time 12 four-agent resets took, and
# the unguarded timing probe exceeded its 3000 s budget. The edge DISTRIBUTION is nearly
# identical at 3, 4 and 6 agents, so the driver is not window density but the OWNER
# multiplier: the attribution space is a clique partition times a choice of partner, and
# six agents means five partners. No edge cap can reach that, and the cap that would
# (6 edges) rejects 79% of draws -- the 2026-08-26 k=7 guard rejected 94%, sampled an
# unrepresentative sparse tail, and its row was discarded rather than reported.
SPECS=(
  # Job 1 -- scale past three agents, at FOUR. This is where the 2026-08-26 headroom
  # measurement peaked for three shared nodes, so it is the informative rung anyway.
  "attr4a 4 2 16 sf --per_agent_reward 0 7"
  "attr4a 4 2 16 sf --per_agent_reward 1 7"
  # THE CONTROL, and it is not optional. Three agents runs cheaply both ways, so the same
  # configuration guarded and unguarded measures what the guard costs instead of assuming
  # it costs nothing. Compare against attr3a_peragent below, which is unguarded.
  "attr3a_guarded 3 2 12 sf --per_agent_reward 0 7"
  "attr3a_guarded 3 2 12 sf --per_agent_reward 1 7"
  # Job 2 -- reward ablation. The per-agent arm doubles as job 1's 3-agent reference.
  "attr3a_peragent 3 2 12 sf --per_agent_reward 0 -"
  "attr3a_peragent 3 2 12 sf --per_agent_reward 1 -"
  "attr3a_peragent 3 2 12 sf --per_agent_reward 2 -"
  "attr3a_shared 3 2 12 sf - 0 -"
  "attr3a_shared 3 2 12 sf - 1 -"
  "attr3a_shared 3 2 12 sf - 2 -"
  # Job 3 -- private sets of three. Window size 6, the top of the usable range.
  "attr3a_p3 3 3 12 sf --per_agent_reward 0 -"
  "attr3a_p3 3 3 12 sf --per_agent_reward 1 -"
  "attr4a_p3 4 3 16 sf --per_agent_reward 0 -"
  "attr4a_p3 4 3 16 sf --per_agent_reward 1 -"
  # Job 4 -- Erdos-Renyi against the existing scale-free results, matched.
  "attr3a_er 3 2 12 er --per_agent_reward 0 -"
  "attr3a_er 3 2 12 er --per_agent_reward 1 -"
)

SPEC=${SPECS[$((SGE_TASK_ID - 1))]}
read -r ARM N_AGENTS PRIVATE BUDGET GRAPH REWARD SEED MAX_EDGES <<< "$SPEC"
[ "$REWARD" = "-" ] && REWARD=""
# "-" means unguarded, and DensityGuardedEnv(max_edges=None) is the ordinary
# environment exactly -- so both halves of the three-agent control share a code path.
GUARD=""
[ "$MAX_EDGES" != "-" ] && GUARD="--max_edges $MAX_EDGES"

OUT="results/attr_scale/${ARM}_s${SEED}.json"
SCORE="results/attr_scale/${ARM}_s${SEED}_scored.json"

echo "=== $ARM seed $SEED : n_agents=$N_AGENTS private=$PRIVATE budget=$BUDGET graph=$GRAPH reward='$REWARD' guard='$MAX_EDGES' : $(date) ==="

COMMON="--n_agents $N_AGENTS --private_size $PRIVATE --n_shared 3 --budget $BUDGET \
  --backend attributed --graph_model $GRAPH --claim_bar 1.0 --reward_criterion claims \
  --policy_arch gnn --turn_order round_robin --episode_mix confounded --disclose_regime \
  --n_obs 60 --n_int 20 --observe_belief_channels --observe_partner_counts --vary_only"

if [ -f "$OUT" ]; then
  echo "$OUT exists -- skipping training"
else
  python -m scripts.ma_train --seed "$SEED" --arm "$ARM" $COMMON $REWARD $GUARD \
    --train_episodes "$TRAIN_EPISODES" --eval_episodes "$EVAL_EPISODES" \
    --no_wandb --out "$OUT"
fi

# The four reported metrics. `--shared_reward` must MATCH the training arm: the scorer
# rebuilds the environment, and a mismatched reward criterion would score the baselines on
# a different task from the learner.
SCORE_REWARD="--per_agent_reward"
[ -z "$REWARD" ] && SCORE_REWARD="--shared_reward"

if [ -f "$SCORE" ]; then
  echo "$SCORE exists -- skipping scoring"
else
  python -m scripts.attr_score --n_agents "$N_AGENTS" --private_size "$PRIVATE" \
    --n_shared 3 --budget "$BUDGET" --graph_model "$GRAPH" $SCORE_REWARD $GUARD \
    --policy "results/attr_scale/${ARM}_s${SEED}.pt" \
    --episodes "$EVAL_EPISODES" --seed "$SEED" --out "$SCORE"
fi

echo "=== done $(date) ==="
