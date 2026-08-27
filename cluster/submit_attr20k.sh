#!/bin/bash -l
# THE CONVERGENCE EXPERIMENT, 4-agent half. See scripts/run_attr20k_local.sh for the
# reasoning; the three 3-agent seeds run on the student's machine and these two run here,
# where a 20,000-episode 4-agent run costs about 5.9 h against roughly 8.3 h locally.
#
# The 4-agent arm is the most informative of the set: it carried the LARGEST deficit against
# a correctly-configured greedy (-0.239 +/- 0.021), so it has the most room to move if
# undertraining is the explanation.
#
# CONFIG IDENTICAL to the 4,000-episode attr4a runs apart from --train_episodes, including
# --per_agent_reward and --max_edges 7. The shared reward measured better, but changing two
# things at once is what made the generator result unattributable this morning.
#
#$ -N attr20k
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l mem=6G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/
set -u

source ~/envs/sa_env/bin/activate
cd ~/ma_attr
mkdir -p logs results/attr_20k ~/.tmp
export PYTHONPATH=. TMPDIR=~/.tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SEED=$((SGE_TASK_ID - 1))
EPISODES=${EPISODES:-20000}
OUT="results/attr_20k/attr4a20k_s${SEED}.json"
SCORE="results/attr_20k/attr4a20k_s${SEED}_scored.json"

COMMON="--n_agents 4 --private_size 2 --n_shared 3 --budget 16 \
  --backend attributed --graph_model sf --claim_bar 1.0 --reward_criterion claims \
  --per_agent_reward --policy_arch gnn --turn_order round_robin --episode_mix confounded \
  --disclose_regime --n_obs 60 --n_int 20 --observe_belief_channels \
  --observe_partner_counts --vary_only --max_edges 7"

echo "=== attr4a20k seed $SEED, $EPISODES episodes : $(date) ==="
if [ -f "$OUT" ]; then
  echo "$OUT exists -- skipping training"
else
  python -m scripts.ma_train --seed "$SEED" --arm attr4a20k $COMMON \
    --train_episodes "$EPISODES" --eval_episodes 150 --no_wandb --out "$OUT"
fi

# Scored at the bar the task is GRADED on. The 0.7 default inverted today's headline.
if [ -f "$SCORE" ]; then
  echo "$SCORE exists -- skipping scoring"
else
  python -m scripts.attr_score --n_agents 4 --private_size 2 --n_shared 3 --budget 16 \
    --graph_model sf --per_agent_reward --max_edges 7 --greedy_bar 1.0 \
    --policy "results/attr_20k/attr4a20k_s${SEED}.pt" \
    --episodes 150 --seed "$SEED" --out "$SCORE"
fi
echo "=== done $(date) ==="
