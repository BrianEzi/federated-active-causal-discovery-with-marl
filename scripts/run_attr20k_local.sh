#!/bin/bash
# THE CONVERGENCE EXPERIMENT: does attribution learn, given enough episodes?
#
# Measured 2026-08-27: the attribution policies barely condition on their observation
# (I(S;A)/H = 0.035 / 0.188 / 0.291 against greedy's 1.000) and finished training at
# entropy ~1.05, where the 20,000-episode version_space runs finish at 0.30-0.50 and reach
# solve 1.000. That is the same failure mode as the generator confound resolved earlier the
# same day, where a 3,000-episode run appeared to lose to greedy and needed 20,000.
#
# CONFIG IS IDENTICAL TO THE 4,000-EPISODE RUNS apart from --train_episodes. The question is
# whether episodes alone fix it, so nothing else may move -- including the reward, even
# though the shared reward measured better; a like-for-like comparison is the whole point.
#
# Three 3-agent shared-reward seeds here; the two 4-agent seeds go to Myriad, where they run
# faster and do not compete with job 6 for this machine's cores.
#
# Scores itself at --greedy_bar 1.0, the bar the task is actually graded on. The 0.7 default
# is what inverted today's headline.
set -u
cd "$(dirname "$0")/.."
PY="C:/Workspace/MSc Project/.venv/Scripts/python.exe"
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUTDIR=results/attr_20k
mkdir -p "$OUTDIR" logs
EPISODES=${EPISODES:-20000}

COMMON="--n_agents 3 --private_size 2 --n_shared 3 --budget 12 \
  --backend attributed --graph_model sf --claim_bar 1.0 --reward_criterion claims \
  --policy_arch gnn --turn_order round_robin --episode_mix confounded --disclose_regime \
  --n_obs 60 --n_int 20 --observe_belief_channels --observe_partner_counts --vary_only"

for SEED in 0 1 2; do
  OUT="$OUTDIR/attr3a_shared20k_s${SEED}.json"
  [ -f "$OUT" ] && { echo "$OUT exists -- skipping"; continue; }
  echo "launching 3-agent shared seed $SEED, $EPISODES episodes"
  (
    "$PY" -m scripts.ma_train $COMMON \
      --train_episodes "$EPISODES" --eval_episodes 150 --seed "$SEED" \
      --no_wandb --arm attr3a_shared20k --out "$OUT"
    "$PY" -m scripts.attr_score --n_agents 3 --private_size 2 --n_shared 3 --budget 12 \
      --graph_model sf --shared_reward --greedy_bar 1.0 \
      --policy "$OUTDIR/attr3a_shared20k_s${SEED}.pt" \
      --episodes 150 --seed "$SEED" \
      --out "$OUTDIR/attr3a_shared20k_s${SEED}_scored.json"
  ) > "logs/attr20k_s${SEED}.log" 2>&1 &
done
wait
echo "ATTR20K-LOCAL-DONE"
