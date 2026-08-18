#!/usr/bin/env bash
# Overnight queue part 2, 2026-08-18. Restores what part 1 cut on a bad estimate.
#
# Part 1's scope cut assumed ~40 minutes per training seed. That figure counted the
# reference evaluations (random and greedy over 400 episodes) as training. Real training is
# ~7 minutes per seed, confirmed by checkpoint mtimes 7 minutes apart. So the two phases
# dropped from part 1 -- the second clamp-cost arm and the extra seeds -- fit comfortably and
# are restored here.
#
# Still strictly sequential, and this must only be launched once part 1 has drained.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
mkdir -p logs results/ma/checkpoints

EPISODES=4000
EVAL=400

log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { log "START $1"; shift; "$@"; log "  exit $?"; }

# The clamp-cost arm part 1 dropped. With 0.15 already run, 0.05 turns a single point into a
# dose-response curve, which is what distinguishes "the price changed behaviour" from "the
# price happened to land somewhere that changed behaviour".
run "4 clamp_cost 0.05" \
  python scripts/ma_train.py --score_rule joint_conf --seeds 0 1 2 \
    --train_episodes $EPISODES --eval_episodes $EVAL \
    --clamp_cost 0.05 --tag jc_clamp005 \
    --out results/ma/train_clamp005.json

# More seeds at the default setting. The 2026-08-17 spread was 0.165-0.560 over three seeds,
# which is far too wide to quote. Seeds 3-9 take the default arm to ten.
#
# Deliberately at the DEFAULT rather than at whichever clamp cost looks best: choosing the
# arm to reinforce after seeing the results would be selecting on the outcome.
run "5 extra seeds under joint_conf" \
  python scripts/ma_train.py --score_rule joint_conf --seeds 3 4 5 6 7 8 9 \
    --train_episodes $EPISODES --eval_episodes $EVAL \
    --tag joint_conf_extra \
    --out results/ma/train_joint_conf_seeds3to9.json

run "6 role differentiation, all policies" \
  python scripts/ma_role_analysis.py --episodes 300 \
    --out results/ma/role_analysis_all.json

run "7 cross-rule matrix, all policies" \
  python scripts/ma_cross_rule.py --eval_episodes $EVAL \
    --rules subset joint_conf --out results/ma/cross_rule_all.json

log "PART 2 COMPLETE"
