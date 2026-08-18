#!/usr/bin/env bash
# Overnight queue, 2026-08-18. STRICTLY SEQUENTIAL.
#
# Sequential is not stylistic: on 2026-08-17 I launched a second training run believing the
# first had died, two processes competed for CPU for ~15 hours, and every per-seed timing in
# that run became meaningless. One job at a time.
#
# SCOPE WAS CUT after measuring the actual rate. The first version of this queue was 16
# training runs at 6000 episodes -- roughly 16 hours against maybe 7 available. Rather than
# let it run and deliver four half-finished phases, it was stopped ~15 minutes in and
# rewritten. Two changes:
#
#   6000 -> 4000 training episodes. Justified from the 2026-08-17 curves: split into
#   quarters, mean solve rate ran 0.641/0.700/0.703/0.800, 0.703/0.637/0.734/0.700 and
#   0.641/0.713/0.641/0.775. There is real but modest gain late, and one seed is flat, so
#   the last third is the cheapest thing to give up.
#
#   The clamp-cost sweep drops from two arms to one, and the extra-seeds phase is dropped
#   entirely. Cross-rule evaluation is the user's stated blocker and it needs BOTH training
#   arms complete, so it is protected ahead of everything else.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
mkdir -p logs results/ma/checkpoints

EPISODES=4000
EVAL=400

log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { log "START $1"; shift; "$@"; log "  exit $?"; }

# --- Phase 1: both training arms, WITH checkpoints (protects phase 2) ---------------
run "1a train under subset" \
  python scripts/ma_train.py --score_rule subset --seeds 0 1 2 \
    --train_episodes $EPISODES --eval_episodes $EVAL \
    --out results/ma/train_subset.json

run "1b train under joint_conf" \
  python scripts/ma_train.py --score_rule joint_conf --seeds 0 1 2 \
    --train_episodes $EPISODES --eval_episodes $EVAL \
    --out results/ma/train_joint_conf_v2.json

# --- Phase 2: the blocker -----------------------------------------------------------
run "2a cross-rule matrix" \
  python scripts/ma_cross_rule.py --eval_episodes $EVAL \
    --rules subset joint_conf --out results/ma/cross_rule.json

run "2b role differentiation" \
  python scripts/ma_role_analysis.py --episodes 300 \
    --out results/ma/role_analysis.json

# --- Phase 3: one clamp-cost arm ----------------------------------------------------
# 0.15 rather than 0.05: with step_cost 0.05 and a terminal reward of 1.0, a 0.05 clamp
# surcharge is a 3% swing on a 6-round episode and would likely be lost in the seed noise
# already measured (solve rates 0.165-0.560). If a price on clamping cannot change
# behaviour at 0.15 it will not at 0.05, so the larger arm is the informative one to run
# when only one fits.
run "3 clamp_cost 0.15" \
  python scripts/ma_train.py --score_rule joint_conf --seeds 0 1 2 \
    --train_episodes $EPISODES --eval_episodes $EVAL \
    --clamp_cost 0.15 --tag jc_clamp015 \
    --out results/ma/train_clamp015.json

run "3b role differentiation under clamp cost" \
  python scripts/ma_role_analysis.py --episodes 300 \
    --out results/ma/role_analysis_clamped.json

log "ALL PHASES COMPLETE"
