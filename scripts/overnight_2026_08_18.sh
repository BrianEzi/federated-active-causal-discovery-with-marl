#!/usr/bin/env bash
# Overnight queue, 2026-08-18. STRICTLY SEQUENTIAL.
#
# Sequential is not a stylistic choice: on 2026-08-17 I launched a second training run
# believing the first had died, and two processes competed for CPU for ~15 hours, making
# every per-seed timing in that run meaningless. One job at a time, each waited on.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Phase 1: train under each rule, WITH checkpoints -------------------------------
# subset is the rule that failed on 2026-08-16 (every seed 0.000 on confounded episodes).
# It is retrained here not because it is expected to work but because the cross-rule matrix
# needs a policy trained under it -- "the valley is a learning obstacle, not a scoring one"
# is only testable if both arms exist.
log "phase 1a: training under subset (3 seeds)"
python scripts/ma_train.py --score_rule subset --seeds 0 1 2 \
    --train_episodes 6000 --eval_episodes 400 \
    --out results/ma/train_subset.json > logs/ma_train_subset.log 2>&1
log "phase 1a done (exit $?)"

log "phase 1b: training under joint_conf (3 seeds)"
python scripts/ma_train.py --score_rule joint_conf --seeds 0 1 2 \
    --train_episodes 6000 --eval_episodes 400 \
    --out results/ma/train_joint_conf_v2.json > logs/ma_train_joint_conf.log 2>&1
log "phase 1b done (exit $?)"

# --- Phase 2: the cross-rule matrix -------------------------------------------------
log "phase 2: cross-rule evaluation"
python scripts/ma_cross_rule.py --eval_episodes 400 \
    --rules subset joint_conf \
    --out results/ma/cross_rule.json > logs/ma_cross_rule.log 2>&1
log "phase 2 done (exit $?)"

# --- Phase 3: attack over-clamping --------------------------------------------------
# The 2026-08-17 policies clamped 84-96% regardless of whether clamping could help. A price
# on clamping is the minimal non-circular fix. Three costs, 3 seeds each. 0.00 duplicates
# phase 1b deliberately as an internal replication check.
log "phase 3: clamp-cost sweep"
for COST in 0.05 0.15; do
    log "  clamp_cost=$COST"
    python scripts/ma_train.py --score_rule joint_conf --seeds 0 1 2 \
        --train_episodes 6000 --eval_episodes 400 \
        --clamp_cost "$COST" --tag "jc_clamp$COST" \
        --out "results/ma/train_clamp$COST.json" \
        > "logs/ma_train_clamp$COST.log" 2>&1
    log "  clamp_cost=$COST done (exit $?)"
done

# --- Phase 4: more seeds at the best setting ----------------------------------------
# Deliberately LAST. Which setting deserves more seeds depends on phase 3, so this runs at
# the default and the extra seeds are added to the arm already known to work rather than to
# one chosen after seeing phase 3 -- picking the arm after the fact would be selecting on
# the outcome.
log "phase 4: extra seeds under joint_conf"
python scripts/ma_train.py --score_rule joint_conf --seeds 3 4 5 6 \
    --train_episodes 6000 --eval_episodes 400 \
    --tag joint_conf_extra \
    --out results/ma/train_joint_conf_seeds3to6.json \
    > logs/ma_train_extra_seeds.log 2>&1
log "phase 4 done (exit $?)"

log "ALL PHASES COMPLETE"
