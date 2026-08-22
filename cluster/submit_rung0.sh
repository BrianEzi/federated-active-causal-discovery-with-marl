#!/bin/bash -l
# RUNG 0 of the n-agent refactor spec: two agents on refactored ma/topology.py must
# reproduce the pre-refactor numbers, or the refactor is wrong and nothing downstream is
# meaningful (docs/N_AGENT_REFACTOR_SPEC.md section 5).
#
# PINNED to --prior_p 0.5, deliberately. The pre-refactor baselines (results/ma_fixed/
# tb_both_s0-9.json, tb_clamp_s0-9.json, commit 504e767) were measured before the
# 2 ln(d)/d prior existed at all, i.e. at the old fixed default. Comparing against the NEW
# default prior would confound two independent changes -- the topology refactor and the
# prior change -- into one number, and a mismatch would not say which one caused it. Pin
# the prior here; re-baseline at the new default is a separate, later, labelled step.
#
# SAME SEEDS (0-9) as the baseline, deliberately. sample_dag draws from the same RNG stream
# regardless of how allowed_edges is computed internally, and the two edge masks were
# proven bit-identical at two agents (tests/test_ma_topology.py::
# test_the_new_rule_reproduces_the_old_one_exactly_at_two_agents). So same seed should mean
# same graphs, and the comparison isolates training-loop noise from mask correctness.
#
#   arms 1-10   rung0_tb_both    seeds 0-9, both action modes  -- compare vs tb_both_s0-9
#   arms 11-20  rung0_tb_clamp   seeds 0-9, clamp-only         -- compare vs tb_clamp_s0-9
#
#$ -N ma_rung0
#$ -cwd
#$ -t 1-20
#$ -l h_rt=04:00:00
#$ -l mem=4G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -e
mkdir -p logs results/rung0
source ~/envs/sa_env/bin/activate
cd ~/ma_tb
export PYTHONPATH=.
export TMPDIR=~/.tmp
mkdir -p ~/.tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TASK=$((SGE_TASK_ID - 1))
ARM_INDEX=$((TASK / 10))
SEED=$((TASK % 10))

COMMON="--n_obs 1000 --n_int 100 --budget 10 --train_episodes 2000 --eval_episodes 150 \
        --turn_order round_robin --disclose_regime --prior_p 0.5"

case $ARM_INDEX in
  0) ARM=rung0_tb_both;  EXTRA="" ;;
  1) ARM=rung0_tb_clamp; EXTRA="--clamp_only" ;;
esac

OUT="results/rung0/${ARM}_s${SEED}.json"
if [ -f "$OUT" ]; then
  echo "$OUT already exists -- skipping"
  exit 0
fi

echo "=== $ARM seed $SEED : $(date) ==="
echo "cmd: python -m scripts.ma_train --seed $SEED --arm $ARM $COMMON $EXTRA --out $OUT"
python -m scripts.ma_train --seed "$SEED" --arm "$ARM" $COMMON $EXTRA --out "$OUT"
echo "=== done $(date) ==="
