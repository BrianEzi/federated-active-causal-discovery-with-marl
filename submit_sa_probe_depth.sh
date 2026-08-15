#!/bin/bash -l
# Phase 1 depth probe. Does multi-hop aggregation lift the per-node scorer's 0.89 ceiling?
#
# The scorer does one round of neighbour aggregation, so a node's logit sees only its own
# edges -- while the oracle's score depends on that node's DESCENDANTS, which is
# inherently multi-hop. Supervised, this costs minutes; inferring it from RL runs costs a
# night and confounds it with everything else.
#
# 24 tasks: d in {4,5} x episodes in {300,1000,3000,9000} x 3 seeds. Each task collects
# once and trains depth 1, 2 and 3 on that same data, so the comparison is matched by
# construction. Three seeds because the decision threshold is 0.03 and a single-seed pilot
# already showed 0.014 of spread between depths on identical data.

#$ -N sa_probe_depth
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -t 1-24
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/probe_depth
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ARGS=$(python -m scripts.probe_depth --cli "${SGE_TASK_ID}")
echo "=== task ${SGE_TASK_ID}: ${ARGS} ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.probe_observation ${ARGS}

echo "=== done $(date) ==="
