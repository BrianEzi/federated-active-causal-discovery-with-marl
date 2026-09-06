#!/bin/bash -l
# ARRAY SUBMIT for long single training runs -- the shape of the post-submission programme:
# the joint adjacency+orientation version space, no-skeleton at 36,000 episodes, k_v past 30,
# and the hypothesis-test substitutions. All of those are many independent long jobs, which is
# exactly what an SGE array is for and exactly what a six-worker laptop is not.
#
# PROVEN END TO END ON 6 SEP 2026, and the proof is the point of this file existing. Before
# today `~/ma_tb` sat at 3d72be7 (31 Aug) -- **410 commits behind** -- so any job submitted here
# would have silently run the engine as it was before the evaluation-RNG fix, the deterministic
# grids and a week of factored-belief work. It is now at parity with `origin`. Re-check that
# before trusting this script again:
#
#     ssh myriad 'cd ~/ma_tb && git fetch -q origin && git log --oneline -1 && \
#                 git rev-list --count HEAD..origin/explore/constraint-based'
#
# A non-zero count means the cluster is behind and the results will not match the laptop's.
#
# MEASURED TIMING, so h_rt is not a guess. A k12 principal-cell training run did 320 episodes
# in 3m55s on a LOGIN node: 0.73 s/episode, so 12,000 episodes is about 2.5 h and 36,000 about
# 7.4 h. Login nodes are shared and throttled, so a compute node should beat that -- but it has
# not been measured, so h_rt below is set from the login-node figure with headroom rather than
# from an optimistic guess. The first real array task should be timed and this comment updated.
#
# WHY A JOB LIST RATHER THAN PASTED COMMANDS. `cluster/submit_oracle_medium.sh` pastes its
# eighteen commands verbatim from `sweep.py --emit jobs`, and the note there is right that a
# sweep command must never be hand-retyped. This script keeps that rule and generalises it: the
# list is generated on the laptop, committed, and read here by line number, so the command that
# runs on the cluster is byte-identical to the one that was reviewed.
#
#   1. generate:  <your generator> > cluster/jobs/<name>.txt      (one full command per line)
#   2. commit and push, then pull on myriad
#   3. edit -t below to match the line count, then:  qsub cluster/submit_long_training.sh
#
#$ -N ma_long
#$ -cwd
#$ -t 1-1
#$ -l h_rt=12:00:00
#$ -l mem=16G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -e
JOBLIST=${JOBLIST:-cluster/jobs/long_training.txt}

cd ~/ma_tb
mkdir -p logs
source ~/envs/sa_env/bin/activate

# Single-threaded per task, because the array gives each task its own slot: letting BLAS spawn
# threads inside a 1-slot allocation oversubscribes the node and slows every task on it.
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CMD=$(sed -n "${SGE_TASK_ID}p" "$JOBLIST")
[ -n "$CMD" ] || { echo "no command on line ${SGE_TASK_ID} of ${JOBLIST}"; exit 1; }

echo "task ${SGE_TASK_ID} on $(hostname) at $(date -Is)"
echo "repo $(git log --oneline -1)"
echo "cmd  ${CMD}"
# Time every task, so the h_rt above stops being an estimate after the first array.
time eval "${CMD}"
echo "task ${SGE_TASK_ID} finished at $(date -Is)"
