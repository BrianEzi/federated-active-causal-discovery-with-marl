#!/bin/bash -l
# k=20/k=30, ORACLE evidence, 12,000 episodes -- DECISIONS_AND_OUTSTANDING_2026_08_31.md
# section 5: seeds 1/2 of both cells are UNDER-TRAINED at 4,000 episodes, not collapsed
# (window rate still climbing at the last checkpoint; k30s50n04b150_s0's own window rate
# went 0.27 -> 0.91 -> 1.00 across its last fifty updates). Seed 0 of each is already fine
# and is not re-run here -- only what needs more training.
#
# Separate output directory (`results/sweep/oracle_long/`) from the 4,000-episode sweep, so
# the two are never confused and neither `[ -f "$OUT" ]` guard can accidentally skip the
# other's work.
#
# RESUMABLE, and this is why it matters more here than anywhere else on this branch: k=30 at
# 4,000 episodes already measured at ~12.3h; at 12,000 episodes a single run is plausibly
# ~35h+, which no Myriad slot below `h_rt=12:00:00` covers in one pass. Re-submitting this
# exact script is the intended recovery -- `scripts/resume_or_start.sh` picks up from the
# highest `_resume_uNNNN.pt` each time. Expect to resubmit this 2-3 times before the k=30
# seeds finish; that is normal, not a failure.
#
# 6 tasks: k30 seeds 0-2, k20 seeds 0-2. Commands pasted verbatim from
# `scripts/sweep.py --emit jobs --evidence oracle --episodes 12000 ... | grep -E "k20s50|k30s50"`
# then piped through the same resume_or_start.sh substitution as the sampled sweep.
#
#$ -N oracle_long
#$ -cwd
#$ -t 1-6
#$ -l h_rt=12:00:00
#$ -l mem=16G
#$ -pe smp 1
#$ -o logs/
#$ -e logs/

set -uo pipefail
mkdir -p logs results/sweep/oracle_long
source ~/envs/sa_env/bin/activate
cd ~/ma_tb
export PYTHONPATH=.
export TMPDIR=~/.tmp
mkdir -p ~/.tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CMD=$(sed -n "${SGE_TASK_ID}p" oracle_long_jobs_array.txt)
echo "=== task $SGE_TASK_ID : $(date) ==="
echo "cmd: $CMD"
eval "$CMD"
echo "=== done $(date) ==="
