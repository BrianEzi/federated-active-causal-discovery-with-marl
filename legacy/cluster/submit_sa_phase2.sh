#!/bin/bash -l
# Phase 2, E1 + E2: the lever sweep repeated on a working network, and again on the broken
# one for comparison.
#
# 66 tasks = 33 configurations x 2 architectures, 3 seeds each (198 runs). Tasks 1-33 are
# per-node (E1), 34-66 are flat (E2), with only `arch` differing between the halves.
#
# n_obs=5000 throughout, because GATE 1 does not pass at d=5 below that. One arm per
# architecture deliberately keeps n_obs=1000 as a negative control and is tagged
# NEGCONTROL so it can never be misread as a normal result.
#
# Walltime: the baseline costs ~42 ms per environment step after the 2026-08-15 hot-path
# work, so ~1.5 h for three seeds. The train_episodes=12000 arm is the longest at roughly
# double that. 8 h leaves margin without over-requesting.

#$ -N sa_phase2
#$ -cwd
#$ -l h_rt=08:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -t 1-66
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp results/phase2
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Offline: compute nodes have no outbound internet, so an online init would hang rather
# than fail. scripts/sync_wandb.py uploads afterwards from the login node.
export WANDB_MODE=offline
export WANDB_SILENT=true

ARGS=$(python -m scripts.sweep_phase2 --cli "${SGE_TASK_ID}")
echo "=== task ${SGE_TASK_ID} ==="
echo "${ARGS}"
echo "host $(hostname)  started $(date)"

python -u -m scripts.run_experiment ${ARGS}

echo "=== done $(date) ==="
