#!/bin/bash -l
# Probe, take two: flat vs per-node at MATCHED data sizes.
#
# The first probe compared the two architectures at 600 episodes (flat 0.528, per-node
# 0.814) but ran the cluster version flat-only at 3000, where flat reached 0.766. So part
# of the original gap was data quantity, not architecture. This sweeps the data size for
# BOTH architectures so the comparison is like-for-like at every point, and the claim
# becomes one about sample efficiency rather than a single headline number.

#$ -N sa_probe2
#$ -cwd
#$ -l h_rt=06:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -t 1-8
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/probe2
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# tasks 1-4: d=4 at four data sizes; tasks 5-8: d=5 at the same four.
IDX=$(( (SGE_TASK_ID - 1) % 4 ))
D=$(( (SGE_TASK_ID - 1) / 4 + 4 ))
EPISODES=$(echo "300 1000 3000 9000" | cut -d' ' -f$((IDX + 1)))

echo "=== probe2 d=${D} episodes=${EPISODES} ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.probe_observation \
  --d "${D}" --episodes "${EPISODES}" --epochs 80 --arch both \
  --out ~/sa_runs/probe2/probe_d${D}_e${EPISODES}.json

echo "=== done $(date) ==="
