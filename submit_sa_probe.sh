#!/bin/bash -l
# Supervised decodability probe: can the oracle's choice be read out of what the agent sees?
#
# Answers the question the RL results cannot: whether the agent's failure is about the
# REPRESENTATION or about the ALGORITHM. One task per d, since d=6 is far slower than d=4.

#$ -N sa_probe
#$ -cwd
#$ -l h_rt=06:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -t 1-3
#$ -o logs/
#$ -e logs/

mkdir -p logs ~/.tmp ~/sa_runs/probe
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

D=$((SGE_TASK_ID + 3))          # tasks 1,2,3 -> d = 4,5,6
# Fewer episodes at d=6: one posterior update there costs ~0.7s.
if [ "${D}" -eq 6 ]; then EPISODES=300; else EPISODES=3000; fi

echo "=== probe d=${D}, ${EPISODES} episodes ==="
echo "host $(hostname)  started $(date)"

python -u -m scripts.probe_observation \
  --d "${D}" --episodes "${EPISODES}" --epochs 80 \
  --out ~/sa_runs/probe/probe_d${D}.json

echo "=== done $(date) ==="
