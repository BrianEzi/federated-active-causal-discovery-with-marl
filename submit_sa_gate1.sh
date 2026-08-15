#!/bin/bash -l
#$ -N sa_gate1
#$ -cwd
#$ -l h_rt=04:00:00
#$ -l mem=12G
#$ -pe smp 2
#$ -t 1-3
#$ -o logs/
#$ -e logs/
mkdir -p logs ~/.tmp ~/sa_runs/gate1
source ~/envs/sa_env/bin/activate
cd /home/ucabbse/marl_sa
export TMPDIR=~/.tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=$((SGE_TASK_ID + 3))
python -u -m scripts.check_gate1 --d "${D}" --n_obs 1000 5000 20000 \
  --episodes 200 --out ~/sa_runs/gate1/gate1_d${D}.json
