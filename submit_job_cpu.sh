#!/bin/bash -l

# SGE Directives
#$ -N marl_causal_cpu
#$ -cwd
#$ -l h_rt=02:00:00
#$ -pe smp 4
#$ -o logs/
#$ -e logs/

mkdir -p logs
source /home/ucabbse/envs/marl_env/bin/activate
cd /home/ucabbse/marl_causal

# WandB Telemetry Mode
export WANDB_MODE=offline

# Execute MARL Causal Discovery base training pipeline on High-Speed CPU
python -m src.train \
  --num_agents 2 \
  --num_variables 4 \
  --batch_size 32 \
  --num_episodes 1000 \
  --estimator_type analytic \
  --intervention_type soft_shift \
  --reward_density dense \
  --obs_feedback true \
  --save_file \
  --use_wandb \
  --wandb_project "federated-causal-marl"
