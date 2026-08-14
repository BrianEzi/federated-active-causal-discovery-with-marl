#!/bin/bash -l

# SGE Directives for Single-Agent Array Job on Myriad HPC
#$ -N single_agent_exp
#$ -cwd
#$ -l h_rt=01:00:00
#$ -pe smp 2
#$ -t 1-9
#$ -o logs/
#$ -e logs/

mkdir -p logs
mkdir -p ~/.tmp
source /home/ucabbse/envs/marl_env/bin/activate
cd /home/ucabbse/marl_causal

export TMPDIR=~/.tmp
export WANDB_DISABLE_SERVICE=true
export WANDB_MODE=offline
export JAX_PLATFORMS=cpu

# Array of estimators: avici, bayes_optimal, learned
ESTIMATORS=("avici" "bayes_optimal" "learned")
SEEDS=(42 7 13)

# SGE_TASK_ID ranges 1-9
TASK_IDX=$((SGE_TASK_ID - 1))
EST_IDX=$((TASK_IDX / 3))
SEED_IDX=$((TASK_IDX % 3))

ESTIMATOR=${ESTIMATORS[$EST_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUTDIR="diag_runs/single_agent_${ESTIMATOR}_s${SEED}"
mkdir -p "$OUTDIR"

echo "=== Running Single-Agent Task $SGE_TASK_ID: Estimator=$ESTIMATOR, Seed=$SEED ==="

python -m src.train \
  --num_agents 1 \
  --num_variables 4 \
  --initial_budget 5.0 \
  --action_cost 1.0 \
  --batch_size 16 \
  --num_episodes 200 \
  --estimator_type "$ESTIMATOR" \
  --intervention_type hard \
  --reward_density dense \
  --use_rnn \
  --uncertainty_coef 2.0 \
  --seed "$SEED" \
  --save_file

# Copy output to experiment directory
cp training_metrics.csv "$OUTDIR/" 2>/dev/null || true
cp evaluation_trace.json "$OUTDIR/eval_trace_temp0.0.json" 2>/dev/null || true
cp checkpoints/best_ippo_params.pkl "$OUTDIR/" 2>/dev/null || true

# Run temperature sweep evaluation on the trained checkpoint
python -m scripts.temperature_sweep_eval \
  --ckpt_path "$OUTDIR/best_ippo_params.pkl" \
  --out_dir "$OUTDIR" \
  --num_agents 1 \
  --seed "$SEED" \
  --estimator_type "$ESTIMATOR" \
  --intervention_type hard

echo "=== Single-Agent Task $SGE_TASK_ID Complete! ==="
