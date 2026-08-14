#!/bin/bash -l
# Re-score the three Track B (uncertainty-bonus) checkpoints with the corrected oracle
# metric and the corrected Bayes estimator (commit b218f8b onward).
#
# This is an EVAL-ONLY pass -- no retraining. Track B trained under --estimator_type avici,
# and the equal-variance shortcut was only reachable via --estimator_type bayes_optimal
# (src/evaluator_env.py:441), so those policies faced the intended task and their
# ended_at_zero figures stand. What was wrong is the oracle-agreement metric, which
# evaluate.py computes through the Bayes posterior regardless of the run's own estimator.
# Re-running the sweep against the already-saved checkpoints is therefore sufficient.
#
# Corrected traces land in diag_runs/uncertainty_bonus_s*/rescored/ , kept separate from
# each run's own (retracted) traces so both remain available for comparison.

#$ -N trackb_rescore
#$ -cwd
#$ -l h_rt=01:00:00
#$ -pe smp 4
#$ -l mem=8G
#$ -t 1-3
#$ -o logs/
#$ -e logs/

mkdir -p logs
mkdir -p ~/.tmp
source /home/ucabbse/envs/marl_env/bin/activate
cd /home/ucabbse/marl_trackb

export TMPDIR=~/.tmp
export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export JAX_PLATFORMS=cpu

SEEDS=(42 7 13)
SEED=${SEEDS[$((SGE_TASK_ID - 1))]}
RUN_DIR=/home/ucabbse/marl_causal/diag_runs/uncertainty_bonus_s${SEED}
OUT_DIR=${RUN_DIR}/rescored_v2

mkdir -p "${OUT_DIR}"

echo "=== Track B re-score: seed ${SEED} ==="
echo "checkpoint: ${RUN_DIR}/checkpoints/best_ippo_params.pkl"
echo "output:     ${OUT_DIR}"

# --seed MUST match the original sweep's (submit_uncertainty_bonus.sh used --seed $SEED,
# not a fixed 42). evaluate.py keys the environment off PRNGKey(seed + graph_idx), so a
# different seed evaluates a DIFFERENT set of SCM parameter draws -- the comparison against
# the original traces would be against different problem instances entirely. Getting this
# wrong on the first attempt produced an apparent ended_at_zero jump that was pure
# instance mismatch.
python -m scripts.temperature_sweep_eval \
  --checkpoint_path "${RUN_DIR}/checkpoints/best_ippo_params.pkl" \
  --output_dir "${OUT_DIR}" \
  --temperatures 0.0,0.2,0.5,1.0 \
  --seed ${SEED}

echo "=== done seed ${SEED} ==="
