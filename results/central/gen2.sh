#!/usr/bin/env bash
# (1) seeds 3-5 at k=12 to firm up the null; (2) A vs E at k=20, the cell where learning
# actually earns its advantage and where success stops saturating. A is retrained rather
# than reused so A and E are one build.
COMMON="--n_obs 60 --n_int 20 --turn_order round_robin --backend factored \
--policy_arch gnn_portable --vary_only --graph_model sf --sf_m 2 --claim_bar 1.0 \
--reward_criterion claims --per_agent_reward --episode_mix confounded --normalise_returns \
--vs_evidence oracle --eval_episodes 200 --no_wandb --force --turn_aware_credit"
OBS="--observe_belief_channels --observe_partner_counts"
i=0
emit () {  # arm cell flags seed
  i=$((i+1)); f=$(printf "results/central/jobs2/%02d.sh" $i)
  cat > "$f" <<EOF
#!/usr/bin/env bash
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $(pwd)
[ -f "results/central/v2_$2_$1_s$4.json" ] || .venv/bin/python scripts/ma_train.py --arm v2_$2_$1 --seed $4 $3 --out results/central/v2_$2_$1_s$4.json
EOF
  chmod +x "$f"
}
K12="--n_agents 4 --private_size 6 --n_shared 6 --budget 50 $COMMON --train_episodes 4000"
K20="--n_agents 4 --private_size 10 --n_shared 10 --budget 75 $COMMON --train_episodes 12000"
for s in 3 4 5; do
  emit A k12 "$K12 --local_epochs 4" $s
  emit E k12 "$OBS $K12 --local_epochs 0" $s
done
for s in 0 1 2; do
  emit A k20 "$K20 --local_epochs 4" $s
  emit E k20 "$OBS $K20 --local_epochs 0" $s
done
echo "wrote $i jobs"
