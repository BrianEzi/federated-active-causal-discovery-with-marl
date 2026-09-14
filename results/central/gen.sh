#!/usr/bin/env bash
# Federation-cost ladder v2, k=12 n=4 sigma=0.5 b150, 3 seeds.
#
# CHANGES FROM v1, all found by running it:
#  * arm C (global conjunction reward) DROPPED. `per_agent_reward=False` gives the all-agents
#    conjunction, which confounds "global credit" with "sparse reward" -- it trained to
#    success 0.03-0.05 because the conjunction gives no gradient at 4 agents, not because
#    centralising credit is bad. No dense-global reward exists in the codebase to separate them.
#  * arm D respecified. `--local_epochs 1` is NOT the centralised optimiser -- ma/policy.py:856
#    says so explicitly ("E=1 IS NOT EQUIVALENT TO THE POOLED PATH"). `--local_epochs 0`
#    selects the data-pooling path, which is the centralised arm. Folded into E.
#  * `--observe_owner_channel` DROPPED. All three observation flags together crash
#    ma/policy.py:499 (partner table view). It is the per-pair OWNERSHIP belief, which is
#    attribution-specific and irrelevant here.
#  * arm A retrained here rather than reused, so A/B/E are one build. The two existing runs of
#    this exact config (sweep 0.977 mean, fedavg_proper/E4_credit 0.870 mean) disagree, which
#    is its own open question -- not one to resolve by picking the flattering baseline.
BASE="--n_agents 4 --private_size 6 --n_shared 6 --budget 50 --n_obs 60 --n_int 20 \
--turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only \
--graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
--episode_mix confounded --normalise_returns --vs_evidence oracle \
--train_episodes 4000 --eval_episodes 200 --no_wandb --force --turn_aware_credit"
OBS="--observe_belief_channels --observe_partner_counts"
i=0
for a in A B E; do
  case $a in
    A) FLAGS="--local_epochs 4" ;;        # federated baseline, blind to partners
    B) FLAGS="$OBS --local_epochs 4" ;;   # F1 information partition removed
    E) FLAGS="$OBS --local_epochs 0" ;;   # F1 + F3 -- centralised controller
  esac
  for s in 0 1 2; do
    i=$((i+1)); f=$(printf "results/central/jobs/%02d.sh" $i)
    cat > "$f" <<EOF
#!/usr/bin/env bash
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $(pwd)
[ -f "results/central/v2_k12_${a}_s${s}.json" ] || .venv/bin/python scripts/ma_train.py --arm v2_k12_${a} --seed ${s} $BASE $FLAGS --out results/central/v2_k12_${a}_s${s}.json
EOF
    chmod +x "$f"
  done
done
echo "wrote $i jobs"
