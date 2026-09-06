#!/bin/bash
# THE DECISIVE TEST: do the observation channels explain why partial-oracle training works?
#
# THE PUZZLE. At the principal cell with a rate-compensated budget (effective beta 1.52), a five
# per cent withholding rate destroys learning: rho=0.95 gives window rate 0.558 over three seeds
# and rho=0.90 gives 0.496, against 0.994 at rho=1.00 from a twelve-seed control at the same
# effective beta. The k=8 fleet tolerated rho all the way to 0.50. With budget now matched, the
# k=8 and k=12 fleets differ in three things: window size (12 vs 8), episodes (12,000 vs 8,000 --
# k=12 has MORE, so it cannot explain k=12 doing worse), and `observe_belief_channels` +
# `observe_reprobe_signal` (ON in the k=8 fleet, OFF here).
#
# WHY THE CHANNELS ARE PLAUSIBLE. Under a partial oracle a query can come back unanswered, and
# without these features the agent has no observation distinguishing "asked and got nothing"
# from "did not ask". The reprobe signal exists exactly to mark unresolved-but-already-probed
# pairs. Remove both and a withheld answer is invisible rather than merely unhelpful.
#
# WHY THE EXISTING ABLATION DOES NOT ANSWER IT. `p85_b70_k8_{on,off}_fixed` compares the flags at
# k=8, rho=0.85, budget 70 -- effective beta 2.57. All six runs pass the floor (0.745-0.883) and
# the flags make no difference, because at that pressure the task is easy enough that nothing
# separates. It was run in a regime that could not detect the effect it tested for.
#
# THE ARM THIS ADDS. Only three runs are needed: the channels-OFF arm at k=12, rho=0.95,
# budget 53 ALREADY EXISTS as `results/rho12b*/rho0.95_s{0,1,2}.json`. This is the matched ON
# arm -- same cell, same budget, same seeds, both flags on, which is the configuration the k=8
# fleet used.
#
# Both flags together on purpose. If ON passes, the mechanism is the observation features and a
# follow-up separates which; if ON also fails, both are ruled out at once and window size is
# what remains. Separating them first would spend six runs to answer a question that may not
# arise.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results/channels12 logs/channels12
WORKERS=${WORKERS:-3}

COMMON="--n_agents 4 --private_size 6 --n_shared 6 --budget 53 --n_obs 60 --n_int 20 \
--turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only \
--graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward \
--episode_mix confounded --normalise_returns --vs_evidence oracle --evidence_power 0.95 \
--train_episodes 12000 --eval_episodes 200 --no_wandb --force --turn_aware_credit \
--local_epochs 4 --observe_belief_channels --observe_reprobe_signal"

jobs=$(mktemp)
for s in 0 1 2; do
  out="results/channels12/on_rho0.95_s${s}.json"
  [ -f "$out" ] && continue
  echo "$s $out" >> "$jobs"
done
echo "$(date +%H:%M:%S)  channels test: $(wc -l < "$jobs") runs (ON arm; OFF arm already measured)"

run_one() {
  read -r s out <<< "$1"
  .venv/bin/python -u scripts/ma_train.py --arm k12chan --seed "$s" $COMMON --out "$out" \
    > "logs/channels12/on_s${s}.log" 2>&1
  echo "$(date +%H:%M:%S)  done seed $s"
}
export -f run_one
export COMMON
[ -s "$jobs" ] && cat "$jobs" | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$jobs"
echo "$(date +%H:%M:%S)  CHANNELS TEST COMPLETE"
