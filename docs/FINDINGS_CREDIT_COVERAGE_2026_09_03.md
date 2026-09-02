# The $k_v=12$ credit ablation is not a failed replication, it is an incomplete one

3 Sep 2026, 02:xx.

## The claim in the chapter

`sec:res_credit` and `tab:credit` report turn-aware credit assignment at $k_v=8$: removing it
leaves the pooled arm unchanged ($0.00137$ against $0.00160$) and costs the federated arm a
factor of eighteen ($0.01917$ against $0.00106$). The chapter has said, in one form or another,
that the effect "does not replicate at $k_v=12$ on the seeds available".

## Why it cannot replicate as run

`results/credit/` holds, at $k_v=12$:

| cell | seeds present |
|---|---|
| pooled, credit on | 3 |
| pooled, credit off | 3 |
| federated, credit on | 3 |
| **federated, credit off** | **1** |

The cell with one seed is the one that carries the entire effect at $k_v=8$. Three cells at
three seeds and the decisive cell at one is not a replication that failed; it is a comparison
that was never in a position to succeed or fail.

## What is running

The two missing seeds, with every flag copied from the surviving run's own config rather than
retyped: `--n_agents 4 --private_size 6 --n_shared 6 --budget 50 --n_obs 60 --n_int 20
--turn_order round_robin --backend factored --policy_arch gnn_portable --vary_only
--graph_model sf --sf_m 2 --claim_bar 1.0 --reward_criterion claims --per_agent_reward
--episode_mix confounded --normalise_returns --vs_evidence oracle --train_episodes 4000
--local_epochs 4`, with `turn_aware_credit` off by omission.

Queued behind the generator control. Two runs at 4,000 episodes.

## What may be written when they land

If the federated credit-off cell degrades at $k_v=12$ as it does at $k_v=8$, the claim widens
from one window size to two and the mechanism -- a correctness fix that only bites under the
federated optimiser -- gains its first replication.

If it does not degrade, that is a real negative and the honest statement is that the effect is
specific to $k_v=8$, which is what the chapter currently implies without the evidence to imply
it.

Either way the sentence in `sec:res_credit` has to change, because "does not replicate on the
seeds available" describes a sample size rather than a result.

## Budget caveat, stated in advance

These runs are at 4,000 episodes, matching the existing cells so the comparison is
within-budget. Given that 14 of 18 sweep cells changed winner between 4,000 and 12,000
episodes, a null at 4,000 here would not establish a null at convergence. The claim available
from this ablation is about the fixed budget it is run at, and the chapter should say so.
