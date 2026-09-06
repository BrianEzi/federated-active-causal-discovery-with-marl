# Spec: the centralisation ceiling (RQ3 rung 0)

Agent B, 6 Sep 2026. **Not implemented. This is the agreement-before-coding step**, per the
standing rule from 21 Aug -- four bugs in one day were all silent design decisions taken
mid-implementation. Nothing here runs until Brian or agent A approves it.

---

## 1. What the rung is, and what it is not

**Brian's definition (5 Sep):** all data pooled AND a central controller directing the agents'
intervention commands. The agents still exist and still reach only their own nodes; what is
centralised is the *decision*, not the *visibility*.

**Agent C's original spec was different and is rejected on measurement.** It asked for K=1,
Z_1 = V -- one agent *seeing* everything. That arm cannot be run:

    federation 4x6+6   1821/2000 draws confounded   91.05%
    single     1x24+6      0/2000 draws confounded    0.00%

Latent confounding here is a bidirected pair in an agent's projected MAG, i.e. a common cause
outside its window. A controller whose window is the whole graph has no outside, so the regime
the ladder trains and scores on is definitionally empty for it, and `_sample_mixed_dag` raises
after 200 failed draws. Separately its edge mask would allow 870 directed edges against the
federation's 438 and it would be scored over 435 pairs against 219 -- a different graph family
on a different denominator. Under Brian's definition none of this arises, because the topology
is untouched.

## 2. The change

**Topology: unchanged.** `federated_topology(4, 6, 6)`. Windows, visibility, edge mask,
confounding rate and covered-pair set are all exactly the federation's. This is what makes the
rung comparable, and it is why no `Topology` change is needed -- the earlier blocker assumed
the visibility had to change.

**Information: already exists.** Arm E (`pooled`) is the pooled-information, pooled-optimiser
arm: `--local_epochs 0 --observe_belief_channels --observe_partner_counts`.

**The only new thing is joint action selection.** Today `IndependentPPO.policy(agent)` returns a
closure that reads `env.observation(agent)` and samples from `self.nets[agent]`
(`ma/policy.py:1160-1170`); each agent decides alone. The ceiling arm replaces the *set* of
closures with one controller that sees every agent's observation and emits every agent's action
in one decision.

Two implementations, and the choice should be made deliberately:

* **(A) Sequential joint policy, no retraining.** One controller, but the action for agent `i`
  is chosen from the concatenation of all agents' observations. Requires a new network whose
  input is the concatenated observation, so it must be trained -- 3 seeds at the chosen cell.
* **(B) Argmax-over-agents coordination wrapper, no new training.** Keep the trained arm-E
  networks; at each turn let the controller pick which agent acts and with which action, by
  scoring every agent's candidate actions under its own net and taking the global best. This
  measures "what does centralised *sequencing* buy on top of pooled information", not "what
  does a centralised policy buy". Cheap -- evaluation only.

**Recommendation: (B) first.** It is evaluation-only, answers a well-posed question, and if it
buys nothing then (A) is very unlikely to. (A) is a training job and should not be started in
submission week.

## 3. The cell

**Not the principal cell.** At k12s50n04b150 the metric is saturated: the six seeds I measured
give federated-minus-pooled of -0.000167 +/- 0.000281 (best) and +0.000164 +/- 0.000110 (final),
with four of six cells at exactly 0.000000 in each convention. A third rung there measures
nothing.

**Use beta 0.5-0.7** from agent A's constrained-budget axis (`results/budget_tight/`, budgets
17/20/24), where the arms separate: the learned advantage peaks at beta=0.6 at +0.472 joint
recovery over myopic, and oracle-cover is beaten below beta~0.8. That is the only region
measured so far where a ceiling could show a gap that is not noise.

## 4. What would be reported

The rung's value is a BOUND, so it should be stated as one: "pooling information, reward and
optimisation, and centralising the sequencing on top, buys X +/- SE over the federated arm at
beta=0.6". Paired per-episode over 200 episodes, both checkpoint conventions, competence floor
reported. If X is inside noise at a cell chosen *because* it separates, that is a stronger null
than the same null at a saturated cell -- and it is the honest use of this rung.

## 5. Open questions for whoever approves this

1. (B) then (A), or (B) only? My view: (B) only, this week.
2. Which beta -- 0.5, 0.6 or 0.7? 0.6 is where the learned advantage peaks; 0.5 is where
   oracle-cover is beaten hardest.
3. Does the controller in (B) get to override the turn order, or only the action within an
   agent's turn? These are different claims and I would rather be told than choose.
