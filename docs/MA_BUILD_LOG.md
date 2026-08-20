# Two-agent build log — 2026-08-16 overnight

Open-ended task: get two agents learning, tracking every decision. This file is the audit
trail. Decisions I made on the user's behalf are marked **[MY CALL]** with the reasoning
and, where one exists, the cheaper alternative I rejected.

Design of record is `docs/MA_DESIGN.md`. Where this file contradicts it, this file is
newer and says so explicitly.

---

## Decisions taken before writing any code

### [MY CALL] Starting topology is `(1,1,3)`, not `(1,1,2)`

The user left this open and asked me to get as far as I could, so I took it.

Evidence, all measured today: at `(1,1,2)` only **13 of 207** graphs give any agent a
bidirected edge (6.3%), and it is always the same pair — the confounding mechanism the
whole two-agent case exists to study is nearly absent. `(1,1,3)` gives 13.4%, three shared
pairs, and is still exactly enumerable (11,649 global graphs; 543 DAGs per agent window).
No sampling anywhere, so nothing about the inference machinery is approximate.

Rejected: starting at `(1,1,2)` for speed. A null result there would be uninterpretable —
"coordination didn't help" and "the phenomenon was too rare to learn from" would look
identical.

### [MY CALL] Each agent's hypothesis space is DAGs over its own window

Justified by today's confinement result: no bidirected edge can touch a private node, so
an agent's belief needs no MAG machinery over `Z_A` — the only place confounding can
appear is inside `X`. An agent scores DAGs over `O_A` with BGe on its own columns.

This is deliberately **misspecified under confounding**, and that is the point rather than
a defect: where a `z_B` confounds two shared nodes, no DAG over `O_A` is correct, the
posterior cannot concentrate on the truth, and only B intervening can fix it. The gap
between what A reaches alone and what A reaches with B is the quantity of interest.

### [MY CALL] Interventions by the other agent on its PRIVATE nodes are not disclosed

The privacy-faithful choice, and it has a cost I am recording rather than hiding.

Under a single shared system, when B does `do(z_B)` the rows still arrive in A's columns.
A is not told an intervention happened, so A scores those rows as observational. That is a
genuine misspecification: `do(z_B)` changes the marginal law of any shared node `z_B`
points into, and A's model of that node is marginal over `z_B`.

Alternatives rejected: (a) disclose the target — breaks the privacy constraint outright;
(b) disclose "a private intervention occurred" as one bit — leaks existence of private
structure and still leaves A unable to score the rows correctly, so it pays a privacy cost
for no inferential gain.

Interventions on **shared** nodes ARE disclosed to both, since `X` is visible to both and
disclosure reveals nothing private.

### [MY CALL] Separate budgets, simultaneous actions, no collision rule

Straight from `MA_DESIGN.md` §7 — recording it here only because it is load-bearing for
the environment's step signature: `step` takes a pair of actions, not one.

### Acyclicity exchange: kept as a correctness guard, NOT used for inference

Measured today at ~0.005 bits per disclosed bit. It stays in the protocol because the
per-episode maximum cyclic mass reaches 0.24, so it occasionally matters, but it is not
part of the belief update and is not credited with any inferential value.

---

## Build order

1. `ma/env.py` — two-agent environment, per-agent beliefs, single system.
2. Gates — the two-agent analogues of GATE 1 and GATE 2, run before any RL.
3. `ma/baselines.py` — random and greedy-EIG, per agent.
4. `ma/policy.py` — independent PPO per agent. No CTDE, per the supervisor constraint.
5. Training run at `(1,1,3)`.

Nothing proceeds past a failing gate. If a gate fails I stop and report rather than tune
until it passes.

---

## What actually happened

### GATE 1 and GATE 3 passed cleanly

    unconfounded + singleton MEC    76.8% identify from observation alone
    unconfounded + tied MEC          0.0%  (exactly, as theory requires)
    confounded                       0.0%  mean posterior mass on truth 7.5e-08
    unconfounded                    13.3%  mean posterior mass on truth 0.328

Tied graphs cannot reach the 0.7 threshold observationally -- class-mates tie exactly,
capping mass at 1/|class| <= 0.5 -- so 0.0% is the predicted number, not a happy accident.
Confounding is not a mild degradation: it is total.

### GATE 4 failed. Twice. And the failures were the most useful part of the night.

**Failure 1.** B intervening on its own private node rescued A in 0 of 29 confounded
episodes, mean posterior mass 0.0000, identical to B doing nothing. MA_DESIGN section 4
predicted the opposite.

**[FINDING] A randomised `do()` does not cut confounding.** `sa/scm.py` deliberately assigns
a *random* value per sample rather than a constant, because a constant is collinear with
the intercept and destroys the ability to see descendants move -- correct, and load-bearing,
for the single-agent case. But a randomised confounder is still a variance source that A
cannot see. `do(z_B ~ N(0,2))` replaces one latent common cause with another.

Measured, dataset drawn entirely under the intervention:

    scale 2.0   identified 0.000   mean mass 0.0000
    scale 1.0   identified 0.000   mean mass 0.0000
    scale 0.1   identified 0.164   mean mass 0.3701
    scale 0.0   identified 0.178   mean mass 0.3879

So the two purposes of intervening need OPPOSITE value distributions: varying reveals
orientation, clamping cuts confounding. **[MY CALL]** I added both as explicit modes --
`VARY` and `CLAMP` -- doubling each agent's action space. This is a real change to the
design and the user should confirm it, but proceeding without it makes coordination
impossible rather than merely hard.

Note what this does to the greedy baseline: expected information gain is computed over the
agent's own hypothesis space, where the other agent's confounding cannot be represented, so
CLAMP has *no value the oracle can see*. Greedy will never clamp to help a partner. That is
precisely the room a learned policy has.

**Failure 2.** With CLAMP available, GATE 4 still failed -- 0 of 38.

**[FINDING] Clamping is necessary but not sufficient; pooling destroys it.** In the
environment A holds 2000 confounded observational rows plus at most 1600 clean clamped
rows in one undifferentiated dataset. No single DAG fits a mixture of two regimes, and
under my no-disclosure decision A is never told the regime changed, so it cannot separate
them. Isolated, the mechanism works; pooled, it vanishes.

Confirmed by separating the arms explicitly, on confounded episodes:

    pooled       identified 0.000   mean mass 0.0002
    regime only  identified 0.162   mean mass 0.3604
    regime + A's own interventions inside the clean regime
                 identified 1.000   mean mass 0.9374

**[FINDING, and the headline] The minimum viable disclosure for coordination is a REGIME
BIT, not the ancestral order.** MA_DESIGN section 5 derived `|X|^2` bits as the disclosure
the design needs; measured earlier tonight, those bits are worth ~0.005 bits each and are a
correctness guard, not an enabler. The disclosure that actually unlocks coordination is one
bit per round -- *"I have clamped something you cannot see"* -- naming no variable, no
count, and revealing no structure beyond the fact that a clamp occurred. It takes A from
0% to 100%.

**[MY CALL] I reversed my own no-disclosure decision** and added `disclose_regime` to the
environment, defaulting on. The decision recorded at the top of this file -- that the
one-bit option "leaks existence of private structure and still leaves A unable to score
the rows correctly" -- was wrong in its second half. A cannot score the rows correctly, but
it does not need to: it can *condition* on the clean regime, which is valid inference.

Implementation is deliberately the simple one: where clean rows exist the agent uses only
those. A joint two-regime score would be strictly better and is the obvious next
improvement; conditioning on a subset is correct but wasteful.

### GATE 4 passes once the regime bit exists

    A's identification rate, CONFOUNDED episodes (n=38)
        solo          0.000  [0.000, 0.092]   mass 0.0000
        partner       0.000  [0.000, 0.092]   mass 0.0000
        oracle-clamp  0.974  [0.865, 0.995]   mass 0.8636
        oracle-vary   0.000  [0.000, 0.092]   mass 0.0000

    A's identification rate, UNCONFOUNDED episodes (n=362, control)
        solo          0.845     partner  0.843
        oracle-clamp  0.994     oracle-vary  0.823

`oracle-vary` sitting at exactly 0.000 alongside `oracle-clamp` at 0.974 is the cleanest
single demonstration of the mode finding: identical policy, identical target, identical
budget, differing only in whether the intervened value varies.

Worth stating rather than glossing: clamping also helps on UNCONFOUNDED episodes, 0.994
against 0.845. Removing a variance source the agent cannot model is useful generally, not
only when it creates a bidirected edge. So the clamp is not a pure coordination signal, and
any claim that a learned policy "clamps to help its partner" must show clamping is
*selective* to confounded episodes rather than merely present.

That is why the training script pre-registers the selectivity test (P2) as the one that
matters, and treats a uniform clamp rate as the weaker result it is.

---

## Training: independent PPO did NOT learn to coordinate

3 seeds, 6000 episodes each, evaluated on 500 held-out episodes against shared references.

              overall   confounded   unconfounded   clamp rate
    random     0.392       0.027         0.455         0.458
    greedy     0.568       0.000         0.667         0.000
    learned s0 0.366       0.000         0.430         0.012
    learned s1 0.398       0.000         0.467         0.263
    learned s2 0.364       0.000         0.427         0.000

**All three pre-registered predictions failed.**

  P1 (learned beats greedy) -- FAILED. Learned is worse everywhere, 0.36-0.40 against 0.568.
  P2 (clamps selectively on confounded episodes) -- FAILED. Seed 1, the only seed that kept
     clamping at all, clamps slightly LESS on confounded episodes (0.232) than unconfounded
     (0.269). Seeds 0 and 2 drove clamping to ~0.
  P3 (level with greedy on unconfounded) -- FAILED. Learned is well below greedy there too.

Confounded solve rate is **0.000 across all three seeds**. The coordination behaviour that
GATE 4 proved is available -- 0.974 when B clamps deliberately -- was not found by learning.

### Why: the payoff is non-monotone, and the gradient near zero points the wrong way

Measured directly, sweeping the probability that B clamps its private node each round:

    disclose_regime   p(clamp)   A solves | confounded   A solves | unconfounded
        False           any          0.000                    ~0.82
        True            0.00         0.000                     0.815
        True            0.25         0.393                     0.721
        True            0.50         0.750                     0.919
        True            1.00         1.000                     0.991

Read the unconfounded column. Going from never clamping to clamping a quarter of the time
**costs** 0.815 -> 0.721. Only past roughly half does it recover and then exceed the
baseline. So a learner starting from near-zero clamping sees a negative gradient, and the
large payoff sits on the far side of a valley it has no reason to cross.

**[FINDING] The cause is my own regime rule, not PPO.** I implemented "where clean rows
exist, use only those" and documented it as *correct but wasteful*. It is worse than
wasteful: when B clamps occasionally, A throws away 2000 good observational rows in favour
of a few hundred clean ones, and on the 85% of episodes where A is not confounded that is a
straight loss. The rule creates the valley.

The fix is the improvement I already flagged and deferred: score BOTH regimes jointly for a
shared DAG rather than conditioning on one. Then clean rows are added information instead of
a replacement, occasional clamping is weakly positive, and the valley should disappear.

This is a real result either way -- "the coordinated equilibrium exists but independent
learners with a terminal reward cannot reach it under a subset-conditioning belief" -- but
it should be re-run with the joint score before anyone concludes anything about PPO.

### Caveat on the p=1.0 row

"B always clamps its private node" maximises A's identification (1.000) and is also best for
A on unconfounded episodes. But B spends its entire budget doing it, so B's own
identification is not measured in that sweep. The genuine joint optimum has to trade the two,
and I have not measured B's side of it. Do not read 1.000 as a solved game.

---

# 2026-08-17 morning — the joint two-regime score

One-hour box, unsupervised. Goal: build the scoring rule flagged last night as the fix for
the valley, and measure whether it works.

## Four rules, compared on identical episodes

Built `ma/score_regimes.py` with all four so the choice is settled by measurement:

  POOLED      one dataset, one score, regime bit ignored (pre-disclosure behaviour)
  SUBSET      clean rows only where they exist (what the env does now; makes the valley)
  JOINT       same structure both regimes, independent parameters, log-scores added
  JOINT_CONF  DAG + a subset S of shared PAIRS marked confounded; S applied to the dirty
              regime only, as an added edge oriented along the DAG's own topological order

JOINT_CONF is only tractable because of the confinement result — every bidirected edge has
both endpoints in `X`, so S ranges over 3 pairs at `(1,1,3)`, giving 543 x 8 = 4344 exact
hypotheses. Marginalising S out returns a posterior over DAGs.

**[MY CALL] JOINT_CONF does not reduce to POOLED when no clean rows exist.** The other three
do. Without clean data an agent genuinely cannot separate a real shared edge from a
confounding artefact, and the rule exists to represent that rather than assume it away. The
consequence is intended and visible: JOINT_CONF starts LOWER at p(clamp)=0 because it
spreads mass over hypotheses the other rules silently exclude. What it buys is that clamping
then resolves the ambiguity, which is a much steeper gradient towards the coordinated
behaviour. Recording it because it is a real modelling choice, not a detail — it trades
baseline accuracy for a learning signal, and the user may prefer the other trade.

## Two bugs found and fixed while building

1. **Global log-sum-exp underflowed whole rows to `-inf`.** Marginalising S used one global
   shift, so every entry of a weaker DAG's row underflowed to zero and `log(0)` deleted the
   hypothesis outright instead of ranking it. Now shifted per row. This would have silently
   removed exactly the hypotheses that are hardest to distinguish.

2. **The `JOINT_CONF` branch ignored the empty-regime fallback.** `groups` was computed and
   then not used on that path, so the fallback was dead code and the p(clamp)=0 behaviour
   was accidental rather than chosen. Restructured so the branch is explicit, and the
   choice above is now deliberate.

## Early signal (n=20, too small to conclude from — full run at n=300 in flight)

    rule          unconfounded curve over p(clamp)      confounded payoff
    pooled        0.667 0.722 0.722 0.778   no valley        +0.000
    subset        0.667 0.556 0.667 1.000   VALLEY           +1.000
    joint         0.667 0.722 0.833 0.833   no valley        +0.000
    joint_conf    0.167 0.667 0.944 1.000   no valley        +0.500

Matches the pre-registered predictions so far: JOINT removes the valley but gains nothing on
confounded episodes, because the dirty regime still prefers a structure that mimics the
confounding and carries most of the rows. Only JOINT_CONF has both properties. Treat these
numbers as a smoke test, not a result — only 2 confounded episodes.

## Full comparison, 300 episodes (29 confounded, 271 unconfounded)

    rule          unconfounded curve over p(clamp)   valley?     confounded payoff
    pooled        0.815 0.838 0.804 0.808            no          +0.000
    subset        0.815 0.454 0.708 0.956            YES -0.362  +0.931
    joint         0.815 0.852 0.841 0.856            no          +0.000
    joint_conf    0.244 0.686 0.908 0.982            no          +0.690

Both pre-registered predictions held.

**JOINT alone is not enough, and this is the informative negative.** It removes the valley
exactly as expected -- clean rows are added rather than substituted -- but the confounded
payoff is *identically zero* at every clamp probability. The reason is the one predicted:
the dirty regime still prefers a structure that mimics the confounding, and it carries most
of the rows, so sharing a structure across regimes just lets the dirty regime win. Fixing
the gradient does not fix the target.

**JOINT_CONF has both properties.** Monotone on unconfounded episodes (0.244 -> 0.982) and a
+0.690 payoff on confounded ones. Modelling confounding explicitly, as a flag per shared
pair, is what makes the clean regime able to *disambiguate* rather than merely *outvote*.

That the confinement result is what makes this tractable is worth stating plainly: without
it the hypothesis space would be MAGs and the score would not decompose. With it, S ranges
over 3 pairs and the whole space is 4344 exact hypotheses.

### Two caveats, both against my own conclusion

1. **JOINT_CONF costs a lot when nobody clamps**: 0.244 against 0.815 for the other rules.
   That is the honest price of admitting confounding might be present, and it is a real
   cost, not a presentational one. If a learner never discovers clamping, JOINT_CONF leaves
   it strictly worse off than the rule it replaces. The gradient is favourable everywhere,
   which is why I still prefer it, but this is a genuine trade and the user may disagree.

2. **The valley detector flagged POOLED with "worst step -0.033"**, which is noise: n=271
   gives a standard error near 0.024, so a 0.033 step is well inside sampling variation. My
   +/-0.02 tolerance is too tight for this sample size. POOLED's curve is flat, not valleyed.
   Recording it because an automated pass/fail that fires on noise is exactly the kind of
   thing that later gets quoted as a finding.

### [MY CALL] JOINT_CONF is now the environment default

`MAConfig.score_rule`, defaulting to `joint_conf`. Belief update cost went 0.02s -> 0.05s
per step, which is acceptable. Retraining with it is running now.

## Retraining under JOINT_CONF — coordination is learned, stability is not

3 seeds, 6000 episodes, 400 held-out eval episodes (61 confounded, 339 unconfounded).
References scored under the SAME rule, so this is internally like-for-like.

                 overall   confounded            unconfounded   clamp (conf/unconf)
    random        0.268    0.049 [0.017,0.135]      0.307        0.477
    greedy        0.190    0.000 [0.000,0.059]      0.224        0.000
    learned s0    0.380    0.213 [0.129,0.331]      0.410        0.891 / 0.834
    learned s1    0.560    0.344 [0.237,0.470]      0.599        0.870 / 0.835
    learned s2    0.165    0.115 [0.057,0.218]      0.174        0.952 / 0.958
    median        0.380    0.213

**[RESULT] The coordination result is robust across seeds.** Every seed solves confounded
episodes at a rate whose 95% interval excludes greedy's 0.000, and two of three exclude
random's 0.049. Under SUBSET last night every seed scored exactly 0.000 there and clamping
collapsed towards zero. The clamp behaviour is now the dominant action, 84-96%.

This is the first evidence that the coordinated behaviour is reachable by learning and not
only by a hand-built oracle arm.

### Three things that must be said against it

1. **Overall performance is unstable.** 0.165 / 0.380 / 0.560 across seeds. Seed 2 lands
   BELOW random (0.268). A median of 0.380 against greedy's 0.190 is a real gain, but with
   a spread that wide, three seeds is not enough to quote a number.

2. **Selectivity (P2) essentially failed.** Clamp differences confounded-minus-unconfounded
   are +0.057, +0.036, **-0.006**. The agents learned to clamp; they did not learn *when*.
   Seed 2 clamps 95.7% of the time on episodes where clamping is pointless, spends its
   budget on it, and takes 5.72 steps to seeds 0 and 1's ~5.0. Over-clamping is the failure
   mode, and it is the direct explanation for seed 2 being the worst arm.

   This matches the p(clamp)=1.0 optimum in the scoring sweep, so clamping always is not
   irrational — but it means the result is "learned that clamping helps", NOT "learned to
   help its partner when its partner needs it". The stronger claim is not supported.

3. **The references moved, and cross-rule comparison is invalid.** Greedy fell from 0.568
   under SUBSET to 0.190 under JOINT_CONF, because JOINT_CONF costs baseline accuracy when
   nobody clamps (0.815 -> 0.244 in the scoring sweep). So this is a beat of greedy WITHIN
   the new belief model, not a beat of last night's greedy. Quoting 0.380 against 0.568
   would be wrong. A cross-rule evaluation -- policies trained under one rule, scored under
   another -- is needed before any of this goes in the thesis.

### Process error, recorded

I declared the first training launch dead and started a second. It was not dead; I read its
log seconds after launch, before anything had flushed. Both ran for ~15 hours competing for
CPU, which is why seed 1 reports 875 minutes of "training time" against seed 0's 47 and
seed 2's 34. **The per-seed timings in this run are meaningless** and should not be used for
any cost claim. The results themselves are unaffected -- the seeds are independent and the
contention only changed wall-clock.

### Next, in priority order

1. Cross-rule evaluation, to make the greedy comparison honest.
2. More seeds -- the spread demands it before any number is quoted.
3. Attack over-clamping: it is the difference between "clamping helps" and "coordination".
   A per-clamp cost, or a shaped signal for clamping only when the agent's own belief shows
   an unresolved shared-pair ambiguity, are the obvious candidates. The second risks
   hand-coding the answer and should be treated carefully.

### Checkpointing added (blocker for the cross-rule evaluation)

`ma_train.py` never persisted the trained networks, so the three JOINT_CONF pairs from the
run above are gone and the cross-rule evaluation -- score a policy trained under one belief
rule against another -- would have meant retraining every arm from scratch.

`IndependentPPO.save` / `.load` now store both agents' state dicts plus the rule the pair
was TRAINED under, and `ma_train.py` writes one checkpoint per seed. The trained-under rule
is stored deliberately: evaluating a policy under a different rule is the entire point of
the comparison, and silently mixing the two would be very easy to do.

Verified by round-trip: weights restore exactly into a freshly-constructed agent with a
different seed.

The three existing JOINT_CONF policies are NOT recoverable -- they were trained before this
existed. The cross-rule evaluation will need one retraining pass per rule, and that is a
cost of my own omission, not of the experiment.

---

# 2026-08-18 overnight — cross-rule evaluation and over-clamping

Unsupervised, full night. Plan set before starting, in priority order given by the user's
own list: (1) cross-rule evaluation, (2) more seeds, (3) attack over-clamping.

## First: made the experiments affordable

`RegimeScorer` was 543 x 8 x 4 Python iterations per belief update, rebuilding parent-set
tuples and hashing them every time. All of that is a pure function of the graph space, so it
is now precomputed once into a `(hypothesis, node) -> slot` index and a belief update is two
array gathers. Step cost 0.05s -> 0.022s, a 2.3x speedup that roughly doubles how many
experiments fit in the night.

**Verified rather than assumed.** `tests/test_score_regimes.py` keeps the straightforward
implementation as a reference and asserts the fast path reproduces it to 1e-12 across all
four rules and three clamp regimes -- compared over the WHOLE posterior, not the mass on the
truth, because a bug that shuffles low-mass hypotheses would be invisible to a summary
statistic. Also asserts the slot packing never collides (a collision would silently score
one hypothesis as another) and that JOINT_CONF marginalises the confounded-subset dimension
rather than maximising it.

One test bug found and fixed while writing it: the helper drew B's non-clamping actions
uniformly, which hits "clamp my private node" about 1/8 of the time, so the
"no clean rows" fixture was not actually clean. The scorer was fine; the test was wrong.

## Reconsidering "over-clamping" — it may be the wrong diagnosis

I reported the 2026-08-17 result as "learned to clamp but not *when*", reading the 84-96%
clamp rate as indiscriminate. On reflection that framing is probably wrong, and it matters
because it points at the wrong fix.

From the scoring sweep, p(clamp)=1.0 was the best setting for agent A on **both** confounded
and unconfounded episodes. So having your partner clamp constantly is genuinely optimal from
the receiving side, and a high clamp rate is not by itself irrational.

What a per-agent clamp rate cannot see is that **a clamping agent is not experimenting**. It
spends its budget holding a variable still for someone else and learns nothing about its own
graph. If both agents clamp in the same round, the round is wasted for both. Seed 2 --
clamp 0.957, solve 0.165, below random -- looks exactly like that.

So the coordinated solution is not "clamp less", it is **role differentiation**: one agent
clamps while the other experiments, and they swap. That is a property of the JOINT action
distribution and is invisible to the per-agent rates I have been quoting.

`scripts/ma_role_analysis.py` measures it against the right null -- two agents clamping
independently at their own observed marginal rates. Two agents each clamping 90% of the time
with no coordination give P(exactly one clamps) = 0.18, so anything near that is "both clamp
a lot" rather than "they take turns". Pre-registered expectation: seeds 0 and 1 above the
independence baseline, seed 2 at or below it. If ALL seeds sit at the baseline, no
differentiation is being learned and the coordination claim is weaker than I have stated.

The clamp-cost sweep in tonight's queue is still worth running -- it is the minimal
non-circular pressure against wasted rounds -- but this diagnostic is what will say whether
the cost is even addressing the right failure.

## Timing anomaly, diagnosed

`train_subset.json` reports per-seed training times of 428s, 403s and **46188s** -- the last
being 12.8 hours inside a phase that took about 20 minutes end to end. The checkpoint mtimes
settle it: 07:57:10, 08:03:58, 08:10:36, i.e. ~7 minutes per seed, all three consistent.

Cause: the machine suspended and resumed, and Windows' `perf_counter` keeps counting through
sleep. `ma_train.py` now records `train_cpu_seconds` from `process_time` alongside the wall
clock, so the two disagreeing is visible in the record instead of a suspended laptop being
quoted as a compute cost.

**Consequence I should act on rather than just note:** my "~40 minutes per seed" estimate,
which is why I cut the queue from 16 runs to 6, was wrong. It counted the reference
evaluations (random and greedy over 400 episodes) as if they were training. Actual training
is ~7 minutes per seed. The scope cut was made on a bad measurement, and once the current
queue finishes there is room to put the dropped phases back.

## Cross-rule evaluation — the blocker, resolved, and my prediction was wrong

Every policy scored under every belief rule, same episodes. Medians over 3 seeds.

CONFOUNDED solve rate      scored under:   subset    joint_conf
    random                                  0.016        0.049
    greedy                                  0.000        0.000
    subset-trained policy                   0.000        0.000
    joint_conf-trained policy               0.016      **0.262**

ALL episodes               scored under:   subset    joint_conf
    random                                  0.370        0.268
    greedy                                  0.542        0.190
    subset-trained policy                   0.290        0.113
    joint_conf-trained policy               0.033        0.495

**X1 confirmed.** Greedy is 0.000 on confounded episodes under every rule. It never clamps,
so no belief rule can hand it clean rows to condition on. The failure is structural, not a
scoring artefact.

**X2 FALSIFIED, and this is the important one.** I predicted a JOINT_CONF-trained policy
would keep most of its confounded advantage when scored under SUBSET, because the behaviour
earning it -- clamping -- produces clean rows SUBSET can also use. It does not: 0.262 drops
to **0.016**, and its overall rate collapses to **0.033**, far below random's 0.370. The
policy clamps 88% of the time, and under SUBSET clamping means discarding every
observational row for a handful of clean ones. The same behaviour that is near-optimal under
one belief model is catastrophic under the other.

**X3 confirmed.** A SUBSET-trained policy gains nothing from being scored under JOINT_CONF
(0.000 confounded, and 0.290 -> 0.113 overall). It never learned to clamp, and JOINT_CONF
without clean rows is strictly worse than the alternatives.

### What this actually licenses me to claim

I had been heading towards "the rule change bought a learning signal rather than an
inference advantage". That framing is now dead. The honest version:

  **Performance is a property of the (policy, belief rule) PAIR, not of either alone.**
  Neither component transfers. A policy trained under one rule is worse than random under
  the other, and a rule without a matching policy is worse than the rule it replaced --
  greedy drops 0.542 -> 0.190 simply by switching the belief model.

The claim that survives is the one the matrix was designed to support, read DOWN a column
with the belief model held fixed:

  **Under JOINT_CONF, the learned policy reaches 0.262 on confounded episodes where greedy
  reaches 0.000, random 0.049, and a policy trained under the other rule 0.000.**

That is a fair comparison and it stands. What it is NOT is evidence that the agents learned
a generally better experimental strategy. They learned a strategy that is co-adapted to a
particular belief model, and outside it that strategy is actively harmful.

For the thesis this is a better result than the one I predicted, because it is specific: the
belief representation and the policy have to be designed together, and the "coordination" on
display is inseparable from the inference machinery that makes clamping legible.

## Role differentiation: measured, and my framing was wrong for the second time

    policy              clampA  clampB  P(both)  P(one)  indep    diff    alt   solve
    joint_conf_seed0     0.695   0.903    0.650   0.297  0.343  -0.046  0.231   0.507
    joint_conf_seed1     0.917   0.761    0.729   0.220  0.282  -0.062  0.179   0.497
    joint_conf_seed2     0.807   0.995    0.806   0.191  0.195  -0.005  0.014   0.440
    subset_seed{0,1,2}   ~0      0.000    0.000     --     --   +0.000  0.000   ~0.29

Differentiation is NEGATIVE in every seed: the agents clamp *together* more often than two
independent clampers at the same marginal rates would. Alternation is low. By the
pre-registered reading that is "both clamp a lot", not "they take turns", and my expectation
that seeds 0 and 1 would sit above the baseline is falsified.

But the premise behind that expectation was wrong, so the test was aimed at the wrong thing.
I assumed simultaneous clamping wastes the round for both. What do they actually clamp?

    policy              both-clamp rounds   both clamp their OWN private node
    joint_conf_seed0            644                       0.992
    joint_conf_seed1            751                       0.995
    joint_conf_seed2            816                       0.999

    A clamps its own private node on 100.0% / 100.0% / 99.8% of its clamps; B on 98.6% /
    99.5% / 100.0%.

**Simultaneous clamping is mutual service, not waste.** A's hidden set is exactly `{z_B}`
and B's is `{z_A}`. So when both clamp their own private node, each one is cutting precisely
the confounder the OTHER cannot see, and both get clean rows in the same round. Taking turns
would be strictly worse -- it would leave one agent's confounding intact every round.

### The strongest form of the result

Clamping your own private node does **nothing for you**. A's rows are clean only when `z_B`
is clamped; A clamping `z_A` leaves A's own regime bit false and removes a variance source A
was not confounded by. The entire benefit accrues to the partner -- an agent A cannot see,
whose belief it has no access to, whose observations it never receives.

And the agents do it essentially always, learned from a shared scalar reward alone, with no
CTDE, no communication channel beyond the one-bit regime flag, and no term in the reward
that names helping.

That is the coordination claim in its strongest available form, and unlike the earlier
framings it is supported by a direct measurement of what the policies do rather than
inferred from an aggregate rate.

### Two framings of mine now retracted

1. **"Over-clamping is the failure mode."** Wrong. High clamp rates are the coordinated
   solution, not a pathology. Seed 2's poor overall score (clamp 0.995, solve 0.440) is the
   weakest of the three but is still well above greedy under the same rule.
2. **"They should learn role differentiation / take turns."** Wrong, and it would have been
   a worse policy. The clamp-cost sweep still in the queue was designed to discourage a
   behaviour that turns out to be correct, so its result should be read as "what does a
   price on the RIGHT behaviour destroy" rather than "does the price fix over-clamping".

## Clamp-cost 0.15: total collapse, and it exposes a reward-design fragility

    seed   solve   confounded   clamp rate   mean steps   final entropy
      0    0.005      0.000        0.000        0.99          0.003
      1    0.010      0.000        0.000        1.01          0.003
      2    0.005      0.000        0.000        0.99          0.003

    (default arm, same rule, no clamp cost: solve ~0.47, mean steps ~5.1)

Not merely "clamps less" -- the agents **stop acting entirely**. Mean episode length 1.0 and
policy entropy 0.003 means both learned to PASS on the first move, which ends the episode at
zero reward. They prefer a guaranteed 0 to a costly attempt.

This confirms the retraction above from the other direction: a price on clamping does not
refine the behaviour, it removes it, and removing it takes everything else with it. Under
JOINT_CONF without clean rows the belief model is poor by construction (0.244 against 0.815
in the scoring sweep), so once clamping is priced out there is no path to the terminal reward
worth paying step costs for. **Clamping is load-bearing, not incidental.**

### [GAP] The two-agent training has no under-acting canary

The single-agent work carries a `no_under_acting` check precisely because step costs plus a
hard-to-reach terminal reward create a give-up attractor, and it fired usefully there. The
two-agent trainer has no equivalent, so this run reported "clamp_fraction 0.000" as though it
were a behavioural finding when the real story was "both agents passed immediately". I read
the numbers correctly only because mean_steps happened to be in the eval output.

That is a hole in the harness, not just in this run. A `mean_steps` floor should be a
first-class canary on every two-agent result, and until it is, any low clamp rate here needs
checking against episode length before it is interpreted.

### Consequence for the 0.05 arm

Part 2 runs clamp_cost 0.05. Given 0.15 collapses to pure passing, 0.05 is now the more
informative point: it distinguishes "any price destroys it" from "there is a price at which
clamping becomes selective". Registering the expectation before it runs -- I expect partial
degradation rather than collapse, because 0.05 doubles the cost of a clamped action rather
than quadrupling it, but after two wrong framings tonight I hold that loosely.

## Clamp-cost dose-response: no price produces selectivity

    clamp_cost   solve (median)   confounded      clamp rate   mean steps
       0.00           0.495       0.20-0.28        ~0.89          5.1
       0.05           0.247       0.08-0.25        0.79-0.99      5.3     canary OK
       0.15           0.007       0.000            0.000          1.0     total inaction

My registered expectation for 0.05 was "partial degradation rather than collapse", and that
is what happened -- but the interesting part is WHICH thing degraded. The clamp rate barely
moves (0.89 -> ~0.90). What falls is the solve rate, roughly by half.

So the price does not make clamping selective. It taxes a behaviour the agents keep doing
anyway, because clamping remains individually correct under this belief model, and the tax
comes straight out of the score. Then somewhere between 0.05 and 0.15 the arithmetic flips
and the whole episode stops being worth attempting, giving the passing collapse.

**There is no intermediate regime where clamping becomes discriminating.** That is the
substantive finding, and it closes the "attack over-clamping" line rather than advancing it:
over-clamping was never the problem (see the mutual-service result above), so pricing it can
only destroy value. Recorded as a null with the mechanism, not filed away.

The remaining honest question about selectivity is different, and this sweep cannot answer
it: an agent cannot tell whether its PARTNER is confounded -- that fact lives entirely in the
partner's window. Clamping unconditionally may simply be the optimal policy available to an
agent with no way to know when it is needed. If so, "learned to clamp but not when" was
never a fair criticism, because "when" is not observable from where the agent stands.

---

# Final overnight results, 2026-08-19

## Ten seeds at the default setting

    seed    all    conf   clamp  steps
       0  0.497   0.279   0.879   5.07
       1  0.495   0.197   0.870   5.16
       2  0.407   0.262   0.894   5.18
       3  0.328   0.197   0.653   4.87
       4  0.225   0.164   0.975   5.71
       5  0.215   0.098   0.960   5.59
       6  0.297   0.246   0.976   5.44
       7  0.005   0.000   0.000   0.99   <- CANARY FIRED (under-acting)
       8  0.182   0.082   0.984   5.70
       9  0.405   0.164   0.702   4.70

    all episodes: median 0.312, mean 0.306, sd 0.154
    confounded:   median 0.180, mean 0.169, sd 0.088
    confounded beats greedy (0.000) in 9/10 seeds and random (0.049) in 9/10

**The headline, at n=10 rather than n=3:** confounded solve rate median **0.180** against
greedy **0.000** and random **0.049**. This is the claim that survives everything measured
tonight.

**Instability is real and quantified.** One seed in ten (seed 7) collapses into passing
immediately -- solve 0.005, mean episode length 0.99 -- with no clamp cost applied at all.
The under-acting canary added earlier tonight caught it automatically, which is the whole
point of adding it. The three-seed spread I could not interpret on 2026-08-17 was not
mysterious: this training is bimodal, mostly landing near 0.2-0.5 with an occasional total
collapse.

## The full cross-rule matrix

    CONFOUNDED       scored under:   subset   joint_conf     n
    random                            0.016      0.049       1
    greedy                            0.000      0.000       1
    subset-trained                    0.000      0.000       3
    joint_conf-trained                0.000      0.180      10
    jc + clamp cost 0.05              0.000      0.148       3
    jc + clamp cost 0.15              0.000      0.000       3

    ALL              scored under:   subset   joint_conf
    random                            0.370      0.268
    greedy                            0.542      0.190
    subset-trained                    0.290      0.113
    joint_conf-trained                0.022      0.312
    jc + clamp cost 0.05              0.025      0.247
    jc + clamp cost 0.15              0.020      0.005

Every joint_conf-trained policy scores **0.000 confounded and ~0.02 overall** under SUBSET --
far below random's 0.370. The co-adaptation finding from the 3-seed matrix holds at 10 seeds
and is if anything stronger.

## Role structure, all 19 policies

    group              n   clampA  clampB  P(both)    diff   solve
    joint_conf        10    0.862   0.898    0.729  -0.002   0.317
    jc+cost 0.05       3    0.989   0.603    0.603  -0.004   0.253
    jc+cost 0.15       3    0.000   0.000    0.000  +0.000   0.007
    subset-trained     3    0.025   0.000    0.000  +0.000   0.290

Differentiation is ~0 or negative everywhere: no turn-taking, as established, because
simultaneous mutual clamping is the better structure.

Note `jc+cost 0.05` broke the symmetry -- A clamps 0.989, B only 0.603. A modest price
produced an ASYMMETRIC equilibrium where one agent bears more of the altruistic cost. With
n=3 that is a suggestion rather than a finding, but it is the only sign all night of the two
agents adopting distinct roles, and it appeared exactly where clamping was made costly. Worth
a targeted run.

## What I would do next

1. **The instability is the biggest threat to the result.** 1-in-10 total collapse, and an
   sd of 0.154 on a median of 0.312. Entropy regularisation or a warmup that delays the step
   cost are the obvious candidates. Nothing else should be tuned until this is understood.
2. **Chase the asymmetric equilibrium at clamp cost 0.05** with more seeds. If a small price
   reliably produces role specialisation, that is a genuinely new result rather than a tax.
3. **Do NOT pursue selectivity further.** Three separate attempts tonight say clamping is
   near-universally correct here, and an agent cannot observe whether its partner is
   confounded, so "clamp only when needed" may be unattainable in principle in this setup.

## 2026-08-19 -- PHASE 0 and PHASE 1 complete

### Phase 0: reference frozen

`scripts/ma_freeze_reference.py` -> `tests/fixtures/ma_reference_posteriors.npz`
(100 episodes, 4 rounds, both agents, all four rules, 14.2 MB).

Stores the belief INPUTS (samples, known, clean) as well as the 543-vector outputs. A
fixture holding only posteriors would let a Phase 1 bug hide behind a differently-sampled
dataset.

Actions are a seeded UNIFORM policy, not greedy. Greedy never clamps (measured today,
clamp_fraction 0.000), so a greedy-driven fixture would contain zero clean rows and would
exercise exactly one branch of the four rules.

[GATE PASSED] Two independent captures are bit-identical (sha256 fc47a423b76c24d6).

### Phase 1: the DP belief

`ma/belief_dp.py` + `tests/ma/test_belief_dp.py`. 10 tests, all passing.

[GATE PASSED] Edge marginals AND true-DAG probability match the frozen enumerated fixture
to < 1e-10 for pooled, subset and joint.
[GATE PASSED] k=10 window completes -- 4.2e18 DAGs, where enumeration is not slow but
impossible. The DP has bought what it was chosen for.
[GATE PASSED] Confinement asserted, not assumed.

[CORRECTED, and it is a wart in the OLD code, not the DP] **joint_conf was never modular.**
`RegimeScorer._dirty_parents` orients each confounding edge "along the DAG's topological
order", but a DAG has many topological orders and `_topological_order` picks one by an
arbitrary tie-break (lowest available index). For two shared nodes INCOMPARABLE in the DAG
the orientation is therefore decided by node numbering, and the two orientations score
differently under BGe. So the hypothesis being scored depended on an implementation detail.

That is also exactly what breaks modularity: node v's dirty parent set depends on the DAG's
global order rather than on v's own parents, so no per-(node, parent-set) table expresses it.

[DECIDED] Reformulated: make the orientation part of the hypothesis and marginalise it out.
A hypothesis is (DAG H, set P of ORDERED pairs declared confounding), P's edges required
present in H; clean regime scores H minus P, dirty regime scores H. Modular for fixed P, so
3^(pairs) DP passes. Gains: acyclicity free (the DP only emits DAGs), no arbitrary
tie-break, confinement preserved. Cost 3^pairs vs 2^pairs -- 27 vs 8 at |X|=3.

Consequence: the Phase 1 gate SPLITS. pooled/subset/joint held to 1e-10; joint_conf cannot
be, because it is deliberately a different hypothesis space. Checked for internal
consistency instead, and to be compared to the old rule by MEASUREMENT, not identity.

[MEASURED] Only 25 of the 27 assignments are usable. The three shared pairs of |X|=3 form a
triangle and its 2 cyclic orientations admit no acyclic completion. Left in, they make the
DP's alternating inclusion-exclusion cancel to an exactly zero partition function -- which
is how they were found, via FloatingPointError rather than a wrong number.

[CORRECTED, mine] First run of the gate failed at 9.5e-01 on the true-DAG probability. My
bug: `DPPosterior.log_prob_dag`'s third argument is `log_z`, not `k`. Passing k=4 subtracts
a constant instead of normalising, and produces a plausible-looking probability rather than
an obvious error. Fixed and commented at the call site.

### Still open before Phase 2

- The joint_conf old-vs-new comparison is a MEASUREMENT that has not been run yet. Until it
  is, no claim about which rule is better under the new formulation.
- Regime bit still needs Mirco's ruling (draft written 2026-08-19).
- [DECIDED, user, revised 2026-08-19] Test WITHOUT the regime bit FIRST, then with it.
  The no-bit arm is the baseline, so the with-bit arm has a clear reference to move against
  and any bug that shows up in the harder arm can be attributed rather than guessed at.
  (This reverses the earlier with-bit-first ordering. The reasoning is better: the no-bit
  arm is also the SIMPLER system, so it fails in fewer ways, and a broken baseline is much
  easier to diagnose than a broken treatment.)

## 2026-08-19 -- PHASES 2 to 6 implemented

New modules, all on the DP belief:

  ma/env2.py        Phase 2 environment. n_obs=1000 (not 100), budget=5 per agent,
                    disclose_regime=False by DEFAULT -- the no-bit arm is the baseline.
  ma/baselines2.py  Phase 4. pass / random_vary / random_clamp / forced_clamp / greedy.
  ma/policy2.py     Phase 5. Independent PPO, no CTDE, shared scalar reward.
  ma/evaluate2.py   Phase 6. The three-part [U14] criterion.
  scripts/ma_gates2.py  Phase 3. GATE 1, 2, 3.

[MEASURED] Cross-validation of joint_conf. `enumerated_posterior` reconstructs the full
543-vector posterior from the DP's OWN local score tables, and its implied edge marginals
match `WindowBeliefDP.edge_marginals` to 1.065e-12 over 6 episodes x 2 agents. Two
independent code paths through the reformulated rule agree. This matters because the DP
path could not be checked against the frozen fixture for joint_conf -- the hypothesis space
deliberately changed -- so this is the substitute check.

[CORRECTED, mine] I reintroduced the global-log-sum-exp bug in `enumerated_posterior` on
the first attempt: a single global shift underflows the weaker DAGs to zero, log(0) = -inf,
and hypotheses are deleted rather than ranked. Caught by a divide-by-zero warning, not by a
wrong number. Fixed to a PER-DAG shift. This is the third time this exact bug has appeared
in this project; it is now commented as such at all three sites.

[DECIDED] The greedy oracle ENUMERATES, and the limit is explicit (MAX_ENUMERATED_K = 5,
raising a message that names the reason). It needs a full window posterior for its
descendant-partition criterion, and no trustworthy DP posterior sampler exists -- the MH
sampler is under-mixed at 5.8% acceptance and its settings are an admitted stopgap. The
oracle is a REFERENCE POINT, not part of the method, so it does not need to scale. Stated
rather than hidden.

[NOTE] Greedy is structurally INDIFFERENT between VARY and CLAMP: both cut the target's
incoming edges, so both induce the same descendant partition and the same one-step gain.
The tie-break therefore decides, which is the mechanism behind the measured
clamp_fraction 0.000. This is now documented in the class that causes it.

[DECIDED] Phase 6 credit set. A DAG counts as a correct answer when it (a) matches the
truth on every edge touching a private node -- identifiable, since the agent may intervene
on its own private nodes -- and (b) lies in the truth's Markov equivalence class. The union
of both agents' MAP graphs is then checked for ACYCLICITY and global equivalence. Union is
by OR, the permissive choice: it can create cycles but never hide them, so the check sees
the worst case.

[MEASURED, smoke only] 8 evaluation episodes after a 32-episode PPO run: success 0.500
against threshold_identified 0.375, union_acyclic 1.000. The three-part criterion is more
permissive than the raw threshold, as intended -- it credits equivalence-class members the
threshold rejects. Not a result; a sign the plumbing is connected.

[CORRECTED, my commentary not the code] I wrote that part 1 of [U14] is "vacuous at
(1,1,3) because there are no private-private edges". That is the LITERAL reading and it is
not what was meant. The intended reading, confirmed 2026-08-19: an agent's "private graph"
is everything involving its private variables -- private-private edges AND the boundary
edges between its private nodes and the shared set. Those are identifiable to FULL DAG
resolution precisely because the agent holds intervention authority over its own private
nodes. Shared-shared edges are the only ones relaxed to CPDAG, because those are what may
need the partner.

The split is the federation claim itself: each agent is SELF-SUFFICIENT inside its own
region, and the shared boundary is the only place coordination is required.

`ma/evaluate2.py:credit_set` already implements the intended reading -- it requires an exact
match on every edge INCIDENT to a private node, in both directions (`dag[p, :]` and
`dag[:, p]`), which is the boundary-inclusive version. So the code was right and the note
was wrong. At (1,1,3) part 1 is therefore NOT vacuous: each agent has 3 boundary edges to
get fully oriented.

## 2026-08-19 (evening) -- identification criterion, three attempts

The joint_conf identification criterion was wrong twice before it was right, and the
sequence is worth keeping because each failure looked like a different problem.

[CORRECTED #1] `P(H == truth)` -- TOO HARSH. A hypothesis is (DAG H, confounding set P)
with P's edges REQUIRED PRESENT in H, so a confounded pair shows up as an edge in H that is
labelled confounding. The true DAG contains no such edge, so it took mass only under the
empty assignment -- the single hypothesis that refuses to model the confounding. Result:
EXACTLY 0.000 for the affected agent on every confounded episode at every budget. This read
as a GATE 3 coordination failure for two runs before I traced it.

[CORRECTED #2] `P(H \ P == truth)` -- TOO GENEROUS, and it produced a GATE 1 LEAK of 0.2387
against a target of 0.0402. With no clean rows the confounding label is UNFALSIFIABLE: any
extra edge may be added and called confounding for the price of a BGe complexity penalty,
and every such superset maps back to the same base graph. Also exposed an overflow --
normalising per assignment trusts each masked table's Z, and the signed sink recurrence is
least reliable exactly there, producing "probabilities" of 1e131. Now a ratio of sums in
log space, which needs only the total.

[CORRECTED #3] `P(H \ P == truth AND P correct)` -- 0.0000 observationally. Demanding the
right confounding set with no clean data means ruling out 24 rival assignments blind.

[DECIDED] The criterion in the code is #3, because it is what [U14] actually asks: an agent
that cannot tell a confounded pair from a causal edge has not recovered the structure.
Orientation of the modelling edge is not part of the claim -- both orientations express the
same "u and v share a hidden cause" -- so both are credited.

[DECIDED] GATE 1 now runs under `pooled`, a genuine DAG model, NOT under joint_conf.
GATE 1 asks a question about the ENVIRONMENT: is the task solvable without acting? Its
target is a DAG-derived quantity (the singleton-MEC fraction). joint_conf is not a DAG
model, and its observational identification is near zero by design, so comparing the two
measures the RULE rather than the environment. Under pooled the same environment gives
0.0547 against a target of 0.0442.

    criterion                        unconfounded obs-only    target
    P(H == truth)                          0.0256            0.0402
    P(H \ P == truth)                      0.2387            0.0402   <- leak
    P(H \ P == truth AND P correct)        0.0000            0.0402
    pooled (DAG model)                     0.0547            0.0442   <- the right test

## Performance work

[MEASURED] Two optimisations to the belief update, both verified against the 1e-10 gates:
  - local score tables were computed TWICE per refresh (once by joint_conf_marginals, once
    by joint_conf_dag_probability). Cached. ~15%.
  - `local_score` recomputes sufficient statistics on every call, so local_table made 16
    full O(n k^2) passes per node -- 64 per table at k=4 -- where 4 suffice. Switched to
    `local_score_from_stats` with one pass per node.
  After both, `local_score` no longer appears in the profile's top ten at all; the cost is
  now entirely the 25 DP passes (edge_marginals_onepass, ~70% of an episode).

[MEASURED, NULL] Pruning negligible assignments does NOTHING at |X| = 3. All 25 carry more
than 1e-14 of the mass (top weights 1.0, 0.500, 0.500, 0.025, 0.018, 0.012...). The guard is
kept because it is nearly free and should matter at |X| = 4 with 729 assignments, but that
is an expectation and is NOT measured. It must not be cited as a speedup.

[NOTE] Absolute timings taken this evening are contaminated -- gate runs were competing for
CPU throughout. Relative profiles are still informative; wall-clock numbers are not.

## 2026-08-19 (late) -- gates v5, and a retraction

    GATE 1  observational (unconfounded, n=507) 0.0394 CI 0.0237-0.0592  target 0.0402  PASS
    GATE 2  unconfounded: greedy 0.074 CI 0.051-0.096  vs random 0.147 CI 0.118-0.178  FAIL
    GATE 3  confounded (n=56): never-clamp 0.000  vs clamping 0.393  headroom +0.393    PASS

### GATE 3 passes, and under the strict criterion it is a cleaner result than v4's

v4, under the leaky criterion, gave 0.143 vs 0.571. v5, requiring the confounding claim to
be correct, gives **0.000 vs 0.393**. The never-clamping arm collapses to exactly zero, which
is the mechanism made visible: without clean rows the confounding claim is UNFALSIFIABLE, so
a pair that never clamps cannot in principle satisfy the criterion, at any budget, ever.
Clamping is what makes the claim testable. That is a much stronger statement than "clamping
helps".

### [RETRACTED] "The greedy oracle never clamps -- that is the myopic objective behaving
### correctly." This was an artefact of the baseline's own code.

`ma/baselines.py:GreedyAgentPolicy` builds its action set as

    self.candidate_actions = [view.actions.index((node, VARY)) for node in view.authority]

VARY only. The old greedy COULD NOT CLAMP. Its measured clamp_fraction of 0.000 was true by
construction, and the comment three lines above it -- "a self-interested agent will therefore
never clamp to help its partner" -- states the assumption that the measurement then returned.
I reported that number repeatedly as a discovery about myopic information gain. It was not.

What is actually true, and is the better claim: myopic EIG is INDIFFERENT between VARY and
CLAMP, because both cut the target's incoming edges and induce the same descendant partition.
The objective does not determine the mode at all, so behaviour is decided entirely by the
tie-break. `ma/baselines2.py:GreedyAgent` offers both modes and breaks ties uniformly, giving
clamp_fraction **0.526**.

### GATE 2 inverts, and I do NOT yet know why

Greedy 0.074 against random 0.147 on unconfounded episodes -- the reference is worse than the
floor. My first explanation was that greedy fails to generate clean rows. MEASURED, AND IT IS
WRONG:

    policy         clamps own PRIVATE node   mean clean rows for A
    greedy               0.116                     40
    random_clamp         0.068                     20

Greedy clamps its own private node MORE often than random and produces TWICE the clean rows,
and still scores half as well. So the cause is not clean-data generation. Candidate
explanations not yet tested: greedy concentrates its interventions on a few high-EIG targets
(the known repeat-target pathology), and ruling out 24 rival confounding assignments may need
BREADTH of intervention rather than depth. That is a hypothesis, not a finding.

[DECIDED] GATE 2 is BLOCKED pending that diagnosis. It is not obviously the environment's
fault: a reference policy that is worse than random may simply be the wrong reference for
this task. But "my baseline is bad" and "my environment is bad" are exactly the two things a
gate is supposed to distinguish, so no training runs until it is understood.

## 2026-08-20 (overnight) -- the random baseline was two synchronised agents

[CORRECTED] Every call site built agent A and agent B with the SAME seed
(`RandomAgent(n, seed=args.seed + 1)` for both). Their action lists are structurally
parallel -- own private node first, then the shared nodes in the same order -- so identical
RNG streams meant identical action INDICES, and therefore the same shared target almost
every round.

    measured collision rate (both agents target the same node):
        random_clamp, shared seed        0.784
        random_clamp, per-agent seeds    0.227     <- independent uniform expects ~0.19
        greedy                           0.352

`random_clamp` is the PRIMARY floor [U16], so this is not cosmetic. A perfectly synchronised
pair is a different policy: it systematically wastes one of its two moves per round, which
makes the floor easier to beat for a reason that has nothing to do with choice quality.

Fixed with `_agent_seed(seed, name)`, so A and B get distinct streams from one seed at every
call site at once.

### The GATE 2 inversion: three explanations tried, two falsified

GATE 2 v5 had greedy at 0.074 BELOW random at 0.147 on unconfounded episodes.

1. "Greedy fails to generate clean rows." FALSIFIED. Greedy clamps its own private node
   MORE often than random (0.116 vs 0.068) and produces twice the clean rows (40 vs 20).
2. "Greedy repeats targets, and ruling out 24 rival confounding assignments needs breadth."
   PARTIALLY SUPPORTED but too small to carry it: 2.10 distinct targets per episode against
   random's 2.45, repeat rate 0.300 against 0.183.
3. Criterion decomposition shows the per-agent marginals are nearly IDENTICAL --
   base-graph-correct mass 0.834 (greedy) vs 0.782 (random), strict mass 0.355 vs 0.363 --
   so a 2x difference in the JOINT event has to come from the correlation between the two
   agents, not from either agent alone. Greedy's agents compute the same objective over
   overlapping authority and collide on 0.352 of rounds against random's 0.227.

That third line is now the live hypothesis, and the synchronised-seed bug means the v5
GATE 2 numbers were measured against a broken floor anyway. GATE 2 must be re-run before
anything is concluded from it.

[NOTE] Training runs already in flight (local seeds 0-7, Myriad 176251) began before this
fix. The LEARNED policy is unaffected -- PPO carries its own RNG and never touches
`make_baselines` -- but the baseline arms inside those reports are measured against the
synchronised floor and must be re-evaluated afterwards rather than quoted.

## 2026-08-20 (overnight, cont.) -- a bug the cross-check caught, and gates v7

[CORRECTED] **The clean regime was credited for the confounding edges.** A hypothesis is
(DAG H, confounding set P) with P's edges present in H and STRIPPED AGAIN for the clean
regime -- clean rows are the ones with no hidden variable transmitting variance, so the
confounding edge must not be scored there. `WindowBeliefDP._assignment_weights` reads the
clean table at `parents \ P`; `enumerated_posterior` read it at the full parent set, so the
confounded hypothesis fit the clean data too well. Disagreement 3.55e-02.

INVISIBLE UNTIL THE TEST FORCED A REGIME SPLIT. With no clean rows the clean table is all
zeros (the empty-regime guard), so stripping changes nothing and the two paths agree to
1e-12 -- which is exactly the "cross-validation" I reported earlier. The same lesson as the
subset-DP sampler: test data that cannot exercise a branch proves nothing about it. The
cross-check now asserts a genuine clean/dirty split occurs, so it cannot go vacuous again
the same way.

`GreedyAgent` scores on `enumerated_posterior`, so the greedy baseline was running on a
wrong posterior on every episode where anyone clamped.

### Gates v7 (fixed posterior, independent agent seeds)

    GATE 1  observational (unconfounded, n=335) 0.0388 CI 0.0209-0.0597  target 0.0420  PASS
    GATE 2  unconfounded: greedy 0.072 CI 0.047-0.099  vs random 0.091 CI 0.061-0.122  FAIL
    GATE 3  confounded (n=38): never-clamp 0.000  vs clamping 0.184  headroom +0.184    PASS

[MEASURED] The posterior fix removed most of the GATE 2 inversion: greedy vs random went
from 0.074/0.147 to 0.072/0.091, with CIs now OVERLAPPING. Greedy is no longer clearly worse
than random -- it is indistinguishable from it.

[DECIDED] GATE 2 cannot pass with this reference, and the reason is the reference rather
than the environment. Greedy maximises the entropy of the descendant-set partition, which is
a DAG-model quantity. Under joint_conf with the strict criterion the binding constraint is
resolving the CONFOUNDING, and the myopic EIG objective contains no term for it -- it cannot
distinguish an intervention that will produce clean rows for the partner from one that will
not. So the myopic oracle optimises the wrong quantity here, and lands at random's level.
That is a finding about myopic experimental design in a federated setting, not a defect in
the environment, and GATE 1 and GATE 3 both pass.

[NOTE] GATE 3's headroom fell from +0.393 to +0.184 once the random agents were desynchronised.
The +0.393 was measured against a synchronised floor. +0.184 is the honest number, and the
never-clamp arm is still EXACTLY 0.000.

### Cluster

Job 176251 (40 tasks) runs no-bit as tasks 1-20 and with-bit as 21-40, and Myriad is
scheduling ~2 tasks at a time, so the with-bit arm would not start for many hours. Submitted
job **176272**, with-bit only, 20 tasks, so the priority arm runs now. 176251 continues and
supplies the no-bit control.


## 2026-08-20 (midday) -- the REPORTED metric cannot credit a confounded episode

[MEASURED] Random policy, 100 episodes, budget 8, regime bit on:

    U14 reported success | unconfounded   ~0.59 (n=85)
    U14 reported success | CONFOUNDED      0.000 (n=15)

[EXPLAINED] Same confusion as the reward bug, third occurrence. `enumerated_posterior`
under joint_conf indexes graphs by **H, the AUGMENTED graph** -- the one containing the
confounding edges, because the dirty regime reads H's own parents and the assignment
requires those edges present. `credit_set` then compares H against the true CAUSAL graph.

On a confounded episode the true causal graph contains no confounding edge, so it matches H
only under the EMPTY assignment -- the single hypothesis that refuses to model the
confounding, and the one that fits confounded data worst. Every other assignment produces an
H with an extra edge, which has a different skeleton and therefore a different Markov
equivalence class, so it is excluded from the credit set.

The metric is therefore structurally incapable of scoring the case the two-agent design
exists to study. The ~0.50 headline figures are carried ENTIRELY by unconfounded episodes.

[CONSEQUENCE] Every two-agent success number reported so far -- with-bit 0.583, no-bit
0.055, the paired +0.013 over random -- is an unconfounded-only number wearing a general
label. The with-bit vs no-bit CONTRAST may still stand, since both arms are scored the same
way, but nothing about coordination under confounding can be read from them.

[DECIDED] `reward_criterion` default held at "identified". The "u14" path is implemented and
verified (agrees with evaluate2.success on 25/25 episodes) but NOT adopted: it reintroduces
window enumeration into the training loop, and it should not be adopted before the question
below is answered, because that determines what the criterion ranges over.

[OPEN, and it is the design question] Should the success criterion be defined over H (the
augmented graph) or over H \ P (the causal graph)? The causal graph is what the thesis
claims to recover, which argues for H \ P plus a correct confounding claim -- exactly the
criterion adopted for `true_mass`. That would make the reward and the report agree AND make
confounded episodes scorable. It is the third time this distinction has caused a bug, so it
belongs in the code's structure rather than in a comment.

---

## 2026-08-20 (afternoon) -- the first two-agent numbers that can score confounding

Everything below is measured under the criterion corrected at 12:11/12:35/12:41 today. It
is the first evidence about the two-agent case that is not conditioned on unconfounded
episodes.

[MEASURED] **With the regime bit, learning beats random, and the intervals separate.**
Seed 0, 2000 training episodes, budget 8, 150 evaluation episodes:

    learned         0.467   CI 0.387-0.547    3.85 steps   clamp 0.776
    random_clamp    0.200   CI 0.140-0.260    7.23 steps   clamp 0.500
    random_vary     0.073   CI 0.033-0.113    7.61 steps   clamp 0.000
    greedy          0.173   CI 0.113-0.233    7.27 steps   clamp 0.516

Three separate things fall out of that one table.

1. The learned policy is strictly better than its own random floor -- non-overlapping
   intervals, not a point-estimate gap -- and it gets there in HALF the steps (3.85 vs
   7.23). It is not buying success by spending more budget.
2. **GATE 2's failure reproduces here independently.** Greedy 0.173 sits BELOW random 0.200.
   That is a second measurement, on a different harness, of the thing the gate found: at two
   agents the myopic oracle is not a good reference. It is now hard to read this as a
   sampling accident.
3. **GATE 3's effect reproduces here too, for free.** random_clamp 0.200 against
   random_vary 0.073 is the same clamp-vs-never-clamp contrast the gate isolates, measured
   inside the training evaluation with no extra runs. Being ALLOWED to clamp is worth more
   than doubling, before any learning happens.

[MEASURED] **The no-bit arm still collapses, and that is still correct behaviour.** Entropy
0.025-0.036 by update 80, solve rate 0.000. The metric fix does not rescue it, which
matters: the collapse was never a measurement artefact. It is the step cost. At 0.05 per
step and ~7.7 steps, acting has negative expected value against 0.000 for passing, so
passing is optimal and the policy finds it. The `ma_cost0` control on Myriad is what
separates "disclosure makes it learnable" from "the reward design made acting irrational".

[MEASURED] **Disclosure changes the BELIEF, not only the policy.** The no-bit arm's own
random floor is 0.027-0.047, against 0.200 for the with-bit arm's random floor. Same policy
class, same budget, ~5x difference. So the regime bit is doing most of its work upstream of
learning, by sharpening the posterior. Two consequences: (a) cross-arm comparisons of the
learned rate are meaningless, each arm must be scored against ITS OWN random floor, which is
what the report now does; (b) the with-bit/no-bit contrast is not a clean test of "does the
bit make coordination learnable" -- it confounds a belief effect with a learning effect.

[CORRECTED] **GATE 3's recorded numbers predate the fix and are not being carried forward.**
`results/ma2/gates_withbit_v7.json` was written at 01:12 today; the criterion was corrected
at 12:11. The gate scores CONFOUNDED EPISODES ONLY -- precisely the regime the old criterion
could not credit. Re-running from scratch (`scripts/ma_gate3_recheck.py`, 2000 attempted
episodes so the confounded subsample is large).

[MEASURED] **GATE 3 PASSES under the corrected criterion, with MORE headroom than before.**

    never-clamp    0.012   CI 0.000-0.030    n=169
    mixed-clamp    0.249   CI 0.183-0.314    n=169
    headroom      +0.237   (was +0.184 at n=38 under the old criterion)

Intervals do not overlap, on 4.4x the confounded sample. The design's central premise --
coordination is both necessary and available -- is re-established rather than inherited. The
direction is worth noting: correcting the metric made the headroom BIGGER, not smaller. The
old criterion demanded the exact true DAG together with the exact confounding set, which is
strictly harder than [U14] and was suppressing the arm that CAN clamp more than the arm that
cannot. I had assumed the error ran the other way, which is precisely why it had to be
measured instead of reasoned about.

[MEASURED, and it belongs in the thesis as a caveat] **Single-agent GATE 1 does not pass at
d=5 or d=7.** Across all 42 runs in `results/gnn_budget_exact/`:

    d=5  n_obs=1000    rate 0.0400   target 0.0893    passed 0/12
    d=5  n_obs=100     rate 0.0000   target 0.0893    passed 0/12
    d=7  n_obs=1000    rate 0.0167   target 0.0779    passed 0/9
    d=7  n_obs=100     rate 0.0000   target 0.0779    passed 0/9

Every failure is on the BELOW-target side, and the gate is two-sided for a reason. Above
target would be a leak -- the task solvable without intervening -- and would invalidate the
comparison. Below target means the posterior cannot reach the 0.7 identification threshold
even on graphs that are uniquely identifiable: the task is HARDER than intended, not easier.
The policy comparison stands, since all arms face the same environment. What does not stand
is any claim about absolute identification rates. The fix is either more data or a threshold
derived per setting instead of inherited from the d=3/d=4 case -- goes in the parameter
audit.

[MEASURED] **The zero-cost control splits the collapse from the bit, and the answer is not
the one the collapse suggested.** No regime bit, `step_cost=0`, everything else identical,
corrected criterion:

    learned         0.073   CI 0.033-0.120    8.03 steps   entropy 1.83
    random_clamp    0.080   CI 0.040-0.127    7.59 steps
    random_vary     0.073   CI 0.033-0.113    7.61 steps
    greedy          0.060   CI 0.027-0.100    7.62 steps

Two findings, and they point in opposite directions:

1. **The collapse WAS the reward design.** 8.03 steps against 0.00, entropy 1.83 against
   0.02. Remove the price on acting and the agent acts. Nothing subtle about it.
2. **Removing the price does not make the task learnable.** 0.073 against a random floor of
   0.080 -- indistinguishable. An agent that acts freely, explores fully, and is never
   punished for it STILL cannot beat random without the regime bit.

So the step cost explains the collapse and does NOT explain away the bit. This is the
control I expected to weaken the regime-bit claim, and it strengthens it instead. The
mechanism is the belief, not the policy: without disclosure the posterior cannot separate
confounding from a genuine shared edge, and no policy can recover what the belief cannot
represent. Note the ceiling -- EVERY arm sits at 0.06-0.08, including greedy with an exact
posterior. That is a property of the no-bit belief, not of any policy in it.

This retires hypothesis "the no-bit arm might learn if acting were free", which had been
open since 2026-08-19.

[MEASURED] **GATE 2 resolved as a finding: the myopic oracle fails under decentralisation
because both agents WANT THE SAME EXPERIMENT, not because they collide by coin flip.**

500 attempted episodes, unconfounded only (n=455), budget 3, with the regime bit:

    random          solve 0.062   CI 0.042-0.084    collisions 0.188 of 1339 rounds
    greedy          solve 0.064   CI 0.042-0.088    collisions 0.372 of 1329 rounds
    greedy_split    solve 0.040   CI 0.022-0.059    collisions 0.366 of 1330 rounds

    target-level tie in 0.068 of decisions
    0 of 74 collisions had a tie for BOTH agents

`greedy_split` gives the two agents opposite tie-breaking conventions -- A takes the
lowest-indexed tied action, B the highest. It uses NO communication: each applies a fixed
convention to its own action list, so nothing crosses the federation boundary. If the
collisions were coin flips landing the same way, this fixes them.

It does not. The collision rate moves 0.372 -> 0.366, and solve does not improve.

The argmax diagnostic explains it, and this is the real finding. A tie-break can only
separate two agents where a tie EXISTS, and at the level of which variable to target they
almost never have one -- a target-level tie in 6.8% of decisions, and **0 of 74 collisions**
involved a tie for both agents. The 2-element argmax ACTION set that appears in ~94% of
rounds is VARY and CLAMP on the SAME node, which is the already-documented indifference
between modes, not a choice between targets.

So each agent independently computes a UNIQUE best target, and it is the same target,
because the objective is identical and the shared variables are visible to both. No local
convention can separate them.

[DECIDED] **Greedy is retired as the two-agent reference.** Random becomes the floor
everything is scored against. This is not a concession -- it is the honest reading of three
independent measurements that agree: the gate itself (0.072 vs 0.091), the training
evaluation on 8 seeds (greedy at or below random on 7 of 8), and this experiment
(0.064 vs 0.062). Keeping a reference that cannot discriminate would be the error.

[CORRECTED] My own earlier framing, "both greedy agents ... pick the same target and waste
the round", was right about the behaviour and wrong about the cause. I called it a collision,
which implies chance. It is systematic agreement. The distinction matters because it decides
whether the fix is a coordination convention (it is not) or a change of objective (it is).

[MEASURED] **Ten seeds, with the regime bit, corrected metric. The instability is gone.**

    seed   learned                  random   sep   steps   greedy
      0    0.467 [0.387,0.547]      0.200    yes   3.85    0.173
      1    0.327 [0.253,0.400]      0.160    yes   3.22    0.127
      2    0.500 [0.420,0.573]      0.200    yes   4.19    0.153
      3    0.320 [0.247,0.393]      0.213     -    2.01    0.153
      4    0.520 [0.447,0.600]      0.200    yes   4.74    0.140
      5    0.400 [0.327,0.480]      0.193    yes   3.43    0.127
      6    0.373 [0.300,0.453]      0.153    yes   3.25    0.173
      7    0.467 [0.387,0.547]      0.160    yes   5.63    0.107
      8    0.313 [0.240,0.387]      0.207     -    2.79    0.127
      9    0.373 [0.293,0.453]      0.140    yes   2.90    0.107

    learned  median 0.387   mean 0.406   sd 0.073
    random   median 0.197
    separated (non-overlapping intervals)  8/10
    learned > random by point estimate     10/10
    greedy <= random                        9/10
    COLLAPSED SEEDS                         0/10

Three things worth pulling out.

1. **0 of 10 seeds collapsed.** The 2026-08-19 run recorded "1-in-10 seeds collapses into
   passing immediately, sd 0.154 on a median of 0.312" and called that instability the
   biggest threat to the result. It does not reproduce: sd is 0.073 on a median of 0.387,
   and every seed acts. Two candidate explanations -- the corrected reward is a better
   learning signal than the old exact-DAG criterion, or the old sd was inflated by the
   metric scoring confounded episodes at 0.000 -- and they are not separable from these runs
   alone. Either way the instability claim does not survive.
2. **Learned beats random on 10/10 by point estimate, 8/10 by non-overlapping intervals.**
   The two that overlap (s3, s8) are also the two lowest-acting seeds (2.01 and 2.79 steps
   against a mean of 3.6), so the failure mode that remains is under-acting, not
   mis-acting.
3. **greedy <= random on 9/10**, a third independent confirmation of the GATE 2 finding.

[MEASURED] **Zero-cost control at five seeds -- the conclusion holds and tightens.**

    seed  learned  random  steps  entropy  greedy
      0    0.073   0.080   8.03    1.86    0.060
      1    0.020   0.027   8.79    2.01    0.013
      2    0.047   0.047   8.07    1.82    0.040
      3    0.087   0.100   7.94    1.66    0.087
      4    0.060   0.047   8.51    1.99    0.060

    learned mean 0.057   random mean 0.060   ahead on 1/5   steps mean 8.27

No seed collapses (8.3 steps, entropy 1.66-2.01, against 0.00 steps and 0.02 entropy at
step_cost=0.05), and no seed learns anything (ahead of its own floor on 1 of 5, by 0.013).
Greedy tracks both at 0.013-0.087. Every arm sits in the same 0.02-0.10 band, so the ceiling
belongs to the no-bit BELIEF and not to any policy in it.
