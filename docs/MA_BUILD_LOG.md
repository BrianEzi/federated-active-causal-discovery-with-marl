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
