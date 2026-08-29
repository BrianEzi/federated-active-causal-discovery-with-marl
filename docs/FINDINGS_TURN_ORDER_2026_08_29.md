# Turn order: round-robin vs random — for the discussion and limitations chapter

Written 29 Aug 2026, prompted by re-reading Mooij, Magliacane & Claassen (JMLR 2020) and
asking whether JCI's warning about deterministic context relations condemns our round-robin
protocol. **It does not, and the empirical answer runs the other way.** But the question
exposes a real limitation that belongs in the write-up.

---

## 1. The answer

**Keep round-robin.** `scale21` and `rndturn` are identical in every recorded config field
except `turn_order` — both 8 agents, 2 private + 4 shared, budget 24, factored backend,
`reward_scale=0.214`, confounded episode mix.

| run | turn order | learned | greedy | margin | free-rider index |
|---|---|---|---|---|---|
| `scale21_s0` | round_robin | **0.620** | 0.480 | **+0.140** | 0.714 |
| `scale21_s1` | round_robin | **0.687** | 0.560 | **+0.127** | 0.740 |
| `rndturn_s0` | random | 0.373 | 0.367 | +0.006 | 0.140 |
| `rndturn_s1` | random | 0.393 | 0.300 | +0.093 | 0.140 |

Round-robin roughly doubles learned joint success AND widens the margin over greedy. Greedy
falls too (0.48/0.56 → 0.37/0.30), so random turns make the task harder for both arms rather
than removing an advantage from ours.

**The confound that nearly wrecked this comparison.** The obvious pairing, `rndturn` against
`a08`, is void: `a08` has `reward_scale=None` and a final entropy of 1.814 against a maximum
of 1.946 — **it never trained**. `rndturn` carries the reward-scale fix and `a08` does not,
so that difference is the reward scale, not the protocol. `scale21` is the only clean
control. This is the third time on this project that a comparison was nearly built from a
neighbouring run's settings.

## 2. The mechanism, and it is not behavioural

`free_rider_index` is `min(interventions) / max(interventions)` across agents — **higher is
more even**. Under random turns it reads 0.140, which looks like catastrophic free-riding.
It is not behaviour at all. Drawing the actor uniformly makes the per-agent move count
Multinomial(24, 1/8), and pure sampling with **no policy whatsoever** gives:

| quantity | pure sampling | measured under random turns |
|---|---|---|
| E[min moves] | 0.83 | — |
| E[max moves] | 5.67 | — |
| E[min/max] | **0.158** | **0.140** |
| P(some agent gets ZERO moves) | **0.299** | — |

The measured index is what the dice alone produce. And the last row is decisive: **about 30%
of episodes give at least one agent no moves at all.** Joint success is zero-tolerance across
every window, and an agent that never acts can never settle its own private nodes — no
partner can, because `Topology.allowed_edges` makes cross-private edges impossible. So under
random turns roughly a third of episodes are **structurally unwinnable for any policy**.

That is the whole effect. Round-robin does not flatter the learner; it removes a sampling-luck
confound that has nothing to do with coordination skill.

## 3. RETRACTED: the JCI round-robin warning

An earlier note argued that JCI §4.1's prohibition on deterministic relations between context
variables condemns round-robin, since whose turn it is is a deterministic function of the
round. **That was wrong, on three counts.**

1. **We never test conditional independence among context variables.**
   `cb.versionspace.estimated_reveal` runs a two-sample comparison on SYSTEM variables between
   row groups. JCI's pathology is a failure of faithfulness in the CONTEXT distribution — a
   property no part of our pipeline relies on.

2. **Actor identity is disclosed by design, so round-robin leaks nothing extra.**
   `ma/env.py` hands it over explicitly: `window.belief.observe_partner(actor, moved)`. The
   privacy claim was never about which AGENT acted; it is about which VARIABLE, and
   `cb/attribution.py` states it directly — "an outsider can never learn which of A's
   variables it was". Round-robin makes the actor predictable from the round index, but the
   actor was already broadcast.

3. **By JCI's own analysis round-robin is the FAVOURABLE case, not the pathological one.**
   One intervention per round is precisely their **diagonal design** (Table 1), and §3.4.4
   proves a diagonal design contains no conditional independences in the context distribution
   — which is exactly the condition that makes JCI Assumption 3 safe. The Sachs pathology is a
   context variable that is a deterministic FUNCTION OF OTHERS (`C_α = ¬(C_θ ∨ C_ι)`), tested
   in a design that does condition among contexts. Neither condition holds here.

## 4. The limitation that IS real, and belongs in the write-up

Round-robin hands every agent exactly `budget / n_agents` moves. That is a fairer protocol
than any real consortium would operate, and it is doing work for us:

> **Under random turns the learned margin over greedy falls from +0.140 / +0.127 to
> +0.006 / +0.093.** Part of the measured coordination advantage depends on a protocol that
> guarantees equal opportunity.

The honest framing for the thesis is that these are two different questions, and we answer
only the first:

- **Given equal opportunity, does the policy divide labour better than a myopic rule?**
  Yes — that is the coordination result, and round-robin is the right protocol for isolating it.
- **Can the policy cope when opportunity itself is unequal and stochastic?**
  Largely untested. The margin shrinks, and we cannot separate "the policy handles scarcity
  badly" from "30% of episodes were unwinnable" without a control that conditions on the
  realised move allocation.

**The control that would separate them, if there is time:** re-score the random-turn runs on
the subset of episodes where every agent drew at least one move. If the margin recovers to the
round-robin level, the drop is entirely the unwinnable episodes and the coordination claim is
untouched. If it does not, the policy genuinely depends on predictable turn-taking, and that
is a limitation to state outright. Eval-only on the existing `rndturn_s*.pt` checkpoints.

## 5. Caveats on these numbers

- **Two seeds per arm.** Enough to see a 2x effect, not enough to quote a margin.
- The `free_rider_index` name is misleading — it measures effort EVENNESS, and a low value
  under random turns is arithmetic, not free-riding. Rename it or define it wherever quoted.
- MI gate not yet reported for `rndturn`; `scale21` and `a08norm` both trained on entropy
  evidence (1.34/1.27 and 1.25/1.10 against a 1.946 maximum), `a08` did not (1.81/1.68/1.86).

## 6. One thing the same table shows, unrelated to turn order

`a08norm` (`normalise_returns=True`) reaches **0.665 / 0.695** — the best 8-agent numbers on
record, above `scale21`'s 0.620 / 0.687 with the hand-tuned constant. The principled fix beats
the magic number, under identical round-robin conditions.
