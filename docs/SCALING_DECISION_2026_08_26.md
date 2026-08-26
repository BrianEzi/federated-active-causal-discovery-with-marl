# The regime bit is the scaling blocker — three ways forward

**Status: DECISION NEEDED. Nothing here is implemented.** Written 2026-08-26, ~03:30, after
the engine work removed every compute wall and the ladder still produced a floor.

## 1. The measurement

Rung 0 (2 agents, 1 private each, 3 shared), 250 episodes, budget 10:

| `disclose_regime` | prior | random_clamp | greedy | pass |
|---|---|---|---|---|
| **True** | 0.5 | **0.356** | 0.212 | 0.008 |
| **True** | 0.6437 | **0.328** | 0.244 | 0.012 |
| False | 0.5 | 0.052 | 0.040 | 0.008 |
| False | 0.6437 | 0.020 | 0.016 | 0.012 |

*(banked `results/ma_fixed/tb_clamp_s0.json`: 0.387 / 0.240 / 0.007 — reproduced by row 1.)*

Rung 1 (3 agents), no bit possible, 150 episodes: **learned 0.000, greedy 0.007.** Training
saw its first success at episode 76 and the solve rate declined from 0.016 to 0.008 over
1500 episodes. It is not under-trained; there is nothing to learn toward.

**One factor explains almost all of it.** The prior is irrelevant when the bit is on
(0.356 vs 0.328); the budget effect is small. The bit is worth 7–16x.

## 2. Why the bit cannot simply be switched on

`disclose_regime` publishes `clean = n_clamped / len(hidden)` — how many of an agent's hidden
nodes were clamped this round. `_assignment_weights` mixes the clean and dirty score tables
with weight `q = 1 - clean`, **using the same weight for every confounding edge under test**.

At two agents with one private node each, `len(hidden) == 1`, so the fraction is 0 or 1 and
is an exact identity. From three agents it is a genuine fraction and the mixture is unsound:
a hypothesis about a specific latent is scored identically whether that latent or a different
one was severed. The guard in `ma/env.py` refuses that combination, correctly.

So the validated two-agent result rests on a mechanism that is exact only at two agents.

**A tempting fix that does NOT work.** Under turn-taking the acting agent is public and
`disclose_signals` already broadcasts its action category, so at one private node per agent
every other agent can *derive* which hidden node was clamped, with no new disclosure. True,
free — and insufficient. An assignment says `(u,v)` is confounded but never names the latent
responsible, so knowing `h_j` was clamped does not say whether *that pair's* edge was severed.
One of `m` clamped still gives `f = 1/m`, the unsound middle.

## 3. The three options

### A — Attribute confounded pairs to latents *(principled; a real design change)*

A pair's state becomes `absent` or `(orientation, owning hidden node)` — `1 + 2m` states
rather than 3. Per-row cleanliness is then exact and the mixture is never entered.

**Affordable only because of the screen.** `(1+2m)^pairs` is `9^10` at five agents and cannot
be enumerated, but the screen never enumerates: it costs `1 + 2m*pairs` partition calls (81
rather than 21 at `m=4`) and keeps 64 assignments either way.

- *For:* fixes the mechanism at every rung; no new disclosure beyond what turn order already
  reveals; reuses machinery that now exists.
- *Against:* changes what a hypothesis IS, so every joint_conf consumer needs review, and the
  identification criterion has to be restated over the richer space. Metric reachability must
  be re-tested — a criterion crediting only unattributed pairs would be unearnable.
- *Cost:* about a day, most of it in the consumers and the tests, not the DP.

### B — Ring visibility, so every agent has exactly one hidden node *(cheap; changes the setup)*

`Topology.visibility` already expresses overlapping visibility. A ring in which agent `i`
sees everything except agent `i+1`'s private node gives `widest_hidden == 1` at any number of
agents, so the bit stays exact and the guard is satisfied untouched.

- *For:* runnable tonight, no engine change, scales to any agent count.
- *Against:* it is a different federation model. "Each agent hides its private nodes from all
  others" becomes "from exactly one other", which weakens the privacy claim considerably and
  is not what the problem statement describes. Reviewers will ask, and the answer is thin.

### C — Run the no-bit ladder as the floor, and let disclosure be the fix *(no new code)*

This is what is queued now. It measures how far scale gets with no regime information at all,
which is exactly the `none` arm of `DISCLOSURE_SPEC.md` §10.

- *For:* honest, already running, and it makes the disclosure result legible — the projection
  is there to carry precisely this information in a form that scales.
- *Against:* every rung reports near-zero, so the ladder shows no agent beating any baseline.
  As a thesis chapter it is "the method does not scale without disclosure", which is a real
  finding but not the one the ladder was built to produce.

## 4. Recommendation

**A, with C already running underneath it.** C costs nothing more — it is queued and will
land — and it is the control that makes A's result meaningful. A is the only option that
reaches "agents beat baselines at 3 and 5 agents" without weakening the federation claim.

B is the fallback if A runs over: it produces numbers, but at a cost to the privacy argument
that the write-up would have to carry.

The freeze is 31 August. A is roughly a day, which fits, but only if it starts immediately
and only if the metric-reachability test is written FIRST — a criterion that cannot be earned
over the attributed space would waste the whole run, and that failure mode has already cost
this project once (529 tests green on an unearnable metric).
