# What it would actually take to make this differentially private

Written 31 Aug 2026 for the future-work section. Not implemented, and deliberately so — but
the shape of the problem is specific enough to be worth stating precisely, because the
obvious answer ("add DP-SGD to the federated training") addresses the **least** important of
three leakage channels in this system.

## The headline: three channels, and classical DP-FL covers one

| # | channel | what an honest adversary learns | does DP-FedAvg help? |
|---|---|---|---|
| A | the model updates | what site *i* contributed to the shared policy | **yes — this is what it is for** |
| B | the environment's disclosure signals | per round, whether each partner intervened on a PRIVATE node, a SHARED node, or passed; which shared node; and one regime bit | **no** — these travel in the clear, outside the weights |
| C | the learned structure, and attribution in particular | **which partner owns a specific latent confounder** | **no** — and attribution exists in order to reveal this |

A thesis that added DP to the training and stopped there would be doing privacy theatre. B
and C are where the private variables actually leak, and they leak *by design*, because the
coordination signal is the thing that makes the task solvable.

---

## Channel A — the model updates. The tractable part.

**What is already in place.** `_fedavg_update` averages weights size-weighted, nothing but
weights leaves a site, and the server applies an adaptive optimiser (FedAdam/FedYogi) to the
averaged delta. That last part matters: **DP-FedAdam is Reddi et al.'s framework plus noise**,
so the structure the DP variant needs already exists rather than having to be retrofitted.

**What is missing, concretely:**

1. **Per-client update clipping.** We clip *gradients* per step (`clip_grad_norm_(..., 0.5)`),
   which is not the same thing. DP requires clipping the whole client delta
   `Δ_i = w_i − w_server` to a fixed norm `S`, so that one site's influence on the aggregate
   is bounded. About ten lines in `_fedavg_update`.
2. **Gaussian noise on the aggregate**, `N(0, σ²S²/n)` added to the averaged delta before the
   server step — i.e. inside `_server_step`, which already exists.
3. **A privacy accountant.** RDP or PLD (Opacus, or `dp_accounting`). Non-trivial here because
   we run **250 communication rounds**; ε composes over all of them, and without
   subsampling amplification the budget is spent quickly. With 4 sites there is no meaningful
   client subsampling — the amplification that makes cross-device FL tolerable is unavailable
   at cross-**silo** scale, which is our regime.
4. **A decision about the unit of privacy** — see below, and this is the part that is not
   engineering.

The reference is McMahan et al. 2018 (*Learning Differentially Private Recurrent Language
Models*) for DP-FedAvg, and Reddi et al. 2021 for the adaptive server this would sit on.

**Expected utility cost.** Our per-site batches are small — 144 rows per client per round at
the baseline cell with turn-aware credit on — and DP noise scales against the clipping norm
rather than the batch, so the signal-to-noise ratio here is poor. Since the measured gap
between the *best* federated arm and the worst is already about 0.07 in success and 0.45 in
MI, DP noise at any useful ε would plausibly swamp the entire effect being studied. Worth
saying: **this system is small enough that DP would be measurable mainly as damage.**

---

## Channel B — the disclosure signals. The hard part, and it is not noise.

Every round, each agent's observation carries (`ma/env.py::observation`):

- `_signal_onehot` — for each partner, a trit: intervened on a **private** node, on a
  **shared** node, or passed. Governed by `disclose_signals` (default **True**).
- `disclosed[agent]` — which **shared** nodes each partner targeted (`disclose_shared_targets`,
  default True). This one is defensible on its own terms: shared columns are visible to both
  parties anyway, so a partner could infer the target from the data.
- `regime_bit` — one bit: "something you cannot see was intervened on" (`disclose_regime`).
- `_partner_counts` — cumulative per-partner intervention counts (`observe_partner_counts`).

**The private-vs-shared trit is the leak.** Over a budget of B rounds it tells every partner,
per round, whether site *i* was working on something they cannot see. That is a direct
statement about private activity, and no amount of noise on the model weights touches it.

**Why it cannot simply be removed.** Measured 22 Aug: with `disclose_regime` off, **10 of 10
seeds collapse to the pass-only floor** (0.007), paired +0.540, CI [+0.515, +0.565]. And read
carefully — the *random* baseline falls too (0.380 → 0.040), and random reads no observations
at all. So the bit changes **what is identifiable**, not merely what the policy can condition
on. The defensible claim is "unsolvable without it", not "the agent exploits the channel".

**What a DP treatment would look like here.** Not central DP but **local** DP on the signal:
randomised response on the trit, with flip probability set by the per-round ε, composed over
the budget. This is a genuinely different mechanism from DP-FedAvg and would need its own
utility study — and the measurement above says the utility floor is a cliff rather than a
slope, so the interesting question is *where* the cliff is as a function of ε.

That is a self-contained piece of work and probably the most publishable part of this section.

---

## Channel C — attribution, which is in direct tension with privacy

`cb/attribution.py` exists to answer: **which partner owns the latent that confounds this
pair of my variables?** Its evidence is `observe_partner(actor, moved)` — when a partner acts
privately and a pair in my window moves, that partner owns a confounder of that pair.

This is not an incidental leak. **It is the mechanism.** A site-level DP guarantee over
latent variables and a working attribution mechanism are mutually exclusive: attribution
succeeds exactly to the degree that a partner's private structure is inferable.

Two honest positions, and the thesis should pick one explicitly:

1. **Attribution is out of scope for privacy.** Confounder ownership is a legitimate shared
   output of the consortium — the sites *agree* to reveal it, in the way clinical consortia
   agree to publish which cohort a signal came from. Privacy then covers records and model
   updates, not the causal structure itself, and this should be stated as the trust model
   rather than left implicit.
2. **Attribution is the privacy cost**, quantified. The V-curve in `intervene_scale` already
   gives a dial: attribution detection runs from **22.0% at scale 1.0 to 92.5% at 2.0**. That
   is an operating curve between coordination quality and disclosure, and it could be
   presented as one.

---

## The question underneath all three: DP may be the wrong primitive

Differential privacy protects **records** — rows, individuals. This setting is **vertically**
partitioned: every site holds *all* rows and its own *columns*. The privacy concern is
therefore about **attributes**, not records: "do not reveal which variables I hold, their
values, or their causal relationships to yours." Standard (ε, δ)-DP does not express that.

The vertical-FL literature reaches for **secure aggregation, homomorphic encryption and
secret sharing** rather than DP for exactly this reason, and federated causal discovery
inherits the mismatch. A rigorous treatment would either:

- adopt record-level DP and be explicit that it protects individuals **within** a site while
  saying nothing about the site's columns — which is a real guarantee, just not the one the
  federated framing implies; or
- move to secure aggregation for the weights and treat DP as the defence against
  *inference* from the released structure, which is closer to the actual threat model.

There is also a specific technical hazard worth a sentence: constraint-based discovery makes
**discrete** decisions from test statistics. Noise on a conditional-independence test flips an
edge, and constraint-based orientation propagates that error through the rest of the graph.
DP causal discovery (PrivPC, EM-PC and successors) exists precisely because naive noise
addition degrades far worse here than in a continuous-output setting. That literature is the
right anchor for this section.

---

## One caveat that keeps the section honest

Our SCMs are **synthetic and resampled every episode**. There is no fixed dataset to protect,
so a DP implementation here would demonstrate a *mechanism* and its *utility cost* — it would
not protect anything real. Any evaluation would have to be framed that way: an ε–utility
curve on a simulator, not a privacy guarantee over data. Making that claim carefully is
itself part of the work.

## Summary of what would have to be built

| item | effort | value |
|---|---|---|
| per-client clipping + Gaussian noise in `_server_step` | ~1 day | the standard, and it fits the existing FedAdam structure |
| RDP/PLD accountant over 250 rounds, no subsampling amplification | ~1 day | necessary before any ε can be quoted |
| ε–utility curve against the arms already measured | ~1 day compute | the actual result |
| local DP (randomised response) on the disclosure trit | ~2 days | **the novel part** — and where the cliff is, is the question |
| a stated trust model for attribution | writing | required either way; the tension is real and unavoidable |
