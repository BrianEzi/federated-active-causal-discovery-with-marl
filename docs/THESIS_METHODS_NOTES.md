# Thesis methods notes

Method descriptions written at the level of precision a methods section needs, staged here as
they're built rather than written from memory at the end. Each entry names the exact
implementation (file, function) so a later pass can check the prose against the code rather
than trust it.

---

## Return normalisation for multi-agent PPO with a shared trunk

**Where implemented:** `ma/policy.py`, `IndependentPPO._return_scale` and `_advantages`.
Flag: `PPOConfig.normalise_returns` (default `False`, verified byte-identical to the
unflagged path when off — `tests/ma/test_normalise_returns.py`).

**Motivation.** In PPO the policy loss uses advantages, which are standardised per batch
before use. The critic's loss, `MSE(values, returns)`, is not — it trains directly on the
scale of the raw discounted return. In a setting where reward magnitude grows with a
structural parameter (here, agent count: per-agent return grew from 1.66 to 11.86 across
two to eight agents), the critic's squared-error loss grows quadratically in that same
parameter, while the policy-gradient term stays O(1). With agents sharing a trunk, the
value loss increasingly dominates the shared gradient at higher agent counts, and the
policy signal is drowned out. A previous fix used a hand-tuned constant
(`MAConfig.reward_scale = 0.214`, chosen empirically); this is the same mechanism made
principled and self-tuning.

**Method.** Let $r_t$ be the reward at timestep $t$ within a training batch, $\gamma$ the
discount factor, and $\text{done}_t \in \{0,1\}$ the episode-termination flag. Define the
per-timestep discounted return

$$G_t = r_t + \gamma\, G_{t+1} \,(1 - \text{done}_t)$$

computed backward through the batch. From the batch's $\{G_t\}$, compute the sample mean
and variance, then merge them into a running estimate maintained across all training
batches and shared across all agents (since the scale is a property of the task, not of
any individual agent's realised luck), using Chan's parallel-variance formula for
combining $(\text{mean}, \text{variance}, n)$ triples exactly:

$$
\delta = \bar{G}_{\text{batch}} - \bar{G}_{\text{running}}, \qquad
\bar{G}_{\text{running}}' = \bar{G}_{\text{running}} + \delta \cdot \frac{n_{\text{batch}}}{n_{\text{total}}}
$$

$$
M_2' = M_{2,\text{running}} + M_{2,\text{batch}} + \delta^2 \cdot \frac{n_{\text{running}}\, n_{\text{batch}}}{n_{\text{total}}}, \qquad
\sigma^2_{\text{running}}{}' = \frac{M_2'}{n_{\text{total}}}
$$

The reward stream fed into GAE is then divided by $\sigma_{\text{running}} =
\sqrt{\sigma^2_{\text{running}}}$, floored at $10^{-6}$ (not clamped to 1) so that a batch of
all-zero rewards — the normal state before any episode has reached its terminal signal —
is left at zero rather than divided by near-zero noise. Scaling is applied **before** the
GAE backward recursion, not to the returns after the fact: the critic's bootstrap term
$V(s_{t+1})$ is itself a prediction in already-scaled units (because the critic was trained
on scaled targets), and adding an unscaled reward to a scaled bootstrap would introduce a
unit mismatch. Advantages and returns proceed through the standard GAE recursion
unchanged once the reward stream is rescaled.

**What is deliberately not done.** The transformation is a pure positive rescaling, never a
recentring — no reward mean is subtracted. Under a finite-horizon episode carrying a
terminal success bonus, subtracting a mean would introduce a per-step reward for merely
surviving, which changes the location of the optimum. A positive multiplicative rescaling
changes no ranking: it cannot alter the argmax of the policy nor the ordering of the
advantages, only the numerical scale the critic is regressing to.

**Empirical result, this codebase.** At 8 agents (federated causal-discovery environment,
factored belief backend), normalisation raised joint identification success from 0.100
(untrained — mean mutual information between observation and action, $I(S;A)/H$, at 0.033,
below the 0.15 floor taken to indicate a policy that has not conditioned on its
observation at all) to 0.665–0.695 across two seeds, exceeding the hand-tuned constant's
0.620–0.687. At 6 agents it moved the policy from *losing* to a fair heuristic baseline by
0.440 to *beating* it by 0.200, and did so without the roughly 30-point cost the hand-tuned
constant's sibling technique (a difference-reward formulation) incurred at low agent
counts — i.e., this is a rescaling that recovers the failing regime without trading away
the regime that already worked. All arms independently confirmed trained via the mutual
information floor before being compared. Full numbers, seeds, and paired statistics:
`docs/logs/SA_EXPERIMENT_LOG.md`, 2026-08-28 entries; source note:
`docs/NOTE_RETURN_NORMALISATION_2026_08_28.md`.

**Caveat for the write-up.** The running estimate mixes early-training statistics (largely
zero, before the policy discovers reward) with late-training statistics (larger, as the
policy starts succeeding), so the effective divisor drifts over the course of a single
training run rather than being fixed. This was not compared against alternatives (e.g., a
decayed/windowed running estimate, or per-agent rather than pooled statistics) — the
pooled, undecayed choice was made to avoid introducing a further tunable hyperparameter,
and worked; whether a different pooling choice would work better is untested.
