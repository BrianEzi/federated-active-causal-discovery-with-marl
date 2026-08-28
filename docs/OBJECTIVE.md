# The top-line objective

Written 28 Aug 2026. This is the document every other one answers to. If an experiment does
not move a claim in here, it is playground work — useful for building theory or iterating
fast, but not a result.

---

## 1. The problem, in the words of someone who would fund it

Several institutions each hold **different measurements about overlapping populations**, and
cannot pool raw data — regulation, patient consent, commercial sensitivity. The causal
structure they care about **crosses their boundaries**: the thing that explains a pattern at
site A is very often a variable held at site B. Experiments that would settle it are
expensive, slow, and sometimes invasive.

The objective is to let those institutions **jointly recover causal structure that no one of
them could recover alone**, spending a scarce experimental budget, with **no central
coordinator and no raw data leaving any site**.

Target application, and the regime every design choice should be judged against:
**biomedical consortia** — many sites, few affordable experiments, high noise, strict privacy.
Not a handful of richly instrumented partners.

## 2. The structural insight that makes this project distinctive

**A bidirected edge is usually someone else's variable.**

When a site sees two of its variables move together with no local explanation, standard
causal discovery writes a bidirected edge: "hidden common cause, unknown." In a federated
setting that hidden cause is rarely mysterious — it is typically held by a partner. Latent
confounding is not a nuisance here; it is the *signature of the federation itself*, and it is
informative about what the partners hold.

So the object of interest is not "is there a latent?" but **whose latent is it?**

If agent A intervenes on one of its own private variables and a partner's confounded pair
resolves, that pair's hidden common cause has been **located** — to A. The bidirected arc can
be redrawn as an edge into A's variable. Neither party saw the other's data. This is
**attribution**, and it is the centrepiece of the thesis.

Why it matters beyond graph-drawing: attribution is how an agent learns **which other
authority is influencing its own observations**. That is the thing a consortium member
actually wants to know, and it is actionable — it tells you who to call.

## 3. What an EXCEPTIONAL version of this project demonstrates

One sentence, and everything else is support:

> **N sites, each holding private variables plus a shared interface, under realistic noisy
> inference, sharing no raw data and using no central coordinator, collectively recover
> cross-boundary latent confounding, ATTRIBUTE each latent to the site that owns it, and
> reach a target confidence in materially fewer experiments than uncoordinated or
> centrally-scheduled baselines.**

Each clause is load-bearing and each is separately falsifiable:

| clause | why it is in the sentence | where it stands |
|---|---|---|
| *no raw data shared* | the premise of the setting | held by construction |
| *no central coordinator* | the supervisor's constraint, and the honest form of "federated" | held, but see the round-robin caveat below |
| *realistic noisy inference* | oracle evidence is a playground, not a claim | **the weakest link** — see §5 |
| *attribute each latent to its owner* | the novel contribution | built, learners not yet convincing |
| *materially fewer experiments* | what a consortium actually buys | **metric not yet framed this way** — see §4 |

## 4. Frame it as penalty minimisation, not success rate

We have been reporting **success rate at a fixed budget**. The application cares about the
inverse: **how many expensive experiments did you need**. A hospital director understands
"reached the answer in 11 experiments instead of 19"; nobody outside this repo understands
"0.627 joint identification rate."

The reframe: an episode accrues a **penalty per experiment run**, and the objective is to
minimise total penalty subject to reaching a confidence target. Consequences:

- `rounds_to_identification` — already implemented in `ma/env.py` and currently unused as a
  headline — becomes the primary metric.
- The comparison becomes **experiments saved against a baseline**, which is a ratio and is
  therefore scale-free across configurations. That sidesteps the normalisation objection that
  currently dogs the iso-budget figure.
- A policy can be well ahead on PACE while tying on a thresholded rate, so this may show an
  advantage the current metric hides.
- Different experiments can later carry different costs, which is true in the application and
  is where an active learner should beat a covering heuristic outright.

## 5. Why the toy setting exists, and what it cannot carry

Oracle evidence hands the belief the true ancestry, so identification reduces to a forced
set-cover problem. That is deliberate: it is fast to iterate on, and it gave us exact results
that a noisy engine could not — the required cover is computable in closed form, the
difference reward is exact rather than estimated, and counterfactuals are replays.

**But it cannot carry the top-line claim.** Under oracle evidence there is no statistical
inference, no effect size, no power, and therefore no value-of-information problem. It is the
furthest thing from a consortium deciding under noise. Any claim of the form "the policy
learns good experimental design" must be demonstrated under **sampled evidence**, where the
simulator's existing heterogeneity — edge weights 0.5–2.0 with random sign, per-node noise
scales 0.5–1.5, attenuation along chains — actually reaches the agent.

Known gap as of 28 Aug: the learned advantage over greedy **disappears** under sampled
evidence (0.874 vs 0.868, against 0.610 vs 0.470 under oracle). Closing or honestly bounding
that is the highest-value remaining work.

## 6. The novelty claim, verified against the literature

Checked 28 Aug 2026. State it in exactly this shape — the boundary matters, and overclaiming
here is the fastest way to lose an examiner.

**Established, cite rather than claim:**
- *Relative latent variables* — observed by some clients, not others — are a named concept in
  vertical federated causal discovery.
  [A Survey on Federated Causal Discovery and Inference](https://arxiv.org/html/2606.23741v1), §3.4.3:
  "clients often observe overlapping but non-identical variable sets, creating both *absolute*
  latent variables (unobserved by all) and *relative* latent variables (observed by some but
  not others)."
- Bidirected edges denote latent confounding (MAG/PAG literature).
- Interventions resolve confounding, including single-vertex interventions and soft
  interventions.
  [Characterization and Learning of Causal Graphs with Latent Variables from Soft Interventions](https://papers.nips.cc/paper/9581-characterization-and-learning-of-causal-graphs-with-latent-variables-from-soft-interventions.pdf)
- Vertical federated causal structure learning exists.
  [Horizontal and Vertical Federated Causal Structure Learning via Higher-order Cumulants](https://arxiv.org/abs/2507.06888)

**Not found, and therefore the contribution:**
1. **No method attributes a latent confounder to a particular client.** The 2026 survey,
   fetched directly, does not identify any existing method that determines which client
   possesses the unobserved confounder behind a bidirected edge. Existing work detects *that*
   confounding exists; none establishes *whose* it is.
2. **No decentralised active experiment-selection policy that does this** — i.e. agents that
   learn which intervention to run in order to establish latent ownership, without a
   coordinator.

Phrase the claim as: *existing work detects that a bidirected edge implies latent confounding,
and vertical-FL work names relative latents; we determine which participant owns that latent,
and learn a decentralised policy that selects interventions to establish it.*

## 7. Standing caveats that the top-line sentence must survive

- **Round-robin is itself a coordination mechanism.** `active_agent` is a deterministic
  function of the round and `budget_left` encodes the round exactly, so a policy can compute
  whose turn it is. A global deterministic schedule is arguably the centralised element we
  claim not to have. Measured 28 Aug: random turn order costs the learned policy and greedy
  the same (−0.247 each), so the advantage is not schedule-exploitation — but the framing
  point stands and should be stated, not hidden.
- **A two-line positional convention beats the learned policy at eight agents**
  (0.880 vs 0.627, duplicating 23× less shared work). Coordination is learned at ≤4 agents and
  is not learned at 8. Bound the claim.
- **Oracle evidence is not the application.** See §5.
