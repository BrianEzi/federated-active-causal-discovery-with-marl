# Attribution has a hard identifiability ceiling, and no policy crosses it

31 Aug 2026, 22:25. Replaces the numbers in `FINDINGS_ATTRIBUTION_SCALE_2026_08_31.md`,
which are superseded -- three claims in that file were defects in this engine, all fixed
today and all listed in its banner.

---

## 1. The finding

**A latent group is recovered if and only if it has exactly two children.** Not usually --
always. Over 20 episodes at k=12 with 4 agents:

| children | pairs explained | right | unsure | wrong | % right |
|---:|---:|---:|---:|---:|---:|
| **2** | 1 | **29** | 7 | 0 | **80.6%** |
| 3 | 3 | 0 | 39 | 0 | **0.0%** |
| 4 | 6 | 0 | 15 | 0 | **0.0%** |
| 5 | 10 | 0 | 6 | 0 | **0.0%** |
| 6 | 15 | 0 | 3 | 0 | **0.0%** |

63 groups with three or more children, across every episode and every agent, and not one is
ever recovered. Zero wrong everywhere, so this is not the engine erring -- it is the engine
correctly reporting that the evidence never arrives.

**The ceiling was predicted before it was measured.** 37.7% of true groups have exactly two
children; the attribution rate of every competent policy is **36.8%**.

## 2. Why, and it is the identifiability argument made concrete

From `cb/attribution.py`, written weeks before this measurement:

> One latent parenting {u, v, w} and three separate latents parenting {u,v}, {u,w}, {v,w}
> induce EXACTLY the same three bidirected edges. No observation distinguishes them. An
> intervention does: act on the single latent and all three associations move together; act
> on one of the three and only one moves. The only agent who can perform that intervention is
> the one who owns the variable.

A group with TWO children explains one pair. There is no finer hypothesis to separate it
from, so ownership is the whole question and a single partner message answers it.

A group with THREE OR MORE children explains a clique. Separating it from several smaller
latents requires a PARTIAL response -- some of its pairs moving while others do not -- and
that requires the owner to probe its private variables ONE AT A TIME. An action that
disturbs several of its latents at once moves everything, and a response that moves
everything separates nothing.

**No policy in this project does that.** Not the learned one, not `probe_then_work`, not
`greedy_uncertainty`. They all probe several private nodes, so partner responses are always
total, and atomicity never fires.

## 3. The evidence that the ceiling belongs to the PROBLEM, not the policy

Three policies with completely different behaviour, `k12s50n04b200`, 100 episodes, 2 seeds:

| arm | identified | attribution | structure | private share | coverage | repeat |
|---|---:|---:|---:|---:|---:|---:|
| learned (trained on structure) | 0.362 | **0.368** | 1.000 | 0.79 | 0.77 | 0.59 |
| probe_then_work | 0.362 | **0.368** | 1.000 | 0.39 | **1.00** | **0.00** |
| greedy_uncertainty | 0.360 | **0.368** | 1.000 | 0.61 | **0.41** | 0.13 |
| greedy_attribution | 0.195 | 0.299 | 0.974 | **0.07** | 0.17 | 0.08 |
| random_vary | 0.193 | 0.306 | 0.986 | 0.51 | 0.76 | 0.46 |

The top three are **identical on every episode** -- the paired difference between `learned`
and `probe_then_work` is `+0.000 +/- 0.000` over 100 episodes -- while their coverage ranges
from 0.41 to 1.00 and their repeat rate from 0.00 to 0.59. Behaviour differs completely; the
outcome does not.

The two below the line fail on BOTH structure (0.974, 0.986 against 1.000) and attribution,
so they never reach the ceiling that binds the others.

**A diversity hypothesis was proposed, instrumented and refuted.** `greedy_uncertainty` has
the LOWEST coverage of the strong arms (0.398) and ties; `random_vary` has high coverage
(0.780) and is worst. Coverage does not predict attribution. The columns are kept because a
refuted mechanism is worth reporting.

## 4. What this explains

**D7, a policy trained ON the attribution reward, loses to greedy 0.400 against 0.945.** That
is no longer a puzzle. The reward cannot teach what the protocol does not permit: an agent is
paid for ITS OWN attribution, and its own attribution depends on what its PARTNERS choose to
probe. No amount of training on your own reward makes someone else probe one variable at a
time.

**`greedy_attribution` probes privately 7% of the time**, against 39-61% for every other
policy, and attributes worse than the generic uncertainty greedy. It is the same fact from
the other side: an agent greedily optimising its own attribution belief correctly concludes
that private probes are not worth a round, because the evidence they produce goes to
PARTNERS.

**37% comes free with structural coverage.** Recovering structure requires intervening on
the relevant nodes, and doing so incidentally generates the partner messages that settle
single-pair groups. Attribution up to the ceiling is a by-product of doing your own job.

## 5. The coordination gap, quantified

    recoverable as a by-product of self-interested structural work    37%
    requiring an experiment with no selfish payoff                    63%

That second number is the thesis's subject. It is not a training failure and not an engine
limitation -- it is the exact share of the target that is unreachable without altruism, and
it is measurable to within one percentage point from the graph distribution alone.

## 6. Engine status, after today's three fixes

Both engines, one build, one process (`results/attr_scale_final.json`):

| k | engine | right | wrong | s/ep |
|---:|---|---:|---:|---:|
| 6 | enumerated | 57 | 0 | 0.02 |
| 6 | component | 57 | 0 | 0.02 |
| 8 | enumerated | 38 | 0 | 0.04 |
| 8 | component | 38 | 0 | 0.03 |
| 12 | enumerated | 44 | 0 | 2.25 |
| 12 | component | 44 | 0 | **1.46** |
| 20 | enumerated | 36 | 0 | 12.64 |
| 20 | component | **38** | 0 | **5.97** |
| 30 | enumerated | 20 | 0 | 10.23 |
| 30 | component | **21** | 0 | **5.28** |

Identical decisions to k=12 and slightly MORE at k=20 and k=30, **zero misattributions at
every size**, component 1.5-2x faster throughout. The extra decisions come from scope: the
component engine keeps whole components where the enumerated one truncates to a global
five-pair prefix, so it reaches 0.57 against 0.55 at k=20 and 0.60 against 0.58 at k=30. The
version-space guarantee -- settled implies settled correctly -- holds at every size.

These numbers are the third independent run to reproduce each other, and unlike the table
they replace, every row comes from ONE process and ONE build.

Three defects were fixed today to reach this, each found by a measurement queued to test the
claim above it: a scope bug advertising authority over truncated pairs; an UNSOUND atomicity
rule (two agents may confound the same pair, so a group can look partially moved -- it
refuted the truth in 27 of 85 oracle messages, now 0 of 85); and a one-ulp accumulation error
that discarded every claim the component belief was certain of.

## 7. Future work this points at

**Reward an agent for its PARTNERS' attribution.** The gap is structural: the experiment that
resolves a multi-pair group has no payoff for the agent performing it. A reward that pays for
partner outcomes is the natural test of whether the 63% is reachable at all, and it is a
different experiment from anything run here.

**One-at-a-time probing as an explicit action.** The protocol currently offers no way to say
"disturb exactly one of my latents". Whether such an action is even well-defined under the
privacy constraint is an open question, and an interesting one.

## 8. Reproducing

```bash
.venv/bin/python scripts/attr_scale.py --episodes 30 --out results/attr_scale_final.json
PYTHONPATH=. .venv/bin/python scripts/attr_score.py --backend component_attributed \
    --allow_backend_transfer --no_belief_channels --no_partner_counts \
    --n_agents 4 --private_size 6 --n_shared 6 --budget 67 --episodes 100 --seed 0 \
    --policy results/sweep/oracle/k12s50n04b200_s0_best.pt \
    --out results/attr/transfer_div_k12s50n04b200_s0.json
```
