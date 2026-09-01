# Attribution is bounded by IDENTIFIABILITY, not by scale

1 Sep 2026, rewritten 09:20 after the matched-budget control refuted the first version of
this file. What that version claimed, and what actually holds, are both below -- the
correction is the substance, so it is not hidden in a footnote.

---

## 1. The result

Recovery rate of a true latent group by the number of window nodes it explains, against the
number of PARTNERS. 200 episodes per cell, k=12, sigma=0.50, oracle evidence, component
backend, **rounds per agent held constant at 15** so coverage does not confound partner count.
**Zero misattributions in every cell.**

| partners | 2 children (1 pair) | 3 children | 4 children | overall |
|---:|---:|---:|---:|---:|
| **1** | 100% | **64%** | **39%** | 0.488 |
| **2** | 80% | **0%** | **0%** | 0.284 |
| **3** | 77% | **0%** | **0%** | 0.330 |
| **7** | 72% | **0%** | **0%** | 0.424 |

**Two facts, and the first is the thesis-relevant one.**

* **A group explaining two or more pairs is unrecoverable from two partners onward, at ANY
  budget.** 64% and 39% at one partner, zero at two, and zero at seven even when the budget
  is doubled to keep coverage matched. This is an identifiability barrier, not a resource one.
* **Single-pair groups decline only gently** with partner count: 100 -> 80 -> 77 -> 72%.
  Ownership among more candidates costs something, but not much.

## 2. Why

A pair's ownership hypothesis space is the set of non-empty owner subsets, `2^n - 1`. With ONE
partner it is a singleton -- ownership is forced, `attributions_for` returns exactly one
canonical candidate for any pair set, and everything the belief has settled resolves. That is
why the one-partner column is limited only by COVERAGE (100/64/39% is the chance that all 1, 3
or 6 pairs of the group get settled in budget), not by identifiability.

From two partners onward, separating a clique from several smaller latents requires a PARTIAL
response -- some of the group's pairs moving while others do not. That requires the owner to
probe its private variables ONE AT A TIME. No policy here does, so partner responses are
always total, atomicity never fires, and multi-pair groups go to zero and stay there however
much budget is spent.

Single-pair groups have no clique ambiguity, so ownership evidence alone settles them.

## 3. THE CORRECTION, and how it was caught

The first version of this file claimed the collapse was **exponential hypothesis-space
growth** -- `2^n - 1` owner sets (127 at seven partners) against a budget split more ways --
and reported single-pair recovery falling to **5%** at seven partners.

**That was budget starvation, not the hypothesis space.** The partner sweep held the budget
fixed at 60 rounds for every partner count, so eight agents got 7.5 rounds each against four
agents' 15. Holding rounds-per-agent constant instead:

| partners | budget | single-pair recovery | overall |
|---:|---:|---:|---:|
| 7 | 60 | **5%** | 0.028 |
| 7 | **120** | **72%** | **0.424** |

Attribution recovers almost completely. The exponential-space story predicted it would not.

The control was queued at 01:00, before the first version was written, precisely because
"exponential space" and "thin evidence" both predicted the collapse and the sweep could not
separate them. It should have run before the claim was published, not after.

**What this changes:** the barrier is identifiability, not scale. More partners costs a little
(80 -> 72% on single-pair groups); needing a partial response costs everything.

## 3b. Coverage is the dominant term, and it is a STEP FUNCTION

The correction in section 3 raised an obvious question it did not answer: if the collapse was
coverage, how much coverage is enough? Swept at k=12, n=4, 200 episodes per cell:

| budget | turns per agent | window positions reached | single-pair recovery | overall |
|---:|---:|---:|---:|---:|
| 30 | 7.5 | ~7 of 12 | **5%** | 0.020 |
| 60 | 15 | **12 of 12** | **77%** | 0.330 |
| 120 | 30 | 12 of 12 | 77% | 0.330 |
| 240 | 60 | 12 of 12 | 77% | 0.330 |

**Attribution requires FULL window coverage and gains nothing whatever beyond it.** The 60,
120 and 240 cells return not merely similar numbers but IDENTICAL ones -- 349 of 1056 in every
case -- which is the signature of an experiment that has stopped changing.

That identity is diagnostic rather than suspicious. The driver sweeps window positions
round-robin, so at k=12 with four agents a budget of 60 gives each agent 15 turns for 12
positions: full coverage plus three wasted repeats. Budget 240 gives 60 turns -- the same 12
positions plus 48 repeats -- and **under oracle evidence a repeat reveals nothing**, which is
the same fact that makes oracle-trained policies fail to transfer to sampled data. The extra
budget is provably inert.

**This single term explains every earlier observation.** Recovery tracks turns-per-agent
against window size, and nothing else:

| observation | turns/agent | positions | recovery |
|---|---:|---:|---:|
| n=4, budget 60 | 15 | 12 | 77% |
| n=8, budget 60 | 7.5 | 12 | **5%** |
| n=8, budget 120 | 15 | 12 | **72%** |
| n=4, budget 30 | 7.5 | 12 | **5%** |
| k=30, budget 100 | 25 | 30 | **30%** |

The two 5% cells arrive from opposite directions -- too many partners, and too small a budget
-- at the same turns-per-agent. The partner count was never the mechanism.

## 3c. The three terms, separated

| term | effect | evidence |
|---|---|---|
| **Coverage** | step function: below full window coverage, collapse; above it, nothing | budget sweep, 4 cells |
| **Partner count** | gentle decline once coverage is matched: 100 / 80 / 77 / **72%** at 1 / 2 / 3 / 7 partners | matched-budget control |
| **Group size** | absolute: **0%** for any group explaining 2+ pairs, from 2 partners, at any budget | every cell measured |

Only the third is an identifiability barrier. The first is a resource question with a sharp
threshold, and the second is mild. **The 23% of single-pair groups unresolved at full coverage
is the genuine ownership-ambiguity floor** -- three partners, and no budget removes it.

## 4. A closed-form predictor, which survived the correction

    attribution  ~=  P(resolve | single-pair group)  x  share of groups that are single-pair

Groups explaining two or more pairs contribute nothing beyond one partner. Against the
matched-budget control (`scripts/attr_model.py`):

| partners | P(resolve\|1-pair) | share 1-pair | predicted | measured | residual |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.798 | 0.389 | 0.310 | 0.284 | -0.026 |
| 3 | 0.765 | 0.477 | 0.365 | 0.330 | -0.034 |
| 7 | 0.718 | 0.639 | 0.459 | 0.424 | -0.035 |

and against the fixed-budget sweep, which also varies sigma and window size:

| cell | P(resolve\|1-pair) | share | predicted | measured | residual |
|---|---:|---:|---:|---:|---:|
| k12 s.25 n4 | 0.941 | 0.845 | 0.795 | 0.755 | -0.041 |
| k12 s.75 n4 | 0.782 | 0.335 | 0.262 | 0.267 | +0.005 |
| k20 s.50 n4 | 0.667 | 0.321 | 0.214 | 0.196 | -0.018 |

**Largest residual 0.041**, over measured values from 0.196 to 0.755, varying partner count,
shared fraction, window size and budget.

**With adequate coverage the first factor is nearly constant at ~0.76**, so the law reduces to

    attribution  ~=  0.76  x  (share of latent groups that explain exactly one pair)

and that share is computable from the topology and graph model with **no simulation at all**.

**The one-partner cell is under-predicted by 0.263, by design** -- the only regime where
multi-pair groups contribute. A model that fails exactly where its derivation says it must is
better evidence than one that fits everywhere.

## 4b. Reach -- the engine runs to k=50

Component backend, 4 agents, 30 episodes, zero misattributions throughout:

| k | pairs in window | right | wrong | scope | s/episode |
|---:|---:|---:|---:|---:|---:|
| 30 | 435 | 21 | **0** | 0.60 | 5.3 |
| 40 | 780 | 33 | **0** | 0.68 | 5.1 |
| 50 | 1225 | 27 | **0** | 0.56 | 9.4 |

The enumerated-ownership backend cannot be run at these sizes -- the attribution space alone
is 8.4e10 hypotheses at k=20. Cost grows with the SETTLED PAIR COUNT rather than with k, which
is why k=40 is no dearer than k=30: both settle around 7-12 pairs per window at this budget.

**Recovery falls with window size** (77% at k=12 to 30% at k=30) and that is the coverage term
of section 3b, not a limit of the engine: a fixed budget buys fewer turns per position as the
window grows.

## 5. What it explains

**D7's three-seed negative.** A policy trained ON the attribution reward scores 0.400 / 0.355
/ 0.205 against greedy's 0.945 / 0.955 / 0.935. Only single-pair groups are recoverable and
they come free with structural coverage, so the attribution term is a constant plus noise --
no gradient, because no achievable behaviour changes the outcome. **The reward cannot teach
what the evidence cannot support.**

**`greedy_attribution`'s 7% private-probing rate.** An agent optimising its own attribution
belief correctly declines to probe privately, because that evidence goes to PARTNERS.

## 6. The claim this licenses

> Federated attribution of latent confounders is limited by identifiability rather than by the
> number of sites. Groups explaining a single confounded pair are recoverable at ~76%
> regardless of site count once the budget covers the window; groups explaining a clique are
> unrecoverable from two sites onward at any budget, because separating them requires a
> partner to probe its private variables one at a time -- an experiment with no payoff for the
> partner performing it. The recoverable share is therefore predictable in advance from the
> topology alone.

## 7. Caveats

* Partner sweep at one window size (k=12) and one shared fraction (sigma=0.50); k=20, sigma
  variants measured separately at fixed budget. **k=30 not measured** -- the control pre-empted
  it and it is queued.
* Oracle evidence only.
* "No policy probes one variable at a time" is measured over the policies in this project, not
  proved impossible. Whether such an action is even definable under the privacy constraint is
  open, and is the natural future-work question.

## 8. Reproducing

```bash
# partner sweep, fixed budget
.venv/bin/python scripts/attr_ceiling.py --cells 12:0.5:2:60,12:0.5:3:60,12:0.5:4:60,12:0.5:8:60 \
  --episodes 200 --out results/attr_ceiling.json
# the control -- rounds per agent held at 15
.venv/bin/python scripts/attr_ceiling.py --cells 12:0.5:2:30,12:0.5:3:45,12:0.5:4:60,12:0.5:8:120 \
  --episodes 200 --out results/attr_ceiling_matched_budget.json
.venv/bin/python scripts/attr_model.py --results results/attr_ceiling_matched_budget.json
```
