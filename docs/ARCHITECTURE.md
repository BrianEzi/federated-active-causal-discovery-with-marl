# Architecture

How an episode runs, and where each component sits.

## The setting

One structural causal model is partitioned vertically among `K` agents. Each agent holds a
private block of variables and shares an interface with the others; its **window** is its
private block plus the shared interface. Windows overlap on the interface and nowhere else.

An edge may exist only where some agent observes both endpoints. That rule has a
consequence the whole design rests on: any latent confounding an agent encounters is
induced by the partition, and is therefore resolvable by intervening on the shared
interface. Global recovery from local experiments is well posed under it.

## An episode

1. A graph is drawn, and `ma/scm.py` builds a linear Gaussian SCM over it with a per-node
   noise scale. Each agent receives `n_obs` observational rows restricted to its window.
2. Each agent seeds a belief over the ancestral marks between pairs in its window. Pairs
   the skeleton calls non-adjacent are closed immediately; the rest start undetermined.
3. Rounds run until the shared intervention budget is spent. On each round an acting agent
   selects a variable to intervene on, or passes. The intervention is hard: the variable's
   structural equation is replaced, so its parents are disconnected.
4. `n_int` interventional rows are drawn. `cb/citest.py` compares them against the
   unintervened rows and answers, per pair, whether the intervened variable is an ancestor
   of the other. The belief narrows accordingly.
5. Targets on shared variables are disclosed to all agents after the round; targets on
   private variables are not, though a one-bit signal marks that some private intervention
   occurred.
6. At the end of the episode each agent's belief is scored, and the per-site beliefs are
   pooled by intersection to give the global result.

## Components

| module | responsibility |
|---|---|
| `ma/topology.py` | the partition: who owns what, which edges may exist, what each agent sees |
| `ma/scm.py` | the generative model, noise families and mechanism families |
| `ma/projection.py` | the ground-truth ancestral graph for a window, by latent projection |
| `ma/env.py` | the environment: episode loop, budget, turn order, disclosure, reward |
| `cb/citest.py` | the only contact with data; conditional independence tests |
| `cb/versionspace.py`, `cb/factored.py` | the belief, enumerated and factored |
| `cb/orient.py` | skeleton to partial ancestral graph |
| `cb/attribution.py` | which hidden variable explains which confounded pairs, and whose |
| `ma/policy.py` | independent PPO, optionally with federated averaging |
| `ma/baselines.py` | the reference policies every result is measured against |
| `ma/evaluate.py` | replaying episodes under a fixed policy and scoring recovery |

## Why the belief is a version space

A posterior would be the conventional choice and is what `crosscheck/` implements for
comparison. It is not used for the reported results because posteriors do not compose under
the privacy constraint: the product of two sites' posteriors is not the posterior of the
pooled data, and correcting for that means exchanging likelihoods.

Mark sets do compose. Each site's set contains the truth, so the intersection of two sites'
sets also contains it, and pooling is exact without any site revealing what it saw.
`docs/BELIEF.md` develops this, including what the representation gives up in exchange.

## Two belief backends

`cb/versionspace.py` holds the belief as an explicit set of candidate structures. It is
exact and supports an exact optimum, and it becomes unusable past roughly a six-variable
window because the candidate count grows as 3^(edges).

`cb/factored.py` keeps a separate mark set per pair. This is what the reported cells use.
It is sound because the evidence is itself pairwise, and it is an outer approximation
because joint constraints between pairs are discarded — so the belief never excludes the
truth, but can stay more uncertain than a fully joint representation would.

## Evidence regimes

The environment supports three, and the distinction matters for reading any result:

- **oracle** — ancestry queries are answered exactly. The belief never touches data.
- **partial oracle** — a fraction of queries returns nothing, chosen at random. This is a
  training regime, not a claim about deployment.
- **sampled** — answers are estimated from the interventional rows by the tests in
  `cb/citest.py`, at a fixed significance level.

Soundness holds under oracle evidence by construction. Under sampled evidence it holds only
while the tests are calibrated, and `docs/BELIEF.md` records the regime where they are not.
