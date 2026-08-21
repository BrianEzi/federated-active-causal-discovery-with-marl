# Prototypes — verified, not yet integrated

Working code from the 2026-08-15 session that removes the d=6 ceiling. Everything here has
been **measured against ground truth**, but none of it is wired into `sa/` yet, so it lives
apart from the package rather than pretending to be part of it.

Import paths assume the repo root is on `PYTHONPATH` and that this directory is too, since
the files import each other.

---

## What each file is

| file | status | what it does |
|---|---|---|
| `subset_dp.py` | **verified** | exact posterior normalising constant Z without enumerating DAGs |
| `subset_dp_edge_marginals.py` | **verified, slow** | edge marginals via `d(d-1)` constrained DP runs |
| `mh_sampler.py` | **verified** | samples DAGs from the posterior; needed for the oracle |
| `verify_sampler_correctness.py` | test | sampled graph frequency vs exact posterior (total variation) |
| `verify_oracle_sampling.py` | test | the acceptance test: sampled oracle's CHOICES vs the exact oracle |
| `measure_mh_scaling.py` | measurement | chain length needed to hold oracle error, by d |
| `measure_oracle_sample_need.py` | measurement | how many posterior samples the oracle needs |
| `BROKEN_gibbs_sampler.py` | **DO NOT USE** | parent-set Gibbs; systematically wrong, root cause unfound |
| `BROKEN_combined_sampler.py` | **DO NOT USE** | Gibbs + reversal; same fault |

The two broken files are kept deliberately, in the same spirit as `KnownVarianceScore` in
`sa/score.py`: the experiment log cites measurements taken from them, and someone may want
to find the root cause. They are prefixed so they cannot be picked up by accident.

---

## Established results

**Subset DP is exact, not approximate.** Decomposing each DAG by its sinks with
inclusion-exclusion gives a recurrence over subsets of nodes. Against enumeration, where
ground truth exists:

| quantity | d | difference |
|---|---|---|
| log Z | 3, 5, 6 | 0.0 |
| log Z | 4 | 4.6e-13 |
| edge marginals | 4, 5, 6 | ≤ 7.2e-14 |

Cost at d=6: Z 294 → 2 ms, edge marginals 733 → 65 ms. Z alone reaches d=11 in 0.46 s.
Enumeration is impossible past d=6 (1.14 billion DAGs at d=7).

Numerical note: the recurrence alternates in sign, and a single global score shift makes it
fail outright at d=6 (`log Z = -inf`). Shifting **each node** by its own maximum — exact,
because the score decomposes per node — fixes it completely: the measured growth ratio, the
largest intermediate magnitude over the final answer, stays below 1 at every d from 3 to 11.

**MH sampling supplies the oracle.** Reachability is not decomposable, so the DP cannot
produce the descendant-set distribution the greedy EIG oracle needs. Sampling DAGs and
computing descendants per sample does. Verified two ways — directly against the exact
posterior (total variation 0.0037–0.02 at d=4, improving as the posterior concentrates),
and through the oracle's actual choices:

| d | draws | agreement with exact oracle | regret vs Monte Carlo floor |
|---|---|---|---|
| 4 | 16000 | 96.8% | 0.0× (at the floor) |
| 5 | 16000 | 100.0% | 0.0× (at the floor) |
| 6 | 16000 | 95.7% | 2.5× |

Required chain length grows roughly 4× per node past d=5. Affordable because the oracle is
used only for evaluation — building the greedy reference and scoring actions — not in
training, so it runs on ~300 evaluation episodes rather than 6000 training ones.

---

## THE NEXT FIX — edge marginals in one pass

**This is the bottleneck now, and it is the next thing to do.**

`subset_dp_edge_marginals.py` gets `P(u → v)` by re-running the whole DP with node `v`
restricted to parent sets containing `u`, then taking `Z_forced / Z`. Correct, and already
faster than enumeration at d=6 — but it is `d(d-1)` separate full DP runs, so it scales as
`d² · 3^d`. Projected per environment step: ~1 s at d=8, ~3.5 s at d=9, ~14 s at d=10.

So having removed the enumeration wall, this is what now caps `d` at roughly 8.

**The fix.** All `d(d-1)` marginals should come from one or two passes rather than 90
separate ones. Two routes, in order of expected effort:

1. **Reuse the DP table.** The recurrence already computes `f(A)` for *every* subset `A` on
   the way to `f(V)`. An edge marginal is a sum over DAGs containing a specific edge, and
   that quantity should be recoverable from the same table combined with a complementary
   backward pass — the standard forward/backward structure of subset DP. This is the
   approach the literature takes; search **Koivisto & Sood (2004)**, which computes all
   edge posteriors in `O(2^d · d²)` rather than one run per edge.
2. **Fast subset convolution** to bring the recurrence itself from `O(3^d)` to
   `O(2^d · d²)`. Search **Björklund, Husfeldt, Kaski & Koivisto, "Fourier meets Möbius"**.
   Independent of (1) and composes with it.

Together these should move the practical ceiling from d≈8 to d≈12–15.

**Acceptance test before use:** `verify_sampler_correctness.py`-style direct comparison
against enumerated edge marginals at d=4, 5 and 6. Checking a new implementation through a
downstream consumer instead of directly is what let a broken sampler look like a mixing
problem for three rounds tonight.

---

## Also outstanding

The score-table batching in `sa/score.py` gave 39.7× at d=10 and is already integrated.
It is a **constant factor** on an exponential term, so it buys roughly five more nodes; the
subset DP is the change in growth class and is what actually removes the wall.
