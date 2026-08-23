"""Exact posterior over DAGs computed WITHOUT enumerating them.

Enumeration is a wall, not a slope. The number of labelled DAGs goes 543 (d=4), 29,281
(d=5), 3,781,503 (d=6), 1.14 billion (d=7) -- so every quantity in `sa/posterior.py` that
sweeps the graph list stops existing one node past the current setting. This module
replaces that sweep with a dynamic program over *subsets* of nodes, which costs `O(3^d)`
instead of `O(#DAGs)`.

**It is exact, not an approximation.** Verified against enumeration in `tests/test_dp.py`:
`log Z`, every edge marginal, and `P(true DAG | data)` all agree to ~1e-12 at d = 3, 4, 5
and 6, on *environment* data rather than on synthetic noise -- see the warning below for
why that distinction is the whole ballgame.

How it works
------------
Every DAG on a node set `A` can be decomposed by its **sinks** -- nodes with no children.
A DAG has at least one sink, so summing over "which nodes are sinks" and recursing on the
rest covers every DAG, but double-counts graphs with several sinks. Inclusion-exclusion
over the sink set fixes that exactly:

    f(A) = sum over nonempty S subset A of  (-1)^(|S|+1) * f(A\\S) * prod_{i in S} alpha_i(A\\S)

where `alpha_i(B)` is the total weight of all parent sets for node `i` drawn from `B`.
`f(V)` is the partition function Z. This is Robinson's sink recurrence; see
docs/THEORY_NOTES.md, and Koivisto & Sood (2004) for the modern treatment.

Everything runs in **signed log space**
---------------------------------------
This is not defensive programming. A first version ran the recurrence in ordinary doubles,
rescaling each node's weights by that node's own maximum, and it verified perfectly against
enumeration at d=3,4,5,6 -- on data drawn from independent normals. On the first contact
with real environment data it returned `Z = 0` at **d=4**.

The reason is structural, not a rounding accident. Rescaling can only be done per node,
because that is the only thing that factorises; but the sum of per-node maxima is the score
of a configuration in which *every* node takes its unconstrained best parent set, and those
choices are jointly cyclic. No DAG attains it. The shortfall is the total information each
node shares with the others, so it grows with both `d` and the sample count:

    gap (nats)    n=1000    n=5000    n=20000
    d=4              834      4,612      18,233
    d=5            1,821      8,888      35,999
    d=6            3,892     19,404      78,306

A double underflows past 745. So the plain-arithmetic version could not have worked at any
size actually used -- it looked correct only because independent columns make the gap
vanish, and independent columns are exactly what a causal discovery environment does not
produce. In log space the quantity being represented is the gap's logarithm, and no shift
is needed at all.

The lesson, recorded because it nearly cost a night: **verifying against ground truth is
not enough if the inputs are unrepresentative.** The acceptance test was right, the test
data was not.

What it can and cannot represent
--------------------------------
The recurrence needs the prior to be **modular** -- a product of per-node terms. The
Erdos-Renyi prior already is one:

    P(G) ~ p^|E| (1-p)^(pairs-|E|)  =  (p/(1-p))^|E| * const

and `|E| = sum_i |Pa_i|` decomposes per node. Uniform-over-DAGs is ER at p=0.5.

`scale_free_prior` is **not** modular -- its Gini reweighting is a function of the whole
degree sequence -- so `for_prior` raises rather than silently scoring a different model,
which is the failure that would be hardest to notice.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from crosscheck.scoretable import LocalScorer

NEG_INF = float("-inf")


# --------------------------------------------------------------------------------------
# Signed log-space arithmetic
# --------------------------------------------------------------------------------------

def _signed_add(log_a: float, sign_a: float,
                log_b: float, sign_b: float) -> Tuple[float, float]:
    """`(log|a+b|, sign(a+b))` from the same representation of `a` and `b`."""
    if sign_a == 0.0:
        return log_b, sign_b
    if sign_b == 0.0:
        return log_a, sign_a
    if log_a >= log_b:
        hi, lo, hi_sign, lo_sign = log_a, log_b, sign_a, sign_b
    else:
        hi, lo, hi_sign, lo_sign = log_b, log_a, sign_b, sign_a
    value = hi_sign + lo_sign * math.exp(lo - hi)
    if value == 0.0:
        return NEG_INF, 0.0
    return hi + math.log(abs(value)), math.copysign(1.0, value)


def log_zeta(log_w: np.ndarray, d: int) -> np.ndarray:
    """Subset log-sum-exp over the last axis of length `2^d`.

    The log-space form of the fast zeta transform: turns per-parent-set log weights into
    `log alpha_i(B) = log sum over P subset B of w_i(P)` for all `2^d` subsets at once, in
    `O(d * 2^d)`. All entries are non-negative in the original scale, so plain `logaddexp`
    suffices and no signs are needed here.
    """
    a = np.array(log_w, dtype=np.float64, copy=True)
    for bit in range(d):
        a = a.reshape(-1, 2, 1 << bit)
        a[:, 1, :] = np.logaddexp(a[:, 1, :], a[:, 0, :])
        a = a.reshape(-1, 1 << d)
    return a


def signed_log_moebius_transpose(log_a: np.ndarray, sign_a: np.ndarray,
                                 d: int) -> Tuple[np.ndarray, np.ndarray]:
    """In-place SUPERSET log-sum over the last axis. The adjoint of `log_zeta`.

    Signed, because the cotangents it propagates come from an alternating recurrence and
    genuinely take both signs -- unlike the forward transform's weights.
    """
    log_a = np.array(log_a, dtype=np.float64, copy=True)
    sign_a = np.array(sign_a, dtype=np.float64, copy=True)
    for bit in range(d):
        log_a = log_a.reshape(-1, 2, 1 << bit)
        sign_a = sign_a.reshape(-1, 2, 1 << bit)
        la, sa = log_a[:, 0, :], sign_a[:, 0, :]
        lb, sb = log_a[:, 1, :], sign_a[:, 1, :]

        hi = np.maximum(la, lb)
        finite = np.isfinite(hi)
        # `where=` rather than `np.where(...)`: both operands are -inf whenever the whole
        # entry is empty, and -inf minus -inf is a NaN that np.where would compute (and
        # warn about) before discarding.
        da = np.full_like(la, NEG_INF)
        db = np.full_like(lb, NEG_INF)
        np.subtract(la, hi, out=da, where=finite)
        np.subtract(lb, hi, out=db, where=finite)
        value = sa * np.exp(da) + sb * np.exp(db)
        sign = np.sign(value)
        with np.errstate(divide="ignore"):
            log_a[:, 0, :] = np.where(sign != 0, hi + np.log(np.abs(value)), NEG_INF)
        sign_a[:, 0, :] = sign

        log_a = log_a.reshape(-1, 1 << d)
        sign_a = sign_a.reshape(-1, 1 << d)
    return log_a, sign_a


# --------------------------------------------------------------------------------------
# The recurrence
# --------------------------------------------------------------------------------------

def log_partition_table(log_alpha: np.ndarray, d: int
                        ) -> Tuple[List[float], List[float], float]:
    """`(log|f(A)|, sign f(A), log peak)` for every subset `A`.

    The third return value is the cancellation diagnostic: the largest, over all subsets
    `A`, of `log(biggest term in f(A)'s sum) - log|f(A)|`. Above about 36 (`log 1/eps`) the
    answer has no significant digits left, because the recurrence alternates in sign and
    the surviving total is dwarfed by what cancelled to produce it.

    It is computed **per subset**, which the first version got wrong: comparing every
    intermediate against the final `f(V)` conflates cancellation with the fact that smaller
    subsets carry far fewer likelihood terms and so are astronomically larger. That
    version reported a "growth" of e^121000 on runs whose answers were exact to 1e-12.

    Carried as Python lists of floats rather than numpy arrays: the loop is `O(3^d)` scalar
    operations, where numpy's per-element overhead dominates by an order of magnitude.
    """
    alpha = log_alpha.tolist()
    size = 1 << d
    log_f = [NEG_INF] * size
    sign_f = [0.0] * size
    log_f[0] = 0.0
    sign_f[0] = 1.0
    worst = 0.0
    popcount = [bin(m).count("1") for m in range(size)]

    for A in range(1, size):
        acc_log, acc_sign = NEG_INF, 0.0
        peak = NEG_INF
        S = A
        while S:
            rest = A ^ S
            rest_sign = sign_f[rest]
            if rest_sign != 0.0:
                total = log_f[rest]
                bits = S
                while bits:
                    low = bits & -bits
                    total += alpha[low.bit_length() - 1][rest]
                    bits ^= low
                if total != NEG_INF:
                    if total > peak:
                        peak = total
                    sign = rest_sign if popcount[S] & 1 else -rest_sign
                    acc_log, acc_sign = _signed_add(acc_log, acc_sign, total, sign)
            S = (S - 1) & A
        log_f[A] = acc_log
        sign_f[A] = acc_sign
        if acc_sign != 0.0 and peak != NEG_INF:
            worst = max(worst, peak - acc_log)
    return log_f, sign_f, worst


def log_backward(log_alpha: np.ndarray, log_f: List[float], sign_f: List[float],
                 d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse-mode sweep through the sink recurrence. Returns `dZ/dalpha`, signed-log.

    Subsets are visited in DECREASING order, the reverse of the forward pass: `f(A)`
    depends only on strictly smaller sets, so by the time `A` is reached its cotangent has
    received every contribution it will get.

    Per-node factors use prefix/suffix sums rather than subtracting one term from the
    total. Subtracting would be shorter, but `log alpha_i(rest)` is `-inf` whenever every
    parent set in `rest` has vanishing weight, and `-inf` minus `-inf` is a NaN that would
    propagate into a marginal still looking like a probability.
    """
    alpha = log_alpha.tolist()
    size = 1 << d
    full = size - 1
    log_fbar = [NEG_INF] * size
    sign_fbar = [0.0] * size
    log_fbar[full] = 0.0
    sign_fbar[full] = 1.0
    log_abar = [[NEG_INF] * size for _ in range(d)]
    sign_abar = [[0.0] * size for _ in range(d)]
    popcount = [bin(m).count("1") for m in range(size)]

    for A in range(full, 0, -1):
        g_sign = sign_fbar[A]
        if g_sign == 0.0:
            continue
        g_log = log_fbar[A]
        S = A
        while S:
            rest = A ^ S
            members = []
            bits = S
            while bits:
                low = bits & -bits
                members.append(low.bit_length() - 1)
                bits ^= low
            k = len(members)
            values = [alpha[j][rest] for j in members]

            prefix = [0.0] * (k + 1)
            for i in range(k):
                prefix[i + 1] = prefix[i] + values[i]
            suffix = [0.0] * (k + 1)
            for i in range(k - 1, -1, -1):
                suffix[i] = suffix[i + 1] + values[i]

            signed = g_sign if popcount[S] & 1 else -g_sign

            if prefix[k] != NEG_INF:
                log_fbar[rest], sign_fbar[rest] = _signed_add(
                    log_fbar[rest], sign_fbar[rest], g_log + prefix[k], signed)

            if sign_f[rest] != 0.0:
                base = g_log + log_f[rest]
                term_sign = signed * sign_f[rest]
                for i, j in enumerate(members):
                    contribution = prefix[i] + suffix[i + 1]
                    if contribution == NEG_INF:
                        continue
                    log_abar[j][rest], sign_abar[j][rest] = _signed_add(
                        log_abar[j][rest], sign_abar[j][rest],
                        base + contribution, term_sign)
            S = (S - 1) & A

    return np.array(log_abar), np.array(sign_abar)


# --------------------------------------------------------------------------------------
# The posterior
# --------------------------------------------------------------------------------------

class DPPosterior:
    """Exact DAG posterior by subset DP -- the enumeration-free counterpart to
    `PosteriorEngine`.

    Holds no graph list, so it is constructible at any `d`. The data-dependent work is a
    `[d, 2^(d-1)]` local score table, shared with the enumerated path via `LocalScorer` so
    the two cannot drift apart.
    """

    def __init__(self, d: int, score, log_edge_odds: float = 0.0):
        self.d = d
        self.score = score
        self.scorer = LocalScorer(d, score)
        self.log_edge_odds = float(log_edge_odds)

        # Prior contribution of each parent set, in log space. Constant factors (the
        # `(1-p)^pairs` term) are dropped: they multiply every DAG equally, so they cancel
        # in the posterior and in every edge-marginal ratio.
        self.log_prior_term = self.log_edge_odds * self.scorer.parent_sizes.astype(float)

        # mask -> parent-set index, per node. Entries for masks containing the node
        # itself are never read; -1 makes a misuse crash rather than score a neighbour.
        self._mask_to_index = np.full((d, 1 << d), -1, dtype=np.int64)
        for node in range(d):
            for i in range(self.scorer.n_parent_sets):
                self._mask_to_index[node, self.scorer.parent_masks[node, i]] = i

        # Masks that are legal parent sets, per node -- used to scatter into mask order.
        self._valid = np.zeros((d, 1 << d), dtype=bool)
        for node in range(d):
            self._valid[node, self.scorer.parent_masks[node]] = True

    # -- construction -------------------------------------------------------------------

    @classmethod
    def for_prior(cls, d: int, score, kind: str = "erdos_renyi",
                  p: Optional[float] = 0.5) -> "DPPosterior":
        """Build with the same prior family `sa/priors.py` uses, or refuse.

        Refusing is the point: `scale_free` is not modular, and a DP that quietly ignored
        the reweighting would produce a posterior under a *different prior* than the
        enumerated path, while agreeing with it well enough at small d to look correct.
        """
        if kind == "uniform":
            return cls(d, score, log_edge_odds=0.0)
        if kind == "erdos_renyi":
            if p is None:
                p = 0.5
            if not 0.0 < p < 1.0:
                raise ValueError(f"p must be in (0, 1), got {p}")
            return cls(d, score, log_edge_odds=float(np.log(p) - np.log1p(-p)))
        raise ValueError(
            f"prior {kind!r} is not modular, so the subset DP cannot represent it exactly. "
            "Only 'uniform' and 'erdos_renyi' decompose per node; see the module docstring."
        )

    # -- weights ------------------------------------------------------------------------

    def log_weights(self, samples: np.ndarray, intervened: np.ndarray) -> np.ndarray:
        """[d, 2^(d-1)] log of score x prior, per (node, parent set)."""
        return self.scorer.table(samples, intervened) + self.log_prior_term

    def _log_weights_masked(self, log_w: np.ndarray) -> np.ndarray:
        """`[d, 2^d]` log weights in MASK order; illegal masks are `-inf`."""
        full = np.full((self.d, 1 << self.d), NEG_INF, dtype=np.float64)
        for node in range(self.d):
            full[node, self.scorer.parent_masks[node]] = log_w[node]
        return full

    def _alpha(self, log_w: np.ndarray,
               force: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """`log alpha`. `force = (child, parent)` restricts `child` to parent sets
        containing `parent`, which is how a single edge marginal is obtained as a ratio."""
        masked = self._log_weights_masked(log_w)
        if force is not None:
            child, parent = force
            drop = ((np.arange(1 << self.d) >> parent) & 1) == 0
            masked[child, drop] = NEG_INF
        return log_zeta(masked, self.d)

    # -- quantities ---------------------------------------------------------------------

    def log_partition(self, log_w: np.ndarray) -> float:
        """log Z, the log normalising constant of the unnormalised posterior."""
        return self.log_partition_diagnostic(log_w)[0]

    def log_partition_diagnostic(self, log_w: np.ndarray) -> Tuple[float, float]:
        """`(log Z, log growth)`.

        `log growth` is the worst per-subset cancellation in the table: the largest term
        entering any `f(A)`, divided by that same `f(A)`, in logs. Above about 36
        (`log 1/eps`) the answer is numerical noise. Reported in logs because the ratio
        itself overflows.
        """
        log_f, sign_f, worst = log_partition_table(self._alpha(log_w), self.d)
        full = (1 << self.d) - 1
        if sign_f[full] <= 0.0:
            raise FloatingPointError(
                f"subset DP produced a non-positive partition function at d={self.d} "
                f"(sign {sign_f[full]}). The alternating recurrence has cancelled past "
                "double precision.")
        return float(log_f[full]), float(worst)

    def edge_marginals(self, log_w: np.ndarray) -> np.ndarray:
        """[d, d] edge marginals by `d(d-1)` constrained DP runs.

        `out[u, v] = P(u -> v | data)`. Correct but `O(d^2 * 3^d)`; kept as the independent
        reference that validates `edge_marginals_onepass`, which is what callers should use.
        """
        d = self.d
        log_z = self.log_partition(log_w)
        out = np.zeros((d, d), dtype=np.float64)
        for child in range(d):
            for parent in range(d):
                if parent == child:
                    continue
                log_f, sign_f, _ = log_partition_table(
                    self._alpha(log_w, force=(child, parent)), d)
                forced = log_f[(1 << d) - 1]
                if sign_f[(1 << d) - 1] > 0.0 and forced != NEG_INF:
                    out[parent, child] = float(np.exp(forced - log_z))
        return out

    def edge_marginals_onepass(self, log_w: np.ndarray,
                               check: bool = False) -> np.ndarray:
        """[d, d] edge marginals from ONE forward and ONE backward pass.

        **The idea.** `Z` is a polynomial in the parent-set weights `w`, and it is
        *multilinear*: every DAG contributes `prod_i w_i(Pa_i)`, which uses each node's
        weights exactly once. So

            c_v(P) := dZ / dw_v(P)

        is the total weight of everything *except* node `v`'s own choice, summed over all
        DAGs in which `v`'s parents are exactly `P`. Hence `w_v(P) c_v(P)` is the posterior
        mass of that choice, and

            P(u -> v)  =  ( sum over P containing u of w_v(P) c_v(P) ) / Z.

        All `d * 2^(d-1)` derivatives come out of one reverse-mode sweep through the same
        recurrence that produced `Z`, because reverse-mode AD costs a constant multiple of
        the forward pass regardless of how many inputs there are: `O(d * 3^d)` in place of
        `O(d^2 * 3^d)`.

        Chosen over the Koivisto & Sood forward/backward construction, which reaches
        `O(2^d d^2)` but assumes an **order-modular** prior -- a different model, biased
        toward graphs with many topological orderings. This keeps the prior intact.

        `check=True` verifies Euler's identity `sum_P w_v(P) c_v(P) == Z` for every node.
        It holds because `Z` has degree exactly 1 in each node's weights, and no misindexed
        backward pass survives it. Its real value is that it needs **no ground truth**, so
        it still works at `d` where enumeration does not. Off by default: it is worth
        running once, not on every environment step.
        """
        d = self.d
        masked = self._log_weights_masked(log_w)
        log_alpha = log_zeta(masked, d)
        log_f, sign_f, _ = log_partition_table(log_alpha, d)
        full = (1 << d) - 1
        log_z = log_f[full]
        if sign_f[full] <= 0.0:
            raise FloatingPointError(
                f"subset DP produced a non-positive partition function at d={d}.")

        log_abar, sign_abar = log_backward(log_alpha, log_f, sign_f, d)
        log_wbar, sign_wbar = signed_log_moebius_transpose(log_abar, sign_abar, d)

        # Posterior mass of each (node, parent set) choice, in signed log space.
        log_mass = masked + log_wbar
        sign_mass = np.where(np.isfinite(masked), sign_wbar, 0.0)

        if check:
            for node in range(d):
                total_log, total_sign = self._signed_sum(
                    log_mass[node], sign_mass[node], np.ones(1 << d, dtype=bool))
                if total_sign <= 0 or abs(total_log - log_z) > 1e-6:
                    raise AssertionError(
                        f"Euler identity violated at node {node}: "
                        f"sum_P w c = {total_sign} x exp({total_log}), log Z = {log_z}")

        bit_set = ((np.arange(1 << d)[None, :] >> np.arange(d)[:, None]) & 1).astype(bool)
        out = np.zeros((d, d), dtype=np.float64)
        for child in range(d):
            for parent in range(d):
                if parent == child:
                    continue
                total_log, total_sign = self._signed_sum(
                    log_mass[child], sign_mass[child], bit_set[parent])
                if total_sign > 0:
                    out[parent, child] = float(np.exp(total_log - log_z))
        return out

    @staticmethod
    def _signed_sum(log_values: np.ndarray, signs: np.ndarray,
                    keep: np.ndarray) -> Tuple[float, float]:
        """Signed log-sum-exp over the selected entries."""
        selected = keep & (signs != 0) & np.isfinite(log_values)
        if not selected.any():
            return NEG_INF, 0.0
        logs = log_values[selected]
        sgn = signs[selected]
        hi = float(logs.max())
        value = float(np.sum(sgn * np.exp(logs - hi)))
        if value == 0.0:
            return NEG_INF, 0.0
        return hi + math.log(abs(value)), math.copysign(1.0, value)

    def log_prob_dag(self, log_w: np.ndarray, adjacency: np.ndarray,
                     log_z: Optional[float] = None) -> float:
        """log P(G | data) for one specific DAG.

        This is the quantity `is_identified` thresholds, and the only thing the pipeline
        needs from the posterior besides edge marginals: the *true* graph's mass. Pass
        `log_z` to avoid recomputing the partition function when scoring several graphs.
        """
        adjacency = np.asarray(adjacency) > 0.5
        if log_z is None:
            log_z = self.log_partition(log_w)
        total = 0.0
        for node in range(self.d):
            mask = int(np.dot(adjacency[:, node], 1 << np.arange(self.d)))
            index = self._mask_to_index[node, mask]
            if index < 0:
                raise ValueError(f"node {node} is its own parent -- not a DAG")
            total += log_w[node, index]
        return float(total - log_z)


def independent_edge_log_weights(marginals: np.ndarray, scorer, d: int) -> np.ndarray:
    """`[d, 2^(d-1)]` log weights for the independent-edge approximation to a posterior.

    This is the belief the `edge_marginal_greedy` baseline reasons with: every edge treated
    as an independent Bernoulli at its marginal probability, renormalised over DAGs. It is
    exactly what edge marginals discard -- the correlations between edges -- so the gap
    between that baseline and the full-posterior greedy measures the cost of the
    compression, and it is the fair opponent for an agent whose observation is edge
    marginals.

    The point of building it as a *log-weight table* rather than a distribution over graphs
    is that the independent-edge product is **modular**: a DAG's weight factorises into a
    term per (node, parent set), because whether edge `j -> i` is present depends only on
    `i`'s parent set. So the same subset DP and the same MH sampler consume it unchanged,
    and no DAG list is needed. Renormalisation over DAGs is then automatic -- both the DP
    and the sampler only ever put mass on acyclic graphs.

    Rejection sampling was the obvious alternative and is not viable: at d=7 only about one
    in four thousand independent-edge draws is acyclic when the marginals are diffuse.
    """
    marginals = np.clip(np.asarray(marginals, dtype=np.float64), 1e-9, 1 - 1e-9)
    log_present = np.log(marginals)
    log_absent = np.log1p(-marginals)

    out = np.zeros((d, scorer.n_parent_sets), dtype=np.float64)
    for node in range(d):
        # bits[k, j] = 1 when j is in parent set k of `node`.
        bits = ((scorer.parent_masks[node][:, None] >> np.arange(d)[None, :]) & 1
                ).astype(np.float64)
        # The self term is excluded explicitly. It is tempting to let it vanish on its own
        # -- no parent set contains `node`, and `marginals[node, node]` is 0 -- but the
        # clip above moves that 0 to 1e-9, so "absent" contributes log1p(-1e-9) per node
        # instead of exactly zero. Harmless (a constant offset per row cancels under
        # normalisation) but it makes the table not literally the product it claims to be,
        # and a test comparing against the definition catches the difference.
        present = log_present[:, node].copy()
        absent = log_absent[:, node].copy()
        present[node] = 0.0
        absent[node] = 0.0
        out[node] = bits @ present + (1.0 - bits) @ absent
    return out
