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

from sa.scoretable import LocalScorer

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


def _free_bits(masks: np.ndarray, d: int, n_free: int) -> np.ndarray:
    """`[len(masks), n_free]` -- the set bit positions of each mask's COMPLEMENT, ascending.

    Every mask at one popcount level has a complement of the same size, which is what makes
    the level rectangular and therefore batchable at all.
    """
    bits = (masks[:, None] >> np.arange(d)[None, :]) & 1
    # Stable argsort puts the zero (free) positions first, in increasing bit order.
    return np.argsort(bits, axis=1, kind="stable")[:, :n_free]


def log_partition_table_vec(log_alpha: np.ndarray, d: int
                            ) -> Tuple[List[float], List[float], float]:
    """Vectorised `log_partition_table`. Same recurrence, same answer, ~15x faster at k=13.

    TWO CHANGES FROM THE SCALAR VERSION, and the second is the one that pays.

    **The loop is inverted.** The scalar version fixes `A` and walks the non-empty
    `S subset A`, reading `f(A \ S)`. This version fixes `rest` and PUSHES to every superset
    `A = rest | T`. Same `3^d` pairs, different order.

    **Pushes are batched BY POPCOUNT LEVEL, not one `rest` at a time.** The obvious
    inversion -- one numpy call per `rest` -- was MEASURED AT 1.5x, and below k=11 it was
    SLOWER than the scalar loop it replaced: `2^d` calls on arrays averaging `1.5^d` entries
    is almost pure per-call overhead. Batching by level is what removes it. Every `rest` with
    `|rest| = m` has a complement of size exactly `d - m`, so the whole level is one
    rectangular `[C(d,m), 2^(d-m)]` array, and the doubling that builds
    `PROD_{i in T} alpha_i(rest)` runs `d - m` numpy calls for the entire level. That is
    `O(d^2)` calls in total rather than `2^d` -- about 100 at k=14 against 16384.

    LEGALITY. `f(rest)` is read, so it must be final. Every push into `rest` comes from a
    STRICT subset, which sits at a strictly lower level, so ascending levels finalise each
    `rest` before it is read. This is the same argument the scalar version makes implicitly
    by visiting `A` in increasing numeric order.

    WITHIN a level, many `rest` push to the same `A`, so the scatter must ACCUMULATE rather
    than overwrite. It is reduced per target in two passes -- `np.maximum.at` for the shift,
    then `np.bincount` for the shifted sum -- and only then merged into the running totals.
    THE SHIFT IS PER TARGET, NEVER GLOBAL: one shift across the level would underflow every
    subset whose mass sits far below the level maximum and delete it silently. That exact
    bug, in the assignment mixture, cost a day (see `joint_conf_marginals`).

    Signs are carried separately throughout. The recurrence ALTERNATES, and the surviving
    total is routinely orders of magnitude below the terms that produced it -- which is what
    the returned `worst` diagnostic exists to report.
    """
    size = 1 << d
    log_f = np.full(size, NEG_INF, dtype=np.float64)
    sign_f = np.zeros(size, dtype=np.float64)
    log_f[0] = 0.0
    sign_f[0] = 1.0
    peak = np.full(size, NEG_INF, dtype=np.float64)

    masks = np.arange(size, dtype=np.int64)
    pop = np.zeros(size, dtype=np.int64)
    for m in range(1, size):
        pop[m] = pop[m >> 1] + (m & 1)

    for level in range(d):
        rest = masks[pop == level]
        rest = rest[sign_f[rest] != 0.0]
        if rest.size == 0:
            continue
        n_free = d - level
        free = _free_bits(rest, d, n_free)

        idx = rest[:, None].copy()
        lp = np.zeros((rest.size, 1), dtype=np.float64)
        for j in range(n_free):
            bit = free[:, j]
            step = (np.int64(1) << bit)[:, None]
            add = log_alpha[bit, rest][:, None]
            idx = np.concatenate((idx, idx + step), axis=1)
            lp = np.concatenate((lp, lp + add), axis=1)
        idx = idx[:, 1:]            # T = {} is not a legal S
        lp = lp[:, 1:]

        total = (log_f[rest][:, None] + lp).ravel()
        flat = idx.ravel()
        # (-1)^(|T|+1): positive for odd |T|, and `A ^ rest` is exactly T.
        odd = (pop[flat ^ np.repeat(rest, idx.shape[1])] & 1).astype(bool)
        tsign = np.where(odd, 1.0, -1.0) * np.repeat(sign_f[rest], idx.shape[1])

        # DROP THE VANISHING TERMS BEFORE THE SCATTER, not after. A term is -inf whenever
        # some alpha_i(rest) is -inf -- an entirely empty parent set, which is common once
        # edges are forced. Left in, it reaches `total - shift` as -inf minus -inf, i.e.
        # NaN, and `np.sign(NaN)` is NaN: the entry then reports neither a positive nor a
        # negative nor an absent contribution, and the magnitudes stay correct so the
        # corruption shows up only in the SIGNS. That is exactly how this first failed.
        alive = np.isfinite(total)
        if not alive.any():
            continue
        flat, total, tsign = flat[alive], total[alive], tsign[alive]

        shift = np.full(size, NEG_INF, dtype=np.float64)
        np.maximum.at(shift, flat, total)
        acc = np.bincount(flat, weights=tsign * np.exp(total - shift[flat]),
                          minlength=size)

        with np.errstate(divide="ignore", invalid="ignore"):
            lvl_log = np.where(acc != 0.0, shift + np.log(np.abs(acc)), NEG_INF)
        lvl_sign = np.sign(acc)
        peak = np.maximum(peak, shift)

        hi = np.maximum(log_f, lvl_log)
        finite = np.isfinite(hi)
        da = np.full(size, NEG_INF)
        db = np.full(size, NEG_INF)
        np.subtract(log_f, hi, out=da, where=finite)
        np.subtract(lvl_log, hi, out=db, where=finite)
        value = sign_f * np.exp(da) + lvl_sign * np.exp(db)
        sgn = np.sign(value)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_f = np.where(sgn != 0.0, hi + np.log(np.abs(value)), NEG_INF)
        sign_f = sgn
        log_f[0], sign_f[0] = 0.0, 1.0      # the empty set is the recurrence's base case

    live = (sign_f != 0.0) & np.isfinite(peak) & np.isfinite(log_f)
    worst = float(np.max(peak[live] - log_f[live])) if live.any() else 0.0
    return log_f.tolist(), sign_f.tolist(), worst


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


def _scatter_signed(log_out: np.ndarray, sign_out: np.ndarray, targets: np.ndarray,
                    log_vals: np.ndarray, sign_vals: np.ndarray, size: int) -> None:
    """Accumulate `(log_vals, sign_vals)` into `(log_out, sign_out)` at `targets`, in place.

    Many sources hit the same target, so this REDUCES per target before merging: one
    `np.maximum.at` for the shift, one `np.bincount` for the shifted signed sum. The shift
    is PER TARGET and never global -- a single shift across the batch silently underflows
    every target whose mass sits far below the batch maximum.
    """
    alive = np.isfinite(log_vals) & (sign_vals != 0.0)
    if not alive.any():
        return
    targets, log_vals, sign_vals = targets[alive], log_vals[alive], sign_vals[alive]

    shift = np.full(size, NEG_INF, dtype=np.float64)
    np.maximum.at(shift, targets, log_vals)
    acc = np.bincount(targets, weights=sign_vals * np.exp(log_vals - shift[targets]),
                      minlength=size)
    with np.errstate(divide="ignore", invalid="ignore"):
        batch_log = np.where(acc != 0.0, shift + np.log(np.abs(acc)), NEG_INF)
    batch_sign = np.sign(acc)

    hi = np.maximum(log_out, batch_log)
    finite = np.isfinite(hi)
    da = np.full(size, NEG_INF)
    db = np.full(size, NEG_INF)
    np.subtract(log_out, hi, out=da, where=finite)
    np.subtract(batch_log, hi, out=db, where=finite)
    value = sign_out * np.exp(da) + batch_sign * np.exp(db)
    sgn = np.sign(value)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_out[:] = np.where(sgn != 0.0, hi + np.log(np.abs(value)), NEG_INF)
    sign_out[:] = sgn


def log_backward_vec(log_alpha: np.ndarray, log_f, sign_f, d: int,
                     max_block: int = 6_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised `log_backward`. Same cotangents, same answer, ~20x faster at k=13.

    Batched BY POPCOUNT LEVEL OF `A`, descending, for the same reason the forward pass is
    batched ascending: every subset at one level has the same number of subsets below it
    (`2^n` for `|A| = n`), so a level is one rectangular `[C(d,n), 2^n]` block and the
    doubling that enumerates `rest subset A` runs `n` numpy calls for the whole level.

    LEGALITY. `fbar[A]` is read, and every push into it comes from a STRICT superset, which
    sits at a strictly HIGHER level. Descending levels therefore finalise each `A` before it
    is read -- the mirror of the forward argument, and the reason the scalar version walks
    `A` downwards.

    PREFIX/SUFFIX RATHER THAN SUBTRACTION, kept from the scalar version and load-bearing.
    The per-node cotangent needs `PROD_{i in S, i != j} alpha_i(rest)`, and the obvious
    `total - alpha_j` is NaN whenever `alpha_j(rest)` is `-inf` -- which happens for real,
    not hypothetically, as soon as a forced edge empties a parent set. An exclusive cumulative
    sum from each side never subtracts, so `-inf` propagates as the zero it represents.

    MEMORY IS THE BINDING CONSTRAINT, not time. The working tensor is
    `[C(d,n), 2^n, d]`, whose largest level is ~46M entries at `d = 15`. `max_block` caps
    the entries per chunk and the `A` axis is split to fit; the arithmetic is identical
    either way, since chunks touch disjoint sources and accumulate into shared targets
    through `_scatter_signed`.
    """
    size = 1 << d
    log_f = np.asarray(log_f, dtype=np.float64)
    sign_f = np.asarray(sign_f, dtype=np.float64)

    log_fbar = np.full(size, NEG_INF, dtype=np.float64)
    sign_fbar = np.zeros(size, dtype=np.float64)
    log_fbar[size - 1] = 0.0
    sign_fbar[size - 1] = 1.0
    log_abar = np.full((d, size), NEG_INF, dtype=np.float64)
    sign_abar = np.zeros((d, size), dtype=np.float64)

    masks = np.arange(size, dtype=np.int64)
    pop = np.zeros(size, dtype=np.int64)
    for m in range(1, size):
        pop[m] = pop[m >> 1] + (m & 1)
    bit_index = np.arange(d, dtype=np.int64)

    for level in range(d, 0, -1):
        a_all = masks[pop == level]
        a_all = a_all[sign_fbar[a_all] != 0.0]
        if a_all.size == 0:
            continue
        # `_free_bits` on the COMPLEMENT gives the set bits of A itself.
        set_bits_all = _free_bits(~a_all & (size - 1), d, level)

        per_row = (1 << level) * d
        chunk = max(1, min(a_all.size, max_block // max(per_row, 1)))
        for lo in range(0, a_all.size, chunk):
            a = a_all[lo:lo + chunk]
            set_bits = set_bits_all[lo:lo + chunk]

            rest = np.zeros((a.size, 1), dtype=np.int64)
            for t in range(level):
                rest = np.concatenate((rest, rest + (np.int64(1) << set_bits[:, t])[:, None]),
                                      axis=1)
            s_mask = a[:, None] ^ rest                     # S = A \ rest
            keep = s_mask != 0                             # S must be non-empty
            if not keep.any():
                continue

            in_s = ((s_mask[:, :, None] >> bit_index[None, None, :]) & 1).astype(bool)
            # alpha_j(rest) laid out [n_a, 2^level, d]; 0.0 outside S is the neutral element
            # of a product in log space.
            gathered = np.moveaxis(log_alpha[:, rest], 0, -1)
            m_tensor = np.where(in_s, gathered, 0.0)

            total = m_tensor.sum(axis=2)
            cum = np.cumsum(m_tensor, axis=2)
            prefix = np.concatenate((np.zeros_like(cum[:, :, :1]), cum[:, :, :-1]), axis=2)
            rcum = np.cumsum(m_tensor[:, :, ::-1], axis=2)[:, :, ::-1]
            suffix = np.concatenate((rcum[:, :, 1:], np.zeros_like(rcum[:, :, :1])), axis=2)
            exclude_one = prefix + suffix

            parity_odd = (pop[s_mask] & 1).astype(bool)
            g_sign = sign_fbar[a][:, None]
            signed = np.where(parity_odd, g_sign, -g_sign)
            g_log = log_fbar[a][:, None]

            flat_rest = rest[keep]
            # push 1 -- the cotangent of f(rest) itself
            _scatter_signed(log_fbar, sign_fbar, flat_rest,
                            (g_log + total)[keep], signed[keep], size)

            # push 2 -- the cotangent of each alpha_j(rest), for j in S only
            base = g_log + log_f[rest]
            term_sign = signed * sign_f[rest]
            for j in range(d):
                sel = keep & in_s[:, :, j]
                if not sel.any():
                    continue
                _scatter_signed(log_abar[j], sign_abar[j], rest[sel],
                                (base + exclude_one[:, :, j])[sel], term_sign[sel], size)

    return log_abar, sign_abar


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
        log_f, sign_f, worst = log_partition_table_vec(self._alpha(log_w), self.d)
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
                log_f, sign_f, _ = log_partition_table_vec(
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
        log_f, sign_f, _ = log_partition_table_vec(log_alpha, d)
        full = (1 << d) - 1
        log_z = log_f[full]
        if sign_f[full] <= 0.0:
            raise FloatingPointError(
                f"subset DP produced a non-positive partition function at d={d}.")

        log_abar, sign_abar = log_backward_vec(log_alpha, log_f, sign_f, d)
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
