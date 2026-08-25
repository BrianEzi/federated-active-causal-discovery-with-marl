"""The fast statistics must be scipy, minus only the call overhead.

Profiling (2026-08-25) put 70% of training compute in ~5000 scipy calls per episode
inside `ancestral_evidence`. The replacements are the same formulas; this file is the
gate that keeps them so: p-values equal to scipy within 1e-9 across random inputs, and
the full `ancestral_evidence` verdict matrix identical to a scipy reimplementation over
many random datasets. If either drifts, the optimisation is wrong, not the test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from cb.citest import FisherZ, _brown_forsythe_p, _pearson_p, _welch_p


def test_pvalues_match_scipy_across_random_inputs():
    rng = np.random.default_rng(0)
    for trial in range(300):
        n1, n2 = rng.integers(20, 400, size=2)
        a = rng.normal(rng.uniform(-1, 1), rng.uniform(0.3, 2.0), n1)
        b = rng.normal(rng.uniform(-1, 1), rng.uniform(0.3, 2.0), n2)
        assert np.isclose(_welch_p(a, b),
                          stats.ttest_ind(a, b, equal_var=False)[1], atol=1e-9)
        assert np.isclose(_brown_forsythe_p(a, b), stats.levene(a, b)[1], atol=1e-9)
        x = rng.normal(size=n1)
        y = 0.3 * x + rng.normal(size=n1)
        assert np.isclose(_pearson_p(x, y), stats.pearsonr(x, y)[1], atol=1e-9)


def _scipy_ancestral(test: FisherZ, min_rows: int = 20) -> np.ndarray:
    """The pre-optimisation implementation, kept verbatim as the reference oracle."""
    out = np.zeros((test.k, test.k), dtype=bool)
    for x in range(test.k):
        clamped = test.intervened[:, x]
        if int(clamped.sum()) < min_rows:
            continue
        for y in range(test.k):
            if y == x or test.intervened[:, y].all():
                continue
            free = ~test.intervened[:, y] & ~test.foreign
            others = [c for c in range(test.k) if c not in (x, y)]
            if others:
                free &= ~test.intervened[:, others].any(axis=1)
            a, b = test.data[clamped & free, y], test.data[(~clamped) & free, y]
            if len(a) < min_rows or len(b) < min_rows:
                continue
            if a.std() < 1e-12 and b.std() < 1e-12:
                continue
            a_x = test.data[clamped & free, x]
            if a_x.std() > 1e-12 and a.std() > 1e-12:
                _, p_corr = stats.pearsonr(a_x, a)
            else:
                p_corr = np.nan
            _, p_mean = stats.ttest_ind(a, b, equal_var=False)
            try:
                _, p_var = stats.levene(a, b)
            except ValueError:
                p_var = np.nan
            fired = [q for q in (p_mean, p_var, p_corr) if np.isfinite(q)]
            if fired and min(fired) < test.alpha:
                out[x, y] = True
    return out


def _reference_independent(test: FisherZ, x, y, cond) -> bool:
    """The pre-cache implementation of the CI test, verbatim."""
    cond = [c for c in cond if c not in (x, y)]
    rows = test._rows_for(x, y)
    n_rows = int(rows.sum())
    dof = n_rows - len(cond) - 3
    if n_rows < 20 or dof <= 0:
        return True
    sub = test.data[rows][:, [x, y] + list(cond)]
    if not np.all(sub.std(axis=0) > 1e-12):
        return True
    corr = np.corrcoef(sub, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return True
    try:
        precision = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        return True
    denom = np.sqrt(precision[0, 0] * precision[1, 1])
    if not np.isfinite(denom) or denom <= 0:
        return True
    r = float(np.clip(-precision[0, 1] / denom, -0.999999, 0.999999))
    z = 0.5 * np.log((1 + r) / (1 - r))
    p = 2.0 * stats.norm.sf(abs(np.sqrt(dof) * z))
    return bool(p > test.alpha)


def test_ci_test_verdicts_identical_after_the_corr_cache():
    rng = np.random.default_rng(2)
    from itertools import combinations
    for trial in range(60):
        k = int(rng.integers(3, 6))
        n = 150 * k
        weights = np.triu(rng.uniform(-1.5, 1.5, (k, k)), 1) * (rng.random((k, k)) < 0.5)
        data = np.zeros((n, k))
        intervened = np.zeros((n, k), dtype=bool)
        block = n // (k + 1)
        for j in range(k):
            data[:, j] = data @ weights[:, j] + rng.normal(size=n)
            rows = slice(j * block, (j + 1) * block)
            data[rows, j] = rng.normal(0, 2, block)
            intervened[rows, j] = True
        test = FisherZ(data, intervened)
        for x, y in combinations(range(k), 2):
            others = [c for c in range(k) if c not in (x, y)]
            conds = [(), (others[0],), tuple(others[:2])]
            for cond in conds:
                assert (test.independent(x, y, cond)
                        == _reference_independent(test, x, y, cond)), (trial, x, y, cond)


def test_ancestral_evidence_verdicts_identical_to_scipy():
    rng = np.random.default_rng(1)
    for trial in range(40):
        k = int(rng.integers(3, 6))
        n = 200 * k
        weights = np.triu(rng.uniform(-1.5, 1.5, (k, k)), 1)
        weights *= rng.random((k, k)) < 0.5
        data = np.zeros((n, k))
        intervened = np.zeros((n, k), dtype=bool)
        block = n // k
        for j in range(k):
            noise = rng.normal(size=n)
            data[:, j] = data @ weights[:, j] + noise
            rows = slice(j * block, (j + 1) * block)
            data[rows, j] = rng.normal(0, 2, block)      # vary-style block per node
            intervened[rows, j] = True
        foreign = rng.random(n) < 0.1
        test = FisherZ(data, intervened, foreign=foreign)
        fast = test.ancestral_evidence()
        assert np.array_equal(fast, _scipy_ancestral(test)), f"trial {trial}"
