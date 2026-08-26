"""Conditional independence testing on interventional data.

The engine's only contact with the data. Everything else in `cb/` consumes yes/no answers
from here, which is what makes the nonlinear story a swap rather than a rewrite: replace
this class with a kernel test and the skeleton search, orientation and bootstrap are
unchanged.

WHY PARTIAL CORRELATION FIRST. It assumes linear-Gaussian, exactly what BGe already
assumes, so the first comparison against `crosscheck/` is like-for-like. Adopting a kernel
test at the same time would confound "constraint-based vs Bayesian" with "nonparametric vs
parametric" and neither result would be attributable.

INTERVENTIONS. A hard intervention on X severs X's incoming edges. That has two separate
consequences and both are implemented:

  masking      rows where X was CLAMPED say nothing about X's own parents -- X was set, not
               caused -- but remain valid evidence for X's children. This is the Cooper &
               Yoo (1999) rule the BGe path already implements, so it is a translation of an
               existing decision rather than a new one. Applied by dropping those rows from
               any test whose LEFT-HAND variable is X.

  orientation  if clamping X changes Y's distribution, X is an ancestor of Y. Full stop --
               no equivalence class, no ambiguity. Observational data can never supply this,
               and it is the entire reason the agents spend budget. Exposed here as
               `ancestral_evidence` and consumed by `cb/orient.py`.

The masking rule is deliberately conservative: a row is dropped for a test only when the
LHS variable itself was clamped in that row. Dropping every row containing any intervention
would discard most of the data an active agent generates.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import stats

# Below this many usable rows a test is not attempted and INDEPENDENCE is returned. That
# choice is deliberate and conservative: claiming dependence on no evidence would invent
# edges, which the skeleton search would then have to be argued out of. Absent evidence,
# assume separable.
MIN_ROWS = 20


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's two-sample t-test p-value -- scipy.stats.ttest_ind(equal_var=False),
    without the per-call overhead. Same formulas: Welch statistic, Satterthwaite df."""
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se2 = v1 / n1 + v2 / n2
    if se2 <= 0:
        return np.nan
    t = (a.mean() - b.mean()) / np.sqrt(se2)
    df = se2 * se2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return 2.0 * float(stats.t.sf(abs(t), df))


def _brown_forsythe_p(a: np.ndarray, b: np.ndarray) -> float:
    """scipy.stats.levene's default (center='median', i.e. Brown-Forsythe), two groups."""
    za = np.abs(a - np.median(a))
    zb = np.abs(b - np.median(b))
    n1, n2 = len(za), len(zb)
    ma, mb = za.mean(), zb.mean()
    grand = (za.sum() + zb.sum()) / (n1 + n2)
    denom = ((za - ma) ** 2).sum() + ((zb - mb) ** 2).sum()
    if denom <= 0:
        return np.nan
    w = (n1 + n2 - 2) * (n1 * (ma - grand) ** 2 + n2 * (mb - grand) ** 2) / denom
    return float(stats.f.sf(w, 1, n1 + n2 - 2))


def _pearson_p(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation p-value via the exact t transform -- what scipy.stats.pearsonr
    computes through the beta distribution; identical two-sided p."""
    n = len(x)
    if n < 3 or x.std() <= 1e-12 or y.std() <= 1e-12:
        return np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(min(r, 1.0), -1.0)
    if 1.0 - r * r <= 0:
        return 0.0
    t = r * np.sqrt((n - 2) / (1.0 - r * r))
    return 2.0 * float(stats.t.sf(abs(t), n - 2))


class FisherZ:
    """Partial-correlation independence test with per-row intervention masking.

    `data` is `[n, k]` for ONE agent's window; `intervened` is the matching `[n, k]` mask
    marking which entries were set by intervention rather than generated.

    The correlation matrix is recomputed per row-subset rather than cached once, because
    the subset depends on which variable is on the left-hand side of the test. Caching one
    matrix over all rows would silently reintroduce the rows the mask exists to remove.
    """

    def __init__(self, data: np.ndarray, intervened: Optional[np.ndarray] = None,
                 alpha: float = 0.01, foreign: Optional[np.ndarray] = None,
                 skeleton_alpha: Optional[float] = None,
                 exclude_foreign: bool = False):
        self.data = np.asarray(data, dtype=float)
        self.n, self.k = self.data.shape
        self.intervened = (np.zeros_like(self.data, dtype=bool) if intervened is None
                           else np.asarray(intervened) > 0.5)
        # `foreign[row]` -- a variable OUTSIDE the window is known to have been intervened
        # in this row. The federated case: another agent clamped one of its private nodes,
        # and this agent was told THAT it happened, never WHICH node (the regime bit,
        # `disclose_regime`). Such rows are a different regime for any descendant of the
        # clamped node, so they are excluded from the two-sample contrasts
        # (`ancestral_evidence`, `pair_power`) exactly like rows where a known third
        # variable was clamped -- otherwise the foreign clamp's effect on y is attributed
        # to whichever x is under test, which is bug 5 wearing a mask the agent cannot see
        # through. Without disclosure the mask is all-False and the contamination is the
        # honest price of privacy.
        self.foreign = (np.zeros(self.n, dtype=bool) if foreign is None
                        else np.asarray(foreign) > 0)
        self.alpha = float(alpha)
        # SEPARATE THRESHOLD FOR THE SKELETON, because the two uses have opposite error
        # costs. A missed edge in the skeleton is a wrong adjacency claim outright -- the
        # largest single error category measured (22 of 45 confident errors, 2026-08-25) --
        # while a spurious ORIENTATION signal corrupts the type claims, and the alpha sweep
        # showed type errors doubling from 16 to 32 as alpha went 0.01 -> 0.10. One shared
        # threshold cannot serve both. Defaults to `alpha`, so behaviour is unchanged unless
        # this is set deliberately.
        self.skeleton_alpha = float(alpha if skeleton_alpha is None else skeleton_alpha)
        # DROP FOREIGN ROWS FROM THE SKELETON'S INDEPENDENCE TESTS as well as from the
        # orientation channels, which already gate on `foreign`. A foreign intervention
        # puts those rows in a different REGIME: a foreign clamp removes a hidden common
        # cause and can destroy a dependence the observational rows carry, while a foreign
        # vary drives children and can create one. Pooling regimes is what JCI says not to
        # do naively (BIBLIOGRAPHY.md §19).
        #
        # MEASURED 2026-08-26, and the answer is NO. Paired, 30 episodes, greedy:
        #     pooled (default)    missed 12  invented 2  type wrong 14  identified 0.211
        #     foreign excluded    missed 12  invented 3  type wrong 17  identified 0.156
        # Exclusion recovers no edges at all and costs identification, because it costs
        # ROWS -- and the engine's failure mode here is under-powered tests, not regime
        # contamination. The JCI argument is sound and simply does not bite at this row
        # count. Off by default; kept so the question stays answerable rather than
        # re-litigated. See `docs/FINDINGS_2026_08_26.md` §13.
        self.exclude_foreign = bool(exclude_foreign)
        self.calls = 0
        self._cache: dict = {}
        # How many variables are intervened in each row -- lets any "no third variable
        # clamped" mask be one vector comparison instead of a column gather (2026-08-25).
        self._n_intervened = self.intervened.sum(axis=1)
        self._groups = None            # lazy sufficient statistics; see _suffstats

    # -- sufficient statistics -----------------------------------------------------------

    def _suffstats(self):
        """Per group: row count, column sums, cross-products, and the group's foreign flag.

        THE ALGORITHMIC HEART OF THE 2026-08-25 SPEEDUP. Rows fall into a handful of
        groups by WHICH variables were intervened (observational block, one group per
        intervened set). Any pair's usable-row correlation matrix is then assembled from
        the kept groups' sums in O(k^2), instead of an O(rows x k^2) corrcoef per pair.
        Products are taken on data CENTERED by the global column means -- covariance is
        invariant to any fixed shift, and centering keeps the one-pass formula
        numerically close to corrcoef's two-pass result.

        The FOREIGN flag joins the grouping key (2026-08-26) so that whole groups can be
        dropped when `exclude_foreign` is set. It is part of the key rather than a separate
        mask because the statistics are per group: a group mixing foreign and native rows
        could not be split afterwards without going back to the rows.
        """
        if self._groups is None:
            key = np.column_stack([self.intervened, self.foreign.astype(self.intervened.dtype)])
            patterns, inverse = np.unique(key, axis=0, return_inverse=True)
            center = self.data.mean(axis=0)
            shifted = self.data - center
            counts, sums, prods = [], [], []
            for g in range(len(patterns)):
                rows = shifted[inverse == g]
                counts.append(len(rows))
                sums.append(rows.sum(axis=0))
                prods.append(rows.T @ rows)
            self._groups = (patterns[:, :-1].astype(bool), np.asarray(counts),
                            np.asarray(sums), np.asarray(prods),
                            patterns[:, -1].astype(bool))
        return self._groups

    def _pair_corr(self, x: int, y: int):
        """(n_rows, [k, k] correlation matrix) over the rows usable for pair (x, y).

        A GROUP is dropped when either variable was intervened in it: a test is only
        meaningful when both sides were free to vary as the mechanism dictates, and
        independence is symmetric so the rule must be too.

        THIS IS THE ONLY ROW FILTER. A `_rows_for` method sat below this one until
        2026-08-26 stating the same rule per row, and was called by nothing -- so a reader
        looking for "where are rows selected" found a plausible answer that had no effect on
        any number. Deleted rather than wired up. `exclude_foreign` is the live question it
        was standing in for.
        """
        key = ("corr", self.exclude_foreign, x, y) if x <= y else \
              ("corr", self.exclude_foreign, y, x)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        patterns, counts, sums, prods, foreign = self._suffstats()
        keep = ~(patterns[:, x] | patterns[:, y])
        if self.exclude_foreign:
            keep = keep & ~foreign
        n = int(counts[keep].sum())
        if n < MIN_ROWS:
            hit = (n, None)
        else:
            s = sums[keep].sum(axis=0)
            p = prods[keep].sum(axis=0)
            cov = (p - np.outer(s, s) / n) / (n - 1)
            sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = cov / np.outer(sd, sd)
            corr[~np.isfinite(corr)] = np.nan
            hit = (n, corr)
        self._cache[key] = hit
        return hit

    # -- the test -----------------------------------------------------------------------

    def independent(self, x: int, y: int, cond: Sequence[int] = ()) -> bool:
        """Is `x` independent of `y` given `cond`, on the usable rows?

        The correlation matrix over ALL columns is computed ONCE per pair and cached:
        the usable rows depend only on (x, y), so every conditioning set the skeleton
        tries for that pair reads submatrices of the same cached matrix (2026-08-25,
        from the profile that found the engine spending its time in call overhead).
        """
        self.calls += 1
        cond = [c for c in cond if c not in (x, y)]
        n_rows, full = self._pair_corr(x, y)
        dof = n_rows - len(cond) - 3
        if full is None or dof <= 0:
            return True

        index = [x, y] + list(cond)
        corr = full[np.ix_(index, index)]
        # A clamped variable is constant within its rows, so its column is degenerate
        # even after masking: corrcoef leaves NaNs there, read as independence rather
        # than propagated into the skeleton (same verdict as the old spread guard).
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
        p_value = 2.0 * stats.norm.sf(abs(np.sqrt(dof) * z))
        # `skeleton_alpha`, which defaults to `alpha`. The orientation channels below keep
        # using `alpha` -- see the note in __init__ for why they must differ.
        return bool(p_value > self.skeleton_alpha)

    # -- interventional orientation -----------------------------------------------------

    def ancestral_evidence(self, min_rows: int = MIN_ROWS) -> np.ndarray:
        """`[k, k]` bool: `out[x, y]` iff clamping `x` demonstrably changed `y`.

        For each variable clamped somewhere in the data, compare `y`'s distribution between
        the rows where `x` was clamped and the rows where it was not. A shift means `x` is
        an ANCESTOR of `y` -- not necessarily a parent, and the direction is certain.

        MEAN **AND** VARIANCE, and the second is not optional -- it is the one that fires.
        A CLAMP sets x to a constant 0.0, and observationally E[x] is already 0, so a child
        of x sees NO MEAN SHIFT at all. What it sees is its variance drop, because one of
        its input terms stopped varying. A mean-only test detects nothing on exactly the
        experiment our agents perform.

        This was found by validating against known graphs, not by reading the code: with a
        mean-only test, a plain chain and a textbook collider both came back fully
        confounded, because `orient` step 4 reads "no ancestral evidence in either
        direction" as a latent common cause -- and absent detection is indistinguishable
        from absent effect.

        Welch's t-test for the mean, Levene for the variance; either firing is evidence.
        Levene rather than an F-test because it does not itself assume normality, and the
        clamped subsample is degenerate by construction.

        Still SOUND, not complete: an effect that moves neither mean nor variance is missed.
        Every ancestry reported is real; not every real one is reported.

        BOTH GROUPS EXCLUDE ROWS WHERE ANY THIRD VARIABLE WAS CLAMPED, and this fixed the
        FOURTH bug this test has had (found 2026-08-24, exposed by the per-pair power work).
        The previous comparison group was simply "x free, y free" -- which, in an episode
        with several clamp blocks, mixes in rows where some OTHER node z was clamped. If z
        is an ancestor of y, y's distribution in those rows genuinely differs, and the test
        attributes z's effect to x: on the hidden-confounder validation graph, clamping a
        CHILDLESS SINK was reported as moving a node two edges away. That false entry then
        satisfied the old global power check, so the latent-detection test passed BECAUSE
        of this bug. Restricting both groups to rows where no third variable is intervened
        makes the comparison a clean two-regime contrast; it costs rows, and costing rows
        is sound -- fewer rows means less power, never a wrong attribution.
        """
        out = np.zeros((self.k, self.k), dtype=bool)
        for x in range(self.k):
            clamped = self.intervened[:, x]
            if int(clamped.sum()) < min_rows:
                continue
            for y in range(self.k):
                if y == x or self.intervened[:, y].all():
                    continue
                # y itself free, NO third variable clamped, no FOREIGN clamp -- see the
                # class docstring for the last one.
                free = (~self.intervened[:, y] & ~self.foreign
                        & ((self._n_intervened - self.intervened[:, x]
                            - self.intervened[:, y]) == 0))
                a, b = self.data[clamped & free, y], self.data[(~clamped) & free, y]
                if len(a) < min_rows or len(b) < min_rows:
                    continue
                if a.std() < 1e-12 and b.std() < 1e-12:
                    continue
                # THIRD CHANNEL, FIRST-ORDER, added 2026-08-24: within the intervened
                # rows x's values are EXOGENOUS -- set by the experimenter, not by x's
                # parents -- so any correlation with y there is causation, read directly.
                # This is what a randomised (vary-mode, scale > 0) intervention buys: the
                # mean and variance channels detect a clamp-to-0 only through second-order
                # effects (detectable variance ratio needs |r| >~ 0.5 at our block sizes),
                # while a correlation against randomised values has 1/sqrt(n) power
                # (|r| ~ 0.2 at n=250). On clamp-to-0 data x is constant in these rows and
                # the channel is inert by the spread guard -- purely additive.
                #
                # DIRECT NUMPY FORMULAS, NOT scipy CALLS (2026-08-25). Profiling put 70%
                # of all training compute in this method's ~5000 scipy calls per episode
                # -- each ~1 ms of call overhead around microseconds of arithmetic. The
                # formulas below ARE Welch's t, Brown-Forsythe Levene (scipy's default,
                # center='median'), and Pearson's r with the t-distributed p -- verified
                # verdict-identical against scipy across random datasets in
                # tests/cb/test_fast_stats.py. Only the call overhead was removed.
                a_x = self.data[clamped & free, x]
                p_corr = _pearson_p(a_x, a)
                p_mean = _welch_p(a, b)
                p_var = _brown_forsythe_p(a, b)
                fired = [q for q in (p_mean, p_var, p_corr) if np.isfinite(q)]
                if fired and min(fired) < self.alpha:
                    out[x, y] = True
        return out

    def pair_power(self, min_rows: int = MIN_ROWS, target_power: float = 0.8) -> np.ndarray:
        """`[k, k]` bool: `out[x, y]` iff the clamp on `x` had adequate power to detect an
        effect on `y`, had the observed x-y dependence been causal.

        THE QUESTION THIS ANSWERS, PER PAIR: "if x -> y were real, would our experiment
        have seen it?" Only when the answer is yes does 'we saw nothing' carry evidence --
        which is what `orient` step 4 needs, and what its previous GLOBAL check
        (`ancestral[x].any()`) could not supply: a clamp that visibly moved some third node
        proves the experiment ran, not that it could have detected an effect on `y`.

        THE EFFECT SIZE COMES FROM THE PAIR ITSELF. Step 4 only asks about pairs the
        skeleton kept adjacent, so a dependence between x and y has already been measured.
        Under linear-Gaussian with a hard clamp, if the marginal correlation r were entirely
        causal (x -> y), clamping x would shrink y's variance by the factor 1 - r^2. That is
        the effect the experiment should have shown, so the power question is concrete:
        with the clamped/free row counts actually available, can a variance test at
        `self.alpha` detect a variance ratio of 1 - r^2 with probability `target_power`?

        MARGINAL r, not partial, and that is a choice: ancestry is a TOTAL-effect claim, so
        the total dependence is the right scale. Where the dependence is mediated the
        marginal overstates the DIRECT effect, but a real x -> y ancestry fires
        `ancestral_evidence` through the mediator anyway, so no false confounder results.

        Power is computed on the log-variance-ratio scale, where the sample statistic is
        asymptotically normal with sd ~ sqrt(2/(n1-1) + 2/(n2-1)) for Gaussian data. This is
        an approximation to the Levene test actually used for detection; it errs
        conservative for heavy tails, and conservative here means 'undetermined', never
        'confounded'.

        A weak edge therefore lands in CIRCLE, not in a wrong mark: no power calculation
        conjures detection out of a sample the effect is invisible in. Sound, not complete.
        """
        z_alpha = stats.norm.ppf(1.0 - self.alpha / 2.0)
        z_power = stats.norm.ppf(target_power)
        out = np.zeros((self.k, self.k), dtype=bool)
        # The pure-regime anchor rows are the same for EVERY pair; one correlation matrix
        # replaces k^2 per-pair corrcoef calls (2026-08-25, from the same profile).
        pure = ~self.foreign & ~self.intervened.any(axis=1)
        if int(pure.sum()) < min_rows:
            return out
        pure_data = self.data[pure]
        pure_sd = pure_data.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pure_corr = np.corrcoef(pure_data, rowvar=False)
        for x in range(self.k):
            clamped = self.intervened[:, x]
            for y in range(self.k):
                if y == x:
                    continue
                # The same rows `ancestral_evidence` would compare: y free, no third
                # variable clamped, no foreign clamp, split by clamp-x. Power must be
                # computed on the sample the detection test actually gets, or it is power
                # for a different test.
                free = (~self.intervened[:, y] & ~self.foreign
                        & ((self._n_intervened - self.intervened[:, x]
                            - self.intervened[:, y]) == 0))
                n1 = int((clamped & free).sum())
                n2 = int(((~clamped) & free).sum())
                if n1 < min_rows or n2 < min_rows:
                    continue                      # the detection test itself would not run
                # THE EFFECT-SIZE ANCHOR COMES FROM PURE-REGIME ROWS ONLY -- no window
                # clamp anywhere, no foreign clamp. Found on the first cross-check against
                # the exact engine, not by inspection: pooling regimes DILUTES r, because
                # a clamp upstream of the pair severs the very dependence being measured.
                # On a strongly confounded pair (true r 0.70) the pooled estimate came
                # back 0.345 -- the de-confounding experiment destroyed the anchor that
                # sizes the effect the experiment must be able to detect, power failed,
                # and a detectable confounder was reported undetermined. The dependence a
                # causal reading must explain is the UNPERTURBED one.
                if pure_sd[x] <= 1e-12 or pure_sd[y] <= 1e-12:
                    continue
                r = float(pure_corr[x, y])
                if not np.isfinite(r):
                    continue
                r2 = min(r * r, 0.999999)

                # SECOND-ORDER CHANNEL (clamp-to-0): the variance test's detectable
                # log-ratio at these sample sizes.
                effect = abs(np.log(1.0 - r2))    # |log variance ratio| if r were causal
                se = np.sqrt(2.0 / (n1 - 1) + 2.0 / (n2 - 1))
                if effect / se >= z_alpha + z_power:
                    out[x, y] = True
                    continue

                # FIRST-ORDER CHANNEL (randomised interventions): if x's intervened
                # values vary, the detection instrument is the correlation of y against
                # those exogenous values. Had the pure-regime dependence r been causal,
                # the intervened-rows correlation would be r*s / sqrt(1 + r^2 (s^2 - 1))
                # with s = sigma_intervention / sigma_x(pure) -- the total-effect
                # regression coefficient driven by the intervention's own spread. Power
                # via Fisher z at the rows the correlation test actually gets. Inert on
                # clamp-to-0 data (spread guard), like the detection channel it mirrors.
                x_int = self.data[clamped & free, x]
                sd_int = float(x_int.std())
                sd_x = float(pure_sd[x])
                if sd_int > 1e-12 and sd_x > 1e-12 and n1 > 3:
                    s = sd_int / sd_x
                    r_int = abs(r) * s / np.sqrt(1.0 + r2 * (s * s - 1.0))
                    r_int = min(r_int, 0.999999)
                    z_effect = np.arctanh(r_int) * np.sqrt(n1 - 3)
                    if z_effect >= z_alpha + z_power:
                        out[x, y] = True
        return out

    def clamped_enough(self, min_rows: int = MIN_ROWS) -> np.ndarray:
        """`[k]` bool: was this variable clamped in enough rows to be informative?

        Needed to separate "we clamped x and y did not move" from "we never clamped x".
        Only the first is evidence. Without this distinction every never-experimented pair
        would look like a confounded one, which is the failure mode `orient` step 4 guards.
        """
        counts = self.intervened.sum(axis=0)
        free = (~self.intervened).sum(axis=0)
        return (counts >= min_rows) & (free >= min_rows)
