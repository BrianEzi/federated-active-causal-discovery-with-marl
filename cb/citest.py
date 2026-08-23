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


class FisherZ:
    """Partial-correlation independence test with per-row intervention masking.

    `data` is `[n, k]` for ONE agent's window; `intervened` is the matching `[n, k]` mask
    marking which entries were set by intervention rather than generated.

    The correlation matrix is recomputed per row-subset rather than cached once, because
    the subset depends on which variable is on the left-hand side of the test. Caching one
    matrix over all rows would silently reintroduce the rows the mask exists to remove.
    """

    def __init__(self, data: np.ndarray, intervened: Optional[np.ndarray] = None,
                 alpha: float = 0.01, foreign: Optional[np.ndarray] = None):
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
        self.calls = 0
        self._cache: dict = {}

    # -- row selection ------------------------------------------------------------------

    def _rows_for(self, x: int, y: int) -> np.ndarray:
        """Rows usable for a test between `x` and `y`.

        A row is dropped when EITHER variable was clamped in it. Symmetric because
        independence is symmetric: a test is only meaningful when both sides were free to
        vary as the mechanism dictates.
        """
        key = (x, y) if x <= y else (y, x)
        hit = self._cache.get(key)
        if hit is None:
            hit = ~(self.intervened[:, x] | self.intervened[:, y])
            self._cache[key] = hit
        return hit

    # -- the test -----------------------------------------------------------------------

    def independent(self, x: int, y: int, cond: Sequence[int] = ()) -> bool:
        """Is `x` independent of `y` given `cond`, on the usable rows?"""
        self.calls += 1
        cond = [c for c in cond if c not in (x, y)]
        rows = self._rows_for(x, y)
        n_rows = int(rows.sum())
        dof = n_rows - len(cond) - 3
        if n_rows < MIN_ROWS or dof <= 0:
            return True

        sub = self.data[rows][:, [x, y] + list(cond)]
        # A clamped variable is constant within its rows, so its column can be degenerate
        # even after masking. np.corrcoef would emit a divide warning and NaNs; catch it
        # here and report independence rather than propagating NaN into the skeleton.
        spread = sub.std(axis=0)
        if not np.all(spread > 1e-12):
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
        p_value = 2.0 * stats.norm.sf(abs(np.sqrt(dof) * z))
        return bool(p_value > self.alpha)

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
                free = ~self.intervened[:, y] & ~self.foreign
                others = [c for c in range(self.k) if c not in (x, y)]
                if others:
                    free &= ~self.intervened[:, others].any(axis=1)
                a, b = self.data[clamped & free, y], self.data[(~clamped) & free, y]
                if len(a) < min_rows or len(b) < min_rows:
                    continue
                if a.std() < 1e-12 and b.std() < 1e-12:
                    continue
                _, p_mean = stats.ttest_ind(a, b, equal_var=False)
                try:
                    _, p_var = stats.levene(a, b)
                except ValueError:
                    p_var = np.nan
                fired = [q for q in (p_mean, p_var) if np.isfinite(q)]
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
        for x in range(self.k):
            clamped = self.intervened[:, x]
            for y in range(self.k):
                if y == x:
                    continue
                # The same rows `ancestral_evidence` would compare: y free, no third
                # variable clamped, no foreign clamp, split by clamp-x. Power must be
                # computed on the sample the detection test actually gets, or it is power
                # for a different test.
                free = ~self.intervened[:, y] & ~self.foreign
                others = [c for c in range(self.k) if c not in (x, y)]
                if others:
                    free &= ~self.intervened[:, others].any(axis=1)
                n1 = int((clamped & free).sum())
                n2 = int(((~clamped) & free).sum())
                if n1 < min_rows or n2 < min_rows:
                    continue                      # the detection test itself would not run
                # Dependence measured where BOTH were free -- the same rows the skeleton's
                # adjacency verdict for this pair rests on.
                both_free = self._rows_for(x, y)
                if int(both_free.sum()) < min_rows:
                    continue
                sub = self.data[both_free][:, [x, y]]
                if not np.all(sub.std(axis=0) > 1e-12):
                    continue
                r = float(np.corrcoef(sub, rowvar=False)[0, 1])
                if not np.isfinite(r):
                    continue
                r2 = min(r * r, 0.999999)
                effect = abs(np.log(1.0 - r2))    # |log variance ratio| if r were causal
                se = np.sqrt(2.0 / (n1 - 1) + 2.0 / (n2 - 1))
                out[x, y] = bool(effect / se >= z_alpha + z_power)
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
