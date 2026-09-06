"""Linear Gaussian structural causal model with hard interventions.

The one design decision that matters here is that **every node gets its own noise scale**,
drawn fresh each episode. That is not a detail -- it is the fix for the defect that
invalidated the previous round of results.

With a single shared noise scale, a linear Gaussian SEM becomes fully identifiable from
observational data alone (Peters & Buehlmann 2014, docs/THEORY_NOTES.md #1): the Markov
equivalence class collapses to a point and interventions stop being necessary. The
previous codebase used one scalar `noise_scale` for all nodes and, as a direct
consequence, roughly half its episodes were already solved before the agent acted -- in
many cases with the agent doing nothing at all. Drawing per-node scales restores the
intended regime, where observational data pins down the equivalence class and nothing
more.

Interventions are `hard`: do(X_i = v) replaces node i's structural equation outright, so
i no longer depends on its parents. Effects propagate to i's descendants, which is what
makes an intervention informative about orientation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SCMParams:
    """A concrete SCM: a graph, edge weights, and per-node noise scales."""

    adjacency: np.ndarray  # [d, d], adjacency[i, j] = 1 meaning i -> j
    weights: np.ndarray    # [d, d], weights[i, j] is the coefficient of i in j's equation
    noise_scales: np.ndarray  # [d]

    @property
    def d(self) -> int:
        return int(self.adjacency.shape[0])


def sample_scm_params(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    weight_range: tuple = (0.5, 2.0),
    noise_range: tuple = (0.5, 1.5),
) -> SCMParams:
    """Draw edge weights and per-node noise scales for a given graph.

    Weights avoid a neighbourhood of zero (magnitude in `weight_range`, random sign) so
    that every edge in the graph is actually detectable -- an edge with a near-zero
    coefficient is present in the graph but absent from the data, which would make the
    ground-truth label wrong rather than the task hard.

    `noise_range` must not be degenerate. A single shared value is exactly the
    equal-variance condition that leaks orientation information, so the default spans a
    3x range.
    """
    adjacency = np.asarray(adjacency)
    d = adjacency.shape[0]

    magnitude = rng.uniform(weight_range[0], weight_range[1], size=(d, d))
    sign = rng.choice((-1.0, 1.0), size=(d, d))
    weights = adjacency * magnitude * sign

    noise_scales = rng.uniform(noise_range[0], noise_range[1], size=d)
    return SCMParams(adjacency=adjacency.astype(np.int8), weights=weights, noise_scales=noise_scales)


@lru_cache(maxsize=256)
def _cached_order(payload: bytes, d: int, dtype: str) -> tuple:
    return tuple(int(x) for x in topological_order(
        np.frombuffer(payload, dtype=dtype).reshape(d, d)))


def order_for(adjacency: np.ndarray) -> np.ndarray:
    """`topological_order`, memoised on the adjacency matrix itself.

    The samplers re-derived the order on EVERY draw -- once per environment step, over a
    graph that does not change within an episode -- and the O(d^2) scan was 0.9 s of a
    43 s profiled run. Keyed on the raw bytes, so two runs with the same graph share the
    entry and a different graph cannot collide with it.

    The ORDER, not merely its validity, must be stable: the sampler draws noise node by
    node in this sequence, so a different (equally valid) ordering would consume the RNG
    differently and change the data. That is why this memoises the existing function
    rather than replacing it with a faster algorithm.
    """
    a = np.ascontiguousarray(adjacency)
    return np.array(_cached_order(a.tobytes(), a.shape[0], a.dtype.str), dtype=int)


def topological_order(adjacency: np.ndarray) -> np.ndarray:
    """A valid topological ordering, so each node can be generated after its parents."""
    a = np.asarray(adjacency) > 0.5
    d = a.shape[0]
    remaining = list(range(d))
    order = []
    while remaining:
        # A source among the remaining nodes has no remaining parents.
        sources = [j for j in remaining if not a[remaining, j].any()]
        if not sources:
            raise ValueError("adjacency contains a cycle; cannot order topologically")
        for s in sources:
            order.append(s)
            remaining.remove(s)
    return np.array(order, dtype=int)




MECHANISMS = ("linear", "tanh", "vshape")

# |z| for a zero-mean Gaussian z of standard deviation s has mean s*sqrt(2/pi) and variance
# s**2 * (1 - 2/pi). Centring and dividing by sqrt(1 - 2/pi) therefore returns mean 0 and
# variance s**2 -- the same two moments the linear mechanism would have delivered.
_ABS_MEAN = np.sqrt(2.0 / np.pi)          # 0.7979
_ABS_SD = np.sqrt(1.0 - 2.0 / np.pi)      # 0.6028


def _apply_mechanism(z, mechanism: str = "linear", scale=None):
    """The parent contribution, optionally passed through a nonlinearity.

    linear  z, unchanged: every headline result in the thesis.
    tanh    2*tanh(z/2). An additive-noise model with a nonlinear function of the parents,
            the standard identifiable-ANM form (Hoyer et al. 2009). Chosen for two
            properties: its slope at the origin is exactly 1, so small signals behave as
            in the linear case and the comparison is not confounded by a change of gain;
            and it is MONOTONE, so it attenuates all three detection channels rather than
            zeroing one. This is the ROBUSTNESS case.
    vshape  (|z| - s*sqrt(2/pi)) / sqrt(1 - 2/pi), with s the SCM's own scale for this node.
            The ADVERSARIAL case, added 6 Sep at Brian's request. An even function of the
            parents, which defeats the third detection channel in `cb/citest.py`: the
            Pearson correlation between a randomised intervention's assigned values and the
            child, the highest-power channel this engine has. Those assigned values are
            drawn Gaussian and independent of everything upstream, so for a direct child
            E[x * f(z)] = 0 and the channel reads zero however many rows it is given. The
            mean and variance channels still fire.

            WHY |z| AND NOT z**2, which is the obvious even function and was tried first.
            Squaring is not merely nonlinear, it is super-linear, so a tail value at one
            node becomes a far larger tail value at the next and the graph compounds it.
            Measured in this project's own generator: max|X| over 25 episodes ran to
            3.2e16 against 63.8 under `linear`, and Welch's t overflowed on the resulting
            variances. That corner would have measured floating-point degeneracy, not
            robustness. |z| grows linearly, so the recursion is as stable as the linear
            case while the evenness -- the property the whole corner rests on -- is
            identical.

            SCALED, NOT RAW, and s comes from the SCM rather than from the rows being drawn
            (`_vshape_scales`). Both moments then match the linear mechanism: mean 0 and
            variance s**2. Without that the gain would change alongside the correlation
            channel and the two effects would be inseparable -- the second-variable-moved
            error this project has caught four times. Estimating s per block instead would
            attenuate the signal by ~11% at n_int=20 and ~3% at n_obs=60, a per-block
            artefact that varies with block size.

            Centring is exact only in expectation here, since s is the SCM's scale and not
            the block's, which leaves the mean channel behaving as it does under `linear`.

    Roots are unaffected: with no parents z is 0 and every mechanism here fixes 0.
    """
    if mechanism == "linear":
        return z
    if mechanism == "tanh":
        return 2.0 * np.tanh(z / 2.0)
    if mechanism == "vshape":
        if scale is None:                      # fallback only; callers pass the SCM scale
            scale = float(np.sqrt(np.mean(z * z)))
        if scale < 1e-12:                      # a root, or a node with no parent variance
            return np.zeros_like(z)
        return (np.abs(z) - scale * _ABS_MEAN) / _ABS_SD
    raise ValueError(f"unknown mechanism {mechanism!r}; expected one of {MECHANISMS}")


def _vshape_scales(params: "SCMParams", targets: dict):
    """Per-node standard deviation of the parent contribution under `vshape`.

    Under `vshape` a child is uncorrelated with each of its parents whenever the parent is
    symmetric, so the off-diagonal terms in `Var(X'w)` vanish and the variance recursion
    collapses to a sum of squares that can be walked in topological order.

    NOT EXACT EVERYWHERE, and the deviation is measured rather than assumed. Squaring makes
    a node skewed, and `Var(f(z)) = Var(z)` relies on `E[z**4] = 3 Var(z)**2`, which is the
    Gaussian value. Deep in a graph, where z is a sum of already-skewed terms, the fourth
    moment runs high and the node's variance with it. Measured over three random 8-node
    SCMs at 300,000 rows: most nodes sit within 0.2% of the linear generator's marginal
    standard deviation, and a minority reach +38%. So this corner is deliberately
    adversarial rather than a perfectly controlled single-variable change -- which does not
    affect the comparison, since all three arms face the identical generator.

    `targets` maps an intervened node to the standard deviation it is drawn at (0.0 for a
    clamp). An intervened node contributes that variance instead of its structural one,
    because its equation has been replaced -- so the scale a child's mechanism uses is
    correct in the interventional block as well as the observational one, which is what
    keeps the Brown-Forsythe channel reading a real variance contrast rather than a
    normalisation artefact.
    """
    d = params.d
    var_x = np.zeros(d)
    scales = np.zeros(d)
    for node in order_for(params.adjacency):
        node = int(node)
        w = params.weights[:, node]
        scales[node] = np.sqrt(float(np.sum(w * w * var_x)))
        var_x[node] = (float(targets[node]) ** 2 if node in targets
                       else scales[node] ** 2 + float(params.noise_scales[node]) ** 2)
    return scales


NOISE_DISTS = ("gaussian", "uniform", "t3")


def _draw_noise(rng, scale: float, n: int, dist: str = "gaussian"):
    """Noise of the requested SHAPE at the requested standard deviation.

    STANDARDISED ON PURPOSE. Every distribution here is scaled to unit variance before
    `scale` is applied, so switching `dist` changes the shape of the noise and nothing else.
    Drawing Student-t(3) at `scale` directly would also multiply the noise MAGNITUDE by
    sqrt(3), and the comparison would confound tail weight with signal-to-noise -- the
    second-variable-moved error this project has caught four times.

    gaussian  N(0, 1) * scale.
    uniform   U(-sqrt(3), sqrt(3)) * scale. Bounded, light-tailed; the classic
              linear-non-Gaussian case, under which the DAG is identifiable from
              observational data alone (Shimizu et al. 2006) -- which this engine cannot
              exploit, since it reads only the first two moments and a linear correlation.
    t3        Student-t with 3 degrees of freedom, divided by sqrt(3) (its own standard
              deviation), times scale. Heavy-tailed: finite variance, infinite kurtosis.
              The adversarial case for the correlation channel's p-values.
    """
    if dist == "gaussian":
        return rng.normal(0.0, scale, n)
    if dist == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n) * scale
    if dist == "t3":
        return rng.standard_t(3, n) / np.sqrt(3.0) * scale
    raise ValueError(f"unknown noise_dist {dist!r}; expected one of {NOISE_DISTS}")


def sample(
    params: SCMParams,
    n: int,
    rng: np.random.Generator,
    intervene_node: Optional[int] = None,
    intervene_scale: float = 2.0,
    noise_dist: str = "gaussian",
    mechanism: str = "linear",
) -> tuple:
    """Draw `n` samples, optionally under a hard intervention on `intervene_node`.

    Returns `(samples [n, d], intervened [n, d])`, where `intervened[i, j]` marks that
    node j was set by intervention in sample i. The mask travels with the data because
    the estimator needs it: a hard-intervened node's samples say nothing about its own
    parents, though they remain valid parent values for its children.

    The intervention assigns a *random* value per sample, `X_i ~ N(0, intervene_scale)`,
    rather than one fixed constant. It is still a hard intervention -- the structural
    equation is replaced and the parents are disconnected -- but the assigned value varies.

    [CORRECTED 2026-08-20] An earlier version of this docstring claimed a constant value is
    "collinear with the intercept" so that "the descendants' dependence on it cannot be
    estimated from those samples". THAT IS TOO STRONG, and measuring it says so: a constant
    intervention recovers 93-98% of the information a varying one does (d=4 and d=5, 40 and
    25 random graphs, posterior entropy over the DAG space).

    The reason the strong claim fails is POOLING. Collinearity would bite only if the
    interventional batch were scored on its own. It is not -- it is pooled with the
    observational rows, and the clamped rows sit at a different location in
    (X_i, descendant) space from the observational cloud, so the slope is identified by the
    contrast BETWEEN regimes even though X_i has zero variance WITHIN the clamped batch.
    Consistent with that, clamping at 2, 4 or 16 distinct levels does not close the small
    remaining gap -- so it is not a degrees-of-freedom effect either.

    Varying is still the right default, for two reasons that survive: the residual few
    percent, and the fact that `intervene_scale` above the noise range makes the signal
    stand out. But the two modes are NOT far apart for learning your own structure. Where
    they genuinely diverge is de-confounding for a PARTNER -- see `ma/env2.py`: a randomly
    varying hidden node is still a variance source, so rescue rate is 0.000 at scale 2.0 and
    1.0 and rises only as the scale goes to zero, i.e. as the intervention becomes a
    constant. Clamping is essential there and varying is useless.
    """
    d = params.d
    samples = np.zeros((n, d))
    intervened = np.zeros((n, d))
    scales = (_vshape_scales(params, {} if intervene_node is None
                             else {int(intervene_node): float(intervene_scale)})
              if mechanism == "vshape" else None)

    for node in order_for(params.adjacency):
        node = int(node)
        if node == intervene_node:
            # do(X_node): the structural equation is replaced, parents disconnected.
            samples[:, node] = rng.normal(0.0, intervene_scale, n)
            intervened[:, node] = 1.0
        else:
            parent_contribution = _apply_mechanism(
                samples @ params.weights[:, node], mechanism,
                None if scales is None else float(scales[node]))
            noise = _draw_noise(rng, params.noise_scales[node], n, noise_dist)
            samples[:, node] = parent_contribution + noise

    return samples, intervened


def sample_multi(
    params: SCMParams,
    n: int,
    rng: np.random.Generator,
    intervene_nodes=(),
    intervene_scale: float = 2.0,
    noise_dist: str = "gaussian",
    mechanism: str = "linear",
) -> tuple:
    """As `sample`, but with SEVERAL nodes intervened on at once.

    Needed for the two-agent case, where both agents act in the same round on one shared
    system (docs/MA_DESIGN.md section 7: separate budgets, simultaneous experiments, no
    collision rule). Two agents choosing the same node is not an error -- it is one
    intervention that both of them asked for, and it is handled here by the set semantics
    rather than by an arbitration rule.

    Kept as a separate function rather than folded into `sample` so that every existing
    single-agent result stays byte-identical.
    """
    d = params.d
    # `intervene_nodes` may be a plain iterable of node ids (all sharing
    # `intervene_scale`) or a mapping node -> scale. A scale of 0.0 CLAMPS the node to a
    # constant; a positive scale RANDOMISES it. The distinction is not cosmetic:
    #
    #   randomising  keeps the node varying, which is what lets an intervention reveal
    #                its descendants' dependence on it (see `sample`), but leaves it an
    #                active source of variance for everything it points into;
    #   clamping     removes it as a variance source entirely, which is the only way to
    #                cut a confounding path through it.
    #
    # Measured 2026-08-16: with scale 2.0 or 1.0 a do() on the confounder restores 0.0%
    # of a confounded agent's identification; at scale 0.1 or 0.0 it restores ~18% and
    # lifts mean posterior mass on the truth from 0.0000 to 0.39.
    if isinstance(intervene_nodes, dict):
        targets = {int(k): float(v) for k, v in intervene_nodes.items()}
    else:
        targets = {int(v): float(intervene_scale) for v in intervene_nodes}
    samples = np.zeros((n, d))
    intervened = np.zeros((n, d))
    scales = _vshape_scales(params, targets) if mechanism == "vshape" else None

    for node in order_for(params.adjacency):
        node = int(node)
        if node in targets:
            scale = targets[node]
            samples[:, node] = (rng.normal(0.0, scale, n) if scale > 0.0
                                else np.zeros(n))
            intervened[:, node] = 1.0
        else:
            parent_contribution = _apply_mechanism(
                samples @ params.weights[:, node], mechanism,
                None if scales is None else float(scales[node]))
            noise = _draw_noise(rng, params.noise_scales[node], n, noise_dist)
            samples[:, node] = parent_contribution + noise

    return samples, intervened
