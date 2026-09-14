"""Linear Gaussian structural causal model with hard interventions.

Every node draws its own noise scale, fresh each episode, and the range spans a factor of
three. This is deliberate rather than incidental. A linear Gaussian model whose error
variances are all equal is identifiable from observational data alone (Peters and
Buehlmann, 2014), which would collapse the Markov equivalence class to a point and make
interventions unnecessary. Unequal variances keep the setting in the one family where
observational data fixes the equivalence class and nothing more, so interventions are
required rather than merely helpful.

Interventions are hard: do(X = v) replaces node X's structural equation outright, so X no
longer depends on its parents and the effect propagates to X's descendants. The assigned
value varies per sample by default, which lets a randomised intervention identify effects
through correlation as well as through shifts in mean and variance.

`NOISE_DISTS` and `MECHANISMS` provide the robustness families the experiments sweep. Noise
shapes are standardised to unit variance before the per-node scale is applied, so changing
the shape changes only the shape.
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

    linear z unchanged. Every headline result uses this.
    tanh 2*tanh(z/2). An additive-noise model with a nonlinear parent function, the
            standard identifiable form (Hoyer et al., 2009). Its slope at the origin is 1,
            so small signals behave as in the linear case and the comparison is not
            confounded by a change of gain, and it is monotone, so it attenuates all three
            detection channels rather than removing one. This is the robustness case.
    vshape (|z| - s*sqrt(2/pi)) / sqrt(1 - 2/pi), with s the SCM's own scale for the node.
            The adversarial case. Being an even function of the parents it defeats the
            correlation channel of `cb/citest.py`: a randomised intervention's assigned
            values are drawn independently of everything upstream, so for a direct child
            E[x * f(z)] = 0 and that channel reads zero at any sample size. The mean and
            variance channels still fire.

            |z| rather than z**2 because squaring is super-linear, so a tail value at one
            node becomes a larger one at the next and the graph compounds it; measured
            max|X| reached 3.2e16 against 63.8 under `linear`, which would test
            floating-point behaviour rather than robustness. |z| grows linearly and keeps
            the recursion as stable as the linear case.

            The scale comes from the SCM (`_vshape_scales`) rather than from the rows being
            drawn, so mean and variance both match the linear mechanism and only the
            correlation structure changes. Estimating it per block would attenuate the
            signal by about 11% at 20 rows, which would move signal strength alongside the
            correlation channel and make the two inseparable.

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

    Under an even mechanism a child is uncorrelated with each symmetric parent, so the
    off-diagonal terms of Var(X'w) vanish and the variance recursion becomes a sum of
    squares that can be walked in topological order.

    The recursion is exact only where z is close to Gaussian. Squaring makes a node skewed,
    and Var(f(z)) = Var(z) relies on the Gaussian fourth moment, so deep in a graph the
    variance runs high: measured over three random eight-node SCMs, most nodes sit within
    0.2% of the linear generator's marginal standard deviation and a minority reach +38%.
    The corner is therefore deliberately adversarial rather than a controlled
    single-variable change, which does not affect the comparison because all arms face the
    identical generator.

    `targets` maps an intervened node to the standard deviation it is drawn at, 0.0 for a
    clamp, so a child's scale is correct in the interventional block as well as the
    observational one.
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

    Every distribution is scaled to unit variance before `scale` is applied, so switching
    `dist` changes the shape of the noise and nothing else. Drawing Student-t(3) at `scale`
    directly would also multiply the noise magnitude by sqrt(3), which would confound tail
    weight with signal-to-noise.

    gaussian N(0, 1) * scale.
    uniform U(-sqrt(3), sqrt(3)) * scale. Bounded, light-tailed; the classic
              linear-non-Gaussian case, under which the DAG is identifiable from
              observational data alone (Shimizu et al. 2006) -- which this engine cannot
              exploit, since it reads only the first two moments and a linear correlation.
    t3 Student-t with 3 degrees of freedom, divided by sqrt(3) (its own standard
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

    [CORRECTED] An earlier version of this docstring claimed a constant value is
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
    system (docs/ARCHITECTURE.md section 7: separate budgets, simultaneous experiments, no
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
    # randomising keeps the node varying, which is what lets an intervention reveal
    # its descendants' dependence on it (see `sample`), but leaves it an
    # active source of variance for everything it points into;
    # clamping removes it as a variance source entirely, which is the only way to
    # cut a confounding path through it.
    #
    #: with scale 2.0 or 1.0 a do on the confounder restores 0.0%
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
