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


def sample(
    params: SCMParams,
    n: int,
    rng: np.random.Generator,
    intervene_node: Optional[int] = None,
    intervene_scale: float = 2.0,
) -> tuple:
    """Draw `n` samples, optionally under a hard intervention on `intervene_node`.

    Returns `(samples [n, d], intervened [n, d])`, where `intervened[i, j]` marks that
    node j was set by intervention in sample i. The mask travels with the data because
    the estimator needs it: a hard-intervened node's samples say nothing about its own
    parents, though they remain valid parent values for its children.

    The intervention assigns a *random* value per sample, `X_i ~ N(0, intervene_scale)`,
    rather than one fixed constant. It is still a hard intervention -- the structural
    equation is replaced and the parents are disconnected -- but the assigned value
    varies. This matters for identifiability from finite data: a constant value has zero
    variance, so the intervened column is collinear with the intercept and the
    descendants' dependence on it cannot be estimated from those samples. A varying value
    both shifts and *moves* the descendants, which is what actually separates orientations.
    `intervene_scale` is set above the noise range so the signal stands out.
    """
    d = params.d
    samples = np.zeros((n, d))
    intervened = np.zeros((n, d))

    for node in topological_order(params.adjacency):
        node = int(node)
        if node == intervene_node:
            # do(X_node): the structural equation is replaced, parents disconnected.
            samples[:, node] = rng.normal(0.0, intervene_scale, n)
            intervened[:, node] = 1.0
        else:
            parent_contribution = samples @ params.weights[:, node]
            noise = rng.normal(0.0, params.noise_scales[node], n)
            samples[:, node] = parent_contribution + noise

    return samples, intervened
