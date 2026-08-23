"""Statistics shared across the multi-agent code.

Two functions, lifted verbatim on 2026-08-23 from `sa/gates.py` and `sa/oracle.py`
when `sa/` was dissolved. They were the ONLY things `ma/` needed from those modules,
and both modules dragged in the entire single-agent path (env, evaluate, samplers) to
supply them. Copying the two functions was cheaper than keeping the dependency.

`_partition_entropy` keeps its underscore name so existing call sites are unchanged,
despite now being public API of this module.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    """Percentile bootstrap interval for the mean.

    Every reported number carries one of these. At 8 episodes per condition -- the
    previous round's sample size -- a single episode moved a rate by 12.5 points, and a
    29-point swing was observed from floating-point noise alone. Intervals make that
    visible instead of inviting over-reading.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def _partition_entropy(labels: np.ndarray, weights: np.ndarray, n_groups: int) -> float:
    """H of the outcome partition, in nats -- the expected information gain of one target.

    `labels[k]` is which descendant-set group hypothesis `k` falls into and `weights[k]` is
    its posterior mass (or `1/n_draws` for sampled hypotheses). Zero when every plausible
    graph agrees, however unexplored the node looks.
    """
    mass = np.bincount(labels, weights=weights, minlength=n_groups)
    mass = mass[mass > 0]
    return float(-np.sum(mass * np.log(mass)))
