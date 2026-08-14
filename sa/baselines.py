"""Reference policies. Every learned result is reported against these.

Three of them, each answering a different question:

  `no_intervention` -- how much is solvable without acting at all. Backs GATE 1, and is
                       the control that catches the environment becoming degenerate again.
  `random`          -- the floor. Backs GATE 2: if the oracle cannot beat this, the
                       environment does not reward choosing well and there is nothing to
                       learn.
  `greedy_oracle`   -- the opponent. Best single next experiment, computed exactly.

A policy is a callable `(env, result) -> action`, matching `sa.gates.run_policy`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sa.env import PASS_ACTION, CausalDiscoveryEnv
from sa.oracle import InterventionOracle


def no_intervention_policy(env: CausalDiscoveryEnv, result) -> int:
    """Never acts. The episode ends immediately."""
    return PASS_ACTION


class RandomPolicy:
    """Intervenes on a uniformly random node until the budget runs out.

    Deliberately never passes: this is the floor for *choice quality*, so it should spend
    the same budget as the oracle and differ only in which node it picks. A random policy
    that also passed at random would conflate two different kinds of badness.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: CausalDiscoveryEnv, result) -> int:
        return int(self.rng.integers(env.config.d))


class GreedyOraclePolicy:
    """Intervenes on the node with the highest expected information gain.

    Passes when no node is informative -- every remaining hypothesis agrees about every
    node's descendants, so no intervention can discriminate and spending budget is
    strictly wasteful. That is a real decision the oracle is entitled to make, and it
    makes "should I act at all" part of what the agent has to learn rather than something
    the action space decides for it.
    """

    def __init__(self, space, seed: int = 0):
        self.oracle = InterventionOracle(space)
        self.rng = np.random.default_rng(seed)

    def __call__(self, env: CausalDiscoveryEnv, result) -> int:
        scores, best = self.oracle.best_targets(result.posterior)
        if scores.max() <= 1e-9:
            return PASS_ACTION
        return int(self.rng.choice(np.flatnonzero(best)))


def make_baselines(space, seed: int = 0) -> dict:
    """The standard comparison set, built together so seeds stay aligned."""
    return {
        "no_intervention": no_intervention_policy,
        "random": RandomPolicy(seed=seed),
        "greedy_oracle": GreedyOraclePolicy(space, seed=seed),
    }
