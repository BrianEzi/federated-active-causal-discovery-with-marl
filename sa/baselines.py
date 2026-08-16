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
from sa.oracle import InterventionOracle, SamplingOracle
from sa.posterior import edge_marginals


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
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Restore the RNG so repeated evaluations of this policy are identical.

        Without this, evaluating the same policy twice gives different results because the
        generator has advanced -- which made `random` score gap-closed 0.233 instead of the
        0.000 it is by definition, and `greedy` 1.067 instead of 1.000.
        """
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

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
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """See `RandomPolicy.reset` -- tie-breaking must be reproducible across runs."""
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

    def __call__(self, env: CausalDiscoveryEnv, result) -> int:
        scores, best = self.oracle.best_targets(result.posterior)
        if scores.max() <= 1e-9:
            return PASS_ACTION
        return int(self.rng.choice(np.flatnonzero(best)))


class GreedyOracleDPPolicy:
    """`GreedyOraclePolicy` on the enumeration-free path.

    Identical decision rule -- highest expected information gain, pass when nothing is
    informative, random tie-breaking -- but reading a sampled oracle instead of an
    enumerated one, because at d=7 there is no DAG list to compute descendant sets over.
    The two agree on 95% of choices at d=5 and differ mainly on near-ties, where the exact
    oracle is itself indifferent; see docs/SA_EXPERIMENT_LOG.md for the regret against an
    ideal-sampling floor.

    `result.posterior` here is the `[d, 2^(d-1)]` log-weight table, not a distribution over
    graphs -- see `sa/env_dp.py`.
    """

    def __init__(self, dp, n_draws: int = 4000, seed: int = 0):
        self.oracle = SamplingOracle(dp, n_draws=n_draws, seed=seed)
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """See `RandomPolicy.reset` -- tie-breaking must be reproducible across runs."""
        self.rng = np.random.default_rng(self._seed if seed is None else seed)
        self.oracle.seed = self._seed if seed is None else seed
        self.oracle._calls = 0

    def __call__(self, env, result) -> int:
        scores, best = self.oracle.best_targets(result.posterior)
        if scores.max() <= 1e-9:
            return PASS_ACTION
        return int(self.rng.choice(np.flatnonzero(best)))


class EdgeMarginalGreedyPolicy:
    """Greedy EIG computed from edge marginals alone -- the opponent for a condition-B agent.

    Why this exists as a first-class baseline rather than a diagnostic: an agent whose
    observation is edge marginals is carrying a *lossier belief* than the full-posterior
    oracle. Scoring it against that oracle would conflate two separate deficits -- being a
    worse policy, and knowing less. This policy holds the belief fixed at the agent's level
    and varies only the policy, which is the comparison that isolates the thing being
    measured.

    Mechanism: take the true posterior's edge marginals, rebuild an approximate posterior
    that assumes every edge is independent, then score with the same oracle. The
    independence assumption is exactly what edge marginals discard, so the gap between this
    and `GreedyOraclePolicy` is the measured cost of the compression -- +3% at d=3, +5% at
    d=4, +11% at d=5 in interventions to identify.
    """

    def __init__(self, space, seed: int = 0):
        self.space = space
        self.oracle = InterventionOracle(space)
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._dags = space.dags.astype(np.float64)

    def reset(self, seed: Optional[int] = None) -> None:
        """See `RandomPolicy.reset`."""
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

    def approximate_posterior(self, posterior: np.ndarray) -> np.ndarray:
        """Rebuild a distribution over DAGs from edge marginals alone.

        Each edge is treated as an independent Bernoulli, so a DAG's weight is the product
        over all ordered pairs of (marginal if the edge is present, 1 - marginal if not).
        Renormalised over the DAG space, which is what keeps the result a valid
        distribution despite the independence assumption ignoring acyclicity.
        """
        marginals = np.clip(edge_marginals(self.space, posterior), 1e-9, 1 - 1e-9)
        log_w = (
            self._dags * np.log(marginals) + (1 - self._dags) * np.log1p(-marginals)
        ).reshape(self.space.n_dags, -1).sum(axis=1)
        w = np.exp(log_w - log_w.max())
        return w / w.sum()

    def __call__(self, env: CausalDiscoveryEnv, result) -> int:
        approx = self.approximate_posterior(result.posterior)
        scores, best = self.oracle.best_targets(approx)
        if scores.max() <= 1e-9:
            return PASS_ACTION
        return int(self.rng.choice(np.flatnonzero(best)))


def make_baselines(space, seed: int = 0) -> dict:
    """The standard comparison set, built together so seeds stay aligned.

    `greedy_oracle` is the opponent for a full-posterior (condition A) agent;
    `edge_marginal_greedy` is the opponent for an edge-marginal (condition B) agent.
    """
    return {
        "no_intervention": no_intervention_policy,
        "random": RandomPolicy(seed=seed),
        "greedy_oracle": GreedyOraclePolicy(space, seed=seed),
        "edge_marginal_greedy": EdgeMarginalGreedyPolicy(space, seed=seed),
    }


class EdgeMarginalGreedyDPPolicy:
    """`EdgeMarginalGreedyPolicy` on the enumeration-free path.

    The opponent a condition-B agent is scored against, and therefore the policy that makes
    a d=7 number comparable with the d=4/5/6 curve. Scoring the agent against the
    full-posterior greedy instead would change what the headline measures halfway along the
    size axis -- conflating "worse policy" with "lossier belief" at d=7 only, exactly the
    conflation this baseline exists to prevent.

    Same two steps as the enumerated version: rebuild the independent-edge approximation to
    the belief, then score it with the greedy information-gain rule. Both steps are done
    without a DAG list -- the approximation because it is modular (see
    `sa.dp.independent_edge_log_weights`), the scoring because the oracle samples.
    """

    def __init__(self, dp, n_draws: int = 4000, seed: int = 0):
        self.dp = dp
        self.oracle = SamplingOracle(dp, n_draws=n_draws, seed=seed)
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """See `RandomPolicy.reset` -- tie-breaking and the oracle's draws must both be
        reproducible, or a reference run and an evaluation run of the same policy differ."""
        self.rng = np.random.default_rng(self._seed if seed is None else seed)
        self.oracle.seed = self._seed if seed is None else seed
        self.oracle._calls = 0

    def approximate_belief(self, log_w: np.ndarray) -> np.ndarray:
        """The independent-edge approximation, as a log-weight table the oracle can read."""
        from sa.dp import independent_edge_log_weights
        marginals = self.dp.edge_marginals_onepass(log_w)
        return independent_edge_log_weights(marginals, self.dp.scorer, self.dp.d)

    def __call__(self, env, result) -> int:
        scores, best = self.oracle.best_targets(self.approximate_belief(result.posterior))
        if scores.max() <= 1e-9:
            return PASS_ACTION
        return int(self.rng.choice(np.flatnonzero(best)))
