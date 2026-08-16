"""Which machinery runs the experiment: the enumerated DAG list, or the subset DP.

There are now two implementations of the same physics. Below d=7 the whole DAG space fits
in memory, so beliefs are an array over graphs and the oracle reads descendant sets off a
precomputed table. At d=7 there are 1.14 billion graphs and neither exists: the belief
becomes a `[d, 2^(d-1)]` log-weight table and the oracle samples.

Everything that differs between those two worlds is collected here, so that
`scripts/run_experiment.py`, `sa/evaluate.py` and `sa/policy.py` can each hold one code
path instead of three sets of `if`. They all ask the backend for an environment, an oracle
and a baseline set, and never learn which world they are in.

The two are not assumed equivalent. `tests/test_env_dp.py` runs both on identical seeds and
identical actions and requires the true-graph mass and every edge marginal to agree to
1e-12, and `scripts/gates_dp.py` re-derives GATE 1 and GATE 2 at d=6 on the DP path as a
control against the enumerated answer. d=7 is believable only because d=6 agrees.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sa.env import EnvConfig

# Enumeration is affordable to d=6 (3.78 million graphs, ~30 s to build and a few hundred
# MB). d=7 is 1.14 billion, which is not a slow version of the same thing -- it does not
# fit. So the switch is a cliff, not a tuning knob.
MAX_ENUMERABLE_D = 6


class Backend:
    """Supplies the environment, oracle and baselines for one experiment."""

    def __init__(self, config: EnvConfig, space=None, force_dp: Optional[bool] = None,
                 oracle_draws: int = 4000, seed: int = 0):
        self.config = config
        self.oracle_draws = oracle_draws
        self.seed = seed
        self.use_dp = (config.d > MAX_ENUMERABLE_D) if force_dp is None else bool(force_dp)
        if not self.use_dp and config.d > MAX_ENUMERABLE_D:
            raise ValueError(
                f"asked for the enumerated path at d={config.d}, which would build "
                f"{config.d}-node DAG list -- 1.14 billion graphs at d=7. This is almost "
                "always an argparse default leaking through as False rather than None; "
                "pass force_dp=None to choose automatically.")

        if self.use_dp:
            from sa.dp import DPPosterior
            from sa.env_dp import DPCausalDiscoveryEnv
            from sa.oracle import SamplingOracle
            from sa.score import get_score

            self.space = None
            self.dp = DPPosterior.for_prior(
                config.d, get_score(config.score, config.d),
                kind=config.prior, p=config.prior_p)
            self._env_class = DPCausalDiscoveryEnv
            self._graph_pool = None
            self.oracle = SamplingOracle(self.dp, n_draws=oracle_draws, seed=seed)
            # Shaping needs the entropy of a distribution over graphs, which does not
            # exist here. Measured dead in the Phase 2 sweep anyway, so refusing costs
            # nothing and prevents a silently different objective at d=7.
            self.max_entropy = None
        else:
            from sa.graphs import build_graph_space
            from sa.oracle import InterventionOracle

            self.space = space if space is not None else build_graph_space(config.d)
            self.dp = None
            self.oracle = InterventionOracle(self.space)
            self.max_entropy = float(np.log(self.space.n_dags))

    # -- construction -------------------------------------------------------------------

    def make_env(self):
        """A fresh environment. Cheap on both paths -- the expensive objects (`space`,
        `dp`) are built once here and shared."""
        if self.use_dp:
            from sa.env_dp import DPCausalDiscoveryEnv
            env = DPCausalDiscoveryEnv(self.config, pool_seed=self.seed,
                                       graph_pool=self._graph_pool)
            # Built on the first environment and reused by every later one. Sharing it is
            # not just a saving: the agent and the baselines MUST face the same graph on
            # episode i, or the measured gap is partly the luck of the draw.
            if self._graph_pool is None:
                env.graph_for(0)              # forces the pool to be drawn
                self._graph_pool = env._pool
            return env
        from sa.env import CausalDiscoveryEnv
        return CausalDiscoveryEnv(self.config, space=self.space)

    def make_baselines(self, seed: int = 0) -> dict:
        """The comparison set. `greedy_oracle` is the opponent the result is measured
        against, so it must be the *same decision rule* on both paths -- only the belief it
        reads differs. See `GreedyOracleDPPolicy`."""
        from sa.baselines import (EdgeMarginalGreedyDPPolicy, GreedyOracleDPPolicy,
                                  GreedyOraclePolicy, RandomPolicy,
                                  no_intervention_policy)
        if self.use_dp:
            return {
                "no_intervention": no_intervention_policy,
                "random": RandomPolicy(seed=seed),
                "greedy_oracle": GreedyOracleDPPolicy(
                    self.dp, n_draws=self.oracle_draws, seed=seed),
                "edge_marginal_greedy": EdgeMarginalGreedyDPPolicy(
                    self.dp, n_draws=self.oracle_draws, seed=seed),
            }
        from sa.baselines import EdgeMarginalGreedyPolicy
        return {
            "no_intervention": no_intervention_policy,
            "random": RandomPolicy(seed=seed),
            "greedy_oracle": GreedyOraclePolicy(self.space, seed=seed),
            "edge_marginal_greedy": EdgeMarginalGreedyPolicy(self.space, seed=seed),
        }

    # -- reporting ----------------------------------------------------------------------

    def describe(self) -> str:
        if self.use_dp:
            return (f"d={self.config.d}  subset-DP path (no DAG list); "
                    f"oracle by {self.oracle_draws} MH draws")
        return (f"d={self.config.d}  enumerated path; "
                f"{self.space.n_dags} DAGs / {self.space.n_mecs} classes")

    @property
    def observation_kinds(self) -> tuple:
        """`posterior` is unavailable on the DP path -- the array does not fit."""
        return ("edge_marginals",) if self.use_dp else ("posterior", "edge_marginals")
