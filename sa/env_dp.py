"""The environment on the enumeration-free path, so `d` is no longer capped at 6.

`CausalDiscoveryEnv` is built around an enumerated `GraphSpace`: the true graph is an
index into a DAG list, the prior is a weight per DAG, the belief is an array over all of
them, and identification reads one entry of that array. Every one of those is impossible at
d=7, where there are 1.14 billion DAGs.

This class keeps the same `EnvConfig`, the same episode structure and the same definition
of solved, but represents each of those four things differently:

| quantity | enumerated env | here |
|---|---|---|
| true graph | index into the DAG list | a `[d, d]` adjacency |
| prior | weight per DAG | modular log-odds per edge (`sa/dp.py`) |
| belief | `[n_dags]` posterior | `[d, 2^(d-1)]` log-weight table |
| identified | `posterior[true_index] >= t` | `exp(log_prob_dag(...)) >= t` |

The *numbers* are identical where both exist -- `tests/test_env_dp.py` pins the two
environments against each other at d=4 and d=5, step for step on a shared seed. That
equality is the whole justification for trusting d=7 results from a class that cannot be
compared to anything.

**One capability is genuinely lost, not worked around.** `observation("posterior")` has no
counterpart here, because the object it returns does not fit in memory. The scalable
`edge_marginals` observation was always the condition intended to run at large `d`, and the
gap between the two is a measured result at d<=6 rather than an assumption -- so nothing
depends on having both at d=7.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sa.dp import DPPosterior
from sa.env import PASS_ACTION, EnvConfig, StepResult
from sa.graphs import is_singleton_mec
from sa.sampler import mh_sample
from sa.scm import sample, sample_scm_params
from sa.score import get_score


class DPCausalDiscoveryEnv:
    """Same environment, belief held as a log-weight table instead of a DAG list."""

    def __init__(self, config: EnvConfig, burn_in: int = 20_000,
                 steps_between_episodes: int = 500):
        if config.prior not in ("uniform", "erdos_renyi"):
            raise ValueError(
                f"prior {config.prior!r} is not modular and cannot be represented on the "
                "DP path; see sa/dp.py. Use the enumerated environment for it.")
        self.config = config
        self.dp = DPPosterior.for_prior(
            config.d, get_score(config.score, config.d),
            kind=config.prior, p=config.prior_p)

        # True graphs are drawn from the prior by MH, since there is no list to index into
        # and no closed-form sampler for the Erdos-Renyi prior *over DAGs*. One persistent
        # chain is advanced between episodes rather than restarted, which avoids paying
        # burn-in every reset; `steps_between_episodes` sets how far consecutive episodes'
        # graphs are decorrelated. See `estimate_singleton_fraction` for why the obvious
        # permutation sampler is the wrong distribution.
        self._graph_chain: Optional[np.ndarray] = None
        self._burn_in = burn_in
        self._steps_between = steps_between_episodes

        self._rng: Optional[np.random.Generator] = None
        self.true_adjacency: Optional[np.ndarray] = None
        self.params = None
        self.samples: Optional[np.ndarray] = None
        self.intervened: Optional[np.ndarray] = None
        self.log_w: Optional[np.ndarray] = None
        self.log_z: Optional[float] = None
        self.n_interventions = 0
        self.intervention_counts: Optional[np.ndarray] = None

    # -- prior sampling ---------------------------------------------------------------

    def _next_true_graph(self, rng: np.random.Generator) -> np.ndarray:
        if self._graph_chain is None:
            draws, _ = mh_sample(self.dp.log_prior_term, self.dp._mask_to_index,
                                 self.config.d, 1, burn_in=self._burn_in, thin=1, rng=rng)
            self._graph_chain = draws[0]
        draws, _ = mh_sample(self.dp.log_prior_term, self.dp._mask_to_index, self.config.d,
                             1, burn_in=self._steps_between, thin=1, rng=rng,
                             init=self._graph_chain)
        self._graph_chain = draws[0]
        return self._graph_chain.copy()

    # -- episode lifecycle --------------------------------------------------------------

    def reset(self, seed: Optional[int] = None,
              force_adjacency: Optional[np.ndarray] = None) -> StepResult:
        """Start an episode. `force_adjacency` pins the true DAG, which is how evaluation
        holds the graph fixed across policies -- the counterpart of `force_index`."""
        self._rng = np.random.default_rng(seed)
        cfg = self.config

        if force_adjacency is None:
            self.true_adjacency = self._next_true_graph(self._rng)
        else:
            self.true_adjacency = np.asarray(force_adjacency, dtype=np.int8)

        self.params = sample_scm_params(
            self.true_adjacency, self._rng,
            weight_range=cfg.weight_range, noise_range=cfg.noise_range)

        self.samples, self.intervened = sample(self.params, cfg.n_obs, self._rng)
        self.n_interventions = 0
        self.intervention_counts = np.zeros(cfg.d, dtype=int)
        self._update_belief()
        return self._result()

    def step(self, action: int) -> StepResult:
        if self.log_w is None:
            raise RuntimeError("call reset() before step()")

        cfg = self.config
        if action != PASS_ACTION:
            if not 0 <= action < cfg.d:
                raise ValueError(f"action must be in [0, {cfg.d}) or PASS_ACTION, got {action}")
            new_samples, new_intervened = sample(
                self.params, cfg.n_int, self._rng,
                intervene_node=int(action), intervene_scale=cfg.intervene_scale)
            self.samples = np.vstack([self.samples, new_samples])
            self.intervened = np.vstack([self.intervened, new_intervened])
            self.n_interventions += 1
            self.intervention_counts[action] += 1
            self._update_belief()

        return self._result(passed=(action == PASS_ACTION))

    def _update_belief(self) -> None:
        """One score table, one partition function. The edge marginals are computed lazily
        in `observation`, since a run using the greedy oracle never asks for them."""
        self.log_w = self.dp.log_weights(self.samples, self.intervened)
        self.log_z = self.dp.log_partition(self.log_w)

    # -- state ---------------------------------------------------------------------------

    def true_mass(self) -> float:
        return float(np.exp(self.dp.log_prob_dag(self.log_w, self.true_adjacency,
                                                 log_z=self.log_z)))

    def _result(self, passed: bool = False) -> StepResult:
        mass = self.true_mass()
        identified = mass >= self.config.identify_threshold
        done = identified or passed or self.n_interventions >= self.config.budget
        return StepResult(
            # The log-weight table IS the belief here, and it is what the oracle consumes.
            # Named `posterior` to keep one StepResult shape across both environments.
            posterior=self.log_w.copy(),
            identified=identified,
            done=done,
            n_interventions=self.n_interventions,
            info={
                "true_adjacency": self.true_adjacency.copy(),
                "true_mass": mass,
                # No `mec_size`: it needs the class, which needs the DAG list. The
                # singleton flag -- the only part GATE 1 reads -- is a local test.
                "is_singleton": bool(is_singleton_mec(self.true_adjacency)),
                "passed": passed,
                "budget_left": self.config.budget - self.n_interventions,
            },
        )

    def observation(self, kind: str = "edge_marginals") -> np.ndarray:
        if kind == "posterior":
            raise ValueError(
                "the enumerated posterior does not exist on the DP path -- 1.14 billion "
                "DAGs at d=7. Use 'edge_marginals', the condition designed to scale.")
        if kind != "edge_marginals":
            raise ValueError(f"unknown observation kind {kind!r}")

        budget_left = np.array(
            [(self.config.budget - self.n_interventions) / max(self.config.budget, 1)],
            dtype=float)
        extra = [budget_left]
        if self.config.include_counts:
            extra.append(np.asarray(self.intervention_counts, dtype=float)
                         / max(self.config.budget, 1))

        marginals = self.dp.edge_marginals_onepass(self.log_w)
        off_diagonal = ~np.eye(self.config.d, dtype=bool)
        return np.concatenate([marginals[off_diagonal]] + extra)

    @property
    def observation_dim(self) -> dict:
        d = self.config.d
        extra = 1 + (d if self.config.include_counts else 0)
        return {"edge_marginals": d * (d - 1) + extra}
