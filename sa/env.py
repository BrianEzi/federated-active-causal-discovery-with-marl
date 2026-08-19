"""Single-agent active causal discovery environment.

One agent, full observability, no masks, no federation, no stitching. The agent's only
job is to decide *where to intervene next*, and the episode ends when the true DAG has
been identified or the budget runs out.

Deliberately minimal. Everything that made the previous environment hard to reason about
-- estimator choice, reward density, exploration bonuses, curricula, per-agent
visibility masks -- is gone. What remains is three parameters: how many samples an
observation or intervention buys, how large the budget is, and how confident the
posterior must be before we call the graph identified.

Episode structure:
  reset  -- draw a DAG uniformly from the graph space, draw weights and per-node noise
            scales, collect `n_obs` purely observational samples.
  step   -- the agent picks a node to intervene on (or passes); we draw `n_int` samples
            under that intervention, append them, and recompute the exact posterior.
  done   -- the true DAG has posterior mass >= `identify_threshold`, or budget exhausted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sa.graphs import GraphSpace, build_graph_space
from sa.posterior import PosteriorEngine, edge_marginals, is_identified
from sa.priors import build_prior
from sa.scm import sample, sample_scm_params
from sa.score import get_score

# Sentinel action meaning "do not intervene this step".
PASS_ACTION = -1


@dataclass
class EnvConfig:
    d: int = 3
    # Enough observational data to pin down the Markov equivalence class, so the agent's
    # job is cleanly "orient within the class" rather than "also find the skeleton".
    # Calibrated against GATE 1: at d=3 this puts the observational-only identification
    # rate at ~14% against a theoretical target of 16%.
    n_obs: int = 1000
    n_int: int = 100           # samples drawn per intervention
    # Maximum interventions per episode. Lowered 10 -> 5 on 2026-08-19, and the reason is
    # a measurement rather than a preference: the greedy-vs-random gap is ENTIRELY a
    # budget-scarcity phenomenon. At d=5/n_obs=20000 it is +0.390 at budget 2, +0.180 at
    # 4, +0.015 at 8, and 0.000 by 16 -- both arms sit at 0.99. Budget 10 is past the point
    # where choice quality affects the outcome at all, which is why every earlier sweep
    # found budget to be a null and why GATE 2 would pass trivially there.
    # Gates run tighter still, at 2-3, where discrimination peaks. See results/budget/.
    budget: int = 5
    # Chosen to sit safely above every Markov-equivalence tie cap. A class of size k caps
    # each member at 1/k, so a size-2 class reaches exactly 0.5 -- a 0.5 threshold could
    # therefore declare an unbroken tie "identified". 0.7 cannot be reached while any tie
    # remains, which is the property that makes this criterion mean something.
    identify_threshold: float = 0.7
    score: str = "bge"
    noise_range: tuple = (0.5, 1.5)   # must not be degenerate -- see sa/scm.py
    weight_range: tuple = (0.5, 2.0)
    intervene_scale: float = 2.0
    # Graph prior. Used BOTH to draw the true graph and as the posterior's prior, so
    # generator and inference agree -- pairing a sparse generator with a uniform prior is
    # a misspecification that would look like an estimator bug. `expected_degree` is held
    # fixed rather than `p`, so graphs stay sparse as d grows (see sa/priors.py).
    # p = 0.5 is exactly the uniform-over-DAGs prior; sparsity (p < 0.5) is close to
    # vacuous below d ~ 8, so it stays the default here and becomes a real lever at scale.
    prior: str = "erdos_renyi"
    prior_p: float = 0.5
    # Append per-node intervention counts to the observation.
    #
    # Off by default because the posterior is formally a sufficient statistic: if an
    # intervention taught nothing, the belief is unchanged and the same target really is
    # still the best one. But that argument holds for the OPTIMAL policy, not for a
    # deterministic network whose output barely varies with its input -- such a policy
    # re-picks the same node until the budget runs out, which is what the first results
    # showed (solve rate below random's). Making past actions explicit is the cheapest way
    # to let a policy break that tie, and whether it helps is a measurement, not an
    # assumption.
    include_counts: bool = False
    expected_degree: Optional[float] = None
    scale_free_gamma: float = 1.0


@dataclass
class StepResult:
    posterior: np.ndarray
    identified: bool
    done: bool
    n_interventions: int
    info: dict = field(default_factory=dict)


class CausalDiscoveryEnv:
    """The environment. Construct once, `reset` per episode.

    The graph space and posterior engine are built once and reused, since both depend
    only on `d` and are the expensive part to construct.
    """

    def __init__(self, config: EnvConfig, space: Optional[GraphSpace] = None):
        self.config = config
        self.space = space if space is not None else build_graph_space(config.d)
        self.engine = PosteriorEngine(self.space, get_score(config.score, config.d))
        self.prior = build_prior(
            self.space, kind=config.prior, p=config.prior_p,
            expected_deg=config.expected_degree, gamma=config.scale_free_gamma,
        )

        self._rng: Optional[np.random.Generator] = None
        self.true_index: Optional[int] = None
        self.params = None
        self.samples: Optional[np.ndarray] = None
        self.intervened: Optional[np.ndarray] = None
        self.posterior: Optional[np.ndarray] = None
        self.n_interventions = 0
        self.intervention_counts: Optional[np.ndarray] = None

    # -- episode lifecycle --------------------------------------------------------

    def reset(self, seed: Optional[int] = None, force_index: Optional[int] = None) -> StepResult:
        """Start an episode. `force_index` pins the true DAG, for tests and for
        evaluating every graph exactly once."""
        self._rng = np.random.default_rng(seed)
        cfg = self.config

        if force_index is None:
            # Drawn FROM THE PRIOR, not uniformly -- so the generator and the estimator's
            # assumption are the same distribution.
            self.true_index = int(self._rng.choice(self.space.n_dags, p=self.prior))
        else:
            self.true_index = int(force_index)

        self.params = sample_scm_params(
            self.space.dags[self.true_index],
            self._rng,
            weight_range=cfg.weight_range,
            noise_range=cfg.noise_range,
        )

        self.samples, self.intervened = sample(self.params, cfg.n_obs, self._rng)
        self.n_interventions = 0
        self.intervention_counts = np.zeros(cfg.d, dtype=int)
        self.posterior = self.engine.posterior(self.samples, self.intervened, self.prior)

        return self._result()

    def step(self, action: int) -> StepResult:
        """Intervene on `action` (or `PASS_ACTION` to collect nothing) and update belief.

        Passing is a real choice, not a no-op: it ends the episode's ability to learn
        anything further, so an agent that passes while unidentified has effectively
        given up. It exists so that "should I act at all" is a decision the agent makes,
        rather than one the action space makes for it.
        """
        if self.posterior is None:
            raise RuntimeError("call reset() before step()")

        cfg = self.config
        if action != PASS_ACTION:
            if not 0 <= action < cfg.d:
                raise ValueError(f"action must be in [0, {cfg.d}) or PASS_ACTION, got {action}")
            new_samples, new_intervened = sample(
                self.params, cfg.n_int, self._rng,
                intervene_node=int(action), intervene_scale=cfg.intervene_scale,
            )
            self.samples = np.vstack([self.samples, new_samples])
            self.intervened = np.vstack([self.intervened, new_intervened])
            self.n_interventions += 1
            self.intervention_counts[action] += 1
            self.posterior = self.engine.posterior(self.samples, self.intervened, self.prior)

        return self._result(passed=(action == PASS_ACTION))

    # -- state ---------------------------------------------------------------------

    def _result(self, passed: bool = False) -> StepResult:
        identified = is_identified(self.posterior, self.true_index, self.config.identify_threshold)
        done = identified or passed or self.n_interventions >= self.config.budget
        return StepResult(
            posterior=self.posterior.copy(),
            identified=identified,
            done=done,
            n_interventions=self.n_interventions,
            info={
                "true_index": self.true_index,
                "true_mass": float(self.posterior[self.true_index]),
                "mec_size": int(len(self.space.mec_members(self.true_index))),
                # Indexed directly rather than via `space.is_singleton`, which is a
                # property that materialises an [N] bool array -- 3.8 million entries at
                # d=6 -- on every access, to read a single element once per step.
                "is_singleton": bool(
                    self.space.mec_sizes[self.space.mec_id[self.true_index]] == 1),
                "passed": passed,
                "budget_left": self.config.budget - self.n_interventions,
            },
        )

    def observation(self, kind: str = "posterior") -> np.ndarray:
        """The agent's view of the world.

        `posterior` -- the exact belief over DAGs. A sufficient statistic, so the problem
        is a proper MDP and no recurrence is needed. Does not scale past d ~ 5.

        `edge_marginals` -- d(d-1) edge probabilities plus remaining budget. Scales to any
        d, but lossy: it discards correlations between edges, so the problem becomes
        partially observed. The gap between the two is the thing worth measuring.
        """
        # Normalised to [0, 1]. Raw counts were a real bug: the budget feature sat at 20.0
        # while posterior entries averaged 0.04, a ~500x scale mismatch that saturated the
        # tanh trunk and drowned out the belief the agent is supposed to act on.
        budget_left = np.array(
            [(self.config.budget - self.n_interventions) / max(self.config.budget, 1)],
            dtype=float,
        )
        # Also normalised to [0, 1], for the same reason the budget feature is.
        extra = [budget_left]
        if self.config.include_counts:
            extra.append(np.asarray(self.intervention_counts, dtype=float)
                         / max(self.config.budget, 1))

        if kind == "posterior":
            return np.concatenate([self.posterior] + extra)
        if kind == "edge_marginals":
            marg = self.engine.edge_marginals(self.posterior)
            off_diagonal = ~np.eye(self.config.d, dtype=bool)
            return np.concatenate([marg[off_diagonal]] + extra)
        raise ValueError(f"unknown observation kind {kind!r}")

    @property
    def observation_dim(self) -> dict:
        d = self.config.d
        extra = 1 + (d if self.config.include_counts else 0)
        return {"posterior": self.space.n_dags + extra,
                "edge_marginals": d * (d - 1) + extra}
