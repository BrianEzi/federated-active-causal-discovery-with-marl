"""PHASE 4 -- reference policies for the rebuilt two-agent environment.

Four arms, and the choice of arms is itself a correction. The earlier comparison used a
random policy whose clamping was incidental, which made "the learned agent clamps" look
like a discovery when a coin-flipping policy clamps half the time by construction. So
`random_clamp` is named, explicit, and the PRIMARY floor [U16].

  pass          never acts. What does an agent reach on observational data plus whatever
                its partner happens to do?
  random_vary   uniform over targets, VARY only. Never removes itself as a confounder.
  random_clamp  uniform over (target, mode). The primary floor.
  greedy        myopic expected information gain over the agent's own window.
  forced_clamp  always clamps its own private node. Not a serious policy -- it is GATE 3's
                upper arm, the "coordination is available if you pay for it" reference.

THE GREEDY ORACLE ENUMERATES, AND THAT IS DELIBERATE. It needs a full posterior over the
window to compute the descendant-set partition its criterion is defined on, and no sampler
for the DP posterior exists that is trustworthy here -- the MH sampler is under-mixed
(5.8% acceptance) and its current settings are an admitted stopgap. Enumeration is exact,
and the oracle is a REFERENCE POINT rather than part of the method, so it does not need to
scale. The guard below makes that limit explicit instead of letting it fail quietly at k=6.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ma.belief_dp import JOINT_CONF, MODULAR_RULES
from ma.env2 import CLAMP, VARY, AgentWindow, TwoAgentEnv2
from sa.graphs import build_graph_space, descendants
from sa.oracle import _partition_entropy

MAX_ENUMERATED_K = 5


class _Window:
    """Enumerated hypothesis space for one window, built once and reused."""

    _cache: Dict[int, "_Window"] = {}

    def __init__(self, k: int):
        if k > MAX_ENUMERATED_K:
            raise ValueError(
                f"the greedy oracle enumerates and k={k} is past the limit of "
                f"{MAX_ENUMERATED_K}. This is a property of the BASELINE, not of the "
                f"method: `WindowBeliefDP` carries to k~15-20. Either cap the oracle "
                f"arm at small windows or build a trustworthy posterior sampler first.")
        self.k = k
        self.dags = np.asarray(build_graph_space(k).dags, dtype=np.int8)
        self.n_dags = len(self.dags)
        self.parent_index: List[List[int]] = []
        # Which hypotheses an intervention on `node` cannot tell apart: two graphs giving
        # the node the same descendant set inside the window respond identically.
        self.signatures = np.zeros((self.n_dags, k), dtype=np.int64)
        for i, dag in enumerate(self.dags):
            reach = descendants(dag)
            for node in range(k):
                self.signatures[i, node] = int(
                    np.dot(reach[node], 1 << np.arange(k)))
        self.n_groups = [len(np.unique(self.signatures[:, n])) for n in range(k)]

    @classmethod
    def get(cls, k: int) -> "_Window":
        if k not in cls._cache:
            cls._cache[k] = cls(k)
        return cls._cache[k]


def enumerated_posterior(window: AgentWindow, samples: np.ndarray,
                         known: np.ndarray, clean: np.ndarray, rule: str) -> np.ndarray:
    """Full posterior over the window's DAGs, from the DP's own local score tables.

    Reusing the DP's tables rather than re-deriving scores is what keeps the oracle and the
    agent looking at the SAME belief. Two estimators that disagree would make every
    comparison between them meaningless.
    """
    belief = window.belief
    space = _Window.get(window.k)
    clean = np.asarray(clean, dtype=bool)

    def per_dag(table: np.ndarray, extra_parents=None) -> np.ndarray:
        out = np.zeros(space.n_dags)
        for i, dag in enumerate(space.dags):
            total = 0.0
            for node in range(window.k):
                parents = tuple(np.flatnonzero(dag[:, node]).tolist())
                if extra_parents is not None:
                    parents = tuple(sorted(set(parents) | extra_parents[node]))
                total += table[node, belief.scorer.lookup[node][parents]]
            out[i] = total
        return out

    if rule in MODULAR_RULES:
        log_w = belief.log_weights(samples, known, clean, rule)
        log_post = per_dag(log_w)
    else:
        clean_table = belief.local_table(samples, known, clean)
        dirty_table = belief.local_table(samples, known, ~clean)
        rows = []
        for assignment in belief.assignments:
            required = [set() for _ in range(window.k)]
            for edge in assignment:
                if edge is not None:
                    required[edge[1]].add(edge[0])
            # A DAG must already contain the assignment's edges to have mass under it.
            ok = np.ones(space.n_dags, dtype=bool)
            for v, parents in enumerate(required):
                for u in parents:
                    ok &= space.dags[:, u, v] > 0
            clean_part = per_dag(clean_table)
            dirty_part = per_dag(dirty_table)
            row = clean_part + dirty_part
            row[~ok] = -np.inf
            rows.append(row)
        stacked = np.vstack(rows)
        # PER-DAG shift, not a global one. A single global shift underflows every entry of
        # the weaker DAGs to zero and then log(0) = -inf, silently deleting hypotheses
        # rather than ranking them -- the same bug that cost a day in `score_regimes`, and
        # I reintroduced it here on the first attempt.
        shift = np.max(stacked, axis=0, keepdims=True)
        finite = np.isfinite(shift)
        safe = np.where(finite, shift, 0.0)
        log_post = np.log(np.exp(stacked - safe).sum(axis=0)) + safe.ravel()
        log_post = np.where(finite.ravel(), log_post, -np.inf)

    log_post = log_post - log_post.max()
    weights = np.exp(log_post)
    return weights / weights.sum()


# -- policies ---------------------------------------------------------------------------


class PassAgent:
    def __init__(self, name: str, seed: int = 0):
        self.name = name

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def __call__(self, env: TwoAgentEnv2, result) -> int:
        return env.windows[self.name].pass_index


class RandomAgent:
    """Uniform over targets. `allow_clamp=False` gives the vary-only floor.

    Never passes: this measures CHOICE quality, so it must spend the same budget as the
    oracle and differ only in what it picks. A random policy that also passed at random
    would conflate two different kinds of badness.
    """

    def __init__(self, name: str, seed: int = 0, allow_clamp: bool = True):
        self.name = name
        self.allow_clamp = allow_clamp
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

    def __call__(self, env: TwoAgentEnv2, result) -> int:
        window = env.windows[self.name]
        candidates = [i for i, (node, mode) in enumerate(window.actions)
                      if node != -1 and (self.allow_clamp or mode == VARY)]
        return int(self.rng.choice(candidates))


class ForcedClampAgent:
    """Always clamps one of its own private nodes -- GATE 3's upper arm.

    Clamping your OWN private node does nothing for you: your rows are clean only when the
    OTHER agent clamps. So this arm is pure altruism, and its value is entirely what it
    does for the partner.
    """

    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

    def __call__(self, env: TwoAgentEnv2, result) -> int:
        window = env.windows[self.name]
        private = [i for i, (node, mode) in enumerate(window.actions)
                   if mode == CLAMP and node in window.private]
        if not private:
            return window.pass_index
        return int(self.rng.choice(private))


class GreedyAgent:
    """Myopic expected information gain over the agent's own window.

    Two graphs giving the intervened node the same descendant set within the window are
    indistinguishable by that intervention, so the expected gain is the entropy of the
    partition the intervention induces on the current belief.

    NOTE, and it is the central two-agent finding rather than an implementation detail:
    this criterion is INDIFFERENT between VARY and CLAMP, because both cut the target's
    incoming edges and so induce the same partition. The tie-break therefore decides, and
    the measured consequence is clamp_fraction 0.000 -- a one-step objective has no term
    for what the partner needs.
    """

    def __init__(self, name: str, env: TwoAgentEnv2, seed: int = 0):
        self.name = name
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        window = env.windows[name]
        self.space = _Window.get(window.k)
        self.candidates = [i for i, (node, mode) in enumerate(window.actions)
                           if node != -1]
        self.rule = env.config.score_rule

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

    def __call__(self, env: TwoAgentEnv2, result) -> int:
        window = env.windows[self.name]
        clean = (env.clean[self.name] if env.config.disclose_regime
                 else np.zeros(len(env.samples), dtype=bool))
        posterior = enumerated_posterior(
            window, env.samples[:, window.nodes], env.known[self.name], clean, self.rule)

        scores = np.full(len(self.candidates), -np.inf)
        for slot, action in enumerate(self.candidates):
            node, _mode = window.actions[action]
            position = window.pos[node]
            scores[slot] = _partition_entropy(
                self.space.signatures[:, position], posterior,
                int(self.space.signatures[:, position].max()) + 1)
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        return int(self.candidates[int(self.rng.choice(best))])


def make_baselines(env: TwoAgentEnv2, name: str, seed: int = 0) -> Dict[str, object]:
    return {
        "pass": PassAgent(name, seed),
        "random_vary": RandomAgent(name, seed, allow_clamp=False),
        "random_clamp": RandomAgent(name, seed, allow_clamp=True),
        "forced_clamp": ForcedClampAgent(name, seed),
        "greedy": GreedyAgent(name, env, seed),
    }
