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

from crosscheck.belief_dp import JOINT_CONF, MODULAR_RULES, WindowBeliefDP
from ma.env import CLAMP, VARY, AgentWindow, TwoAgentEnv
from ma.graphs import build_graph_space, descendants
from ma.stats import _partition_entropy

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
        # MARKOV EQUIVALENCE PARTITION, precomputed. Which class a graph belongs to is a
        # property of the graph space and never changes, but `credit_set` recomputed
        # `mec_signature` for all 543 graphs on every call, and `singleton_fraction`
        # recomputed it once per DRAW. Stored as integer class ids so membership is an
        # array comparison rather than a set comparison.
        from ma.graphs import mec_signature
        lookup: Dict[object, int] = {}
        self.mec_id = np.zeros(self.n_dags, dtype=np.int64)
        for i, dag in enumerate(self.dags):
            sig = mec_signature(dag)
            if sig not in lookup:
                lookup[sig] = len(lookup)
            self.mec_id[i] = lookup[sig]
        self._sig_to_id = lookup
        # How many graphs share each class -- `singleton_fraction` needs exactly this.
        self.mec_size = np.bincount(self.mec_id)

    def id_of(self, adjacency: np.ndarray) -> int:
        """Class id of an arbitrary graph on this window, or -1 if it is not one."""
        from ma.graphs import mec_signature
        return self._sig_to_id.get(mec_signature(adjacency), -1)

    @classmethod
    def get(cls, k: int) -> "_Window":
        if k not in cls._cache:
            cls._cache[k] = cls(k)
        return cls._cache[k]



class _PerDagIndex:
    """Precomputed `(DAG, node) -> parent-set slot` tables.

    THE MAPPING IS DATA-INDEPENDENT. Which parent set a DAG gives a node is a property of
    the graph space, not of any dataset, so it can be built once and reused for every
    episode, every rule, and every belief update. `RegimeScorer._build_index` already did
    this for the enumerated path and the DP rewrite did not carry it over.

    What it replaces: `per_dag` looped 543 DAGs x 4 nodes in Python, rebuilding a parent
    tuple with `np.flatnonzero` and doing a dict lookup each time, twice per assignment
    across 25 assignments -- about 109,000 Python iterations per call, measured at 598 ms.
    Evaluation ran this twice per episode, so it was roughly 40% of every training job.

    Three tables:
      `own`        [n_dags, k]                 the DAG's own parent set, per node
      `stripped`   [n_dags, n_assign, k]       parents MINUS that assignment's confounding
                                               edges, which is what the CLEAN regime scores
      `compatible` [n_dags, n_assign]          does the DAG contain the assignment's edges
    """

    _cache: Dict[tuple, "_PerDagIndex"] = {}

    def __init__(self, k: int, assignments, scorer):
        space = _Window.get(k)
        self.k = k
        self.n_dags = space.n_dags
        self.n_assign = len(assignments)
        self.own = np.zeros((space.n_dags, k), dtype=np.int64)
        self.stripped = np.zeros((space.n_dags, self.n_assign, k), dtype=np.int64)
        self.compatible = np.ones((space.n_dags, self.n_assign), dtype=bool)

        required = []
        for assignment in assignments:
            need = [set() for _ in range(k)]
            for edge in assignment:
                if edge is not None:
                    need[edge[1]].add(edge[0])
            required.append(need)

        for i, dag in enumerate(space.dags):
            parents = [tuple(np.flatnonzero(dag[:, node]).tolist()) for node in range(k)]
            for node in range(k):
                self.own[i, node] = scorer.lookup[node][parents[node]]
            for a, need in enumerate(required):
                ok = True
                for node in range(k):
                    if not need[node]:
                        self.stripped[i, a, node] = self.own[i, node]
                        continue
                    if not need[node].issubset(parents[node]):
                        ok = False
                    kept = tuple(p for p in parents[node] if p not in need[node])
                    self.stripped[i, a, node] = scorer.lookup[node][kept]
                self.compatible[i, a] = ok

    @classmethod
    def get(cls, k: int, assignments, scorer) -> "_PerDagIndex":
        key = (k, tuple(assignments))
        if key not in cls._cache:
            cls._cache[key] = cls(k, assignments, scorer)
        return cls._cache[key]


def enumerated_posterior(window: AgentWindow, samples: np.ndarray,
                         known: np.ndarray, clean: np.ndarray, rule: str) -> np.ndarray:
    """Full posterior over the window's DAGs, from the DP's own local score tables.

    Reusing the DP's tables rather than re-deriving scores is what keeps the oracle and the
    agent looking at the SAME belief. Two estimators that disagree would make every
    comparison between them meaningless.
    """
    if not isinstance(window.belief, WindowBeliefDP):
        raise NotImplementedError(
            "enumerated_posterior reads the exact DP's own score tables "
            "(belief.assignments, belief.scorer); there is no posterior to enumerate "
            "under the constraint backend. The greedy baseline and the enumerated "
            "report need their own constraint-side design -- an expected reduction in "
            "bootstrap disagreement, not an expected posterior gain. Deliberately "
            "unimplemented in Phase 1; see docs/CB_IMPLEMENTATION_PLAN.md.")
    belief = window.belief
    space = _Window.get(window.k)
    clean = np.asarray(clean, dtype=bool)
    index = _PerDagIndex.get(window.k, belief.assignments, belief.scorer)
    nodes = np.arange(window.k)

    def per_dag(table: np.ndarray, slots: Optional[np.ndarray] = None) -> np.ndarray:
        """Total log score of every DAG, as one gather-and-sum over the precomputed slots.

        `slots` defaults to each DAG's own parent sets; pass an assignment's STRIPPED
        slots to score the clean regime, which must not be credited for the confounding
        edges.
        """
        if slots is None:
            slots = index.own
        return table[nodes[None, :], slots].sum(axis=1)

    if rule in MODULAR_RULES:
        log_w = belief.log_weights(samples, known, clean, rule)
        log_post = per_dag(log_w)
    else:
        clean_table = belief.local_table(samples, known, clean)
        dirty_table = belief.local_table(samples, known, ~clean)
        rows = []
        # The DIRTY part is the same for every assignment -- it always reads the DAG's own
        # parent sets -- so it is computed once rather than 25 times.
        dirty_part = per_dag(dirty_table)
        for a, assignment in enumerate(belief.assignments):
            # THE CLEAN REGIME MUST NOT BE CREDITED FOR THE CONFOUNDING EDGES.
            #
            # A hypothesis is (DAG H, confounding set P) where P's edges are present in H
            # and STRIPPED AGAIN for the clean regime -- clean rows are the ones with no
            # hidden variable transmitting variance, so the confounding edge should not be
            # there at all. `WindowBeliefDP._assignment_weights` reads the clean table at
            # `parents \ P` for exactly this reason; scoring at the full parent set makes
            # the confounded hypothesis fit the clean data too well.
            #
            # This disagreed with the DP by 3.55e-02 and was invisible until a test forced
            # a genuine clean/dirty split: with no clean rows the clean table is all zeros
            # (the empty-regime guard), so stripping changes nothing and the two paths
            # agreed to 1e-12. The same lesson as the subset-DP sampler -- test data that
            # cannot exercise the branch proves nothing about it.
            clean_part = per_dag(clean_table, index.stripped[:, a, :])
            row = clean_part + dirty_part
            row[~index.compatible[:, a]] = -np.inf
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



def _agent_seed(seed: int, agent: int) -> int:
    """Distinct stream per agent.

    A and B were being constructed with the SAME seed at every call site, so their RNGs
    produced identical sequences. Their action lists are structurally parallel -- own
    private node first, then the shared nodes in the same order -- so identical indices
    meant the two agents chose the SAME shared target almost every round. Measured
    collision rate 0.784 against the ~0.19 expected of two independent uniform agents.

    That is not a cosmetic bug: `random_clamp` is the PRIMARY floor [U16], and a
    perfectly synchronised pair is a different policy from two independent ones -- it
    systematically wastes one of the two moves, and it makes the floor easier to beat for
    the wrong reason.
    """
    return int(seed) * 1000 + int(agent) if int(agent) >= 2 else int(seed) * 2 + int(agent)


# -- policies ---------------------------------------------------------------------------


class PassAgent:
    def __init__(self, agent: int, seed: int = 0):
        self.agent: int = int(agent)

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def __call__(self, env: TwoAgentEnv, result) -> int:
        return env.windows[self.agent].pass_index


class RandomAgent:
    """Uniform over targets. `allow_clamp=False` gives the vary-only floor.

    Never passes: this measures CHOICE quality, so it must spend the same budget as the
    oracle and differ only in what it picks. A random policy that also passed at random
    would conflate two different kinds of badness.
    """

    def __init__(self, agent: int, seed: int = 0, allow_clamp: bool = True):
        self.agent: int = int(agent)
        self.allow_clamp = allow_clamp
        self._seed = _agent_seed(seed, self.agent)
        self.rng = np.random.default_rng(self._seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(
            self._seed if seed is None else _agent_seed(seed, self.agent))

    def __call__(self, env: TwoAgentEnv, result) -> int:
        window = env.windows[self.agent]
        # Under `mode_by_role` the mode is not a choice -- it is fixed by the node's role --
        # so `allow_clamp` has nothing to select on and the arm is "uniform over targets",
        # which is the floor it was always meant to be. Filtering on mode there would
        # return an empty candidate list.
        candidates = [i for i, (node, mode) in enumerate(window.actions)
                      if node != -1 and (self.allow_clamp or window.mode_by_role
                                         or mode == VARY)]
        if not candidates:
            # `allow_clamp=False` in a clamp-only environment. `scripts/ma_train.py` guards
            # this at the call site by only offering `random_vary` when VARY is available,
            # but the class did not, so the failure surfaced as an opaque numpy error from
            # `rng.choice` several frames down. Say what is wrong instead.
            raise ValueError(
                f"random_vary has no legal move for agent {self.agent}: the environment's "
                f"action modes are {window.modes} and this arm excludes clamps. Use "
                f"random_clamp, which is uniform over every action, when the environment "
                f"offers no vary.")
        return int(self.rng.choice(candidates))


class ForcedClampAgent:
    """Always clamps one of its own private nodes -- GATE 3's upper arm.

    Clamping your OWN private node does nothing for you: your rows are clean only when the
    OTHER agent clamps. So this arm is pure altruism, and its value is entirely what it
    does for the partner.
    """

    def __init__(self, agent: int, seed: int = 0):
        self.agent: int = int(agent)
        self._seed = _agent_seed(seed, self.agent)
        self.rng = np.random.default_rng(self._seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(
            self._seed if seed is None else _agent_seed(seed, self.agent))

    def __call__(self, env: TwoAgentEnv, result) -> int:
        window = env.windows[self.agent]
        private = [i for i, (node, mode) in enumerate(window.actions)
                   if mode == CLAMP and node in window.private]
        if not private:
            return window.pass_index
        return int(self.rng.choice(private))


class UncertaintyGreedyAgent:
    """Myopic uncertainty targeting for the CONSTRAINT backend -- the greedy analogue.

    `GreedyAgent` below reads the exact DP's score tables and cannot exist on the
    constraint path (see `enumerated_posterior`). This one is TRUTH-FREE and reads only
    the agent's own bootstrap frequencies: a claim is UNSURE when no answer reaches the
    confidence bar, each authority node is scored by how many unsure claims touch it, and
    the agent intervenes on the argmax -- the node whose experiments would speak to the
    most open questions. Passes when nothing is unsure. Seeded tie-breaks, so evaluation
    stays reproducible.

    Myopic by construction, exactly like the exact-path greedy: it values what is unsure
    NOW, not what an intervention would render decidable later. That is the baseline the
    thesis question names.
    """

    def __init__(self, agent: int, seed: int = 0, bar: float = 0.7):
        self.agent = int(agent)
        self.bar = float(bar)
        self._seed = _agent_seed(seed, self.agent)
        self.rng = np.random.default_rng(self._seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(
            self._seed if seed is None else _agent_seed(seed, self.agent))

    def _unsure_touching(self, belief, k: int) -> np.ndarray:
        counts = np.zeros(k)
        adjacency = np.asarray(belief.adjacency)
        directed = np.asarray(belief.directed)
        bidirected = np.asarray(belief.bidirected)
        for u in range(k):
            for v in range(u + 1, k):
                f_adj = float(adjacency[u, v])
                if max(f_adj, 1.0 - f_adj) < self.bar:
                    counts[u] += 1; counts[v] += 1       # adjacency itself unsettled
                elif f_adj >= self.bar:
                    settled = max(float(directed[u, v]), float(directed[v, u]),
                                  float(bidirected[u, v]))
                    if settled < self.bar:
                        counts[u] += 1; counts[v] += 1   # edge known, type not
        return counts

    def __call__(self, env: TwoAgentEnv, result) -> int:
        window = env.windows[self.agent]
        belief = window.belief.last
        if belief is None:
            return int(self.rng.integers(0, window.n_actions - 1))
        counts = self._unsure_touching(belief, window.k)
        authority_scores = {node: counts[window.pos[node]] for node in window.authority}
        best = max(authority_scores.values())
        if best <= 0:
            return window.pass_index                     # nothing left worth a round
        candidates = [n for n, s in authority_scores.items() if s == best]
        node = int(self.rng.choice(candidates))
        return window.action_index(node, prefer=VARY)


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

    TIE_BREAKS = ("random", "low", "high")

    def __init__(self, agent: int, env: TwoAgentEnv, seed: int = 0,
                 tie_break: str = "random"):
        """`tie_break` decides among actions of EQUAL expected gain, and it matters.

        GATE 2 fails: two greedy agents are no better than random, and the live explanation
        is that they compute the same objective over overlapping authority, pick the same
        shared target, and waste the round -- measured at 0.352 of rounds against random's
        0.227.

        If that is the cause, it is fixable WITHOUT communication. Ties are broken over each
        agent's own action list, so giving the two agents opposite conventions -- A takes the
        lowest-indexed tied action, B the highest -- makes them diverge whenever their tied
        sets overlap, using nothing but a fixed asymmetric convention agreed in advance. No
        observation, parameter or message crosses the federation boundary, so this stays
        inside the no-central-server constraint.

        This is a DIAGNOSTIC arm, not a proposed method: it tests whether collision is really
        what breaks myopic design under decentralisation.
        """
        if tie_break not in self.TIE_BREAKS:
            raise ValueError("tie_break must be one of %s" % (self.TIE_BREAKS,))
        self.agent: int = int(agent)
        self.tie_break = tie_break
        self._seed = _agent_seed(seed, self.agent)
        self.rng = np.random.default_rng(self._seed)
        window = env.windows[self.agent]
        self.space = _Window.get(window.k)
        self.candidates = [i for i, (node, mode) in enumerate(window.actions)
                           if node != -1]
        self.rule = env.config.score_rule

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(
            self._seed if seed is None else _agent_seed(seed, self.agent))

    def __call__(self, env: TwoAgentEnv, result) -> int:
        window = env.windows[self.agent]
        clean = (env.clean[self.agent] if env.config.disclose_regime
                 else np.zeros(len(env.samples), dtype=bool))
        posterior = enumerated_posterior(
            window, env.samples[:, window.nodes], env.known[self.agent], clean, self.rule)

        scores = np.full(len(self.candidates), -np.inf)
        for slot, action in enumerate(self.candidates):
            node, _mode = window.actions[action]
            position = window.pos[node]
            scores[slot] = _partition_entropy(
                self.space.signatures[:, position], posterior,
                int(self.space.signatures[:, position].max()) + 1)
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        if self.tie_break == "low":
            slot = int(best[0])
        elif self.tie_break == "high":
            slot = int(best[-1])
        else:
            slot = int(self.rng.choice(best))
        return int(self.candidates[slot])


class ProbeThenWorkAgent:
    """Probe your own private variables first, then work on the shared ones.

    THE REFERENCE FOR THE ATTRIBUTED ENVIRONMENT, and it has to exist or the comparison is
    rigged. `greedy_uncertainty` scores unsure STRUCTURE claims and knows nothing about
    attribution, so against a learner that is rewarded for attribution it would look
    artificially bad -- the same unfair comparison found and fixed on 2026-08-26 when the
    learner was the blindfolded one.

    Not a serious policy either: it is fixed, ignores the belief entirely, and cannot
    respond to what a partner has already settled. It is the "coordination is available if
    you pay for it" arm, in the same spirit as `forced_clamp`.

    Measured 2026-08-26 at 3 agents x 2 private, scale-free, budget 12: attribution 0.907
    and identification 0.658, against 0.000/0.000 for shared-only and 0.981/0.292 for
    private-only. A private probe pays only the PARTNERS, so the ordering is the whole
    point: neither pure strategy comes close to the mixture.
    """

    def __init__(self, agent: int, seed: int = 0, probe_rounds: Optional[int] = None):
        self.agent = int(agent)
        self.probe_rounds = probe_rounds
        self._seed = _agent_seed(seed, self.agent)
        self.turn = 0

    def reset(self, seed: Optional[int] = None) -> None:
        self.turn = 0

    def __call__(self, env: TwoAgentEnv, result) -> int:
        window = env.windows[self.agent]
        private, shared = list(window.private), list(window.shared)
        # HOW MANY MOVES THIS AGENT HAS ACTUALLY MADE, read from the environment rather
        # than counted here. Under turn-taking the policy is queried EVERY round and the
        # environment discards the inactive agent's move, so a local counter advances once
        # per round instead of once per action -- at three agents it ran three times too
        # fast, the probe phase was over before the agent had acted at all, and the arm
        # scored 0.000 attribution while the standalone version of the same policy scored
        # 0.907. `own_counts` is incremented only when a move is APPLIED, so it cannot
        # drift from what happened.
        index = int(env.own_counts[self.agent].sum())
        probe = len(private) if self.probe_rounds is None else self.probe_rounds
        if index < probe and private:
            return window.action_index(private[index % len(private)], prefer=VARY)
        if not shared:
            return window.pass_index
        return window.action_index(shared[(index - probe) % len(shared)], prefer=VARY)


class _LazyBaselines(dict):
    """Baselines built ON ACCESS, not up front.

    `GreedyAgent` enumerates the window and refuses past k=5, and raises on any non-exact
    backend when called. Building it eagerly meant a caller that only wanted
    `greedy_uncertainty` still crashed -- which took down the frontier sweep at window size
    6 and an attribution run's report. Nothing changes for a caller that asks for an arm
    this environment can supply.
    """

    def __init__(self, builders):
        super().__init__()
        self._builders = builders

    def __getitem__(self, key):
        if key not in self and key in self._builders:
            super().__setitem__(key, self._builders[key]())
        return super().__getitem__(key)

    def __contains__(self, key):
        return key in self._builders

    def keys(self):
        return self._builders.keys()


def make_baselines(env: TwoAgentEnv, agent: int, seed: int = 0) -> Dict[str, object]:
    return _LazyBaselines({
        "pass": lambda: PassAgent(agent, seed),
        "random_vary": lambda: RandomAgent(agent, seed, allow_clamp=False),
        "random_clamp": lambda: RandomAgent(agent, seed, allow_clamp=True),
        "forced_clamp": lambda: ForcedClampAgent(agent, seed),
        "greedy": lambda: GreedyAgent(agent, env, seed),
        "greedy_uncertainty": lambda: UncertaintyGreedyAgent(agent, seed),
        "probe_then_work": lambda: ProbeThenWorkAgent(agent, seed),
    })
