"""Two-agent active causal discovery on one shared system.

Design of record: docs/MA_DESIGN.md, with the decisions taken tonight recorded in
docs/MA_BUILD_LOG.md. The short version:

  ONE SYSTEM. Both agents study the same physical system, so every intervention perturbs
  the same rows. Each agent sees only its own columns -- `Z_A u X` for A, `Z_B u X` for B.

  SEPARATE BUDGETS, SIMULTANEOUS ACTIONS. `step` takes a pair. Both interventions happen
  in the same round on the same system. Two agents choosing the same node is not a
  collision to arbitrate, it is one intervention both of them wanted.

  PER-AGENT BELIEFS. Each agent holds an exact posterior over DAGs on its OWN window,
  scored with BGe on its own columns. No pooling, no central posterior, no CTDE.

  DISCLOSURE. An intervention on a SHARED node is announced to both agents -- `X` is
  visible to both, so this reveals nothing private. An intervention on a PRIVATE node is
  announced to nobody. The other agent still receives the perturbed rows and scores them
  as observational, which is a real misspecification and is the price of the privacy
  constraint. See MA_BUILD_LOG for the alternatives rejected.

  CONFOUNDING. Where a `z_B` is a common cause of two shared nodes, NO DAG over A's window
  is correct, so A's posterior cannot concentrate on the truth however much data it
  collects alone. Only B intervening can break it. That gap is the object of study, and it
  is why the per-agent model is deliberately misspecified rather than accidentally so.

  Confounding cannot touch a private node (verified exhaustively in tests/test_projection.py),
  which is what keeps each agent's hypothesis space a plain set of DAGs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ma.topology import Topology
from sa.graphs import build_graph_space
from sa.scm import sample_multi, sample_scm_params
from ma.score_regimes import JOINT_CONF, RegimeScorer
from sa.score import BGeScore

PASS_ACTION = -1

# Intervention modes. The two-agent case needs BOTH, and that is a finding rather than a
# convenience -- see docs/MA_BUILD_LOG.md.
#
#   VARY   do(v ~ N(0, scale)). The single-agent intervention. The node keeps varying, so
#          its descendants move with it and orientations separate.
#   CLAMP  do(v := 0). The node stops varying, so it stops transmitting variance. This is
#          the ONLY kind that cuts a confounding path, because a randomised value replaces
#          a latent common cause with a different latent common cause.
VARY = "vary"
CLAMP = "clamp"
MODES = (VARY, CLAMP)


@dataclass
class MAConfig:
    topology: Topology
    n_obs: int = 2000
    n_int: int = 200
    # PER AGENT -- separate budgets, not a shared pool. Lowered 8 -> 5 on 2026-08-19 to
    # match the single-agent operating point, for the same measured reason: above ~8 the
    # budget stops binding and every arm converges, so the comparison measures nothing.
    budget: int = 5
    identify_threshold: float = 0.7
    prior_p: float = 0.5
    intervene_scale: float = 2.0     # used by VARY; CLAMP always uses 0.0
    # Announce that SOME variable outside the other agent's window has been clamped --
    # one bit per round, naming nothing. Measured 2026-08-16 to be the difference between
    # 0% and 100% rescue of a confounded agent, and therefore not optional if coordination
    # is to be possible at all. See docs/MA_BUILD_LOG.md.
    disclose_regime: bool = True
    # How an agent scores data spanning two regimes. Measured 2026-08-17 over 300 episodes:
    #
    #   rule        unconfounded curve over p(clamp)   confounded payoff
    #   pooled      0.815 0.838 0.804 0.808            +0.000
    #   subset      0.815 0.454 0.708 0.956            +0.931   <- the valley
    #   joint       0.815 0.852 0.841 0.856            +0.000
    #   joint_conf  0.244 0.686 0.908 0.982            +0.690
    #
    # JOINT_CONF is the only rule that is both monotone (a learner can climb) and pays off
    # under confounding. It costs baseline accuracy when nobody clamps -- 0.244 against
    # 0.815 -- which is the honest price of admitting confounding might be present.
    score_rule: str = JOINT_CONF
    weight_range: tuple = (0.5, 2.0)
    noise_range: tuple = (0.5, 1.5)


@dataclass
class MAStepResult:
    beliefs: Dict[str, np.ndarray]
    identified: Dict[str, bool]
    done: bool
    n_interventions: Dict[str, int]
    info: dict = field(default_factory=dict)


class AgentView:
    """One agent's window, hypothesis space, and bookkeeping.

    The hypothesis space is every DAG over the agent's own nodes. At `(1,1,3)` the window
    is 4 nodes and there are 543 of them, so the posterior is exact and nothing is
    sampled.
    """

    def __init__(self, name: str, topology: Topology):
        self.name = name
        self.nodes: List[int] = list(topology.observed_by(name))
        self.authority: List[int] = list(topology.may_intervene_on(name))
        self.shared: List[int] = list(topology.exposed)
        self.private: List[int] = [n for n in self.nodes if n not in self.shared]
        self.k = len(self.nodes)
        self.pos = {node: i for i, node in enumerate(self.nodes)}
        # Every (target, mode) pair the agent may choose, plus PASS last. Indexed by the
        # integer the policy emits.
        self.actions: List[tuple] = [(node, mode) for node in self.authority
                                     for mode in MODES] + [(PASS_ACTION, None)]
        self.n_actions = len(self.actions)

        space = build_graph_space(self.k)
        self.dags = np.asarray(space.dags, dtype=np.int8)
        self.n_dags = len(self.dags)
        # Parent sets, precomputed once -- the inner loop of every belief update.
        self.parents: List[List[Tuple[int, ...]]] = [
            [tuple(np.flatnonzero(dag[:, node]).tolist()) for node in range(self.k)]
            for dag in self.dags
        ]
        self.score = BGeScore(self.k)
        self.log_prior = np.zeros(self.n_dags)   # uniform over DAGs, matching prior_p=0.5
        self.regime_scorer = RegimeScorer(
            self, [self.pos[node] for node in self.shared])

    def true_index(self, global_adjacency: np.ndarray) -> int:
        """Index of the agent's TRUE induced DAG -- the global graph restricted to its
        window. Well defined precisely because cross-private edges are forbidden, so no
        edge of the global graph is lost by restriction to one window or the other."""
        induced = np.asarray(global_adjacency)[np.ix_(self.nodes, self.nodes)]
        matches = np.flatnonzero((self.dags == induced).all(axis=(1, 2)))
        if len(matches) != 1:
            raise RuntimeError(f"induced graph not found exactly once ({len(matches)})")
        return int(matches[0])

    def posterior(self, samples: np.ndarray, known_intervened: np.ndarray,
                  clean: Optional[np.ndarray] = None,
                  rule: Optional[str] = None) -> np.ndarray:
        """Exact posterior over the agent's DAGs, from its own columns only.

        `known_intervened` is `[n, k]` and marks only the interventions this agent has
        been TOLD about. Rows where a node was secretly intervened on by the other agent
        are scored as observational -- the misspecification described in the module
        docstring.
        """
        # Regime conditioning. `clean` marks rows drawn while every variable hidden from
        # this agent was clamped, so no hidden variable was transmitting variance and the
        # agent's window really is a DAG. Where such rows exist the agent uses ONLY them.
        #
        # Valid but not efficient: it discards the observational rows rather than modelling
        # both regimes jointly. Deliberate -- a joint two-regime score would be the better
        # estimator, and it is the obvious next improvement, but conditioning on a subset
        # is correct inference and keeps the claim clean.
        if rule is not None:
            return self.regime_scorer.log_posterior(
                samples, known_intervened,
                np.zeros(len(samples), dtype=bool) if clean is None else clean, rule)

        if clean is not None and clean.any():
            samples = samples[clean]
            known_intervened = known_intervened[clean]

        cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        log_post = np.array(self.log_prior, dtype=float)

        for i in range(self.n_dags):
            total = 0.0
            for node in range(self.k):
                key = (node, self.parents[i][node])
                if key not in cache:
                    # A hard-intervened node's own rows say nothing about its parents,
                    # so they are dropped for that node's term only -- Cooper & Yoo.
                    keep = known_intervened[:, node] < 0.5
                    cache[key] = self.score.local_score(
                        node, self.parents[i][node], samples[keep])
                total += cache[key]
            log_post[i] += total

        log_post -= log_post.max()
        weights = np.exp(log_post)
        return weights / weights.sum()


class TwoAgentEnv:
    """The two-agent environment. Construct once, `reset` per episode."""

    def __init__(self, config: MAConfig, seed: int = 0):
        self.config = config
        self.topology = config.topology
        self.d = self.topology.d
        self.views = {name: AgentView(name, self.topology) for name in ("A", "B")}
        self._allowed = self.topology.allowed_edges()
        self._rng = np.random.default_rng(seed)

        self.true_adjacency: Optional[np.ndarray] = None
        self.params = None
        self.samples: Optional[np.ndarray] = None
        # Full intervention mask over ALL nodes -- ground truth, never shown to an agent.
        self.intervened: Optional[np.ndarray] = None
        # Per agent, the mask restricted to its window containing only what it was told.
        self.known: Dict[str, Optional[np.ndarray]] = {"A": None, "B": None}
        # Per agent, per row: was every variable hidden from this agent clamped?
        self.clean: Dict[str, Optional[np.ndarray]] = {"A": None, "B": None}
        self.beliefs: Dict[str, Optional[np.ndarray]] = {"A": None, "B": None}
        self.true_index: Dict[str, int] = {}
        self.n_interventions = {"A": 0, "B": 0}

    # -- episode lifecycle --------------------------------------------------------

    def reset(self, seed: Optional[int] = None,
              force_adjacency: Optional[np.ndarray] = None) -> MAStepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.config

        if force_adjacency is None:
            self.true_adjacency = self.topology.sample_dag(self._rng, p=cfg.prior_p)
        else:
            self.true_adjacency = np.asarray(force_adjacency, dtype=np.int8)

        self.params = sample_scm_params(
            self.true_adjacency, self._rng,
            weight_range=cfg.weight_range, noise_range=cfg.noise_range)

        self.samples, self.intervened = sample_multi(self.params, cfg.n_obs, self._rng)
        for name, view in self.views.items():
            self.known[name] = np.zeros((cfg.n_obs, view.k))
            self.clean[name] = np.zeros(cfg.n_obs, dtype=bool)
            self.true_index[name] = view.true_index(self.true_adjacency)

        self.n_interventions = {"A": 0, "B": 0}
        self._update_beliefs()
        return self._result()

    def step(self, action_a: int, action_b: int) -> MAStepResult:
        """Both agents act on the same system in the same round.

        Each action is an INDEX into that agent's `actions` list -- a (target, mode) pair,
        or PASS. Mode matters: see the VARY/CLAMP note at the top of this module.
        """
        cfg = self.config
        chosen = {}
        for name, index in (("A", int(action_a)), ("B", int(action_b))):
            view = self.views[name]
            if not 0 <= index < view.n_actions:
                raise ValueError(f"agent {name}: action {index} out of range")
            chosen[name] = view.actions[index]

        actions = {name: target for name, (target, _) in chosen.items()}
        targets = {}
        for name, (target, mode) in chosen.items():
            if target == PASS_ACTION:
                continue
            scale = cfg.intervene_scale if mode == VARY else 0.0
            # If both agents pick the same node in different modes, the more restrictive
            # one wins: a clamped node is not varying, whatever the other agent asked for.
            # This is not an arbitration rule between agents -- it is what the physical
            # system does when two hands hold the same dial.
            targets[target] = min(scale, targets.get(target, scale))

        if not targets:
            return self._result(passed=True)

        new_samples, new_intervened = sample_multi(
            self.params, cfg.n_int, self._rng, intervene_nodes=targets)

        self.samples = np.vstack([self.samples, new_samples])
        self.intervened = np.vstack([self.intervened, new_intervened])

        # Disclosure. Each agent learns about its own action always, and about the other
        # agent's action ONLY if the target is a shared node.
        for name, view in self.views.items():
            other = "B" if name == "A" else "A"
            disclosed = set()
            if actions[name] != PASS_ACTION:
                disclosed.add(actions[name])
            if actions[other] != PASS_ACTION and actions[other] in view.shared:
                disclosed.add(actions[other])

            block = np.zeros((cfg.n_int, view.k))
            for target in disclosed:
                block[:, view.pos[target]] = 1.0
            self.known[name] = np.vstack([self.known[name], block])

            # The regime bit. Rows are clean for this agent when every node hidden from it
            # was clamped this round. Only the BIT is disclosed -- not which node, not how
            # many, not whether the other agent has private structure at all beyond the
            # fact that it clamped something.
            hidden = [n for n in range(self.d) if n not in view.nodes]
            all_clamped = bool(hidden) and all(
                targets.get(n, None) == 0.0 for n in hidden)
            flag = bool(cfg.disclose_regime and all_clamped)
            self.clean[name] = np.concatenate(
                [self.clean[name], np.full(cfg.n_int, flag, dtype=bool)])

        for name, action in actions.items():
            if action != PASS_ACTION:
                self.n_interventions[name] += 1

        self._update_beliefs()
        return self._result()

    # -- belief and reporting ------------------------------------------------------

    def _update_beliefs(self) -> None:
        for name, view in self.views.items():
            self.beliefs[name] = view.posterior(
                self.samples[:, view.nodes], self.known[name], self.clean[name],
                rule=self.config.score_rule)

    def true_mass(self, name: str) -> float:
        return float(self.beliefs[name][self.true_index[name]])

    def _result(self, passed: bool = False) -> MAStepResult:
        threshold = self.config.identify_threshold
        identified = {name: self.true_mass(name) >= threshold for name in ("A", "B")}
        # Global identification needs nothing extra: cross-private edges are forbidden, so
        # every permitted edge lies in one window or the other, and two correct induced
        # DAGs union to the true global graph. Agreement on X and global acyclicity are
        # then automatic. See MA_DESIGN section 5 and the derivation of 2026-08-16.
        both = identified["A"] and identified["B"]
        out_of_budget = all(self.n_interventions[n] >= self.config.budget
                            for n in ("A", "B"))
        return MAStepResult(
            beliefs={n: self.beliefs[n].copy() for n in ("A", "B")},
            identified=dict(identified),
            done=both or passed or out_of_budget,
            n_interventions=dict(self.n_interventions),
            info={
                "true_mass": {n: self.true_mass(n) for n in ("A", "B")},
                "both_identified": both,
                "passed": passed,
                "budget_left": {n: self.config.budget - self.n_interventions[n]
                                for n in ("A", "B")},
            },
        )

    def observation(self, name: str) -> np.ndarray:
        """Edge marginals over the agent's own window, plus its remaining budget.

        Marginals rather than the raw posterior for the same reason as the single-agent
        `edge_marginals` condition: it is the representation that scales, and it is the
        one the d=7 results were produced with.
        """
        view = self.views[name]
        belief = self.beliefs[name]
        marginals = np.tensordot(belief, view.dags, axes=(0, 0))
        off_diagonal = ~np.eye(view.k, dtype=bool)
        budget_left = np.array(
            [(self.config.budget - self.n_interventions[name])
             / max(self.config.budget, 1)])
        return np.concatenate([marginals[off_diagonal], budget_left])

    def observation_dim(self, name: str) -> int:
        k = self.views[name].k
        return k * (k - 1) + 1

    def action_space(self, name: str) -> List[tuple]:
        """(target, mode) pairs the agent may choose, in a fixed order, PASS last."""
        return list(self.views[name].actions)

    def n_actions(self, name: str) -> int:
        return self.views[name].n_actions
