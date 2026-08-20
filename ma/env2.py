"""PHASE 2 -- the two-agent environment, rebuilt on the DP belief.

Replaces `ma/env.py`. The differences that matter, each traceable to a ruling or a
measurement rather than to preference:

  belief        `WindowBeliefDP` (subset DP) instead of 543-DAG enumeration [U11]. Carries
                to k ~ 15-20 windows; enumeration died at k=6.
  n_obs         1000, not 100. Measured 2026-08-19: at n_obs=100 the posterior NEVER
                reaches the 0.7 identification threshold -- best of 150 episodes was 0.579
                -- so the environment is harder than its own success criterion allows and
                GATE 1 fails on the low side. 1000 leaves ~94% of episodes needing
                interventions while remaining solvable.
  regime bit    DEFAULT OFF [U, 2026-08-19]. The no-bit arm is the baseline and the simpler
                system, so it fails in fewer ways and a bug in it is attributable. It also
                needs no disclosure protocol, so it can run before the supervisor rules on
                whether the bit is admissible at all.
  budget        The DEFAULT here is 5, but training runs pass 8 explicitly and that is
                what every reported result uses. The two gates want opposite budgets:
                discrimination peaks at 2-3 and is gone by 16, while coordination registers
                nothing below 5, because an agent must spend moves clamping for its partner
                AND moves experimenting on itself. Training follows coordination.

KNOWN FLAW, NOT YET FIXED -- THE REWARD IS NOT THE REPORTED METRIC.

`_result` pays +1 when `both_identified`, which requires each agent's posterior to put
>= `identify_threshold` on its EXACT true DAG with the correct confounded pairs named.
`ma/evaluate2.py` reports `success` under the [U14] criterion, which credits any graph that
matches the truth on private-incident edges and is MARKOV EQUIVALENT to it. The second is
roughly twice as forgiving -- measured with a random policy: reward 0.250 vs reported 0.500
with the regime bit, 0.133 vs 0.467 without.

So every reported number is on a metric the agent was never trained for. [U14] is the
criterion that was specified, so it is the REWARD that is wrong, not the report. Fixing it
means changing the reward and re-running, which has not been done.

Everything else follows `docs/MA_PROBLEM_STATEMENT.md`: one shared SCM, vertical partition
with overlap, cross-private edges forbidden, hard interventions in two value modes,
simultaneous actions, separate budgets, shared-node targets disclosed AFTER acting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ma.belief_dp import JOINT_CONF, WindowBeliefDP
from ma.topology import Topology
from sa.scm import sample_multi, sample_scm_params

PASS_ACTION = -1
VARY = "vary"
CLAMP = "clamp"
MODES = (VARY, CLAMP)
AGENTS = ("A", "B")


@dataclass
class MA2Config:
    topology: Topology
    n_obs: int = 1000
    n_int: int = 100
    budget: int = 5                    # PER AGENT -- separate budgets, not a shared pool
    identify_threshold: float = 0.7
    prior_p: float = 0.5
    intervene_scale: float = 2.0       # VARY draws N(0, scale^2); CLAMP always uses 0.0
    score_rule: str = JOINT_CONF
    # One bit per round: "I clamped something you cannot see". OFF by default -- the no-bit
    # arm is the baseline. See the module docstring.
    disclose_regime: bool = False
    # Which SHARED nodes the other agent targeted. Shared columns are visible to both, so
    # this reveals nothing private. Delivered AFTER acting, so it can only condition
    # future moves [U10].
    disclose_shared_targets: bool = True
    step_cost: float = 0.05


@dataclass
class MA2StepResult:
    beliefs: Dict[str, np.ndarray]     # edge marginals per agent
    identified: Dict[str, bool]
    done: bool
    reward: float
    n_interventions: Dict[str, int]
    info: dict = field(default_factory=dict)


class AgentWindow:
    """One agent's view: its columns, its authority, and its DP belief."""

    def __init__(self, name: str, topology: Topology):
        self.name = name
        self.nodes: List[int] = list(topology.observed_by(name))
        self.authority: List[int] = list(topology.may_intervene_on(name))
        self.shared: List[int] = list(topology.exposed)
        self.private: List[int] = [n for n in self.nodes if n not in self.shared]
        self.k = len(self.nodes)
        self.pos = {node: i for i, node in enumerate(self.nodes)}
        self.actions: List[Tuple[int, Optional[str]]] = (
            [(node, mode) for node in self.authority for mode in MODES]
            + [(PASS_ACTION, None)])
        self.n_actions = len(self.actions)
        self.pass_index = self.n_actions - 1
        self.belief = WindowBeliefDP(self.k, [self.pos[n] for n in self.shared])

    def induced(self, global_adjacency: np.ndarray) -> np.ndarray:
        """The global graph restricted to this window. Well defined precisely because
        cross-private edges are forbidden, so no edge is lost by restriction."""
        return np.asarray(global_adjacency)[np.ix_(self.nodes, self.nodes)]

    @property
    def obs_size(self) -> int:
        return self.k * (self.k - 1) + 1 + len(self.shared) + 1


class TwoAgentEnv2:
    """One SCM, two agents, simultaneous hard interventions."""

    def __init__(self, config: MA2Config, seed: int = 0):
        self.config = config
        self.topology = config.topology
        self.windows: Dict[str, AgentWindow] = {
            name: AgentWindow(name, config.topology) for name in AGENTS}
        self._rng = np.random.default_rng(seed)
        self.reset(seed)

    # -- episode ------------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None,
              adjacency: Optional[np.ndarray] = None) -> MA2StepResult:
        cfg = self.config
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.true_adjacency = (np.asarray(adjacency) if adjacency is not None
                               else self.topology.sample_dag(self._rng, p=cfg.prior_p))
        self.params = sample_scm_params(self.true_adjacency, self._rng)
        self.samples, _ = sample_multi(self.params, cfg.n_obs, self._rng)

        self.known: Dict[str, np.ndarray] = {}
        self.clean: Dict[str, np.ndarray] = {}
        self.n_interventions: Dict[str, int] = {}
        self.disclosed: Dict[str, np.ndarray] = {}
        self.regime_bit: Dict[str, float] = {}
        for name, window in self.windows.items():
            self.known[name] = np.zeros((cfg.n_obs, window.k))
            self.clean[name] = np.zeros(cfg.n_obs, dtype=bool)
            self.n_interventions[name] = 0
            self.disclosed[name] = np.zeros(len(window.shared))
            self.regime_bit[name] = 0.0

        self._refresh()
        return self._result(reward=0.0)

    def step(self, action_a: int, action_b: int) -> MA2StepResult:
        cfg = self.config
        actions = {"A": int(action_a), "B": int(action_b)}
        for name, index in actions.items():
            if not 0 <= index < self.windows[name].n_actions:
                raise ValueError(f"action {index} out of range for {name}")

        passed = all(actions[n] == self.windows[n].pass_index for n in AGENTS)
        if passed:
            return self._result(reward=0.0, passed=True)

        # Both act on the SAME system. On a collision the more restrictive assignment wins:
        # a clamp fixes the variable outright, so a simultaneous vary cannot also hold.
        targets: Dict[int, float] = {}
        chosen: Dict[str, Tuple[int, Optional[str]]] = {}
        for name in AGENTS:
            node, mode = self.windows[name].actions[actions[name]]
            chosen[name] = (node, mode)
            if node == PASS_ACTION:
                continue
            scale = 0.0 if mode == CLAMP else cfg.intervene_scale
            targets[node] = min(scale, targets.get(node, np.inf))

        new_samples, _ = sample_multi(self.params, cfg.n_int, self._rng,
                                      intervene_nodes=targets)
        self.samples = np.vstack([self.samples, new_samples])

        for name in AGENTS:
            window = self.windows[name]
            block = np.zeros((cfg.n_int, window.k))
            # An agent always knows its OWN intervention.
            own_node, _ = chosen[name]
            if own_node != PASS_ACTION:
                block[:, window.pos[own_node]] = 1.0
            # And the other's, but only on SHARED nodes -- those columns are visible to
            # both, so this discloses nothing private.
            other = "B" if name == "A" else "A"
            other_node, _ = chosen[other]
            if other_node != PASS_ACTION and other_node in window.shared:
                block[:, window.pos[other_node]] = 1.0
            self.known[name] = np.vstack([self.known[name], block])

            # A batch is CLEAN for this agent when every variable hidden from it was
            # clamped -- only then is its window really a DAG rather than a latent
            # projection.
            hidden = self.topology.hidden_from(name)
            hidden_clamped = bool(hidden) and all(
                targets.get(node, None) == 0.0 for node in hidden)
            self.clean[name] = np.concatenate(
                [self.clean[name], np.full(cfg.n_int, hidden_clamped, dtype=bool)])

            self.disclosed[name] = np.zeros(len(window.shared))
            if cfg.disclose_shared_targets and other_node in window.shared:
                self.disclosed[name][window.shared.index(other_node)] = 1.0
            self.regime_bit[name] = float(hidden_clamped) if cfg.disclose_regime else 0.0

        for name in AGENTS:
            if chosen[name][0] != PASS_ACTION:
                self.n_interventions[name] += 1

        self._refresh()
        cost = cfg.step_cost * sum(
            1 for name in AGENTS if chosen[name][0] != PASS_ACTION)
        return self._result(reward=-cost)

    # -- belief -------------------------------------------------------------------------

    def _refresh(self) -> None:
        """One belief update per agent. `clean` is passed regardless of whether the regime
        bit is DISCLOSED: when it is not, the agent is not told, and the rule reduces to
        scoring everything once. Keeping the mask correct internally means the no-bit arm
        differs from the with-bit arm in exactly one place -- what the agent is told."""
        cfg = self.config
        self.marginals: Dict[str, np.ndarray] = {}
        for name, window in self.windows.items():
            clean = (self.clean[name] if cfg.disclose_regime
                     else np.zeros(len(self.samples), dtype=bool))
            self.marginals[name] = window.belief.edge_marginals(
                self.samples[:, window.nodes], self.known[name], clean, cfg.score_rule)

    def true_mass(self, name: str) -> float:
        window = self.windows[name]
        cfg = self.config
        clean = (self.clean[name] if cfg.disclose_regime
                 else np.zeros(len(self.samples), dtype=bool))
        rule = cfg.score_rule
        if rule == JOINT_CONF:
            # joint_conf has no single log_weights table -- it is a mixture over confounding
            # assignments -- so the true DAG's mass is read from the mixture directly. The
            # TRUE confounded pairs are passed in because identification requires getting
            # the confounding right as well as the causal edges; see the method docstring
            # for the two wrong criteria that preceded this one.
            return float(window.belief.joint_conf_dag_probability(
                self.samples[:, window.nodes], self.known[name], clean,
                window.induced(self.true_adjacency),
                confounded_pairs=self._confounded_positions(name)))
        return float(np.exp(window.belief.log_prob_dag(
            self.samples[:, window.nodes], self.known[name], clean, rule,
            window.induced(self.true_adjacency))))

    def _confounded_positions(self, name: str):
        """Truly confounded shared pairs, as WINDOW positions.

        Read from the generating graph via the latent projection, so it is ground truth and
        never visible to the agent -- it is used only to score identification."""
        from ma.projection import bidirected_pairs
        window = self.windows[name]
        pairs = bidirected_pairs(self.true_adjacency, tuple(window.nodes))
        return tuple((window.pos[u], window.pos[v]) for u, v in pairs)

    # -- observation and result ---------------------------------------------------------

    def observation(self, name: str) -> np.ndarray:
        """Edge marginals, remaining budget, and whatever was disclosed.

        Every feature is on [0, 1]. Raw counts were a real bug once: the budget feature sat
        at 20.0 beside probabilities in [0, 1] and dominated the first layer.
        """
        window = self.windows[name]
        marginals = self.marginals[name]
        off_diagonal = ~np.eye(window.k, dtype=bool)
        budget_left = np.array(
            [(self.config.budget - self.n_interventions[name])
             / max(self.config.budget, 1)])
        return np.concatenate([marginals[off_diagonal], budget_left,
                               self.disclosed[name],
                               np.array([self.regime_bit[name]])])

    def _result(self, reward: float, passed: bool = False) -> MA2StepResult:
        threshold = self.config.identify_threshold
        mass = {name: self.true_mass(name) for name in AGENTS}
        identified = {name: mass[name] >= threshold for name in AGENTS}
        both = all(identified.values())
        out_of_budget = all(self.n_interventions[n] >= self.config.budget
                            for n in AGENTS)
        if both:
            reward += 1.0                       # shared terminal reward [U15]
        return MA2StepResult(
            beliefs={n: self.marginals[n].copy() for n in AGENTS},
            identified=identified,
            done=both or passed or out_of_budget,
            reward=reward,
            n_interventions=dict(self.n_interventions),
            info={"true_mass": mass, "both_identified": both, "passed": passed,
                  "budget_left": {n: self.config.budget - self.n_interventions[n]
                                  for n in AGENTS}},
        )

    # -- convenience --------------------------------------------------------------------

    def n_actions(self, name: str) -> int:
        return self.windows[name].n_actions

    def obs_size(self, name: str) -> int:
        return self.windows[name].obs_size
