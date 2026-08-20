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

# Turn protocols. SIMULTANEOUS is the original and every result before 2026-08-20 was
# measured under it; it is kept so those numbers stay reproducible, not because it is
# preferred. Under the two turn-taking protocols exactly one agent may act per round and
# the other is forced to pass.
SIMULTANEOUS = "simultaneous"
ROUND_ROBIN = "round_robin"
RANDOM_TURN = "random"
TURN_ORDERS = (SIMULTANEOUS, ROUND_ROBIN, RANDOM_TURN)


@dataclass
class MA2Config:
    topology: Topology
    n_obs: int = 1000
    n_int: int = 100
    budget: int = 5                    # PER AGENT -- separate budgets, not a shared pool
    # One agent acts per round, or both. Budget stays PER AGENT under turn-taking, which
    # is deliberate: a shared pool would halve each agent's interventions and with them the
    # number of clean rounds, confounding the protocol change with a data-quantity change.
    # An episode therefore runs up to 2*budget rounds instead of budget rounds, and the
    # total row count doubles. That dilutes the clean FRACTION but not the clean COUNT,
    # and the regime rules score the two regimes separately, so only the count matters.
    turn_order: str = SIMULTANEOUS
    # Which intervention modes an agent may choose. `(CLAMP,)` is the clamp-only arm: it
    # halves the action space without removing the coordination problem, which becomes one
    # of targeting and timing rather than of mode. Kept as an ARM, not a deletion -- whether
    # clamp really dominates vary under turn-taking is the open question, and a policy
    # given both that converges on clamp anyway is the evidence for it.
    action_modes: Tuple[str, ...] = MODES
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
    # WHAT THE AGENT IS ACTUALLY PAID FOR.
    #   "u14"        the specified success criterion: each agent's posterior mass on its
    #                CREDIT SET (private-incident edges exact, Markov equivalent on the
    #                rest) clears the threshold, and the union is acyclic and globally
    #                equivalent. This is what `ma/evaluate2.py` reports.
    #   "identified" the previous reward: mass on the EXACT true DAG with the exact
    #                confounded set. Roughly twice as demanding -- measured with a random
    #                policy, 0.250 against u14's 0.500 with the regime bit and 0.133
    #                against 0.467 without.
    # The old reward was inherited from the single-agent environment, where there was no
    # confounding and no CPDAG relaxation, so "the exact true DAG" was the natural ask.
    # NOW THE DEFAULT. The earlier objection -- that it reintroduced window enumeration
    # into the training loop -- no longer applies: `credit_candidates` enumerates the
    # SHARED subgraph only (25 DAGs at |X| = 3, not 543), which is exponential in |X| and
    # constant in the window size. That is the same axis the confounding enumeration
    # already costs, so no new scaling debt is acquired.
    reward_criterion: str = "u14"


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

    def __init__(self, name: str, topology: Topology,
                 modes: Sequence[str] = MODES):
        self.name = name
        self.modes: Tuple[str, ...] = tuple(modes)
        self.nodes: List[int] = list(topology.observed_by(name))
        self.authority: List[int] = list(topology.may_intervene_on(name))
        self.shared: List[int] = list(topology.exposed)
        self.private: List[int] = [n for n in self.nodes if n not in self.shared]
        self.k = len(self.nodes)
        self.pos = {node: i for i, node in enumerate(self.nodes)}
        self.actions: List[Tuple[int, Optional[str]]] = (
            [(node, mode) for node in self.authority for mode in self.modes]
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
        if config.turn_order not in TURN_ORDERS:
            raise ValueError(f"turn_order must be one of {TURN_ORDERS}")
        if not config.action_modes or any(m not in MODES for m in config.action_modes):
            raise ValueError(f"action_modes must be a non-empty subset of {MODES}")
        # A block marked clean is scored against the BARE DAG -- no confounding at all. With
        # a single hidden node that is exact. With several, one clamp silences only the
        # pathways through the node it clamped and the block is PARTIALLY clean, so scoring
        # it as fully clean is simply wrong. The fix is known and cheap -- give each clamp
        # block its own active confounding subset S_r and marginalise it, which costs
        # R * 2^|S| because the per-block log-scores ADD -- but it is not built, and the
        # ladder does not need it until an agent has two private nodes. Fail loudly rather
        # than score silently wrong data.
        widest = max(len(config.topology.hidden_from(name)) for name in AGENTS)
        if widest > 1:
            raise NotImplementedError(
                f"topology {config.topology.name!r} hides {widest} nodes from an agent; the "
                "regime rules assume a clean block has NO confounding, which holds only for "
                "one hidden node. Per-block confounding subsets are needed first.")
        self.config = config
        self.topology = config.topology
        self.windows: Dict[str, AgentWindow] = {
            name: AgentWindow(name, config.topology, config.action_modes)
            for name in AGENTS}
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

        self._credit_cache: Dict[str, np.ndarray] = {}
        self.round = 0
        self.active: Optional[str] = None
        self.last_chosen: Dict[str, Tuple[int, Optional[str]]] = {
            n: (PASS_ACTION, None) for n in AGENTS}
        self._refresh()
        return self._result(reward=0.0)

    # -- turn taking --------------------------------------------------------------------

    def active_agent(self) -> Optional[str]:
        """Whose turn it is, or None when both act. Round-robin alternates from A; random
        draws from the environment's own stream, so the choice is part of the episode seed
        and an evaluation is reproducible without the policy having to record it."""
        order = self.config.turn_order
        if order == SIMULTANEOUS:
            return None
        # Only agents with budget left are eligible. Without this, round-robin hands the
        # turn to an exhausted agent, whose only legal move is a pass -- which reads as a
        # voluntary pass and ends the episode while the partner still has moves to spend.
        eligible = [n for n in AGENTS
                    if self.n_interventions[n] < self.config.budget]
        if not eligible:
            return None
        if order == ROUND_ROBIN:
            for offset in range(len(AGENTS)):
                candidate = AGENTS[(self.round + offset) % len(AGENTS)]
                if candidate in eligible:
                    return candidate
        return str(self._rng.choice(eligible))

    def step(self, action_a: int, action_b: int) -> MA2StepResult:
        cfg = self.config
        actions = {"A": int(action_a), "B": int(action_b)}
        for name, index in actions.items():
            if not 0 <= index < self.windows[name].n_actions:
                raise ValueError(f"action {index} out of range for {name}")

        # Under turn-taking the inactive agent is FORCED to pass. Its submitted action is
        # discarded rather than rejected: the policy is queried for both agents every round
        # and the environment, not the policy, owns the protocol.
        self.active = self.active_agent()
        if cfg.turn_order != SIMULTANEOUS:
            for name in AGENTS:
                if name != self.active:            # active None => everyone passes
                    actions[name] = self.windows[name].pass_index
        self.round += 1

        # A voluntary pass by the only agent able to act ends the episode, exactly as a
        # mutual pass does under simultaneous play -- it is the same signal, "no one wants
        # another move", read through whoever had the move.
        passed = all(actions[n] == self.windows[n].pass_index for n in AGENTS)
        # What was ACTUALLY applied, after the protocol has had its say. Consumers must
        # tally from this rather than from the submitted actions: under turn-taking the
        # inactive agent still submits a move and it is discarded, so counting submissions
        # double-counts the moves and corrupts any per-move statistic.
        self.last_chosen = {n: self.windows[n].actions[actions[n]] for n in AGENTS}
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
            # ANY, not ALL. With one hidden node the two are identical, which is every
            # result to date. They diverge as soon as an agent has more than one private
            # node, and there ALL is unreachable: an agent gets one action per round and
            # has no authority over the other's private nodes, so "all hidden clamped" can
            # never fire -- the regime machinery would be silently dead. See the guard in
            # __init__ and tests/test_env2_turns.py::test_clean_rounds_are_reachable.
            hidden_clamped = bool(hidden) and any(
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

    def _u14_state(self):
        """Per-agent credit-set mass, and whether the full [U14] criterion holds.

        DP-NATIVE. Nothing here enumerates the window. `credit_candidates` enumerates only
        the SHARED subgraph -- 25 DAGs at |X| = 3 against 543 for the window -- because
        criterion 1 pins every private-incident edge to the truth, leaving the shared block
        as the only freedom. `joint_conf_set_probability` then scores those candidates as
        CAUSAL graphs through the subset DP.

        Verified identical to the enumerated `credit_set` on 40 episodes x 2 agents.

        This is the same object `ma/evaluate2.py` reports, so the reward and the reported
        number cannot drift apart -- which is exactly how they drifted apart before.
        """
        from ma.evaluate2 import credit_candidates, union_graph
        from sa.graphs import is_acyclic, mec_signature

        mass, best_graph = {}, {}
        for name in AGENTS:
            window = self.windows[name]
            truth = window.induced(self.true_adjacency)
            clean = (self.clean[name] if self.config.disclose_regime
                     else np.zeros(len(self.samples), dtype=bool))
            # Cached per episode: the true graph is fixed for its whole duration, so the
            # credit set is too, and it was being rebuilt at every step.
            cached = self._credit_cache.get(name)
            if cached is None:
                cached = credit_candidates(window, truth)
                self._credit_cache[name] = cached
            candidates = cached
            pairs = self._confounded_positions(name)
            mass[name] = float(window.belief.joint_conf_set_probability(
                self.samples[:, window.nodes], self.known[name], clean,
                candidates, pairs))
            # Representative for the union check. Only consulted when the agent is
            # credited, and every credited answer is Markov equivalent to the truth with
            # its private edges exact, so any member is a valid stand-in.
            best_graph[name] = candidates[0] if len(candidates) else truth

        threshold = self.config.identify_threshold
        d = self.topology.d
        union = np.zeros((d, d), dtype=np.int8)
        for name in AGENTS:
            window = self.windows[name]
            graph = np.asarray(best_graph[name])
            for i, u in enumerate(window.nodes):
                for j, v in enumerate(window.nodes):
                    if graph[i, j]:
                        union[u, v] = 1
        both = bool(all(mass[n] >= threshold for n in AGENTS)
                    and is_acyclic(union)
                    and mec_signature(union) == mec_signature(
                        np.asarray(self.true_adjacency)))
        return mass, both

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
        if self.config.reward_criterion == "u14":
            mass, both = self._u14_state()
            identified = {name: mass[name] >= threshold for name in AGENTS}
        else:
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
                  # ROUNDS, not per-agent interventions. Under turn-taking an agent acts
                  # every other round, so `n_interventions` is roughly half the episode
                  # length -- reporting one as the other understates duration by 2x.
                  "rounds": self.round, "active": self.active,
                  "budget_left": {n: self.config.budget - self.n_interventions[n]
                                  for n in AGENTS}},
        )

    # -- convenience --------------------------------------------------------------------

    def n_actions(self, name: str) -> int:
        return self.windows[name].n_actions

    def obs_size(self, name: str) -> int:
        return self.windows[name].obs_size
