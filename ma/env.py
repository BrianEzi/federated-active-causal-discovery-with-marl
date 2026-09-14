"""The federated environment: one SCM, several agents, overlapping variable windows.

A single structural causal model is partitioned vertically. Each agent observes and may
intervene on its own window of variables; windows overlap on a shared interface, and no
agent sees the whole graph. Edges between two agents' private blocks are forbidden, so all
latent confounding an agent can encounter is induced by the partition itself and is
resolvable through the shared interface.

An episode draws a graph, gives every agent observational rows over its window, and then
runs rounds until the shared intervention budget is spent. On each round an agent picks a
variable to intervene on, or passes. Interventions are hard: the variable's structural
equation is replaced, so its parents are disconnected and the effect propagates to its
descendants. Targets on shared variables are disclosed after the round; targets on private
variables are not.

Each agent carries a belief over the ancestral marks between pairs in its window, updated
from the evidence its interventions produce. Two backends implement it, selected by
`belief_backend` and described in `docs/BELIEF.md`.

Reward follows `reward_criterion`. Every result reported in the dissertation used
`"claims"` with `claim_bar=1.0`, under which an agent is paid for pairs its belief has
resolved to a single mark. The `"u14"` default on `MAConfig` predates that and is retained
only so older configurations still load; it scores a Markov-equivalence criterion that no
reported run used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cb.backend import ConstraintBackend
from crosscheck.belief_dp import JOINT_CONF, WindowBeliefDP
from ma.topology import ER, Topology
from ma.priors import connectivity_prior_p
from ma.scm import sample_multi, sample_scm_params

PASS_ACTION = -1

# Belief backends. The env talks to either through the same `edge_marginals` call; they
# differ in what identification means (posterior mass vs replicate credit fraction) and in
# what they can soundly handle (see the capability check in `TwoAgentEnv.__init__`).
EXACT = "exact"
CONSTRAINT = "constraint"
# The deterministic idealisation (cb/versionspace.py): belief is the SET of structures still
# consistent with what interventions have established, with no statistics anywhere. It
# answers the infinite-data question -- can agents learn to divide experiments -- with
# episodes in milliseconds and a computable optimum to measure against. It is NOT a claim
# about finite data; the constraint backend remains the realistic path.
VERSION_SPACE = "version_space"
# Version space over (structure, ATTRIBUTION): who owns each hidden cause.
# See cb/attribution.py. Deterministic, like VERSION_SPACE, and it adds the
# only channel through which one agent can help another in that environment.
ATTRIBUTED = "attributed"
# FACTORED STRUCTURE + ENUMERATED OWNERSHIP. `attributed` couples two enumerations that are
# independent: the structure space (3^edges) and the attribution space over owner assignments.
# Only the first is expensive -- at k=12 the structure space is 5.0e10
# against 482 attribution hypotheses -- so attribution was chained to a structure belief it
# never needed and was cut from the thesis on that basis. This backend keeps the enumerated
# attribution exactly as it was and swaps the structure half for `factored`.
FACTORED_ATTRIBUTED = "factored_attributed"
# FACTORED STRUCTURE + COMPONENT-FACTORED OWNERSHIP. `factored_attributed` still enumerates
# every owner assignment jointly, so it is bounded by the attribution space itself -- 482
# hypotheses at k=12 and 8.4e10 at k=20 -- and holds a GLOBAL cap on how many settled pairs
# it will attribute at all. See cb/component_attribution.py: the candidate set factors
# exactly over connected components of the bidirected graph, so the cap becomes per component
# and the cost becomes a sum rather than a product.
COMPONENT_ATTRIBUTED = "component_attributed"
# Pairwise belief: one small version space PER PAIR, O(k^2) state and update,
# so it carries to window sizes the enumerated belief cannot reach. See
# cb/factored.py for what it gives up -- joint constraints, and with them the
# exact ceiling and exact optimum.
FACTORED = "factored"
BACKENDS = (EXACT, CONSTRAINT, VERSION_SPACE, ATTRIBUTED, FACTORED,
            FACTORED_ATTRIBUTED, COMPONENT_ATTRIBUTED)
# Backends whose belief carries an ATTRIBUTION as well as a structure. They replace the
# bidirected claim with the attribution claim, and they are the only ones for which the
# partner-disclosure channel carries anything -- see `_disclose_partner_responses`.
ATTRIBUTION_BACKENDS = (ATTRIBUTED, FACTORED_ATTRIBUTED, COMPONENT_ATTRIBUTED)
# Backends whose belief exposes bootstrap-shaped claim frequencies, so `cb.claims` and the
# constraint-side greedy read them the same way.
CLAIM_BACKENDS = (CONSTRAINT, VERSION_SPACE, ATTRIBUTED, FACTORED,
                  FACTORED_ATTRIBUTED, COMPONENT_ATTRIBUTED)
VARY = "vary"
CLAMP = "clamp"
MODES = (VARY, CLAMP)

# Turn protocols. SIMULTANEOUS lets every agent act each round and was
# measured under it; it is kept so those numbers stay reproducible, not because it is
# preferred. Under the two turn-taking protocols exactly one agent may act per round and
# the other is forced to pass.
SIMULTANEOUS = "simultaneous"
ROUND_ROBIN = "round_robin"
RANDOM_TURN = "random"
TURN_ORDERS = (SIMULTANEOUS, ROUND_ROBIN, RANDOM_TURN)

# The broadcast signal, one categorical per agent per round, free of charge. It names a
# REGION, never a variable, and carries no value -- see docs/ARCHITECTURE.md section 6.
# Advisory: nothing forces an agent to respect it. Conditional on
# that a peer-to-peer action-type broadcast is admissible; `disclose_signals=False` removes
# it in one flag.
NO_INTERVENTION = "none"
SHARED_SIGNAL = "shared"
PRIVATE_SIGNAL = "private"
SIGNALS = (NO_INTERVENTION, SHARED_SIGNAL, PRIVATE_SIGNAL)


def _is_connected(adjacency: np.ndarray) -> bool:
    """Is the graph one component, ignoring edge direction?

    A DISCONNECTED graph splits the agents into independent subproblems: no path crosses the
    private/shared boundary, so there is no latent confounding and nothing to coordinate
    about. Those episodes cannot test what this project is building, so every multi-agent
    metric is reported split by this flag rather than pooled over both kinds.
    """
    a = np.asarray(adjacency) > 0.5
    d = a.shape[0]
    if d == 0:
        return True
    undirected = a | a.T
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for other in np.flatnonzero(undirected[node]):
            if int(other) not in seen:
                seen.add(int(other))
                frontier.append(int(other))
    return len(seen) == d


@dataclass
class MAConfig:
    """Everything that defines one experimental cell.

    Defaults here are the library defaults, not the reported configuration. Every run
    records the config it actually used, and `scripts/rescore_from_config.py` rebuilds an
    environment from that record rather than from these values, so an evaluation can never
    silently differ from the run it is scoring.
    """

    topology: Topology
    # Observational rows each agent receives at the start of an episode, and interventional
    # rows drawn per intervention.
    n_obs: int = 1000
    n_int: int = 100
    # Interventions available for the whole federation per episode, drawn from a shared
    # pool. Under round-robin this is equivalent to budget / n_agents each, and the two
    # differ only when the turn order is random. A shared pool makes free-riding cost
    # something: a round one agent wastes is a round another does not get.
    budget: int = 10
    # SIMULTANEOUS lets every agent act each round; TURN_ROBIN and TURN_RANDOM let one act.
    turn_order: str = SIMULTANEOUS
    # CLAMP fixes an intervened variable to a constant, VARY draws it randomly. Clamping
    # removes the variable as a variance source, which is the only way to cut a confounding
    # path through it; randomising keeps it informative about its own descendants.
    action_modes: Tuple[str, ...] = (CLAMP,)
    # Clamp on private variables, randomise on shared ones. Overrides `action_modes`, and
    # gives each variable exactly one action so the action space does not double.
    mode_by_role: bool = False
    # Posterior mass on the true structure at which a window counts as identified, for the
    # backends that carry a posterior.
    identify_threshold: float = 0.7
    # Graph generator. `prior_p` is the edge probability for Erdos-Renyi; `sf_m` the
    # attachment parameter for scale-free. The sweep reports scale-free with an
    # Erdos-Renyi control at matched density.
    prior_p: Optional[float] = None
    graph_model: str = ER
    sf_m: int = 2
    # Robustness families (see `ma/scm.py`): noise shape, and the parent-to-child function.
    noise_dist: str = "gaussian"
    mechanism: str = "linear"
    # Where each window's adjacency comes from. "true" supplies it; "estimated" recovers it
    # per episode from the observational rows by conditional-independence testing, at the
    # level and conditioning-set size below. `docs/BELIEF.md` explains why this assumption
    # is load-bearing.
    skeleton_source: str = "true"
    skeleton_alpha: float = 0.05
    skeleton_max_cond: int = 2
    # Evidence regime. "oracle" answers ancestry queries exactly; "sampled" estimates them
    # from the interventional rows at `vs_evidence_alpha`. Under a partial oracle,
    # `vs_evidence_power` is the fraction of queries answered, the rest returning nothing.
    vs_evidence: str = "oracle"
    vs_evidence_alpha: float = 0.001
    vs_evidence_power: float = 1.0
    # Make the answer rate fall with graph distance rather than apply uniformly.
    distance_weighted_power: bool = False
    intervene_scale: float = 2.0       # VARY draws N(0, scale^2); CLAMP always uses 0.0
    score_rule: str = JOINT_CONF
    # What is disclosed between agents. Shared targets and a one-bit private-intervention
    # signal are disclosed after each round. `disclose_regime` additionally announces that a
    # foreign intervention occurred, which repairs a calibration failure under sampled
    # evidence; it is off in every reported run.
    disclose_regime: bool = False
    disclose_shared_targets: bool = True
    disclose_signals: bool = True
    # Reward. "claims" pays for pairs resolved to a single mark and is what every reported
    # run used; "u14" scores a Markov-equivalence criterion and is retained only so older
    # configurations still load. `claim_bar` is the confidence required to commit a mark,
    # and must be 1.0 for the version-space backends, where the truth is always in the set
    # and a lower bar would re-admit committing the wrong mark.
    step_cost: float = 0.0
    reward_criterion: str = "u14"
    claim_bar: float = 0.7
    # Credit each agent for what its own interventions resolved, rather than paying every
    # agent the round's outcome. Under turn-taking only one agent acts per round, so without
    # this an idle agent is rewarded or punished for a peer's choice.
    per_agent_reward: bool = False
    difference_reward: bool = False
    difference_reward_mode: str = "both"
    reward_scale: float = 1.0
    # Optional observation features: per-pair belief state, latent ownership, cumulative
    # partner intervention counts, and a flag marking pairs already probed but unresolved.
    # The last matters under a partial oracle, where a withheld answer is otherwise
    # indistinguishable from a question never asked.
    observe_belief_channels: bool = False
    observe_owner_channel: bool = False
    observe_partner_counts: bool = False
    observe_reprobe_signal: bool = False
    attribution_local_disturbance: bool = True
    claims_require_all_types: bool = True
    claim_penalty: float = 1.0         # settled-wrong weight in the dense reward
    # Belief implementation. EXACT enumerates structures and is sound only for small
    # windows; CONSTRAINT and the version-space backends carry the reported cells. The
    # constructor refuses combinations known to be unsound unless
    # `allow_unsound_backend` is set, which exists so the defect can be demonstrated.
    belief_backend: str = EXACT
    cb_n_boot: int = 50            # bootstrap replicates per refresh; B is the speed knob
    cb_alpha: float = 0.01
    cb_skeleton_alpha: Optional[float] = None
    cb_n_jobs: int = 1
    policy_arch: str = "mlp"
    # Restrict episodes to graphs that do or do not contain partition-induced confounding.
    episode_mix: str = "any"
    oracle_obs_structure: bool = False
    allow_unsound_backend: bool = False

    def __post_init__(self):
        # Resolve the scaling prior ONCE, here, so that everything downstream -- the
        # generator, the posterior's prior, and the config written into every results
        # JSON -- sees the same float. Resolving lazily at each use site is how a
        # generator and its prior drift apart, which is the misspecification
        # `ma/topology.py` exists to prevent.
        if self.prior_p is None:
            self.prior_p = connectivity_prior_p(self.topology.d)


@dataclass
class StepResult:
    beliefs: Dict[int, np.ndarray]     # edge marginals per agent
    identified: Dict[int, bool]
    done: bool
    reward: float
    n_interventions: Dict[int, int]
    info: dict = field(default_factory=dict)


class AgentWindow:
    """One agent's view: its columns, its authority, and its DP belief."""

    def __init__(self, agent: int, topology: Topology,
                 modes: Sequence[str] = MODES, backend: str = EXACT,
                 cb_skeleton_alpha: Optional[float] = None,
                 observe_belief_channels: bool = False,
                 observe_owner_channel: bool = False,
                 observe_partner_counts: bool = False,
                 observe_reprobe_signal: bool = False,
                 attribution_local_disturbance: bool = True,
                 mode_by_role: bool = False,
                 skeleton_source: str = "true", skeleton_alpha: float = 0.05,
                 skeleton_max_cond: int = 2,
                 vs_evidence: str = "oracle", vs_evidence_alpha: float = 0.001,
                 vs_evidence_power: float = 1.0, distance_weighted_power: bool = False,
                 cb_n_boot: int = 50, cb_alpha: float = 0.01, cb_n_jobs: int = 1):
        self.agent: int = int(agent)
        self.topology: Topology = topology
        self.modes: Tuple[str, ...] = tuple(modes)
        self.nodes: List[int] = list(topology.observed_by(self.agent))
        self.authority: List[int] = list(topology.may_intervene_on(self.agent))
        self.shared: List[int] = list(topology.exposed)
        self.private: List[int] = [n for n in self.nodes if n not in self.shared]
        self.k = len(self.nodes)
        self.pos = {node: i for i, node in enumerate(self.nodes)}
        self.mode_by_role = bool(mode_by_role)
        if self.mode_by_role:
            # One action per node, its mode fixed by the node's role. See
            # MAConfig.mode_by_role for why this pairing and not the other.
            self.actions: List[Tuple[int, Optional[str]]] = (
                [(node, VARY if node in self.shared else CLAMP)
                 for node in self.authority]
                + [(PASS_ACTION, None)])
            # The GNN wrapper refuses a multi-mode window because it has no mode head.
            # Under this rule there is no mode CHOICE -- the mode is a function of the
            # node -- so the window reports a single effective mode and the wrapper is
            # satisfied without any silent averaging.
            self.modes = ("by_role",)
        else:
            self.actions = ([(node, mode) for node in self.authority for mode in self.modes]
                            + [(PASS_ACTION, None)])
        self.n_actions = len(self.actions)
        self.pass_index = self.n_actions - 1
        self._observe_channels = bool(observe_belief_channels)
        self._observe_owner_channel = bool(observe_owner_channel)
        self._observe_partner_counts = bool(observe_partner_counts)
        self._observe_reprobe_signal = bool(observe_reprobe_signal)
        shared_positions = [self.pos[n] for n in self.shared]
        if backend == CONSTRAINT:
            # base_seed separates the agents' resample streams; deterministic in the agent
            # id so identical seeded episodes reproduce bit-for-bit.
            self.belief = ConstraintBackend(self.k, shared_positions, n_boot=cb_n_boot,
                                            alpha=cb_alpha, n_jobs=cb_n_jobs,
                                            skeleton_alpha=cb_skeleton_alpha,
                                            base_seed=100003 * (self.agent + 1))
        elif backend == VERSION_SPACE:
            from cb.versionspace import VersionSpaceBackend
            self.belief = VersionSpaceBackend(self.k, shared_positions,
                                              evidence=vs_evidence,
                                              evidence_alpha=vs_evidence_alpha)
        elif backend == FACTORED:
            from cb.factored import FactoredBackend
            self.belief = FactoredBackend(self.k, shared_positions,
                                          evidence=vs_evidence,
                                          evidence_alpha=vs_evidence_alpha,
                                          evidence_power=vs_evidence_power,
                                          distance_weighted_power=distance_weighted_power,
                                          # Per agent, so two windows do not miss the same
                                          # questions in lockstep -- that would be a shared
                                          # blind spot rather than independent weak tests.
                                          power_seed=1000 + int(agent))
        elif backend == COMPONENT_ATTRIBUTED:
            from cb.component_attribution import ComponentAttributedBackend
            self.belief = ComponentAttributedBackend(
                self.k, shared_positions, n_agents=topology.n_agents, agent=self.agent,
                evidence=vs_evidence, evidence_alpha=vs_evidence_alpha,
                local_disturbance=attribution_local_disturbance)
        elif backend == FACTORED_ATTRIBUTED:
            from cb.factored_attribution import FactoredAttributedBackend
            self.belief = FactoredAttributedBackend(
                self.k, shared_positions, n_agents=topology.n_agents, agent=self.agent,
                evidence=vs_evidence, evidence_alpha=vs_evidence_alpha,
                local_disturbance=attribution_local_disturbance)
        elif backend == ATTRIBUTED:
            from cb.attribution import AttributedVersionSpaceBackend
            self.belief = AttributedVersionSpaceBackend(
                self.k, shared_positions, n_agents=topology.n_agents, agent=self.agent,
                local_disturbance=attribution_local_disturbance)
        else:
            self.belief = WindowBeliefDP(self.k, shared_positions)

    def action_index(self, node: int, prefer: Optional[str] = None) -> int:
        """Index of the action targeting `node`, preferring mode `prefer` where it is free.

        Exists because `mode_by_role` makes the mode a FUNCTION of the node, so a caller
        that builds the key `(node, VARY)` and looks it up raises -- which is exactly what
        two baselines did. Asking the window instead keeps every caller correct under any
        mode rule, including ones added later.
        """
        for index, (candidate, mode) in enumerate(self.actions):
            if candidate == node and (prefer is None or self.mode_by_role
                                      or mode == prefer):
                return index
        # `prefer` was not available for this node: fall back to whatever is.
        for index, (candidate, _mode) in enumerate(self.actions):
            if candidate == node:
                return index
        raise ValueError(f"node {node} is not in agent {self.agent}'s action space")

    def induced(self, global_adjacency: np.ndarray) -> np.ndarray:
        """The global graph restricted to this window. Well defined precisely because
        cross-private edges are forbidden, so no edge is lost by restriction."""
        return np.asarray(global_adjacency)[np.ix_(self.nodes, self.nodes)]

    @property
    def obs_size(self) -> int:
        n_others = self.topology.n_agents - 1
        return (self.k * (self.k - 1)
                + 1
                + n_others * len(self.shared)
                + 1
                + n_others * len(SIGNALS)
                # Per-node own-intervention counts. The winning behaviour is
                # "touch each node once, private first" -- unlearnable by a policy that
                # cannot see which nodes it already touched. Own history only: nothing
                # crosses the privacy boundary.
                + self.k
                # Bidirected and adjacency upper triangles, when enabled.
                + (self.k * (self.k - 1) if self._observe_channels else 0)
                # Per-pair x per-agent ownership, when enabled. See `_belief_channels`.
                + ((self.k * (self.k - 1) // 2) * self.topology.n_agents
                   if (self._observe_channels and self._observe_owner_channel) else 0)
                # Cumulative partner counts: per other agent, one column per
                # SHARED node plus one for "private, node unspecified". See
                # MAConfig.observe_partner_counts.
                + (n_others * (len(self.shared) + 1)
                   if self._observe_partner_counts else 0)
                # Worth-reprobing signal (1 Sep): one flag per pair, upper triangle. See
                # MAConfig.observe_reprobe_signal.
                + (self.k * (self.k - 1) // 2 if self._observe_reprobe_signal else 0))


class TwoAgentEnv:
    """One SCM, n agents, simultaneous hard interventions."""

    def __init__(self, config: MAConfig, seed: int = 0):
        if config.turn_order not in TURN_ORDERS:
            raise ValueError(f"turn_order must be one of {TURN_ORDERS}")
        if not config.action_modes or any(m not in MODES for m in config.action_modes):
            raise ValueError(f"action_modes must be a non-empty subset of {MODES}")
        # The guard here is deliberately absent; see the note below.
        #
        # This env previously refused any topology hiding more than one node from an agent.
        # The restriction was NEVER about the science -- it was about one representation.
        # `self.clean[agent]` carries a SCALAR fraction per row batch, `n_clamped /
        # len(hidden)`, and `belief_dp._assignment_weights` mixes the clean and dirty score
        # tables with weight `q = 1 - fraction`. With one hidden node the fraction is 0 or
        # 1 and everything is exact. With two, clamping only one gives 0.5, and EVERY
        # confounding hypothesis is scored with that same 0.5 -- the mixture knows how MANY
        # hidden nodes were clamped, never WHICH. So a hypothesis about h1 scores
        # identically whether h1 or h2 was the node actually clamped.
        #
        # THEREFORE, AND THIS IS THE PART THAT MUST NOT BE LOST:
        #
        # The EXACT (Bayesian) belief path is UNSOUND for `widest_hidden > 1`.
        # It is not slow, or approximate. It scores the wrong hypothesis.
        #
        # It is removed here because this worktree is moving to a constraint-based engine,
        # which has no clean/dirty score mixture at all -- independence tests condition on
        # the actual per-row intervention regime, so the abstraction that loses node
        # identity never exists. Lifting the guard is what unblocks rung 1: three agents
        # with one private node each hides two nodes from every agent.
        #
        # The backend boundary: this is the capability check
        # below: the exact backend cannot handle `widest_hidden > 1`, the constraint
        # backend can, and the env asks rather than hard-coding.
        #
        # The original guard, its wording and its three regression tests are preserved on
        # `main` and in every other worktree. Retrieve with:
        # git show main:ma/env.py
        # See docs/ARCHITECTURE.md for how the backends differ.
        if config.belief_backend not in BACKENDS:
            raise ValueError(f"belief_backend must be one of {BACKENDS}")
        if config.episode_mix not in ("any", "confounded", "unconfounded"):
            raise ValueError("episode_mix must be 'any', 'confounded' or 'unconfounded'")
        if config.reward_criterion == "claims" and config.belief_backend not in CLAIM_BACKENDS:
            raise ValueError(
                "reward_criterion='claims' scores bootstrap claim frequencies; the exact "
                "backend has no replicates. Use the constraint backend, or 'u14'.")
        if config.difference_reward_mode not in ("both", "delta", "bonus"):
            raise ValueError(
                "difference_reward_mode must be 'both', 'delta' or 'bonus', got "
                f"{config.difference_reward_mode!r}")
        if config.oracle_obs_structure and config.belief_backend != CONSTRAINT:
            raise ValueError("oracle_obs_structure requires the constraint backend")
        if config.belief_backend in (VERSION_SPACE, FACTORED):
            if config.reward_criterion != "claims":
                raise ValueError("version_space belief only scores the claims criterion")
            # Below 1.0 a MAJORITY of survivors could carry a wrong answer over the bar and
            # settled-wrong would reappear -- the one thing this backend exists to make
            # impossible. Unanimity is what makes "resolved" mean "resolved correctly".
            if config.claim_bar < 1.0:
                raise ValueError(
                    "version_space requires claim_bar=1.0: the truth is always in the "
                    "space, so a claim is settled correctly exactly when every survivor "
                    "agrees. A lower bar re-admits settled-wrong.")
        widest_hidden = max((len(config.topology.hidden_from(a))
                             for a in config.topology.agents), default=0)
        if (config.belief_backend == EXACT and widest_hidden > 1
                and not config.allow_unsound_backend):
            raise ValueError(
                f"the exact backend is UNSOUND for widest_hidden > 1 (here "
                f"{widest_hidden}): the clean-fraction mixture scores the wrong "
                f"hypothesis -- see the note above and "
                f"tests/test_env_turns.py::test_clean_fraction_cannot_say_WHICH_node_was_"
                f"clamped. Use belief_backend='constraint', or set "
                f"allow_unsound_backend=True to demonstrate the defect.")
        self.config = config
        self.topology = config.topology
        self.windows: Dict[int, AgentWindow] = {
            agent: AgentWindow(agent, config.topology, config.action_modes,
                               backend=config.belief_backend,
                               cb_n_boot=config.cb_n_boot, cb_alpha=config.cb_alpha,
                               cb_skeleton_alpha=config.cb_skeleton_alpha,
                               observe_belief_channels=config.observe_belief_channels,
                               observe_owner_channel=config.observe_owner_channel,
                               observe_partner_counts=config.observe_partner_counts,
                               observe_reprobe_signal=config.observe_reprobe_signal,
                               attribution_local_disturbance=config.attribution_local_disturbance,
                               mode_by_role=config.mode_by_role,
                               vs_evidence=config.vs_evidence,
                               vs_evidence_alpha=config.vs_evidence_alpha,
                               vs_evidence_power=config.vs_evidence_power,
                               distance_weighted_power=config.distance_weighted_power,
                               cb_n_jobs=config.cb_n_jobs)
            for agent in self.topology.agents}
        self._rng = np.random.default_rng(seed)
        self.reset(seed)

    @property
    def agents(self) -> Tuple[int, ...]:
        return self.topology.agents

    @property
    def n_agents(self) -> int:
        return self.topology.n_agents

    # -- episode ------------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None,
              adjacency: Optional[np.ndarray] = None) -> StepResult:
        cfg = self.config
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if adjacency is not None:
            # An explicit graph bypasses the mix: the caller has chosen the episode.
            self.true_adjacency = np.asarray(adjacency)
            self.mix_draws = 1
        else:
            self.true_adjacency, self.mix_draws = self._sample_mixed_dag(cfg)
        self.params = sample_scm_params(self.true_adjacency, self._rng)
        self.samples, _ = sample_multi(self.params, cfg.n_obs, self._rng,
                                       noise_dist=cfg.noise_dist,
                                       mechanism=cfg.mechanism)

        self.known: Dict[int, np.ndarray] = {}
        self.clean: Dict[int, np.ndarray] = {}
        # Per row: was ANY node hidden from this agent intervened, in EITHER mode. The
        # constraint backend's foreign-regime mask (see cb/backend.py). Distinct from
        # `clean`, which counts CLAMPS only -- a varied hidden node still drives its
        # children (measured: vary restores 0% of a confounded agent's
        # identification), so it is not clean, but its rows ARE a different regime, and
        # treating them as observational re-creates bug 6 with the mode swapped. Disclosed
        # under the same `disclose_regime` gate as the clean bit -- one bit per round,
        # "something you cannot see was intervened on".
        self.hidden_intervened: Dict[int, np.ndarray] = {}
        self.n_interventions: Dict[int, int] = {}
        self.disclosed: Dict[int, np.ndarray] = {}
        self.regime_bit: Dict[int, float] = {}
        n_others = self.topology.n_agents - 1
        self.own_counts: Dict[int, np.ndarray] = {}
        # [n_others, n_shared + 1] per agent: how many times each PARTNER has intervened on
        # each shared node, and (last column) on a private node of its own. Cumulative over
        # the episode -- the memory the feedforward policy cannot keep for itself.
        self.partner_counts: Dict[int, np.ndarray] = {}
        for agent, window in self.windows.items():
            self.known[agent] = np.zeros((cfg.n_obs, window.k))
            self.clean[agent] = np.zeros(cfg.n_obs, dtype=float)
            self.hidden_intervened[agent] = np.zeros(cfg.n_obs, dtype=bool)
            self.own_counts[agent] = np.zeros(window.k, dtype=float)
            self.partner_counts[agent] = np.zeros((n_others, len(window.shared) + 1),
                                                  dtype=float)
            self.n_interventions[agent] = 0
            self.disclosed[agent] = np.zeros(n_others * len(window.shared))
            self.regime_bit[agent] = 0.0

        self._credit_cache: Dict[int, np.ndarray] = {}
        self._mag_cache: Dict[int, np.ndarray] = {}
        self._skeleton_cache: Dict[int, np.ndarray] = {}
        self._last_claim_fraction: Optional[float] = None
        self._last_agent_fraction: Optional[Dict[int, float]] = None
        self._agent_rewards: Optional[Dict[int, float]] = None
        for window in self.windows.values():
            # Per-episode resample stream: see ConstraintBackend.set_episode.
            if hasattr(window.belief, "set_episode"):
                window.belief.set_episode(seed if seed is not None else 0)
        if cfg.belief_backend == FACTORED:
            for agent, window in self.windows.items():
                window.belief.reset(self._true_mag(agent),
                                    skeleton=self._episode_skeleton(agent))
        elif cfg.belief_backend in (FACTORED_ATTRIBUTED, COMPONENT_ATTRIBUTED):
            # Needs the GLOBAL graph for the same reason ATTRIBUTED does: the true latent
            # groups are a property of the whole system, not of one window. Truth is used
            # only to prune and to score, never in the observation.
            for agent, window in self.windows.items():
                window.belief.reset(self._true_mag(agent), adjacency=self.true_adjacency,
                                    topology=self.topology)
        elif cfg.belief_backend == ATTRIBUTED:
            # Needs the GLOBAL graph, not only the window's MAG: the true latent groups and
            # the response of each to a partner's action are properties of the whole system.
            # Truth is used only to prune and to score -- oracle-side, exactly as the reward
            # is -- and never reaches the observation vector.
            for agent, window in self.windows.items():
                window.belief.reset(self._true_mag(agent), adjacency=self.true_adjacency,
                                    topology=self.topology)
        elif cfg.belief_backend == VERSION_SPACE:
            # The version space is defined relative to THIS episode's truth, so it has to
            # be rebuilt every reset. Truth is used only to prune -- oracle-side, exactly
            # as the reward is -- and never reaches the observation vector.
            for agent, window in self.windows.items():
                window.belief.reset(self._true_mag(agent))
        if cfg.oracle_obs_structure:
            from ma.projection import observational_skeleton
            for agent, window in self.windows.items():
                window.belief.oracle_skeleton = observational_skeleton(
                    self.true_adjacency, tuple(window.nodes))
        # Experiment-block label per row, for stratified bootstrap resampling.
        self.blocks = np.zeros(cfg.n_obs, dtype=int)
        self.round = 0
        self.rounds_used = 0
        self.active: Optional[int] = None
        # Per-agent behaviour, logged separately and never as a max across agents: an idle
        # agent hides inside an average, and free-riding is exactly what we need to see.
        self.forfeits: Dict[int, int] = {a: 0 for a in self.topology.agents}
        # Clamps split by TARGET REGION. Clamping a shared node does nothing for a partner;
        # only clamping one's own private node de-confounds for them. An aggregate clamp
        # fraction cannot tell those apart, so it cannot measure altruism.
        self.clamps_private: Dict[int, int] = {a: 0 for a in self.topology.agents}
        self.clamps_shared: Dict[int, int] = {a: 0 for a in self.topology.agents}
        # Behavioural coordination metric. How many interventions landed on
        # each shared node, by anyone. Duplicate coverage -- two agents spending rounds on
        # the same shared node -- is the failure a coordinating policy avoids, and it is
        # measurable WITHOUT the belief engine, so it transfers across environments where
        # the identification rate does not. That is the point: it separates "the policy
        # stopped coordinating" from "coordination stopped paying".
        self.shared_touches: Dict[int, int] = {n: 0 for n in self.topology.exposed}
        # ROUNDS TO IDENTIFICATION, per agent. `None` until the window is identified; then
        # the round it first happened. Censored at the budget when it never does. A
        # continuous metric with a true zero, against the binary rate's 1-bit-per-episode.
        # WHO touched WHAT, for the difference reward. Ground truth rather than the
        # disclosed table: this is the environment scoring itself, not an agent reasoning,
        # so it may see everything. Nothing reads it unless `difference_reward` is set.
        self._touched_by: Dict[int, set] = {}
        self.identified_round: Dict[int, Optional[int]] = {
            a: None for a in self.topology.agents}
        self.signals: Dict[int, str] = {a: NO_INTERVENTION for a in self.topology.agents}
        self.done_bit: Dict[int, float] = {a: 0.0 for a in self.topology.agents}
        self.connected = _is_connected(self.true_adjacency)
        self.last_chosen: Dict[int, Tuple[int, Optional[str]]] = {
            a: (PASS_ACTION, None) for a in self.topology.agents}
        self._refresh()
        return self._result(reward=0.0)

    def _episode_skeleton(self, agent: int):
        """The skeleton the belief starts from, or None to seed it from the true MAG.

        Estimated from the OBSERVATIONAL rows only -- `self.samples` before any intervention
        has been applied this episode -- so nothing an observational method could not know
        leaks in. Cached per episode because the adjacency search is the expensive half.
        """
        if self.config.skeleton_source != "estimated":
            return None
        cached = self._skeleton_cache.get(agent)
        if cached is None:
            from scripts.skeleton_ablation import estimate_skeleton
            window = self.windows[agent]
            cached = estimate_skeleton(self.samples[:, window.nodes], window.k,
                                       self.config.skeleton_alpha,
                                       self.config.skeleton_max_cond)
            self._skeleton_cache[agent] = cached
        return cached

    def _draw(self, cfg) -> np.ndarray:
        """One graph from the configured generator. One call site for both models."""
        return self.topology.sample_dag(self._rng, p=cfg.prior_p,
                                        model=cfg.graph_model, m=cfg.sf_m)

    def _sample_mixed_dag(self, cfg) -> Tuple[np.ndarray, int]:
        """Draw DAGs until the episode-mix condition holds. Returns (graph, draws).

        "Confounded" is judged by the authoritative criterion: a bidirected pair in some
        agent's true MAG (`projection.bidirected_pairs`), never `common_source_pairs` --
        see the trap note in ma/projection.py. The cap exists so a topology where the
        condition is near-impossible fails loudly instead of looping forever.
        """
        from ma.projection import bidirected_pairs
        if cfg.episode_mix == "any":
            return self._draw(cfg), 1
        for draw in range(1, 201):
            candidate = self._draw(cfg)
            confounded = any(
                bidirected_pairs(candidate, tuple(w.nodes))
                for w in self.windows.values())
            if confounded == (cfg.episode_mix == "confounded"):
                return candidate, draw
        raise RuntimeError(
            f"episode_mix={cfg.episode_mix!r}: no qualifying graph in 200 draws on "
            f"topology {self.topology.name!r} at prior_p={cfg.prior_p:.3f}")

    # -- turn taking --------------------------------------------------------------------

    def active_agent(self) -> Optional[int]:
        """Whose turn it is, or None when all act. Round-robin alternates through agents;
        random draws from the environment's own stream, so the choice is part of the episode
        seed and an evaluation is reproducible without the policy having to record it."""
        order = self.config.turn_order
        if order == SIMULTANEOUS:
            return None
        # The budget is a shared pool of ROUNDS, so there is no per-agent exhaustion to
        # skip over: whoever the rotation names may act. The episode simply stops when the
        # pool runs out.
        if order == ROUND_ROBIN:
            return self.topology.agents[self.round % self.topology.n_agents]
        return int(self._rng.choice(self.topology.agents))

    def step(self, actions: Dict[int, int]) -> StepResult:
        cfg = self.config
        actions = {agent: int(actions[agent]) for agent in self.topology.agents}
        for agent, index in actions.items():
            if not 0 <= index < self.windows[agent].n_actions:
                raise ValueError(f"action {index} out of range for agent {agent}")

        # Under turn-taking the inactive agent is FORCED to pass. Its submitted action is
        # discarded rather than rejected: the policy is queried for all agents every round
        # and the environment, not the policy, owns the protocol.
        self.active = self.active_agent()
        if cfg.turn_order != SIMULTANEOUS:
            for agent in self.topology.agents:
                if agent != self.active:            # active None => everyone passes
                    actions[agent] = self.windows[agent].pass_index
        self.round += 1

        # THERE IS NO VOLUNTARY TERMINATION. Declining is a forfeit: it burns the round and
        # the episode rolls on. With `step_cost` at zero there is nothing to escape by
        # stopping early -- an episode ending with no solution scores 0, while continuing
        # might still score a discounted +1 -- so early exit is dominated rather than
        # tempting. Removing the mechanism also removes the entire class of rule that
        # collapsed 5/10 seeds on 20 August. `passed` survives as a DIAGNOSTIC only.
        passed = all(actions[a] == self.windows[a].pass_index for a in self.topology.agents)
        self.rounds_used += 1

        # What was ACTUALLY applied, after the protocol has had its say. Consumers must
        # tally from this rather than from the submitted actions: under turn-taking the
        # inactive agent still submits a move and it is discarded, so counting submissions
        # double-counts the moves and corrupts any per-move statistic.
        self.last_chosen = {a: self.windows[a].actions[actions[a]] for a in self.topology.agents}
        for _agent, (_node, _mode) in self.last_chosen.items():
            if _node != PASS_ACTION:
                self._touched_by.setdefault(_node, set()).add(_agent)
        self._record_signals()
        self._tally(cfg)

        if passed:
            # A forfeited round still GENERATES DATA. All agents receive it -- samples are
            # shared, so there is no asymmetry -- and because every round produces a batch,
            # total data volume is constant at `n_obs + budget * n_int` instead of varying
            # with how much the policy chose to act. That confound is present in every
            # earlier definition of this number.
            self._append_observational_batch()
            self._refresh()
            return self._result(reward=0.0, passed=True)

        # All act on the SAME system. On a collision the more restrictive assignment wins:
        # a clamp fixes the variable outright, so a simultaneous vary cannot also hold.
        targets: Dict[int, float] = {}
        chosen: Dict[int, Tuple[int, Optional[str]]] = {}
        for agent in self.topology.agents:
            node, mode = self.windows[agent].actions[actions[agent]]
            chosen[agent] = (node, mode)
            if node == PASS_ACTION:
                continue
            scale = 0.0 if mode == CLAMP else cfg.intervene_scale
            targets[node] = min(scale, targets.get(node, np.inf))

        new_samples, _ = sample_multi(self.params, cfg.n_int, self._rng,
                                      intervene_nodes=targets,
                                      noise_dist=cfg.noise_dist,
                                      mechanism=cfg.mechanism)
        self.samples = np.vstack([self.samples, new_samples])
        self.blocks = np.concatenate(
            [self.blocks, np.full(cfg.n_int, self.blocks[-1] + 1, dtype=int)])

        for agent in self.topology.agents:
            window = self.windows[agent]
            block = np.zeros((cfg.n_int, window.k))
            # An agent always knows its OWN intervention.
            own_node, _ = chosen[agent]
            if own_node != PASS_ACTION:
                block[:, window.pos[own_node]] = 1.0
                self.own_counts[agent][window.pos[own_node]] += 1.0
            # And others', but only on SHARED nodes -- those columns are visible to
            # all, so this discloses nothing private.
            for other in self.topology.agents:
                if other == agent:
                    continue
                other_node, _ = chosen[other]
                if other_node != PASS_ACTION and other_node in window.shared:
                    block[:, window.pos[other_node]] = 1.0
            self.known[agent] = np.vstack([self.known[agent], block])

            # A batch is clean (or partially clean) for this agent when variables hidden
            # from it were clamped.
            hidden = self.topology.hidden_from(agent)
            if not hidden:
                clean_fraction = 0.0
            else:
                n_clamped = sum(1 for node in hidden if targets.get(node, None) == 0.0)
                clean_fraction = float(n_clamped / len(hidden))
            self.clean[agent] = np.concatenate(
                [self.clean[agent], np.full(cfg.n_int, clean_fraction, dtype=float)])
            self.hidden_intervened[agent] = np.concatenate(
                [self.hidden_intervened[agent],
                 np.full(cfg.n_int, any(node in targets for node in hidden), dtype=bool)])

            # Concatenate disclosed shared target vectors for all other partners in canonical order.
            disclosed_blocks = []
            for slot, other in enumerate(o for o in self.topology.agents if o != agent):
                other_block = np.zeros(len(window.shared))
                other_node, _ = chosen[other]
                if cfg.disclose_shared_targets and other_node in window.shared:
                    other_block[window.shared.index(other_node)] = 1.0
                disclosed_blocks.append(other_block)
                # The cumulative version of the same two disclosures, and of nothing else:
                # a shared target under `disclose_shared_targets`, a private intervention
                # under `disclose_signals` (which already broadcasts PRIVATE_SIGNAL), with
                # the node deliberately unnamed.
                if other_node == PASS_ACTION:
                    continue
                if other_node in window.shared:
                    if cfg.disclose_shared_targets:
                        self.partner_counts[agent][slot,
                                                   window.shared.index(other_node)] += 1.0
                elif cfg.disclose_signals:
                    self.partner_counts[agent][slot, -1] += 1.0
            self.disclosed[agent] = (np.concatenate(disclosed_blocks)
                                     if disclosed_blocks
                                     else np.zeros(0))
            self.regime_bit[agent] = float(clean_fraction) if cfg.disclose_regime else 0.0

        self._disclose_partner_responses(chosen)
        self._refresh()
        cost = cfg.step_cost * sum(
            1 for a in self.topology.agents if chosen[a][0] != PASS_ACTION)
        return self._result(reward=-cost)

    def _disclose_partner_responses(self, chosen) -> None:
        """Tell each agent WHICH of its confounded pairs moved when a named partner acted.

        This is the only channel through which one agent can help another under the
        deterministic belief. Every other update prunes on the acting agent's own window
        columns, so without it a partner's private experiment is invisible and apparent
        coordination reduces to division of labour over shared nodes.

        WHAT IS AND IS NOT DISCLOSED. The agent learns which partner acted (already
        broadcast by `disclose_signals`) and which of its OWN pairs responded, which it
        could compute for itself from its own columns given infinite data. It never learns
        which node the partner touched. That asymmetry is the whole privacy claim, and it is
        what makes the recovered object an attribution to an AGENT rather than to a variable.
        """
        from cb.attribution import estimated_moved, response_signature
        # GATED ON THE FAMILY, NOT ON ONE MEMBER. This read `!= ATTRIBUTED` until 31 Aug
        # 2026, so an environment running `factored_attributed` -- the backend that exists
        # precisely to carry attribution past k=5 -- received no partner messages at all and
        # its attribution could never be settled by evidence. It went unnoticed because every
        # factored-attribution number to date came from a driver that calls `observe_partner`
        # directly (`tests/crosscheck/`, `scripts/attr_scale.py`), never through the env.
        if (self.config.belief_backend not in ATTRIBUTION_BACKENDS
                or not self.config.disclose_signals):
            return
        sampled = self.config.vs_evidence == "sampled"
        if sampled:
            # The rows this round produced, against the rows in which nothing hidden was
            # touched. Both sides are needed because the evidence is a CHANGE, not a level.
            fresh = np.zeros(len(self.samples), dtype=bool)
            fresh[-self.config.n_int:] = True
        for actor in self.topology.agents:
            node, _mode = chosen[actor]
            if node == PASS_ACTION or node in self.topology.exposed:
                continue                       # only PRIVATE actions carry this signal
            for agent, window in self.windows.items():
                if agent == actor:
                    continue
                groups = window.belief.true_groups
                if not groups:
                    continue
                if sampled:
                    baseline = ~self.hidden_intervened[agent]
                    pairs = sorted({pair for group in groups for pair in group.pairs()})
                    moved = estimated_moved(self.samples[:, window.nodes], fresh,
                                            baseline, pairs,
                                            alpha=self.config.vs_evidence_alpha)
                else:
                    responded = response_signature(self.true_adjacency, self.topology,
                                                   agent, groups, node)
                    moved = frozenset(pair for group, hit in zip(groups, responded) if hit
                                      for pair in group.pairs())
                window.belief.observe_partner(actor, moved)

    def _append_observational_batch(self) -> None:
        """One batch with nothing intervened on -- what a forfeited round produces."""
        cfg = self.config
        new_samples, _ = sample_multi(self.params, cfg.n_int, self._rng,
                                      noise_dist=cfg.noise_dist,
                                      mechanism=cfg.mechanism)
        self.samples = np.vstack([self.samples, new_samples])
        self.blocks = np.concatenate(
            [self.blocks, np.full(cfg.n_int, self.blocks[-1] + 1, dtype=int)])
        n_others = self.topology.n_agents - 1
        for agent, window in self.windows.items():
            self.known[agent] = np.vstack(
                [self.known[agent], np.zeros((cfg.n_int, window.k))])
            # Nothing hidden was clamped, so the batch is DIRTY for a confounded agent.
            self.clean[agent] = np.concatenate(
                [self.clean[agent], np.zeros(cfg.n_int, dtype=float)])
            self.hidden_intervened[agent] = np.concatenate(
                [self.hidden_intervened[agent], np.zeros(cfg.n_int, dtype=bool)])
            self.disclosed[agent] = np.zeros(n_others * len(window.shared))
            self.regime_bit[agent] = 0.0

    def _tally(self, cfg) -> None:
        """Per-agent accounting, from what was APPLIED, in exactly one place.

        A FORFEIT means "I had the move and declined it" -- not "it was not my turn".
        Counting the inactive agent as forfeiting would make every agent forfeit every round
        it did not hold, which measures the protocol rather than the policy.
        """
        for agent in self.topology.agents:
            if cfg.turn_order != SIMULTANEOUS and agent != self.active:
                continue
            node, mode = self.last_chosen[agent]
            if node == PASS_ACTION:
                self.forfeits[agent] += 1
                continue
            self.n_interventions[agent] += 1
            if node in self.shared_touches:
                self.shared_touches[node] += 1
            if mode == CLAMP:
                if node in self.windows[agent].shared:
                    self.clamps_shared[agent] += 1
                else:
                    self.clamps_private[agent] += 1

    def _record_signals(self) -> None:
        """The free broadcast, derived from what was actually applied this round."""
        for agent in self.topology.agents:
            node, _mode = self.last_chosen[agent]
            if node == PASS_ACTION:
                self.signals[agent] = NO_INTERVENTION
            elif node in self.windows[agent].shared:
                self.signals[agent] = SHARED_SIGNAL
            else:
                self.signals[agent] = PRIVATE_SIGNAL

    def _signal_onehot(self, agent: int) -> np.ndarray:
        """The partners' signals, one-hot blocks concatenated in canonical agent order.
        Zeros when disclosure is switched off."""
        blocks = []
        for other in self.topology.agents:
            if other == agent:
                continue
            out = np.zeros(len(SIGNALS))
            if self.config.disclose_signals:
                out[SIGNALS.index(self.signals[other])] = 1.0
            blocks.append(out)
        return np.concatenate(blocks) if blocks else np.zeros(0)

    def _update_done_bits(self) -> None:
        """Each agent's confidence in ITS OWN answer, from ITS OWN posterior.

        Deliberately NOT the credit-set mass. The credit set is defined against the TRUE
        graph, so its mass is an ORACLE quantity -- and since the reward already computes it
        every step, it would be free to hand over, which is precisely what made this an easy
        mistake to make. Free is not the same as legitimate.

        Concentration is measured on the edge marginals: how far the belief sits from
        maximum uncertainty. Cheap, truth-free, and monotone in how settled the posterior is.
        """
        for agent, window in self.windows.items():
            marginals = self.marginals[agent]
            off_diagonal = marginals[~np.eye(window.k, dtype=bool)]
            # Mean distance from 0.5, rescaled to [0, 1]: 0 is a coin flip on every edge.
            self.done_bit[agent] = float(np.mean(np.abs(off_diagonal - 0.5)) * 2.0)

    # -- belief -------------------------------------------------------------------------

    def _refresh(self) -> None:
        """One belief update per agent. `clean` is passed regardless of whether the regime
        bit is DISCLOSED: when it is not, the agent is not told, and the rule reduces to
        scoring everything once. Keeping the mask correct internally means the no-bit arm
        differs from the with-bit arm in exactly one place -- what the agent is told."""
        cfg = self.config
        self.marginals: Dict[int, np.ndarray] = {}
        for agent, window in self.windows.items():
            # THE ONE BRANCH the backend boundary allows itself. Both backends receive
            # "what the agent was told about rows it cannot account for", gated by the
            # same disclosure flag; they differ in WHICH summary is the right one. The
            # exact mixture needs the clamped fraction (`clean`); the constraint engine
            # needs the mode-agnostic regime flag (`hidden_intervened`), because a varied
            # hidden node is not clean but its rows are still foreign (bug 6's second
            # form, which vary-mode can zero for a confounded agent).
            if cfg.belief_backend in CLAIM_BACKENDS:
                told = (self.hidden_intervened[agent] if cfg.disclose_regime
                        else np.zeros(len(self.samples), dtype=bool))
                self.marginals[agent] = window.belief.edge_marginals(
                    self.samples[:, window.nodes], self.known[agent], told,
                    cfg.score_rule, blocks=self.blocks)
            else:
                told = (self.clean[agent] if cfg.disclose_regime
                        else np.zeros(len(self.samples), dtype=bool))
                self.marginals[agent] = window.belief.edge_marginals(
                    self.samples[:, window.nodes], self.known[agent], told,
                    cfg.score_rule)
        self._update_done_bits()

    def _true_mag(self, agent: int) -> np.ndarray:
        """The window's true MAG, in window positions. Ground truth, cached per episode.

        `latent_projection`, not `window.induced`: a hidden chain u -> h -> v projects to
        a DIRECTED edge u -> v the induced subgraph does not carry, and a hidden common
        cause projects to a bidirected edge. The MAG is what a sound engine converges to,
        so it is what identification must be scored against."""
        cached = self._mag_cache.get(agent)
        if cached is None:
            from ma.projection import latent_projection
            cached = latent_projection(self.true_adjacency,
                                       tuple(self.windows[agent].nodes))
            self._mag_cache[agent] = cached
        return cached

    def true_mass(self, agent: int) -> float:
        window = self.windows[agent]
        cfg = self.config
        if cfg.belief_backend in CLAIM_BACKENDS:
            # The strict analogue of "mass on the exact true DAG": every directed MAG edge
            # recovered, confounding exactly right. See cb/backend.py.
            return window.belief.credit_fraction(self._true_mag(agent), strict=True)
        clean = (self.clean[agent] if cfg.disclose_regime
                 else np.zeros(len(self.samples), dtype=bool))
        rule = cfg.score_rule
        if rule == JOINT_CONF:
            # joint_conf has no single log_weights table -- it is a mixture over confounding
            # assignments -- so the true DAG's mass is read from the mixture directly. The
            # TRUE confounded pairs are passed in because identification requires getting
            # the confounding right as well as the causal edges; see the method docstring
            # for the two wrong criteria that preceded this one.
            return float(window.belief.joint_conf_dag_probability(
                self.samples[:, window.nodes], self.known[agent], clean,
                window.induced(self.true_adjacency),
                confounded_pairs=self._confounded_positions(agent)))
        return float(np.exp(window.belief.log_prob_dag(
            self.samples[:, window.nodes], self.known[agent], clean, rule,
            window.induced(self.true_adjacency))))

    def _u14_state(self):
        """Per-agent credit-set mass, and whether the equivalence criterion holds.

        DP-NATIVE. Nothing here enumerates the window. `credit_candidates` enumerates only
        the SHARED subgraph -- 25 DAGs at |X| = 3 against 543 for the window -- because
        criterion 1 pins every private-incident edge to the truth, leaving the shared block
        as the only freedom. `joint_conf_set_probability` then scores those candidates as
        CAUSAL graphs through the subset DP.

        Verified identical to the enumerated `credit_set` on 40 episodes x 2 agents.

        This is the same object `ma/evaluate2.py` reports, so the reward and the reported
        number cannot drift apart -- which is exactly how they drifted apart before.
        """
        if self.config.belief_backend in CLAIM_BACKENDS:
            # Replicate credit fraction with private-incident edges required, mirroring
            # Criterion 1. The union acyclicity and equivalence check does NOT port -- a
            # replicate PAG has no representative DAG -- so the constraint verdict is
            # per-agent credit only. Documented divergence; see cb/backend.py.
            threshold = self.config.identify_threshold
            mass = {}
            for agent in self.topology.agents:
                window = self.windows[agent]
                private_positions = [window.pos[n] for n in window.private]
                mass[agent] = window.belief.credit_fraction(
                    self._true_mag(agent), private_positions)
            all_identified = bool(all(mass[a] >= threshold
                                      for a in self.topology.agents))
            return mass, all_identified

        from ma.evaluate import credit_candidates
        from ma.graphs import is_acyclic, mec_signature

        mass, best_graph = {}, {}
        for agent in self.topology.agents:
            window = self.windows[agent]
            truth = window.induced(self.true_adjacency)
            clean = (self.clean[agent] if self.config.disclose_regime
                     else np.zeros(len(self.samples), dtype=bool))
            # Cached per episode: the true graph is fixed for its whole duration, so the
            # credit set is too, and it was being rebuilt at every step.
            cached = self._credit_cache.get(agent)
            if cached is None:
                cached = credit_candidates(window, truth)
                self._credit_cache[agent] = cached
            candidates = cached
            pairs = self._confounded_positions(agent)
            mass[agent] = float(window.belief.joint_conf_set_probability(
                self.samples[:, window.nodes], self.known[agent], clean,
                candidates, pairs))
            # Representative for the union check. Only consulted when the agent is
            # credited, and every credited answer is Markov equivalent to the truth with
            # its private edges exact, so any member is a valid stand-in.
            best_graph[agent] = candidates[0] if len(candidates) else truth

        threshold = self.config.identify_threshold
        d = self.topology.d
        union = np.zeros((d, d), dtype=np.int8)
        for agent in self.topology.agents:
            window = self.windows[agent]
            graph = np.asarray(best_graph[agent])
            for i, u in enumerate(window.nodes):
                for j, v in enumerate(window.nodes):
                    if graph[i, j]:
                        union[u, v] = 1
        all_identified = bool(all(mass[a] >= threshold for a in self.topology.agents)
                              and is_acyclic(union)
                              and mec_signature(union) == mec_signature(
                                  np.asarray(self.true_adjacency)))
        return mass, all_identified

    def _confounded_positions(self, agent: int):
        """Truly confounded shared pairs, as WINDOW positions.

        Read from the generating graph via the latent projection, so it is ground truth and
        never visible to the agent -- it is used only to score identification."""
        from ma.projection import bidirected_pairs
        window = self.windows[agent]
        pairs = bidirected_pairs(self.true_adjacency, tuple(window.nodes))
        return tuple((window.pos[u], window.pos[v]) for u, v in pairs)

    # -- observation and result ---------------------------------------------------------

    def observation(self, agent: int) -> np.ndarray:
        """Edge marginals, remaining budget, and whatever was disclosed.

        Every feature is on [0, 1]. Raw counts were a real bug once: the budget feature sat
        at 20.0 beside probabilities in [0, 1] and dominated the first layer.
        """
        window = self.windows[agent]
        marginals = self.marginals[agent]
        off_diagonal = ~np.eye(window.k, dtype=bool)
        # ROUNDS left in the shared pool -- the same number for all agents now, because
        # the budget is shared across agents rather than allocated per agent.
        budget_left = np.array(
            [(self.config.budget - self.rounds_used) / max(self.config.budget, 1)])
        return np.concatenate([marginals[off_diagonal], budget_left,
                               self.disclosed[agent],
                               np.array([self.regime_bit[agent]]),
                               self._signal_onehot(agent),
                               # Own per-node intervention counts, budget-normalised so
                               # the feature stays on [0, 1] (raw counts once dominated
                               # a first layer -- see the docstring above).
                               self.own_counts[agent] / max(self.config.budget, 1),
                               # Confounding and adjacency beliefs, upper triangles. Both
                               # already live on [0, 1]. See `observe_belief_channels`.
                               self._belief_channels(agent),
                               # Cumulative partner counts, same normalisation as own
                               # counts. See `observe_partner_counts`.
                               self._partner_counts(agent),
                               # Worth-reprobing signal. See `observe_reprobe_signal`.
                               self._reprobe_signal(agent)])

    def _partner_counts(self, agent: int) -> np.ndarray:
        """Cumulative per-partner counts, flattened, or an empty array.

        Budget-normalised for the same reason the own-counts are: a raw count next to
        probabilities in [0, 1] once dominated the first layer.
        """
        if not self.config.observe_partner_counts:
            return np.zeros(0)
        return (self.partner_counts[agent] / max(self.config.budget, 1)).reshape(-1)

    def _reprobe_signal(self, agent: int) -> np.ndarray:
        """1.0 per pair that is still UNRESOLVED and has an endpoint already probed this
        episode, else 0.0. Upper triangle, or an empty array. See
        `MAConfig.observe_reprobe_signal`.

        "Unresolved" is read off the marginals already in the observation (directed both
        ways, plus bidirected), generically across belief backends: a settled pair has
        exactly one of the three at 1.0 and the rest at 0.0, an unresolved one carries a
        fractional value on at least one. Deliberately NOT read from `belief.last.possible`
        (factored-only) so this works with any backend that reports the standard marginal
        fields `FactoredBelief`'s own docstring says every belief carries.
        """
        window = self.windows[agent]
        if not self.config.observe_reprobe_signal:
            return np.zeros(0)
        belief = window.belief.last
        if belief is None:
            return np.zeros(window.k * (window.k - 1) // 2)
        rows, cols = np.triu_indices(window.k, k=1)
        directed = np.asarray(belief.directed)
        bidirected = np.asarray(belief.bidirected)
        is_settled = lambda x: np.isclose(x, 0.0) | np.isclose(x, 1.0)
        settled = (is_settled(directed[rows, cols]) & is_settled(directed[cols, rows])
                  & is_settled(bidirected[rows, cols]))
        counts = self.own_counts[agent]
        probed = (counts[rows] > 0) | (counts[cols] > 0)
        return (~settled & probed).astype(float)

    def _belief_channels(self, agent: int) -> np.ndarray:
        """Bidirected and adjacency frequencies, upper triangle, or an empty array.

        The claims criterion scores three channels; the observation carried one. An agent
        that cannot see what it believes about confounding cannot act on it, while the
        greedy baseline reads all three -- which made every learned-vs-greedy number a
        comparison between a blindfolded learner and a sighted rule.
        """
        window = self.windows[agent]
        if not self.config.observe_belief_channels:
            return np.zeros(0)
        n_owner = ((window.k * (window.k - 1) // 2) * self.topology.n_agents
                   if self.config.observe_owner_channel else 0)
        belief = window.belief.last
        if belief is None:
            return np.zeros(window.k * (window.k - 1) + n_owner)
        rows, cols = np.triu_indices(window.k, k=1)
        channels = [np.asarray(belief.bidirected)[rows, cols],
                    np.asarray(belief.adjacency)[rows, cols]]
        if self.config.observe_owner_channel:
            # WHO the agent blames for each confounded pair, not merely THAT it is
            # confounded. `AttributedBelief.owner_channel` has existed since the attributed
            # backend landed, its docstring says it "replaces the single bidirected channel
            # in the observation" -- and nothing ever called it. So every attributed run
            # trained a policy that could see "this pair is confounded" and never "and I
            # blame agent 2", which is the same blindfold this method's own docstring
            # records fixing one level down for confounding.
            #
            # It carries no foreign node identities: only an agent index and a pair of THIS
            # window's positions, so it discloses nothing beyond the ownership this agent
            # has already inferred for itself.
            #
            # NOT PORTABLE, and deliberately so. The width is per-pair x per-agent, and
            # `PortableRoleActorCritic` pools partner blocks to be permutation-INVARIANT --
            # which is the right symmetry for generic behaviour and destroys exactly the
            # identity attribution needs. Its own docstring says per-partner attribution "is
            # not expressible by this variant". Use `policy_arch="gnn"` when this is on.
            owner = getattr(belief, "owner_channel", None)
            channels.append(owner(self.topology.n_agents).reshape(-1) if owner is not None
                            else np.zeros(n_owner))
        return np.concatenate(channels)

    def _result(self, reward: float, passed: bool = False) -> StepResult:
        threshold = self.config.identify_threshold
        claim_info = None
        if self.config.reward_criterion == "claims":
            from cb.claims import score_window
            cfg = self.config
            scores = {}
            attributed = cfg.belief_backend in ATTRIBUTION_BACKENDS
            attribution = {}
            for agent, window in self.windows.items():
                scores[agent] = score_window(
                    window.belief.last, self._true_mag(agent),
                    [window.pos[n] for n in window.private], bar=cfg.claim_bar,
                    require_all_types=cfg.claims_require_all_types,
                    # Under the attributed backend the confounding type claim is REPLACED
                    # by the attribution claim, never scored alongside it -- scoring both
                    # would let an agent bank "something confounds these" while failing the
                    # half that carries the thesis, which is whose it is.
                    confounding_claims=not attributed)
                if attributed:
                    from cb.attribution import score_groups
                    attribution[agent] = score_groups(
                        window.belief.last, window.belief.true_groups, bar=cfg.claim_bar)
            identified = {a: scores[a].identified
                          and (not attributed or attribution[a]["identified"])
                          for a in self.topology.agents}
            all_identified = all(identified.values())
            mass = {}
            for a in self.topology.agents:
                base, n = scores[a].n_right - cfg.claim_penalty * scores[a].n_wrong, scores[a].n_claims
                if attributed:
                    # Attribution claims join the SAME pool as the structure claims, so the
                    # dense reward pays for progress on either and the two cannot be traded
                    # against each other by weighting.
                    row = attribution[a]
                    base += row["right"] - cfg.claim_penalty * row["wrong"]
                    n += row["total"]
                mass[a] = (base / n) if n else 0.0
            mean_fraction = float(np.mean(list(mass.values())))
            # Dense component: what THIS round settled, net of what it unsettled. At
            # reset there is no previous score, so the first delta is zero by definition.
            if self._last_claim_fraction is not None:
                reward += mean_fraction - self._last_claim_fraction
            self._last_claim_fraction = mean_fraction
            if cfg.per_agent_reward:
                # Each agent paid for ITS OWN window: its own dense delta, and its own
                # terminal +1. The shared alternative makes agent i's gradient depend on
                # agent j's luck, which is credit-assignment noise the policy has to
                # average out -- and it compresses the metric exponentially in the number
                # of agents, so every further rung looks worse however well agents learn.
                self._agent_rewards = {}
                for a in self.topology.agents:
                    own = mass[a]
                    previous = (self._last_agent_fraction.get(a)
                                if self._last_agent_fraction is not None else None)
                    delta = 0.0 if previous is None else own - previous
                    mode = self.config.difference_reward_mode
                    if self.config.difference_reward:
                        # Two changes, both aimed at the same fault. The DELTA is paid only
                        # to whoever actually moved this round: under turn-taking a partner's
                        # intervention on a shared node raises this agent's window credit
                        # too, and paying for that is the dense half of the free-riding.
                        # The OUTCOME bonus, which is the larger half, is replaced by the
                        # agent's own causal contribution -- a level, exactly as the +1 was.
                        moved = self.active is None or a == self.active
                        # The delta half: a partner's move on a shared node raises this
                        # agent's window credit too, so paying for it is the dense share of
                        # the free-riding. Gated unless this arm keeps the plain delta.
                        paid_delta = delta if (moved or mode == "bonus") else 0.0
                        # The bonus half: contribution rather than outcome.
                        bonus = (self.difference_credit(a) if mode in ("both", "bonus")
                                 else (1.0 if identified[a] else 0.0))
                        self._agent_rewards[a] = paid_delta + bonus
                    else:
                        self._agent_rewards[a] = delta + (1.0 if identified[a] else 0.0)
                    self._agent_rewards[a] *= self.config.reward_scale
                self._last_agent_fraction = dict(mass)
            claim_info = {a: {"right": s.n_right, "wrong": s.n_wrong,
                              "unsure": s.n_unsure,
                              "required_right": s.required_right,
                              "required_total": s.required_total,
                              **({"attr_right": attribution[a]["right"],
                                  "attr_total": attribution[a]["total"],
                                  "attr_unsure": attribution[a]["unsure"]}
                                 if attributed else {})}
                          for a, s in scores.items()}
        elif self.config.reward_criterion == "u14":
            mass, all_identified = self._u14_state()
            identified = {a: mass[a] >= threshold for a in self.topology.agents}
        else:
            mass = {a: self.true_mass(a) for a in self.topology.agents}
            identified = {a: mass[a] >= threshold for a in self.topology.agents}
            all_identified = all(identified.values())
        # First round at which each window became identified. Latched: a window that comes
        # undone later keeps the round it was first settled, because the metric being
        # reported is "how many experiments did it take", not "was it still true at the end"
        # -- the identification rate already answers the second.
        for a in self.topology.agents:
            if self.identified_round[a] is None and identified[a]:
                self.identified_round[a] = self.rounds_used
        # The SHARED pool is what ends an episode, together with joint success. Declining
        # never ends it -- see docs/ARCHITECTURE.md section 4.
        out_of_budget = self.rounds_used >= self.config.budget
        if all_identified:
            reward += 1.0                       # shared terminal reward
        return StepResult(
            beliefs={a: self.marginals[a].copy() for a in self.topology.agents},
            identified=identified,
            # `passed` is DELIBERATELY absent: declining never terminates an episode.
            # It remains in `info` as a diagnostic only. See TURN_BUDGET_SPEC section 4.
            done=all_identified or out_of_budget,
            reward=reward,
            n_interventions=dict(self.n_interventions),
            info={"true_mass": mass, "both_identified": all_identified, "passed": passed,
                  "agent_rewards": self._agent_rewards,
                  "identified_fraction": float(np.mean([float(v) for v in identified.values()])),
                  "claims": claim_info,
                  # Per agent, never a max across agents: an idle agent hides inside an
                  # average, and free-riding is the thing we most need to see.
                  "interventions": dict(self.n_interventions),
                  "forfeits": dict(self.forfeits),
                  "clamps_private": dict(self.clamps_private),
                  "clamps_shared": dict(self.clamps_shared),
                  "signals": dict(self.signals),
                  "done_bit": dict(self.done_bit),
                  # See the notes on these two in `reset`.
                  "shared_touches": dict(self.shared_touches),
                  "duplicate_coverage": self.duplicate_coverage(),
                  # Read duplicate_coverage AGAINST this: past `len(exposed)` shared
                  # interventions, duplication is forced rather than chosen.
                  "duplicate_coverage_floor": self.duplicate_coverage_floor(),
                  "identified_round": dict(self.identified_round),
                  "connected": bool(self.connected),
                  "mix_draws": self.mix_draws,
                  "rounds_used": self.rounds_used,
                  # ROUNDS, not per-agent interventions. Under turn-taking an agent acts
                  # every other round, so `n_interventions` is roughly half the episode
                  # length -- reporting one as the other understates duration by 2x.
                  "rounds": self.round, "active": self.active,
                  "budget_left": {a: self.config.budget - self.n_interventions[a]
                                  for a in self.topology.agents}},
        )

    # -- behavioural metrics --------------------------------------------------------------

    def difference_credit(self, agent: int) -> float:
        """Credit this agent's OWN interventions caused, as a difference reward.

        `credit(what happened) - credit(what would have happened had this agent passed all
        episode)`. A node a PARTNER also reached stays in the counterfactual set: had this
        agent stayed home the partner would still have reached it, so it is nobody's marginal
        contribution and crediting it is precisely the free-riding the plain reward suffers.

        Exact, not estimated, because under oracle evidence the belief is a deterministic
        function of the intervened set. Factored backend only -- `credit_for_set` replays
        `_apply_ancestry`, which no other backend exposes.
        """
        from cb.factored import credit_for_set

        window = self.windows[agent]
        in_window = set(window.nodes)
        realised, own_only = set(), set()
        for node, actors in self._touched_by.items():
            if node not in in_window:
                continue
            realised.add(window.pos[node])
            if actors == {agent}:
                own_only.add(window.pos[node])
        if not own_only:
            return 0.0
        mag = self._true_mag(agent)
        return (credit_for_set(mag, window.k, realised)
                - credit_for_set(mag, window.k, realised - own_only))

    def duplicate_coverage(self) -> float:
        """Fraction of shared-node interventions that landed on an already-covered node.

        `(spent - distinct) / spent` over the episode so far, on shared nodes only: with
        `spent` interventions covering `distinct` nodes, every intervention past the first
        on a node is a round that bought nothing a partner had not already bought. Zero when
        agents divide the shared surface perfectly; 1 - 1/spent when they all pile onto one
        node. Undefined (0.0) before any shared intervention.

        DELIBERATELY BELIEF-FREE. It reads only what agents DID, so the same number means
        the same thing in the deterministic and the statistical environments -- which is
        what makes it usable as a transfer diagnostic when the identification rate is not
        comparable across them. It is a NECESSARY condition for coordination, not a
        sufficient one: a policy can divide the shared nodes perfectly and still choose the
        wrong ones.
        """
        spent = sum(self.shared_touches.values())
        if spent == 0:
            return 0.0
        distinct = sum(1 for count in self.shared_touches.values() if count > 0)
        return float((spent - distinct) / spent)

    def duplicate_coverage_floor(self) -> float:
        """The SMALLEST duplicate coverage this episode could have had, given the spend.

        WHY THIS IS NEEDED TO READ THE NUMBER ABOVE. There are only `len(exposed)` shared
        nodes, so once an arm has spent more shared interventions than there are shared
        nodes, duplication is FORCED: distinct can never exceed the surface. An arm that
        spends more on the shared surface therefore duplicates more no matter how well it
        coordinates, and comparing raw duplicate coverage across arms that spend differently
        credits the one that spent less. Measured: greedy puts 44-71% of its moves
        on the shared surface against the learned policy's 35-59%, so the two are NOT on the
        same footing (`the pre-squash-submission tag` section 3b).

        Report `duplicate_coverage - duplicate_coverage_floor`, or the two side by side.
        Zero floor means the surface was never saturated and the raw number stands alone.
        """
        spent = sum(self.shared_touches.values())
        if spent == 0:
            return 0.0
        surface = len(self.topology.exposed)
        return float(max(0.0, (spent - surface) / spent))

    def rounds_to_identification(self, censor: Optional[int] = None) -> Dict[int, int]:
        """Per agent, the round its window was first identified; `censor` if it never was.

        Right-censored rather than dropped: excluding the failures would report the mean
        over the episodes a policy happened to solve, which rewards a policy that solves
        few and easy ones. `censor` defaults to `budget + 1` -- one worse than using the
        entire budget, which is the honest ordering ("did not finish" is worse than
        "finished on the last round") without pretending to know how much worse.
        """
        limit = self.config.budget + 1 if censor is None else int(censor)
        return {a: (limit if self.identified_round[a] is None else self.identified_round[a])
                for a in self.topology.agents}

    # -- convenience --------------------------------------------------------------------

    def n_actions(self, agent: int) -> int:
        return self.windows[agent].n_actions

    def obs_size(self, agent: int) -> int:
        return self.windows[agent].obs_size

