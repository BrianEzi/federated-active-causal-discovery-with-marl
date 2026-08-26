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
BACKENDS = (EXACT, CONSTRAINT, VERSION_SPACE)
# Backends whose belief exposes bootstrap-shaped claim frequencies, so `cb.claims` and the
# constraint-side greedy read them the same way.
CLAIM_BACKENDS = (CONSTRAINT, VERSION_SPACE)
VARY = "vary"
CLAMP = "clamp"
MODES = (VARY, CLAMP)

# Turn protocols. SIMULTANEOUS is the original and every result before 2026-08-20 was
# measured under it; it is kept so those numbers stay reproducible, not because it is
# preferred. Under the two turn-taking protocols exactly one agent may act per round and
# the other is forced to pass.
SIMULTANEOUS = "simultaneous"
ROUND_ROBIN = "round_robin"
RANDOM_TURN = "random"
TURN_ORDERS = (SIMULTANEOUS, ROUND_ROBIN, RANDOM_TURN)

# The broadcast signal, one categorical per agent per round, free of charge. It names a
# REGION, never a variable, and carries no value -- see docs/TURN_BUDGET_SPEC.md section 6.
# Advisory: nothing forces an agent to respect it. PROVISIONAL on the supervisor confirming
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
    topology: Topology
    n_obs: int = 1000
    n_int: int = 100
    # TOTAL ROUNDS FOR THE SYSTEM -- a SHARED POOL, not a per-agent allowance. Every round
    # consumes one unit whether the active agent intervenes or declines, which is what makes
    # free-riding cost something: a round A wastes is a round B does not get, and the reward
    # is shared. Under round-robin this is exactly equivalent to a per-agent budget of
    # `budget / n_agents`; the two diverge only under random turn order.
    # NOTE the semantic change from the pre-2026-08-21 meaning ("interventions per agent").
    budget: int = 10
    # One agent acts per round, or both. STALE COMMENT REMOVED 2026-08-22: this block used
    # to say the budget stays PER AGENT under turn-taking. It has been a shared pool since
    # the turn-budget spec (see `budget` above), and the two statements sat three lines
    # apart contradicting each other.
    turn_order: str = SIMULTANEOUS
    # CLAMP ONLY, adopted as the default on 2026-08-22. This is a TRADE WITH A KNOWN PRICE.
    # At ten seeds the cost looked like nothing measurable (+0.018, CI [-0.005, +0.041]).
    # At TWENTY it is +0.021, CI [+0.001, +0.042] -- significant, if barely. So the price is
    # about 2pp, and it buys a halved action space and one fewer axis to sweep as agents are
    # added. Both-modes leads on only 11 of 20 seeds, so this is a small consistent effect
    # rather than a large unreliable one.
    # The coordination problem is untouched -- it becomes one of targeting and timing
    # rather than of mode -- and a policy given both modes converges on clamp anyway
    # (81-91% of clamps on its own private node).
    # Pass `action_modes=MODES` to restore both. `tb_both` remains a live arm.
    action_modes: Tuple[str, ...] = (CLAMP,)
    # MODE BY ROLE (2026-08-26): clamp on your own private nodes, vary on shared ones.
    # Overrides `action_modes` entirely -- each node gets exactly ONE action, so the action
    # space stays the size of the vary-only or clamp-only arms rather than doubling.
    #
    # THE REASONING, WHICH THE MEASUREMENT REFUTED. Identifiability depends on intervention
    # TARGETS, not on the values assigned (Hauser & Buhlmann, JMLR 13, 2012 --
    # BIBLIOGRAPHY.md §19), so in the infinite-data limit clamp and vary are equally
    # powerful and this flag can only be a FINITE-SAMPLE, role-dependent choice. The
    # argument was that the two roles want opposite things: keep variance in a node you are
    # orienting for yourself, and remove it from a node that confounds your PARTNER, since
    # clamping a hidden common cause makes the confounded association vanish for them.
    #
    # MEASURED (statistical backend, 3 agents, budget 6, 40 episodes, greedy):
    #     vary only                    identified 0.308   confounding claims right 0.373
    #     clamp only                   identified 0.333   confounding claims right 0.355
    #     clamp private / vary shared  identified 0.250   confounding claims right 0.309
    # The mixed rule is WORST, and worst on the confounding claims it was meant to help
    # most. The likely reason is a cost the argument left out: clamping your own private
    # node destroys YOUR orientation power on the edges incident to it, and those are
    # required claims for you. The altruistic move is not free, and per-window scoring
    # charges you for it.
    #
    # KEPT, not adopted. The flag stays because the finding is worth reporting and because
    # a reward that actually paid for partner outcomes might reverse it -- but it is off by
    # default, and anyone turning it on should expect it to cost identification.
    mode_by_role: bool = False
    identify_threshold: float = 0.7
    # None means "scale with d" -- `2 ln(d)/d`, the FULL-CONNECTIVITY threshold with an
    # empirical factor of two. See `ma.priors.connectivity_prior_p` for the measurement and
    # for why the percolation threshold `1/d` is the wrong target here. A float overrides.
    # NOTE this CHANGES the graph distribution: at d=5 it is 0.644, not 0.5.
    prior_p: Optional[float] = None
    # HOW THE GENERATING GRAPH IS DRAWN. "er" is the historical default; "sf" is scale-free
    # by preferential attachment, where `sf_m` sets the parents each node takes and thus the
    # density (`prior_p` is ignored under "sf"). See `Topology.sample_dag`: the ER
    # assumption was inherited from a Bayesian prior that had to match the generator, and
    # no engine now in use reads `prior_p`, so this is free to vary.
    graph_model: str = ER
    sf_m: int = 2
    intervene_scale: float = 2.0       # VARY draws N(0, scale^2); CLAMP always uses 0.0
    score_rule: str = JOINT_CONF
    # One bit per round: "I clamped something you cannot see". OFF by default -- the no-bit
    # arm is the baseline. See the module docstring.
    disclose_regime: bool = False
    # Which SHARED nodes the other agent targeted. Shared columns are visible to both, so
    # this reveals nothing private. Delivered AFTER acting, so it can only condition
    # future moves [U10].
    disclose_shared_targets: bool = True
    # The three-category action-type broadcast. Provisional on the supervisor; one flag.
    disclose_signals: bool = True
    # ZERO, and load-bearing. Measured at 0.05: over ~7.7 steps a random-level policy has
    # expected value -0.255 against 0.000 for passing, so PASSING WAS OPTIMAL and every
    # recorded "collapse" was the agent being correct. Efficiency pressure now comes from
    # the finite round budget and from gamma discounting instead.
    # DO NOT re-introduce a step cost without also re-introducing a termination mechanism:
    # the two are coupled, and changing one alone re-opens the collapse. See
    # docs/TURN_BUDGET_SPEC.md section 5.
    step_cost: float = 0.0
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
    #   "claims"     (2026-08-24, constraint backend only) three-outcome claim scoring
    #                (cb/claims.py): dense reward = per-step change in
    #                (right - penalty*wrong)/total over the window's claims, terminal +1
    #                when every agent has all REQUIRED claims settled right and NOTHING
    #                settled wrong. Replaces the all-or-nothing conjunction that turned
    #                95% per-claim accuracy into 36% success and a luck-dominated
    #                training signal.
    reward_criterion: str = "u14"
    claim_bar: float = 0.7
    # Pay each agent for its OWN window instead of the all-agents conjunction. Off by
    # default so every number measured before 2026-08-26 stays reproducible; the shared
    # reward remains what `both_identified` reports either way.
    per_agent_reward: bool = False
    # SHOW THE POLICY THE CHANNELS ITS REWARD IS SCORED ON. The observation carried only
    # DIRECTED edge frequencies, so an agent could not see which pairs it believed
    # CONFOUNDED -- the one claim the thesis is about, always required, and the largest
    # remaining error category. The greedy baseline reads all three channels through
    # cb.claims, so every learned-vs-greedy comparison to date handicapped the learner.
    # Found 2026-08-26 while diagnosing learned < greedy in the deterministic environment.
    # Off by default because it changes obs_size and voids old checkpoints.
    observe_belief_channels: bool = False
    # CUMULATIVE PARTNER-INTERVENTION COUNTS (2026-08-26). Per (other agent x shared node),
    # budget-normalised, plus one column per partner counting its PRIVATE interventions
    # without saying which node.
    #
    # This discloses nothing new. Every shared target is already disclosed in the round it
    # happens (`disclose_shared_targets`) and every private intervention already raises the
    # PRIVATE_SIGNAL bit in the round it happens (`disclose_signals`). The policy is
    # FEEDFORWARD, so a per-round disclosure it cannot retain is information the
    # environment hands over and the agent structurally loses. This converts disclosure
    # into MEMORY -- the same class of fix as the own-intervention counts, which was the
    # last thing to unlock a result.
    #
    # The private column is also the handle for the clique case study: an agent that can
    # count HOW MANY private interventions each partner has made can, in principle, tell
    # which partner's hidden variables its own window responds to. Weakly identifiable at
    # best, and named as further work rather than claimed.
    # Off by default because it changes obs_size and voids old checkpoints.
    observe_partner_counts: bool = False
    # Grade every type claim, not only the private-incident ones. True since 2026-08-26 --
    # the old exemption rested on a false claim about Markov equivalence. See cb/claims.py.
    claims_require_all_types: bool = True
    claim_penalty: float = 1.0         # settled-wrong weight in the dense reward
    # WHICH ENGINE HOLDS THE BELIEF. "exact" is the Bayesian subset DP
    # (`crosscheck/belief_dp.py`); "constraint" is the bootstrap PC/FCI engine (`cb/`).
    # The arms differ in exactly this one flag. Under "constraint", identification is the
    # fraction of bootstrap replicates credited against the window's true MAG -- there is
    # no posterior mass any more; see `cb/backend.py` for the criterion and for the one
    # documented divergence (no cross-agent union check).
    belief_backend: str = EXACT
    cb_n_boot: int = 50            # bootstrap replicates per refresh; B is the speed knob
    cb_alpha: float = 0.01
    # Separate significance threshold for SKELETON search only. None => same as cb_alpha.
    # The alpha sweep (2026-08-25) showed one shared threshold cannot serve both uses:
    # loosening recovers missed edges but doubles orientation errors.
    cb_skeleton_alpha: Optional[float] = None         # CI-test level. SWEPT 2026-08-24 (0.05 halves credit
                                   # via noisier skeletons) and FIXED. Do not revisit
                                   # against results.
    # Bootstrap replicates CAN run on a process pool (bit-identical to serial by
    # construction and pinned by test) -- but at k=4 a replicate is ~4 ms of work and
    # process dispatch LOSES (measured 2026-08-25: 4-way was 2x slower than serial).
    # Default stays serial; raise this on the scale ladder (k=7-9), where a replicate
    # is ~25 ms and the pool pays.
    cb_n_jobs: int = 1
    # WHICH NETWORK THE POLICY USES. "mlp" is the flat ActorCritic behind every banked
    # number; "gnn" is the role-aware per-node wrapper (ma/policy.py) around
    # ma/nets.PerNodeActorCritic. The student wants results from the GNN; the MLP is the
    # attribution arm. On MAConfig rather than PPOConfig because a checkpoint must be able
    # to say what environment/architecture pair it belongs to.
    policy_arch: str = "mlp"
    # WHICH EPISODES TO GENERATE (2026-08-24). "confounded": some agent's window carries
    # a bidirected pair in its true MAG -- the episodes the thesis is about, and only 15%
    # of unconstrained draws. "unconfounded": none does -- the SANITY arm, where zero
    # settled-wrong confounding claims is a pinned requirement, not a hope. "any": the
    # historical behaviour. Rejection sampling with a draw cap; acceptance rate recorded.
    episode_mix: str = "any"
    # ORACLE WARM START (2026-08-25, student-approved): hand each agent the TRUE
    # infinite-observational-data structure of its window -- adjacency and separating
    # sets from `ma.projection.observational_skeleton` -- so the whole task is the part
    # observation cannot do: orienting by experiment and detecting confounders. Nothing
    # interventional leaks: a confounded pair starts adjacent and UNEXPLAINED, exactly as
    # infinite observational data would leave it. Constraint backend only. Keep at least
    # one estimated-skeleton arm in any reported comparison, or the claim silently
    # becomes "given perfect observational preprocessing".
    oracle_obs_structure: bool = False
    # The exact backend is UNSOUND for `widest_hidden > 1` -- it scores the wrong
    # hypothesis (see the long note in `TwoAgentEnv.__init__`). The env refuses that
    # combination unless this is set, which exists for demonstrations of the defect
    # (`tests/test_env_turns.py`), never for producing numbers.
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
                 observe_partner_counts: bool = False,
                 mode_by_role: bool = False,
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
        self._observe_partner_counts = bool(observe_partner_counts)
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
            self.belief = VersionSpaceBackend(self.k, shared_positions)
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
                # Per-node OWN-intervention counts (2026-08-25). The winning behaviour is
                # "touch each node once, private first" -- unlearnable by a policy that
                # cannot see which nodes it already touched. Own history only: nothing
                # crosses the privacy boundary.
                + self.k
                # Bidirected + adjacency upper triangles (2026-08-26), when enabled.
                + (self.k * (self.k - 1) if self._observe_channels else 0)
                # Cumulative partner counts (2026-08-26): per other agent, one column per
                # SHARED node plus one for "private, node unspecified". See
                # MAConfig.observe_partner_counts.
                + (n_others * (len(self.shared) + 1)
                   if self._observe_partner_counts else 0))


class TwoAgentEnv:
    """One SCM, n agents, simultaneous hard interventions."""

    def __init__(self, config: MAConfig, seed: int = 0):
        if config.turn_order not in TURN_ORDERS:
            raise ValueError(f"turn_order must be one of {TURN_ORDERS}")
        if not config.action_modes or any(m not in MODES for m in config.action_modes):
            raise ValueError(f"action_modes must be a non-empty subset of {MODES}")
        # GUARD REMOVED 2026-08-23, deliberately, by the student's instruction.
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
        #   The EXACT (Bayesian) belief path is UNSOUND for `widest_hidden > 1`.
        #   It is not slow, or approximate. It scores the wrong hypothesis.
        #
        # It is removed here because this worktree is moving to a constraint-based engine,
        # which has no clean/dirty score mixture at all -- independence tests condition on
        # the actual per-row intervention regime, so the abstraction that loses node
        # identity never exists. Lifting the guard is what unblocks rung 1: three agents
        # with one private node each hides two nodes from every agent.
        #
        # THE BACKEND BOUNDARY LANDED 2026-08-24 and this became the capability check
        # below: the exact backend cannot handle `widest_hidden > 1`, the constraint
        # backend can, and the env asks rather than hard-coding.
        #
        # The original guard, its wording and its three regression tests are preserved on
        # `main` and in every other worktree. Retrieve with:
        #     git show main:ma/env.py
        # See docs/STRIP_SCOPE.md section 1, and docs/logs/MA_BUILD_LOG.md 2026-08-22.
        if config.belief_backend not in BACKENDS:
            raise ValueError(f"belief_backend must be one of {BACKENDS}")
        if config.episode_mix not in ("any", "confounded", "unconfounded"):
            raise ValueError("episode_mix must be 'any', 'confounded' or 'unconfounded'")
        if config.reward_criterion == "claims" and config.belief_backend not in CLAIM_BACKENDS:
            raise ValueError(
                "reward_criterion='claims' scores bootstrap claim frequencies; the exact "
                "backend has no replicates. Use the constraint backend, or 'u14'.")
        if config.oracle_obs_structure and config.belief_backend != CONSTRAINT:
            raise ValueError("oracle_obs_structure requires the constraint backend")
        if config.belief_backend == VERSION_SPACE:
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
                               observe_partner_counts=config.observe_partner_counts,
                               mode_by_role=config.mode_by_role,
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
        self.samples, _ = sample_multi(self.params, cfg.n_obs, self._rng)

        self.known: Dict[int, np.ndarray] = {}
        self.clean: Dict[int, np.ndarray] = {}
        # Per row: was ANY node hidden from this agent intervened, in EITHER mode. The
        # constraint backend's foreign-regime mask (see cb/backend.py). Distinct from
        # `clean`, which counts CLAMPS only -- a varied hidden node still drives its
        # children (measured 2026-08-16: vary restores 0% of a confounded agent's
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
        self._last_claim_fraction: Optional[float] = None
        self._last_agent_fraction: Optional[Dict[int, float]] = None
        self._agent_rewards: Optional[Dict[int, float]] = None
        for window in self.windows.values():
            # Per-episode resample stream: see ConstraintBackend.set_episode.
            if hasattr(window.belief, "set_episode"):
                window.belief.set_episode(seed if seed is not None else 0)
        if cfg.belief_backend == VERSION_SPACE:
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
        # BEHAVIOURAL COORDINATION METRIC (2026-08-26). How many interventions landed on
        # each shared node, by anyone. Duplicate coverage -- two agents spending rounds on
        # the same shared node -- is the failure a coordinating policy avoids, and it is
        # measurable WITHOUT the belief engine, so it transfers across environments where
        # the identification rate does not. That is the point: it separates "the policy
        # stopped coordinating" from "coordination stopped paying".
        self.shared_touches: Dict[int, int] = {n: 0 for n in self.topology.exposed}
        # ROUNDS TO IDENTIFICATION, per agent. `None` until the window is identified; then
        # the round it first happened. Censored at the budget when it never does. A
        # continuous metric with a true zero, against the binary rate's 1-bit-per-episode.
        self.identified_round: Dict[int, Optional[int]] = {
            a: None for a in self.topology.agents}
        self.signals: Dict[int, str] = {a: NO_INTERVENTION for a in self.topology.agents}
        self.done_bit: Dict[int, float] = {a: 0.0 for a in self.topology.agents}
        self.connected = _is_connected(self.true_adjacency)
        self.last_chosen: Dict[int, Tuple[int, Optional[str]]] = {
            a: (PASS_ACTION, None) for a in self.topology.agents}
        self._refresh()
        return self._result(reward=0.0)

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
        self._record_signals()
        self._tally(cfg)

        if passed:
            # A forfeited round still GENERATES DATA. All agents receive it -- samples are
            # shared, so there is no asymmetry -- and because every round produces a batch,
            # total data volume is constant at `n_obs + budget * n_int` instead of varying
            # with how much the policy chose to act. That confound is present in every
            # number this project produced before 2026-08-21.
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
                                      intervene_nodes=targets)
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

        self._refresh()
        cost = cfg.step_cost * sum(
            1 for a in self.topology.agents if chosen[a][0] != PASS_ACTION)
        return self._result(reward=-cost)

    def _append_observational_batch(self) -> None:
        """One batch with nothing intervened on -- what a forfeited round produces."""
        cfg = self.config
        new_samples, _ = sample_multi(self.params, cfg.n_int, self._rng)
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
            # form, caught 2026-08-24 when vary-mode zeroed a confounded agent).
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
        if self.config.belief_backend in CLAIM_BACKENDS:
            # Replicate credit fraction with private-incident edges required, mirroring
            # [U14]'s criterion 1. The union acyclicity/MEC check does NOT port -- a
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
        # the budget is shared. It was per-agent interventions before 2026-08-21.
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
                               self._partner_counts(agent)])

    def _partner_counts(self, agent: int) -> np.ndarray:
        """Cumulative per-partner counts, flattened, or an empty array.

        Budget-normalised for the same reason the own-counts are: a raw count next to
        probabilities in [0, 1] once dominated the first layer.
        """
        if not self.config.observe_partner_counts:
            return np.zeros(0)
        return (self.partner_counts[agent] / max(self.config.budget, 1)).reshape(-1)

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
        belief = window.belief.last
        if belief is None:
            return np.zeros(window.k * (window.k - 1))
        rows, cols = np.triu_indices(window.k, k=1)
        return np.concatenate([np.asarray(belief.bidirected)[rows, cols],
                               np.asarray(belief.adjacency)[rows, cols]])

    def _result(self, reward: float, passed: bool = False) -> StepResult:
        threshold = self.config.identify_threshold
        claim_info = None
        if self.config.reward_criterion == "claims":
            from cb.claims import score_window
            cfg = self.config
            scores = {}
            for agent, window in self.windows.items():
                scores[agent] = score_window(
                    window.belief.last, self._true_mag(agent),
                    [window.pos[n] for n in window.private], bar=cfg.claim_bar,
                    require_all_types=cfg.claims_require_all_types)
            identified = {a: scores[a].identified for a in self.topology.agents}
            all_identified = all(identified.values())
            mass = {a: scores[a].fraction(cfg.claim_penalty)
                    for a in self.topology.agents}
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
                    self._agent_rewards[a] = delta + (1.0 if identified[a] else 0.0)
                self._last_agent_fraction = dict(mass)
            claim_info = {a: {"right": s.n_right, "wrong": s.n_wrong,
                              "unsure": s.n_unsure,
                              "required_right": s.required_right,
                              "required_total": s.required_total}
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
        # never ends it -- see docs/TURN_BUDGET_SPEC.md section 4.
        out_of_budget = self.rounds_used >= self.config.budget
        if all_identified:
            reward += 1.0                       # shared terminal reward [U15]
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

