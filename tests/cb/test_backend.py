"""The belief-backend boundary, and the constraint identification criterion.

Two kinds of test. The credit criterion is checked DIRECTLY against hand-written
replicates -- the same discipline as `test_orientation_ground_truth.py`, because every
engine bug so far survived every downstream metric. The reachability tests then check the
one thing direct tests cannot: that the reward is EARNABLE through the real environment,
and not free at round zero. A structurally unearnable metric once passed 529 tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from cb.backend import ConstraintBackend
from cb.bootstrap import BootstrapBelief
from cb.orient import CODE_BIDIRECTED, CODE_DIRECTED, CODE_UNDETERMINED
from cb.citest import FisherZ
from ma.env import CLAMP, MAConfig, ROUND_ROBIN, TwoAgentEnv
from ma.projection import BIDIRECTED as MAG_BIDIRECTED
from ma.projection import DIRECTED as MAG_DIRECTED
from ma.topology import two_agent


# -- the credit criterion, directly -----------------------------------------------------

def _backend_with(replicates) -> ConstraintBackend:
    replicates = np.asarray(replicates, dtype=np.int8)
    k = replicates.shape[1]
    backend = ConstraintBackend(k, shared_positions=())
    backend.last = BootstrapBelief(
        directed=np.zeros((k, k)), bidirected=np.zeros((k, k)),
        adjacency=np.zeros((k, k)), n_boot=len(replicates), ci_tests=0,
        truncated_fraction=0.0, replicates=replicates)
    return backend


def _mag_chain_with_confounder():
    """0 -> 1 directed, (1, 2) confounded: one of each claim type."""
    mag = np.zeros((3, 3), dtype=np.int8)
    mag[0, 1] = MAG_DIRECTED
    mag[1, 2] = mag[2, 1] = MAG_BIDIRECTED
    return mag


def _perfect_replicate():
    codes = np.zeros((3, 3), dtype=np.int8)
    codes[0, 1] = CODE_DIRECTED
    codes[1, 2] = codes[2, 1] = CODE_BIDIRECTED
    return codes


def test_a_perfect_replicate_is_credited():
    backend = _backend_with([_perfect_replicate()])
    assert backend.credit_fraction(_mag_chain_with_confounder()) == 1.0
    assert backend.credit_fraction(_mag_chain_with_confounder(), strict=True) == 1.0


def test_wrong_adjacency_is_not_credited():
    codes = _perfect_replicate()
    codes[0, 2] = codes[2, 0] = CODE_UNDETERMINED     # extra edge
    backend = _backend_with([codes])
    assert backend.credit_fraction(_mag_chain_with_confounder()) == 0.0


def test_confounding_must_be_exact_in_both_directions():
    # Missed confounder: the pair is adjacent but left undetermined.
    missed = _perfect_replicate()
    missed[1, 2] = missed[2, 1] = CODE_UNDETERMINED
    # False confounder: the directed edge is called bidirected instead.
    false = _perfect_replicate()
    false[0, 1] = false[1, 0] = CODE_BIDIRECTED
    mag = _mag_chain_with_confounder()
    assert _backend_with([missed]).credit_fraction(mag) == 0.0
    assert _backend_with([false]).credit_fraction(mag) == 0.0


def test_circles_are_allowed_except_where_required():
    """Markov equivalence leaves edges unorientable; the criterion must not punish
    honesty. But a REQUIRED (private-incident) edge left as a circle is not credited --
    an agent must resolve its own private structure."""
    codes = _perfect_replicate()
    codes[0, 1] = CODE_UNDETERMINED                   # honest circle on the directed edge
    codes[1, 0] = CODE_UNDETERMINED
    backend = _backend_with([codes])
    mag = _mag_chain_with_confounder()
    assert backend.credit_fraction(mag) == 1.0                       # nothing required
    assert backend.credit_fraction(mag, required_positions=[0]) == 0.0   # 0 is private
    assert backend.credit_fraction(mag, strict=True) == 0.0


def test_unsound_orientation_is_never_credited():
    codes = _perfect_replicate()
    codes[0, 1] = 0
    codes[1, 0] = CODE_DIRECTED                       # claims 1 -> 0: false
    codes[0, 1] = 3                                   # keep adjacency, circle at other end
    backend = _backend_with([codes])
    assert backend.credit_fraction(_mag_chain_with_confounder()) == 0.0


def test_fraction_is_the_mean_over_replicates():
    backend = _backend_with([_perfect_replicate(),
                             np.zeros((3, 3), dtype=np.int8),      # empty graph: wrong
                             _perfect_replicate(),
                             _perfect_replicate()])
    assert backend.credit_fraction(_mag_chain_with_confounder()) == 0.75


# -- the foreign-regime mask -------------------------------------------------------------

def test_foreign_rows_do_not_attribute_someone_elses_clamp():
    """The federated form of bug 5. A clamp on a variable OUTSIDE the window changes y's
    distribution in rows the window's own mask cannot flag. Without the foreign mask that
    difference is attributed to whatever x is under test; with it those rows leave the
    contrast. x here is causally inert -- nothing may fire either way."""
    rng = np.random.default_rng(7)
    n = 900
    # Hidden h drives y strongly. x is independent of everything.
    h = rng.normal(size=n)
    x = rng.normal(size=n)
    y = 1.5 * h + rng.normal(size=n)
    # Foreign clamp on h in the last third: h held at 0, so y loses most of its variance.
    h[600:] = 0.0
    y[600:] = 1.5 * h[600:] + rng.normal(size=300)
    # This agent's window sees only (x, y); it clamped x itself in the first third.
    x[:300] = 0.0
    data = np.column_stack([x, y])
    intervened = np.zeros((n, 2), dtype=bool)
    intervened[:300, 0] = True
    foreign = np.zeros(n, dtype=bool)
    foreign[600:] = True

    contaminated = FisherZ(data, intervened).ancestral_evidence()
    masked = FisherZ(data, intervened, foreign=foreign).ancestral_evidence()
    assert contaminated[0, 1], "the contamination this test exists to demonstrate"
    assert not masked[0, 1]


# -- through the real environment --------------------------------------------------------

TOPO = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))

# 0 -> 2, 0 -> 3 makes agent A's private node a confounder of B's shared pair (2, 3);
# 1 -> 4 gives B a private edge to resolve; 2 -> 4 connects the graph.
ADJ = np.zeros((5, 5), dtype=int)
ADJ[0, 2] = ADJ[0, 3] = ADJ[1, 4] = ADJ[2, 4] = 1

# Each agent clamps its own private node first -- the headline behaviour -- then the
# shared pair, giving both directions of the (2, 3) power check their own clamp block.
PLAN = {0: [0, 2, 3], 1: [1, 3, 2]}


def _drive(seed: int, criterion: str = "u14"):
    cfg = MAConfig(topology=TOPO, n_obs=400, n_int=250, budget=6,
                   turn_order=ROUND_ROBIN, belief_backend="constraint",
                   cb_n_boot=12, identify_threshold=0.7, disclose_regime=True,
                   reward_criterion=criterion)
    env = TwoAgentEnv(cfg, seed=seed)
    result = env.reset(seed=seed, adjacency=ADJ)
    at_reset = dict(result.identified)
    used = {0: 0, 1: 0}
    while not result.done:
        active = env.active_agent()
        actions = {}
        for agent in (0, 1):
            window = env.windows[agent]
            if agent == active and used[agent] < len(PLAN[agent]):
                actions[agent] = window.actions.index((PLAN[agent][used[agent]], CLAMP))
                used[agent] += 1
            else:
                actions[agent] = window.pass_index
        result = env.step(actions)
    return env, at_reset, result


def test_the_metric_is_earnable_and_not_free():
    """THE reachability test, re-pinned 2026-08-24 on the CLAIMS criterion -- the one
    training now uses. Seed 3 draws a strong confounder (w02*w03 = +2.34); with each
    agent clamping private-then-shared the episode must identify -- and must NOT be
    identified from observational data alone at round zero. If this fails after an
    engine change, the reward has silently become unearnable (or free), and no training
    run can be interpreted until that is understood.

    (The original pin was on the per-replicate u14 translation; block-stratified
    resampling legitimately moved that criterion's frequencies at this seed, and the
    criterion itself was superseded for its conjunction pathology -- see cb/claims.py.)"""
    env, at_reset, result = _drive(seed=3, criterion="claims")
    assert not any(at_reset.values()), "identification must not be free at round 0"
    assert result.info["both_identified"], result.info["true_mass"]
    assert all(result.identified.values())


def test_weak_confounding_reads_as_unidentified_not_as_confounded_elsewhere():
    """Seed 0 draws w02*w03 = -0.29: genuinely undetectable at this data volume. The
    engine must stay silent -- B uncredited, nothing falsely bidirected -- rather than
    invent the confounder or misattribute it."""
    env, _, result = _drive(seed=0)
    assert not result.info["both_identified"]
    bidirected = env.windows[1].belief.bidirected
    mag = env._true_mag(1)
    false_pairs = (bidirected >= 0.5) & ~(mag == MAG_BIDIRECTED)
    assert not false_pairs.any(), bidirected


def test_identical_seeds_reproduce_identical_beliefs():
    """Determinism through the backend's own resample stream: same seed, same episode,
    bit for bit. Without this no fixture or bug report is reproducible."""
    _, _, first = _drive(seed=3)
    _, _, second = _drive(seed=3)
    for agent in (0, 1):
        assert np.array_equal(first.beliefs[agent], second.beliefs[agent])
    assert first.info["true_mass"] == second.info["true_mass"]


def test_exact_backend_is_refused_on_wide_hidden_topologies():
    """The capability check the removed guard became. Exercised at env level in
    tests/test_env_turns.py; pinned here too because cb/ owns the capability claim."""
    wide = two_agent(name="T2_2", a_private=(0, 1), b_private=(2, 3), exposed=(4,))
    with pytest.raises(ValueError, match="UNSOUND"):
        TwoAgentEnv(MAConfig(topology=wide, n_obs=100, n_int=20, budget=2), seed=0)
    env = TwoAgentEnv(MAConfig(topology=wide, n_obs=100, n_int=20, budget=2,
                               belief_backend="constraint", cb_n_boot=4), seed=0)
    assert ConstraintBackend.can_handle_multi_hidden
    assert env.topology is wide


# =======================================================================================
# WHY `claim_bar` IS INERT ON THE FACTORED BACKEND, and why that is worth a test.
#
# The Tier 1 correctness work fixed a real defect: greedy was constructed at the class
# default bar of 0.7 while the task grades at `claim_bar`, measured worth +0.233 to greedy
# and enough to invert a headline. On 30 Aug 2026 the same defect was found still live in
# `scripts/ma_train.py`, which kept its own copy of the baseline registry -- the path the
# whole sweep runs through.
#
# Fixing it changed NOTHING on the factored backend, and this is the reason. A factored
# belief holds a set of surviving marks per pair and weights them uniformly, so every
# frequency it can produce is 1/|marks| summed over a subset -- and the skeleton is oracle
# in BOTH evidence modes, so a pair is known-adjacent or known-absent from reset. The
# reachable lattice is therefore {0, 1/3, 1/2, 1}, with NOTHING in (1/2, 1). Any bar in
# that open interval scores identically to a bar of 1.0.
#
# Two things follow, and both are load-bearing:
#   - existing factored numbers do NOT need re-scoring on account of the greedy bar;
#   - `claim_bar` is not a usable knob here. Anyone sweeping it on this backend would be
#     sweeping a no-op, and would spend a day finding out.
# It is asserted rather than argued because it depends on FactoredBelief's uniform
# weighting, which is an implementation choice that could change.
# =======================================================================================
import numpy as np
import pytest

from ma.baselines import make_baselines
from ma.env import MAConfig, TwoAgentEnv
from ma.topology import federated_topology


@pytest.mark.parametrize("evidence", ["oracle", "sampled"])
def test_factored_frequencies_never_land_between_a_half_and_one(evidence):
    config = MAConfig(topology=federated_topology(3, 4, 4), n_obs=60, n_int=20, budget=24,
                      turn_order="round_robin", belief_backend="factored",
                      action_modes=("vary",), claim_bar=1.0, reward_criterion="claims",
                      policy_arch="gnn_portable", graph_model="sf", sf_m=2,
                      episode_mix="confounded", vs_evidence=evidence)
    env = TwoAgentEnv(config)
    agents = {a: make_baselines(env, a, 0)["greedy_uncertainty"]
              for a in env.topology.agents}

    seen = set()
    for episode in range(5):
        result = env.reset(seed=episode)
        for agent in agents.values():
            agent.reset(episode)
        while not result.done:
            result = env.step({a: agents[a](env, result) for a in env.topology.agents})
            for window in env.windows.values():
                belief = window.belief.last
                for matrix in (belief.adjacency, belief.directed, belief.bidirected):
                    seen |= set(np.round(np.asarray(matrix).ravel(), 9).tolist())

    ambiguous = sorted(v for v in seen if 0.5 < v < 1.0)
    assert not ambiguous, (
        f"a frequency in (1/2, 1) appeared: {ambiguous}. The bar is no longer inert on "
        "this backend, so `claim_bar` now changes factored results and every comparison "
        "must hold it fixed -- see the note above.")
    assert seen <= {0.0, 1.0 / 3.0, 0.5, 1.0} | {round(1 / 3, 9), round(2 / 3, 9)}, (
        f"unexpected frequency lattice: {sorted(seen)}")
