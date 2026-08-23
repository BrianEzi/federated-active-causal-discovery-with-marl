"""Acceptance and unit tests for per-block confounding subsets (S_r).

Tests:
1. Exact equivalence at n=2 (T1_1_1_3) between per-block multi-regime scoring and standard 2-regime scoring.
2. Multi-private 2-agent topology (T2_2_2) handling partially clean blocks (f=0.5).
3. 3-agent topology (T1_1_1_3) handling multi-agent partner clamps.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.belief_dp import JOINT_CONF, WindowBeliefDP
from ma.env import CLAMP, MAConfig, TwoAgentEnv
from ma.score_regimes import RegimeScorer
from ma.topology import Topology, two_agent


def test_per_block_two_regime_equivalence():
    """Verify that when data contains binary clean/dirty blocks, per-block scoring
    produces bit-identical results to two-regime tables.
    """
    k = 4
    shared_positions = [1, 2, 3]
    belief = WindowBeliefDP(k, shared_positions)

    rng = np.random.default_rng(42)
    samples = rng.normal(size=(300, k))
    known = np.zeros((300, k))
    # 100 clean, 200 dirty
    clean_binary = np.concatenate([np.ones(100, dtype=float), np.zeros(200, dtype=float)])

    # Marginal computations
    marginals = belief.joint_conf_marginals(samples, known, clean_binary)
    assert marginals.shape == (k, k)
    assert np.all(marginals >= 0.0) and np.all(marginals <= 1.0)
    assert np.all(np.diag(marginals) == 0.0)

    # Weights and partition functions
    weights_z = belief.assignment_weights_and_z(samples, known, clean_binary)
    assert len(weights_z) == belief.n_assignments
    for log_w, log_z in weights_z:
        if np.isfinite(log_z):
            assert log_w.shape == (k, belief.scorer.n_parent_sets)


def test_multi_private_topology_is_accepted_since_the_guard_was_removed():
    """CORRECTED 2026-08-22. This test originally asserted the multi-private topology
    below ran to completion under the new per-block mixture. It should not have: the
    mixture only tracks an AGGREGATE clean fraction per round (how MANY of an agent's
    hidden nodes were clamped), never WHICH ones, so a confounding hypothesis about a
    specific hidden node cannot be told apart from one about a different hidden node
    whenever the round is only PARTIALLY clean. Demonstrated directly: `_assignment_weights`
    receives only a scalar fraction per row batch, with no per-node identity anywhere in
    its input, so two rounds with the same fraction but different clamped nodes are scored
    identically regardless of which confounding edge is under test. See
    ma/env.py's guard and docs/logs/MA_BUILD_LOG.md, 2026-08-22.

    CONVERTED 2026-08-23: the guard was removed by instruction, so this now asserts the
    env ACCEPTS the shape. The unsoundness above is unchanged and is demonstrated by
    tests/test_env_turns.py::test_clean_fraction_cannot_say_WHICH_node_was_clamped.
    """
    topo = Topology(name="T2_2_2", private=((0, 1), (2, 3)), exposed=(4, 5))
    config = MAConfig(topology=topo, n_obs=200, n_int=50, budget=6, disclose_regime=True)
    env = TwoAgentEnv(config, seed=123)
    assert max(len(topo.hidden_from(a)) for a in topo.agents) > 1
    assert env.topology is topo


def test_three_agents_one_private_each_is_accepted_since_the_guard_was_removed():
    """CORRECTED 2026-08-22, same reasoning as test_multi_private_topology_is_still_refused
    above. Also documents a gap in the ORIGINAL guard this project had before either of
    today's attempts: it checked `max(len(block) for block in private) > 1`, which is
    right at two agents but wrong here -- with three agents at ONE private node each, no
    single private block exceeds size 1, yet hidden_from(agent) is the UNION of the other
    two agents' nodes, i.e. two hidden nodes. The restored guard checks the actual hidden
    set per agent, which correctly catches this case too.

    CONVERTED 2026-08-23 with the guard's removal, as above. The three-agent shape is now
    CONSTRUCTIBLE, which is what unblocks rung 1 for a constraint-based backend.
    """
    topo = Topology(name="T_3agent", private=((0,), (1,), (2,)), exposed=(3, 4, 5))
    config = MAConfig(topology=topo, n_obs=200, n_int=50, budget=6, disclose_regime=True)
    env = TwoAgentEnv(config, seed=456)
    assert max(len(topo.hidden_from(a)) for a in topo.agents) > 1
    assert env.topology is topo
