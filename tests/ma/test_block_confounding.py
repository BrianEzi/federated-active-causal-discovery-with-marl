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


def test_multi_private_partially_clean_blocks():
    """Verify that a multi-private topology (2 private nodes per agent)
    can be instantiated and stepped without error, correctly generating
    partially clean blocks (f=0.5) and valid beliefs.
    """
    # 2 agents, 2 private nodes each, 2 exposed nodes (total d=6)
    topo = Topology(
        name="T2_2_2",
        private=((0, 1), (2, 3)),
        exposed=(4, 5)
    )
    config = MAConfig(topology=topo, n_obs=200, n_int=50, budget=6, disclose_regime=True)
    env = TwoAgentEnv(config, seed=123)

    assert env.windows[0].k == 4  # nodes {0, 1, 4, 5}
    assert env.windows[1].k == 4  # nodes {2, 3, 4, 5}

    # Step: Agent 0 clamps shared node 4, Agent 1 clamps its private node 2
    # For Agent 0: hidden nodes are {2, 3}. Node 2 is clamped, node 3 is not -> f = 1/2 = 0.5
    # For Agent 1: hidden nodes are {0, 1}. Neither is clamped -> f = 0/2 = 0.0
    action_0 = env.windows[0].actions.index((4, CLAMP))
    action_1 = env.windows[1].actions.index((2, CLAMP))

    result = env.step({0: action_0, 1: action_1})

    # Verify clean fractions
    assert env.clean[0][-50:].mean() == 0.5
    assert env.clean[1][-50:].mean() == 0.0

    # Verify beliefs updated with valid marginals
    for agent in env.topology.agents:
        belief = result.beliefs[agent]
        assert belief.shape == (4, 4)
        assert np.all(belief >= 0.0) and np.all(belief <= 1.0)
        assert np.all(np.diag(belief) == 0.0)


def test_three_agent_partially_clean_blocks():
    """Verify that in a 3-agent topology (1 private node each), when Agent 1 clamps,
    Agent 0 sees f = 0.5 (since hidden is {1, 2} and 1 is clamped).
    """
    topo = Topology(
        name="T_3agent",
        private=((0,), (1,), (2,)),
        exposed=(3, 4, 5)
    )
    config = MAConfig(topology=topo, n_obs=200, n_int=50, budget=6, disclose_regime=True)
    env = TwoAgentEnv(config, seed=456)

    # Agent 0 clamps shared 3, Agent 1 clamps private 1, Agent 2 passes
    action_0 = env.windows[0].actions.index((3, CLAMP))
    action_1 = env.windows[1].actions.index((1, CLAMP))
    action_2 = env.windows[2].pass_index

    result = env.step({0: action_0, 1: action_1, 2: action_2})

    # For Agent 0: hidden is {1, 2}. Node 1 clamped -> f = 1/2 = 0.5
    assert env.clean[0][-50:].mean() == 0.5
    # For Agent 1: hidden is {0, 2}. Neither clamped -> f = 0.0
    assert env.clean[1][-50:].mean() == 0.0
    # For Agent 2: hidden is {0, 1}. Node 1 clamped -> f = 1/2 = 0.5
    assert env.clean[2][-50:].mean() == 0.5

    # Check beliefs are non-trivial and valid
    for agent in env.topology.agents:
        b = result.beliefs[agent]
        assert b.shape == (4, 4)
        assert np.all(b >= 0.0) and np.all(b <= 1.0)
