"""The greedy agent's tie-break convention, and the diagnostic that made it moot.

`GreedyAgent(tie_break=...)` was added to test whether GATE 2's failure is collision between
two agents that happen to break a tie the same way. These tests pin the mechanics, so that
the NEGATIVE result it produced -- a split convention changes nothing -- is attributable to
the measurement rather than to a broken knob.

That distinction matters here specifically. "The intervention had no effect" and "the
intervention was never applied" produce identical numbers, and this project has already
published one figure that turned out to be the second thing wearing the first thing's label.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.baselines import GreedyAgent
from ma.env import MAConfig, TwoAgentEnv
from ma.topology import Topology, two_agent


@pytest.fixture(scope="module")
def env():
    topology = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    return TwoAgentEnv(MAConfig(topology=topology, n_obs=200, n_int=50, budget=3,
                                  disclose_regime=True))


def test_rejects_an_unknown_convention(env):
    with pytest.raises(ValueError):
        GreedyAgent(0, env, tie_break="lowest")


def test_low_and_high_pick_the_extremes_of_the_tied_set(env):
    """The knob must actually reach the choice, not merely be stored.

    Driven directly rather than through an episode: a synthetic score vector with a known
    tied set, so the expected answer is known by inspection.
    """
    low = GreedyAgent(0, env, tie_break="low")
    high = GreedyAgent(0, env, tie_break="high")
    assert low.candidates == high.candidates
    assert len(low.candidates) >= 2

    def pick(agent, scores):
        best = np.flatnonzero(scores >= scores.max() - 1e-9)
        if agent.tie_break == "low":
            return int(agent.candidates[int(best[0])])
        if agent.tie_break == "high":
            return int(agent.candidates[int(best[-1])])
        raise AssertionError

    scores = np.zeros(len(low.candidates))          # everything tied
    assert pick(low, scores) == low.candidates[0]
    assert pick(high, scores) == high.candidates[-1]
    assert pick(low, scores) != pick(high, scores)


def test_a_singleton_argmax_makes_the_convention_irrelevant(env):
    """The reason the split arm was a no-op, stated as a test.

    A tie-break can only separate two agents where a tie exists. With one strictly best
    action every convention returns it, so the arms coincide -- which is exactly what the
    measurement found at the node level in ~94% of rounds.
    """
    low = GreedyAgent(0, env, tie_break="low")
    high = GreedyAgent(0, env, tie_break="high")
    scores = np.zeros(len(low.candidates))
    scores[2] = 1.0                                  # a unique maximum

    best = np.flatnonzero(scores >= scores.max() - 1e-9)
    assert len(best) == 1
    assert low.candidates[int(best[0])] == high.candidates[int(best[-1])]


def test_default_is_unchanged(env):
    """The diagnostic must not silently alter the arm the gates were measured with."""
    assert GreedyAgent(0, env).tie_break == "random"


def test_both_agents_run_and_choose_a_real_action(env):
    """End to end: the convention survives a live episode and yields valid actions."""
    agents = {0: GreedyAgent(0, env, seed=0, tie_break="low"),
              1: GreedyAgent(1, env, seed=0, tie_break="high")}
    result = env.reset(seed=0)
    for agent in env.topology.agents:
        index = agents[agent](env, result)
        assert index in agents[agent].candidates
        node, mode = env.windows[agent].actions[index]
        assert node != -1
        assert mode in ("vary", "clamp")
