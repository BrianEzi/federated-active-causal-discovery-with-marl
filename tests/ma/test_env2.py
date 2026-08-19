"""PHASE 2 GATES -- the checks the implementation plan specified for the environment.

Three properties, each of which would be invisible in ordinary results if violated:

  NO LEAK          an agent's observation must be a function of its own columns only. A
                   federation whose observations quietly depend on hidden variables is not
                   a federation, and the failure would look like unusually good performance
                   rather than like a bug.
  DISCLOSURE TIMING what the partner did in round t must not be readable at the moment of
                   choosing round t's action. This is the "before or after acting" question
                   turned into an assertion.
  DETERMINISM      same seed, same episode. Without it no fixture, no regression test, and
                   no bug report is reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.env2 import AGENTS, CLAMP, MA2Config, TwoAgentEnv2, VARY
from ma.topology import Topology


@pytest.fixture(scope="module")
def topology():
    return Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


def make(topology, **kwargs):
    config = MA2Config(topology=topology, n_obs=200, n_int=50, budget=3, **kwargs)
    return TwoAgentEnv2(config)


def test_observation_uses_only_the_agents_own_columns(topology):
    """The leak test.

    Rebuild the episode with the hidden private column REPLACED by noise, leaving every
    observed column byte-identical. A correct observation cannot move. This catches an
    observation assembled from the full sample matrix rather than the window slice -- the
    single most damaging bug this design could have, and one that would present as good
    results rather than as an error.
    """
    env = make(topology)
    env.reset(seed=3)
    before = env.observation("A").copy()

    hidden = topology.hidden_from("A")
    assert hidden, "topology must hide something from A or the test is vacuous"
    rng = np.random.default_rng(0)
    for node in hidden:
        env.samples[:, node] = rng.normal(size=env.samples.shape[0])
    env._refresh()

    assert np.allclose(env.observation("A"), before, atol=1e-12), (
        "A's observation changed when only B's private column changed -- the window slice "
        "is not being respected somewhere in the belief path")


def test_belief_uses_only_the_agents_own_columns(topology):
    """Same property one level deeper: the belief itself, not just the observation vector."""
    env = make(topology)
    env.reset(seed=5)
    before = env.marginals["B"].copy()
    rng = np.random.default_rng(1)
    for node in topology.hidden_from("B"):
        env.samples[:, node] = rng.normal(size=env.samples.shape[0])
    env._refresh()
    assert np.allclose(env.marginals["B"], before, atol=1e-12)


def test_partner_disclosure_is_not_readable_before_acting(topology):
    """Round t's observation must carry round t-1's disclosure, never round t's."""
    env = make(topology, disclose_shared_targets=True)
    env.reset(seed=7)
    # Nothing has happened yet, so nothing can be disclosed.
    assert not env.disclosed["A"].any()
    assert not env.disclosed["B"].any()

    shared_index = env.windows["B"].actions.index((topology.exposed[0], VARY))
    a_private = env.windows["A"].actions.index((topology.a_private[0], VARY))
    env.step(a_private, shared_index)
    # AFTER the step, A may see that B touched a shared node.
    assert env.disclosed["A"].any(), "shared-node targets are supposed to be disclosed"


def test_private_targets_are_never_disclosed(topology):
    """B acting on its own private node must leave A's disclosure vector empty."""
    env = make(topology, disclose_shared_targets=True)
    env.reset(seed=11)
    b_private = env.windows["B"].actions.index((topology.b_private[0], VARY))
    a_private = env.windows["A"].actions.index((topology.a_private[0], VARY))
    env.step(a_private, b_private)
    assert not env.disclosed["A"].any(), (
        "A was told about an intervention on B's PRIVATE node -- that is the federation "
        "constraint violated outright")


def test_regime_bit_is_off_unless_enabled(topology):
    """The no-bit arm is the baseline, so the default must really disclose nothing."""
    env = make(topology, disclose_regime=False)
    env.reset(seed=13)
    b_clamp = env.windows["B"].actions.index((topology.b_private[0], CLAMP))
    a_private = env.windows["A"].actions.index((topology.a_private[0], VARY))
    env.step(a_private, b_clamp)
    assert env.regime_bit["A"] == 0.0
    # The environment still tracks cleanliness internally; it simply does not tell anyone.
    assert env.clean["A"].any(), (
        "the environment should still KNOW the batch was clean -- only the disclosure is "
        "withheld, so the two arms differ in exactly one place")


def test_regime_bit_fires_when_enabled(topology):
    env = make(topology, disclose_regime=True)
    env.reset(seed=13)
    b_clamp = env.windows["B"].actions.index((topology.b_private[0], CLAMP))
    a_private = env.windows["A"].actions.index((topology.a_private[0], VARY))
    env.step(a_private, b_clamp)
    assert env.regime_bit["A"] == 1.0


def test_clamp_wins_a_collision(topology):
    """Both agents may target the same shared node; the more restrictive assignment holds."""
    env = make(topology)
    env.reset(seed=17)
    node = topology.exposed[0]
    a_vary = env.windows["A"].actions.index((node, VARY))
    b_clamp = env.windows["B"].actions.index((node, CLAMP))
    env.step(a_vary, b_clamp)
    new_rows = env.samples[-env.config.n_int:, node]
    assert np.allclose(new_rows, new_rows[0]), (
        "a clamp collided with a vary and the variable still varied")


def test_episodes_are_deterministic_under_a_fixed_seed(topology):
    """Without this, no fixture and no regression test means anything."""
    def run():
        env = make(topology)
        result = env.reset(seed=23)
        trace = [result.info["true_mass"]["A"]]
        for action in (0, 2, 1):
            result = env.step(action, action)
            trace.append(result.info["true_mass"]["A"])
        return np.asarray(trace), env.samples.copy()

    first_trace, first_samples = run()
    second_trace, second_samples = run()
    assert np.array_equal(first_samples, second_samples)
    assert np.allclose(first_trace, second_trace, atol=0)


def test_budget_is_per_agent_not_a_shared_pool(topology):
    """One agent exhausting itself must not end the other's episode."""
    env = make(topology)
    result = env.reset(seed=29)
    a_pass = env.windows["A"].pass_index
    b_act = env.windows["B"].actions.index((topology.b_private[0], VARY))
    for _ in range(env.config.budget):
        result = env.step(a_pass, b_act)
    assert result.n_interventions["A"] == 0
    assert result.n_interventions["B"] == env.config.budget


def test_observation_features_are_all_in_unit_range(topology):
    """Raw counts beside probabilities was a real bug once: the budget feature sat at 20.0
    next to values in [0,1] and dominated the first layer."""
    env = make(topology, disclose_regime=True)
    env.reset(seed=31)
    for _ in range(2):
        env.step(0, 2)
    for name in AGENTS:
        obs = env.observation(name)
        assert obs.shape == (env.obs_size(name),)
        assert np.isfinite(obs).all()
        assert (obs >= -1e-9).all() and (obs <= 1 + 1e-9).all()
