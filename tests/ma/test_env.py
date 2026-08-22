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

from ma.env import CLAMP, MAConfig, MODES, TwoAgentEnv, VARY
from ma.topology import Topology, two_agent
from sa.priors import connectivity_prior_p


@pytest.fixture(scope="module")
def topology():
    return two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


def make(topology, **kwargs):
    # BOTH modes, explicitly. The default became clamp-only on 2026-08-22, but these are
    # MECHANISM gates -- several assert vary-specific semantics (a vary cleans nothing, a
    # clamp wins a collision), and those properties still have to hold for any caller who
    # opts back into `MODES`. Testing them requires the mode to exist.
    kwargs.setdefault("action_modes", MODES)
    config = MAConfig(topology=topology, n_obs=200, n_int=50, budget=3, **kwargs)
    return TwoAgentEnv(config)


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
    before = env.observation(0).copy()

    hidden = topology.hidden_from(0)
    assert hidden, "topology must hide something from A or the test is vacuous"
    rng = np.random.default_rng(0)
    for node in hidden:
        env.samples[:, node] = rng.normal(size=env.samples.shape[0])
    env._refresh()

    assert np.allclose(env.observation(0), before, atol=1e-12), (
        "A's observation changed when only B's private column changed -- the window slice "
        "is not being respected somewhere in the belief path")


def test_belief_uses_only_the_agents_own_columns(topology):
    """Same property one level deeper: the belief itself, not just the observation vector."""
    env = make(topology)
    env.reset(seed=5)
    before = env.marginals[1].copy()
    rng = np.random.default_rng(1)
    for node in topology.hidden_from(1):
        env.samples[:, node] = rng.normal(size=env.samples.shape[0])
    env._refresh()
    assert np.allclose(env.marginals[1], before, atol=1e-12)


def test_partner_disclosure_is_not_readable_before_acting(topology):
    """Round t's observation must carry round t-1's disclosure, never round t's."""
    env = make(topology, disclose_shared_targets=True)
    env.reset(seed=7)
    # Nothing has happened yet, so nothing can be disclosed.
    assert not env.disclosed[0].any()
    assert not env.disclosed[1].any()

    shared_index = env.windows[1].actions.index((topology.exposed[0], VARY))
    a_private = env.windows[0].actions.index((topology.private[0][0], VARY))
    env.step({0: a_private, 1: shared_index})
    # AFTER the step, A may see that B touched a shared node.
    assert env.disclosed[0].any(), "shared-node targets are supposed to be disclosed"


def test_private_targets_are_never_disclosed(topology):
    """B acting on its own private node must leave A's disclosure vector empty."""
    env = make(topology, disclose_shared_targets=True)
    env.reset(seed=11)
    b_private = env.windows[1].actions.index((topology.private[1][0], VARY))
    a_private = env.windows[0].actions.index((topology.private[0][0], VARY))
    env.step({0: a_private, 1: b_private})
    assert not env.disclosed[0].any(), (
        "A was told about an intervention on B's PRIVATE node -- that is the federation "
        "constraint violated outright")


def test_regime_bit_is_off_unless_enabled(topology):
    """The no-bit arm is the baseline, so the default must really disclose nothing."""
    env = make(topology, disclose_regime=False)
    env.reset(seed=13)
    b_clamp = env.windows[1].actions.index((topology.private[1][0], CLAMP))
    a_private = env.windows[0].actions.index((topology.private[0][0], VARY))
    env.step({0: a_private, 1: b_clamp})
    assert env.regime_bit[0] == 0.0
    # The environment still tracks cleanliness internally; it simply does not tell anyone.
    assert env.clean[0].any(), (
        "the environment should still KNOW the batch was clean -- only the disclosure is "
        "withheld, so the two arms differ in exactly one place")


def test_regime_bit_fires_when_enabled(topology):
    env = make(topology, disclose_regime=True)
    env.reset(seed=13)
    b_clamp = env.windows[1].actions.index((topology.private[1][0], CLAMP))
    a_private = env.windows[0].actions.index((topology.private[0][0], VARY))
    env.step({0: a_private, 1: b_clamp})
    assert env.regime_bit[0] == 1.0


def test_clamp_wins_a_collision(topology):
    """Both agents may target the same shared node; the more restrictive assignment holds."""
    env = make(topology)
    env.reset(seed=17)
    node = topology.exposed[0]
    a_vary = env.windows[0].actions.index((node, VARY))
    b_clamp = env.windows[1].actions.index((node, CLAMP))
    env.step({0: a_vary, 1: b_clamp})
    new_rows = env.samples[-env.config.n_int:, node]
    assert np.allclose(new_rows, new_rows[0]), (
        "a clamp collided with a vary and the variable still varied")


def test_episodes_are_deterministic_under_a_fixed_seed(topology):
    """Without this, no fixture and no regression test means anything."""
    def run():
        env = make(topology)
        result = env.reset(seed=23)
        trace = [result.info["true_mass"][0]]
        for action in (0, 2, 1):
            result = env.step({0: action, 1: action})
            trace.append(result.info["true_mass"][0])
        return np.asarray(trace), env.samples.copy()

    first_trace, first_samples = run()
    second_trace, second_samples = run()
    assert np.array_equal(first_samples, second_samples)
    assert np.allclose(first_trace, second_trace, atol=0)


def test_passing_does_not_consume_the_partners_opportunities(topology):
    """Under SIMULTANEOUS action both agents act every round, so a round A wastes by passing
    is still a round B gets to use. Renamed 2026-08-22: the old name said the budget was
    per-agent, which stopped being true at the turn-budget change -- it is a shared pool of
    ROUNDS. Under simultaneous action the two readings coincide, which is why this kept
    passing under a name that contradicted the config it was testing."""
    env = make(topology)
    result = env.reset(seed=29)
    a_pass = env.windows[0].pass_index
    b_act = env.windows[1].actions.index((topology.private[1][0], VARY))
    for _ in range(env.config.budget):
        result = env.step({0: a_pass, 1: b_act})
    assert result.n_interventions[0] == 0
    assert result.n_interventions[1] == env.config.budget


def test_observation_features_are_all_in_unit_range(topology):
    """Raw counts beside probabilities was a real bug once: the budget feature sat at 20.0
    next to values in [0,1] and dominated the first layer."""
    env = make(topology, disclose_regime=True)
    env.reset(seed=31)
    for _ in range(2):
        env.step({0: 0, 1: 2})
    for agent in env.topology.agents:
        obs = env.observation(agent)
        assert obs.shape == (env.obs_size(agent),)
        assert np.isfinite(obs).all()
        assert (obs >= -1e-9).all() and (obs <= 1 + 1e-9).all()


# -- defaults, guarded ------------------------------------------------------------------
#
# Both of these changed on 2026-08-22 and both change measured numbers, so they are pinned
# here rather than left to a docstring. A default that drifts silently is how this project
# lost a budget's meaning once already.


def test_clamp_only_is_the_default(topology):
    """Adopted as a TRADE WITH A KNOWN PRICE: paired over TWENTY seeds both-modes leads by
    +0.021, CI [+0.001, +0.042] -- significant, if barely. The ten-seed figure that motivated
    the change (+0.018, CI [-0.005, +0.041]) was not significant and is withdrawn. Restore
    both modes with `action_modes=MODES`."""
    assert MAConfig(topology=topology).action_modes == (CLAMP,)
    assert MAConfig(topology=topology, action_modes=MODES).action_modes == MODES


def test_prior_p_scales_with_d_and_resolves_once(topology):
    """`None` means `2 ln(d)/d`, resolved in __post_init__ so the generator, the posterior's
    prior and the logged config cannot disagree. An explicit float still wins."""
    cfg = MAConfig(topology=topology)
    assert cfg.prior_p == pytest.approx(connectivity_prior_p(topology.d))
    assert cfg.prior_p != 0.5, "the old fixed default must not survive as a coincidence"
    assert MAConfig(topology=topology, prior_p=0.5).prior_p == 0.5


def test_the_scaling_prior_beats_a_fixed_p_on_connectedness_at_scale():
    """The reason for the rule. At d=30 a fixed p=0.5 gives a mean degree near 14.5, far
    outside the literature's 2-6 band, while ER-2 gives 1% connected graphs. See
    scripts/sa_graph_density.py for the measurement this pins."""
    assert connectivity_prior_p(30) < 0.25 < connectivity_prior_p(10) < connectivity_prior_p(5)
    assert 2.0 < connectivity_prior_p(30) * 29 < 7.0, "mean degree stays in the ER-2..ER-6 band"


def test_three_agent_smoke_environment():
    """A 3-agent topology (1 private node each, 3 exposed) can be constructed, reset,
    stepped, and returns dictionary results indexed by {0, 1, 2}."""
    topo = Topology(name="T_3agent", private=((0,), (1,), (2,)), exposed=(3, 4, 5))
    env = TwoAgentEnv(MAConfig(topology=topo, n_obs=100, n_int=20, budget=6))
    res = env.reset(seed=42)
    assert len(res.beliefs) == 3
    assert set(res.beliefs.keys()) == {0, 1, 2}
    step_res = env.step({0: 0, 1: 0, 2: 0})
    assert len(step_res.beliefs) == 3
    assert len(step_res.identified) == 3
    assert len(step_res.n_interventions) == 3

