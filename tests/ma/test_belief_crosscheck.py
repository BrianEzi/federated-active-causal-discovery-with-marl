"""Two independent implementations of every belief rule must agree.

`WindowBeliefDP` computes edge marginals through the subset DP without ever materialising a
graph. `enumerated_posterior` reconstructs the full 543-vector posterior from the DP's OWN
local score tables and sums it. They share the score tables and NOTHING else -- one walks the
subset lattice, the other lists graphs.

This matters most for `joint_conf`, which is the only rule that CANNOT be checked against the
frozen enumeration fixture: its hypothesis space was deliberately changed when the confounding
orientation became part of the hypothesis rather than a topological tie-break. So this
cross-check is the substitute, and without it that rule rests on a single implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.baselines import _Window, enumerated_posterior
from ma.belief_dp import JOINT, JOINT_CONF, POOLED, SUBSET
from ma.env import MAConfig, TwoAgentEnv
from ma.topology import Topology, two_agent

TOL = 1e-9
RULES = (POOLED, SUBSET, JOINT, JOINT_CONF)


@pytest.fixture(scope="module")
def topology():
    return two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


@pytest.mark.parametrize("rule", RULES)
def test_dp_and_enumeration_agree_on_edge_marginals(topology, rule):
    env = TwoAgentEnv(MAConfig(topology=topology, n_obs=300, n_int=50, budget=3,
                                 score_rule=rule, disclose_regime=True))
    worst = 0.0
    for seed in range(4):
        env.reset(seed=seed)
        # Two rounds, so the clean/dirty split is genuinely exercised rather than every
        # row landing in one regime.
        for action in (1, 3):
            env.step(action, action)
        for name in ("A", "B"):
            window = env.windows[name]
            clean = (env.clean[name] if env.config.disclose_regime
                     else np.zeros(len(env.samples), dtype=bool))
            posterior = enumerated_posterior(
                window, env.samples[:, window.nodes], env.known[name], clean, rule)
            implied = np.tensordot(
                posterior, _Window.get(window.k).dags.astype(float), axes=(0, 0))
            worst = max(worst, float(np.abs(implied - env.marginals[name]).max()))
    assert worst < TOL, f"{rule}: worst disagreement {worst:.3e}"


def test_a_regime_split_actually_occurs(topology):
    """Guards the test above from being vacuous: if no clean rows are ever produced, the
    four rules collapse to the same computation and agreement proves nothing."""
    env = TwoAgentEnv(MAConfig(topology=topology, n_obs=300, n_int=50, budget=3,
                                 score_rule=JOINT_CONF, disclose_regime=True))
    env.reset(seed=0)
    from ma.env import CLAMP
    b_clamp = env.windows["B"].actions.index((topology.private[1][0], CLAMP))
    env.step(0, b_clamp)
    assert env.clean["A"].any() and not env.clean["A"].all(), (
        "need both clean and dirty rows for the regime rules to differ")


def test_marginals_are_a_valid_probability_field(topology):
    for rule in RULES:
        env = TwoAgentEnv(MAConfig(topology=topology, n_obs=300, n_int=50, budget=2,
                                     score_rule=rule))
        env.reset(seed=9)
        env.step(0, 2)
        for name in ("A", "B"):
            m = env.marginals[name]
            assert np.isfinite(m).all()
            assert (m >= -1e-12).all() and (m <= 1 + 1e-12).all()
            assert np.allclose(np.diag(m), 0.0, atol=1e-12)
            # An edge and its reverse cannot both be certain in a DAG posterior.
            assert (m + m.T <= 1 + 1e-9).all()
