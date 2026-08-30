"""The optimal arm. Above k=5 there was no ceiling at all, so every comparison at the sizes
this project reports was learned-vs-heuristic: "beats greedy by X" and never "closes Y% of
the achievable headroom".
"""
import numpy as np
import pytest

from ma.baselines import OracleCoverAgent, make_baselines
from ma.env import MAConfig, TwoAgentEnv
from ma.evaluate import run_arm
from ma.topology import federated_topology


def _env(private=6, shared=6, agents=4, budget=60, evidence="oracle"):
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(agents, private, shared), n_obs=60, n_int=20,
        budget=budget, turn_order="round_robin", belief_backend="factored",
        action_modes=("vary",), claim_bar=1.0, reward_criterion="claims",
        policy_arch="gnn_portable", graph_model="sf", sf_m=2, episode_mix="confounded",
        vs_evidence=evidence))


def _arm(env, name):
    return {a: make_baselines(env, a, 0)[name] for a in env.topology.agents}


def test_it_identifies_every_episode_given_the_budget():
    """If the forced set does not identify the window, the closed form is wrong -- so this
    is really a test of the covering characterisation, executed rather than asserted."""
    env = _env()
    assert run_arm(env, _arm(env, "oracle_cover"), episodes=25, seed=0)["success"] == 1.0


def test_it_is_a_ceiling_on_effort_not_just_on_outcome():
    """A reference that identifies but wastes rounds is not an optimum. It must reach the
    same place in no more moves than the heuristic."""
    env = _env()
    optimal = run_arm(env, _arm(env, "oracle_cover"), episodes=25, seed=0)
    greedy = run_arm(env, _arm(env, "greedy_uncertainty"), episodes=25, seed=0)
    assert optimal["mean_steps"] <= greedy["mean_steps"]
    assert optimal["success"] >= greedy["success"]


def test_it_leaves_no_residual_ambiguity():
    env = _env()
    out = run_arm(env, _arm(env, "oracle_cover"), episodes=20, seed=0)
    assert out["global_soft_shd"] == pytest.approx(0.0, abs=1e-9)


def test_it_works_above_the_enumeration_wall_where_no_other_optimum_exists():
    """k=20 is past MAX_ENUMERATED_K, so the exact-DP greedy and vs_evaluate's exact
    optimum are both unavailable -- which is precisely the gap this arm fills."""
    env = _env(private=10, shared=10, agents=4, budget=90)
    assert run_arm(env, _arm(env, "oracle_cover"), episodes=8, seed=0)["success"] == 1.0


def test_it_stops_once_the_cover_is_complete():
    """Further rounds buy nothing under oracle evidence -- a repeat re-reveals ancestry
    already applied -- so a ceiling that kept acting would overstate the required effort."""
    env = _env(budget=200)
    out = run_arm(env, _arm(env, "oracle_cover"), episodes=15, seed=0)
    assert out["mean_steps"] < 200


def test_it_spends_private_before_shared():
    """A private node is reachable by nobody else; a shared one a partner may also cover.
    Deferring the contended surface is the measured winning ordering."""
    env = _env(private=4, shared=4, agents=3, budget=60)
    agent = 0
    policy = OracleCoverAgent(agent, env, seed=0)
    result = env.reset(seed=3)
    window = env.windows[agent]
    plan = policy._build_plan(env)
    private = [n for n in plan if n in window.private]
    shared = [n for n in plan if n not in window.private]
    assert plan == private + shared


def test_it_refuses_under_sampled_evidence():
    """There the belief is not a function of the intervened SET alone, so no set is
    sufficient with certainty and the required cover does not exist. Returning a number
    anyway would be the quietly-meaningless failure `required_cover.py` refuses."""
    env = _env(evidence="sampled")
    with pytest.raises(ValueError, match="oracle evidence"):
        OracleCoverAgent(0, env, seed=0)


def test_it_beats_a_random_arm_by_a_wide_margin():
    """Sanity: a ceiling that a random policy approaches is not measuring anything."""
    env = _env()
    optimal = run_arm(env, _arm(env, "oracle_cover"), episodes=20, seed=0)["success"]
    random = run_arm(env, _arm(env, "random_vary"), episodes=20, seed=0)["success"]
    assert optimal > random + 0.5
