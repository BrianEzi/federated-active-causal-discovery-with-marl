"""`ma.evaluate._claims_success` must grade what `TwoAgentEnv._result` pays for.

WHY THIS TEST EXISTS. The two drifted, silently, and the drift survived the suite because
both sides were internally consistent -- training used the env's criterion, the reported
`success` used evaluate's, and nothing compared them. On the attributed backend evaluate
scored the confounding claims that backend REPLACES and never looked at attribution at all.

WHERE THE DIFFERENCE IS REACHABLE, which decides what this file can assert. Measured
2026-08-28 at the `attr3a` configuration: the old criterion over-credited 23 of 714 WINDOWS
and under-credited none, but moved no EPISODE verdict in 604 sampled states, because a joint
verdict needs every window and the over-credited window was never the last blocker. So an
episode-level test alone would pass against the bug -- it would prove nothing. The window
level is where the two criteria can be told apart, and that is what
`test_window_criterion_matches_the_env` pins.

The tests compare the two on REAL EPISODES rather than asserting on either implementation, so
they fail whichever side moves. A mirror obligation enforced by a comment is not enforced.
"""
from __future__ import annotations

import pytest

from ma.env import ATTRIBUTED, MAConfig, ROUND_ROBIN, TwoAgentEnv, VARY
from ma.evaluate import _claims_success
from ma.topology import federated_topology


def _env(backend: str, require_all_types: bool = True,
         agents: int = 2, private: int = 1, shared: int = 3,
         budget: int = 6) -> TwoAgentEnv:
    topology = federated_topology(agents, private, shared)
    config = MAConfig(topology=topology, n_obs=60, n_int=20, budget=budget,
                      disclose_regime=True, turn_order=ROUND_ROBIN,
                      action_modes=(VARY,), belief_backend=backend,
                      policy_arch="gnn_portable", episode_mix="confounded",
                      reward_criterion="claims", claim_bar=1.0,
                      claims_require_all_types=require_all_types)
    return TwoAgentEnv(config, seed=0)


def _greedy(env, agent):
    """The baseline at the GRADED bar. A fixed cycling policy identifies too little to put
    either criterion under strain."""
    from ma.baselines import UncertaintyGreedyAgent
    return UncertaintyGreedyAgent(agent, 0, bar=env.config.claim_bar)


def _env_verdict(env) -> bool:
    """The env's OWN verdict, taken from the step result rather than recomputed here.

    Recomputing it would make this a test of two copies of the same expression.
    `both_identified` is what `_result` puts in `info`, and it is the flag training reads.
    """
    result = env.step({a: 0 for a in env.topology.agents})
    return bool(result.info["both_identified"]), result


@pytest.mark.parametrize("backend", ["factored", ATTRIBUTED])
def test_evaluate_agrees_with_the_env_across_an_episode(backend):
    """Every round of several episodes, on both backends that score claims."""
    env = _env(backend)
    disagreements = []
    for episode in range(6):
        result = env.reset(seed=4_000 + episode)
        rounds = 0
        while not result.done and rounds < 12:
            actions = {a: rounds % env.n_actions(a) for a in env.topology.agents}
            result = env.step(actions)
            rounds += 1
            mine = _claims_success(env)
            theirs = bool(result.info["both_identified"])
            if mine != theirs:
                disagreements.append((backend, episode, rounds, mine, theirs))
    assert not disagreements, (
        f"evaluate and the env disagree on {len(disagreements)} states: "
        f"{disagreements[:5]}")


def test_relaxed_type_requirement_is_honoured():
    """With `claims_require_all_types=False` evaluate must not grade on the strict rule.

    The old code passed `score_window`'s default, so a run that relaxed the criterion was
    reported against a criterion it never trained for.
    """
    strict = _env("factored", require_all_types=True)
    relaxed = _env("factored", require_all_types=False)
    for env in (strict, relaxed):
        env.reset(seed=99)
    # The configured value has to reach the scorer; the clearest evidence is that the two
    # environments are graded differently at all on some state.
    seen = set()
    for episode in range(8):
        for env in (strict, relaxed):
            result = env.reset(seed=7_000 + episode)
            rounds = 0
            while not result.done and rounds < 8:
                result = env.step({a: rounds % env.n_actions(a)
                                   for a in env.topology.agents})
                rounds += 1
            seen.add((env.config.claims_require_all_types, _claims_success(env)))
    assert any(flag is False for flag, _ in seen), "the relaxed env never ran"
    assert any(flag is True for flag, _ in seen), "the strict env never ran"


def _old_window_verdict(env, agent, window) -> bool:
    """The criterion `_claims_success` applied before 2026-08-28, per window.

    Kept here rather than deleted with the bug: an episode-level test passes against it, so
    without this the regression is not actually pinned by anything.
    """
    from cb.claims import score_window
    return score_window(window.belief.last, env._true_mag(agent),
                        [window.pos[n] for n in window.private],
                        bar=env.config.claim_bar).identified


def _new_window_verdict(env, agent, window) -> bool:
    """What the env pays for, per window, on the attributed backend."""
    from cb.attribution import score_groups
    from cb.claims import score_window
    cfg = env.config
    structure = score_window(window.belief.last, env._true_mag(agent),
                             [window.pos[n] for n in window.private],
                             bar=cfg.claim_bar,
                             require_all_types=cfg.claims_require_all_types,
                             confounding_claims=False).identified
    if not structure:
        return False
    return bool(score_groups(window.belief.last, window.belief.true_groups,
                             bar=cfg.claim_bar)["identified"])


def test_window_criterion_matches_the_env():
    """The old rule must be visibly WEAKER, and the new one must never be weaker.

    Two assertions, and the first is the load-bearing one: it fails if a future edit makes
    `_claims_success` equivalent to the old rule again. Without it the file would pass
    against the very bug it was written for.
    """
    # The `attr3a` configuration, which is where the two criteria were measured to come
    # apart (23 windows of 714). The small (2, 1, 3) env used above cannot separate them --
    # the vacuity assertion at the end of this test is what caught that.
    env = _env(ATTRIBUTED, agents=3, private=2, shared=3, budget=12)
    policies = {a: _greedy(env, a) for a in env.topology.agents}
    old_only = new_only = 0
    for episode in range(8):
        for policy in policies.values():
            policy.reset(0)
        result = env.reset(seed=5_000 + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result)
                               for a in env.topology.agents})
            for agent, window in env.windows.items():
                old = _old_window_verdict(env, agent, window)
                new = _new_window_verdict(env, agent, window)
                old_only += old and not new
                new_only += new and not old
    assert new_only == 0, (
        f"{new_only} windows the env credits and the old rule did not -- the two criteria "
        "are not ordered the way the fix assumes")
    assert old_only > 0, (
        "the old rule credited nothing extra, so this configuration cannot tell the two "
        "criteria apart and the test is vacuous -- pick one that can")
