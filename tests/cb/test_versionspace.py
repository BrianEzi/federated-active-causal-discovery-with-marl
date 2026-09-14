"""The deterministic backend's guarantees, each checked directly rather than through a run.

The whole value of this environment rests on two properties: the truth never leaves the
version space, and a claim every survivor agrees on is therefore agreed CORRECTLY. If
either breaks, the environment silently becomes an ordinary noisy one with a wrong reward.
"""
from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest

from cb.claims import enumerate_claims
from cb.versionspace import (BI, FWD, NONE, VersionSpaceBackend, equivalence_class,
                             m_separated, marks_from_mag, pairs, reveal, valid_mag)
from ma.env import MAConfig, ROUND_ROBIN, VARY, TwoAgentEnv
from ma.topology import Topology

TOPO = Topology(name="T_3agent_1each", private=((0,), (1,), (2,)), exposed=(3, 4, 5))


def _env(**kw):
    kw.setdefault("belief_backend", "version_space")
    kw.setdefault("reward_criterion", "claims")
    kw.setdefault("claim_bar", 1.0)
    kw.setdefault("episode_mix", "confounded")
    kw.setdefault("budget", 3)
    return TwoAgentEnv(MAConfig(topology=TOPO, n_obs=60, n_int=20,
                                turn_order=ROUND_ROBIN, action_modes=(VARY,), **kw), seed=0)


def test_truth_is_always_in_the_space_however_much_is_revealed():
    env = _env()
    for seed in range(6):
        result = env.reset(seed=seed)
        while not result.done:
            result = env.step({a: 0 for a in TOPO.agents})
        for agent in TOPO.agents:
            truth = marks_from_mag(env._true_mag(agent))
            assert truth in env.windows[agent].belief.last.space, (seed, agent)


def test_resolved_implies_correct_so_settled_wrong_cannot_happen():
    """At bar 1.0 a claim settles only when every survivor agrees, and the truth is one of
    them -- so agreement is agreement WITH the truth. No claim may ever score `wrong`."""
    env = _env()
    for seed in range(6):
        result = env.reset(seed=seed)
        while not result.done:
            result = env.step({a: 1 for a in TOPO.agents})
        for agent in TOPO.agents:
            window = env.windows[agent]
            claims = enumerate_claims(window.belief.last, env._true_mag(agent),
                                      [window.pos[n] for n in window.private], bar=1.0)
            assert not any(c.outcome == "wrong" for c in claims), (seed, agent)


def test_adjacency_is_settled_before_any_intervention():
    """Markov-equivalent MAGs share adjacencies, so observation alone fixes the skeleton and
    every adjacency claim starts resolved. Interventions exist to settle the TYPES."""
    env = _env()
    env.reset(seed=3)
    for agent in TOPO.agents:
        frequencies = env.windows[agent].belief.last.adjacency
        assert np.all(np.isin(frequencies, (0.0, 1.0)))


def test_interventions_never_grow_the_space():
    env = _env(budget=6)
    result = env.reset(seed=4)
    sizes = {a: [len(env.windows[a].belief.last.space)] for a in TOPO.agents}
    while not result.done:
        result = env.step({a: 0 for a in TOPO.agents})
        for a in TOPO.agents:
            sizes[a].append(len(env.windows[a].belief.last.space))
    for a, trail in sizes.items():
        assert all(later <= earlier for earlier, later in zip(trail, trail[1:])), (a, trail)


def test_skeleton_only_enumeration_equals_exhaustive_search():
    """The scaling claim: equivalent MAGs share adjacencies, so searching orientations of
    the true skeleton is not an approximation. Checked against the full mark space at k=4,
    with maximality enforced -- an unenforced maximality check is what once made the
    exhaustive set look LARGER (it admitted non-maximal ancestral graphs)."""
    rng = np.random.default_rng(0)
    k = 4

    def maximal(marks):
        for (u, v), m in zip(pairs(k), marks):
            if m != NONE:
                continue
            others = [c for c in range(k) if c not in (u, v)]
            if not any(m_separated(marks, k, u, v, frozenset(c))
                       for r in range(len(others) + 1) for c in combinations(others, r)):
                return False
        return True

    for _ in range(4):
        while True:
            truth = tuple(int(x) for x in rng.integers(0, 4, len(pairs(k))))
            if valid_mag(truth, k) is not None and any(m != NONE for m in truth):
                break
        fast = set(equivalence_class(truth, k))
        queries = [(x, y, frozenset(c)) for (x, y) in pairs(k)
                   for r in range(k - 1)
                   for c in combinations([n for n in range(k) if n not in (x, y)], r)]
        target = [m_separated(truth, k, x, y, cond) for x, y, cond in queries]
        slow = {marks for marks in product((NONE, FWD, 2, BI), repeat=len(pairs(k)))
                if valid_mag(marks, k) is not None and maximal(marks)
                and [m_separated(marks, k, x, y, cond)
                     for x, y, cond in queries] == target}
        assert fast == slow, truth


def test_reveal_distinguishes_a_cause_from_a_confounder():
    """The one thing an intervention must buy: with 0 -> 1 versus 0 <-> 1, intervening on
    0 tells them apart, because only in the first is 0 an ancestor of 1."""
    k = 2
    caused = (FWD,)
    confounded = (BI,)
    assert reveal(caused, k, 0) != reveal(confounded, k, 0)


def test_backend_refuses_a_bar_below_one():
    with pytest.raises(ValueError, match="claim_bar"):
        _env(claim_bar=0.7)


def test_backend_refuses_a_criterion_it_cannot_score():
    with pytest.raises(ValueError, match="claims"):
        _env(reward_criterion="u14")


def test_reset_is_required_before_use():
    backend = VersionSpaceBackend(3)
    with pytest.raises(RuntimeError, match="reset"):
        backend.edge_marginals(np.zeros((5, 3)), np.zeros((5, 3)))
