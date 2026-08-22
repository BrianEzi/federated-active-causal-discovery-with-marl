"""The regime scorer, and proof that its fast path is the slow path.

`RegimeScorer` precomputes every (hypothesis, node) -> parent-set slot so a belief update
becomes two array gathers instead of 543 x 8 x 4 Python iterations. That optimisation is
only safe if it reproduces the straightforward implementation exactly, so the straightforward
implementation lives here as a reference and the two are compared on real environment data.

Written after the optimisation, deliberately over the whole hypothesis space rather than a
summary statistic: a bug that shifts a handful of low-mass hypotheses would not move the
posterior mass on the truth, and would be invisible to a coarser check.

!! DO NOT MOVE THIS FILE TO legacy/tests/ !!

It imports from `legacy/`, which makes it look like the nineteen retired v1 test files moved
out on 2026-08-22. It is the opposite. Here v1 is the **independent reference oracle** for
CURRENT code: the value of the check is precisely that the reference shares no code with the
thing under test, so a shared bug cannot hide in both. If `legacy/ma_v1/` is ever deleted,
convert this to a frozen fixture FIRST -- never drop the check.
"""
from __future__ import annotations

import numpy as np
import pytest

from legacy.ma_v1.env import AgentView, MAConfig, TwoAgentEnv
from ma.score_regimes import JOINT, JOINT_CONF, POOLED, RULES, SUBSET, RegimeScorer
from ma.topology import Topology, two_agent

T113 = two_agent("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


def reference_log_posterior(scorer, samples, known, clean, rule):
    """The direct implementation: loop hypotheses, build parent sets, sum local scores."""
    view = scorer.view
    clean = np.asarray(clean, dtype=bool)
    cache = {}

    def local(tag, rows, node, parents):
        key = (tag, node, tuple(sorted(parents)))
        if key not in cache:
            keep = known[rows][:, node] < 0.5
            subset_rows = samples[rows][keep]
            cache[key] = (0.0 if len(subset_rows) <= len(parents) + 2
                          else view.score.local_score(node, tuple(sorted(parents)),
                                                      subset_rows))
        return cache[key]

    all_rows = np.ones(len(samples), dtype=bool)
    dirty = ~clean
    has_clean = bool(clean.any())

    if rule in (POOLED, SUBSET, JOINT) and not has_clean:
        groups = [("all", all_rows)]
    elif rule == POOLED:
        groups = [("all", all_rows)]
    elif rule == SUBSET:
        groups = [("clean", clean)]
    elif rule == JOINT:
        groups = [("clean", clean), ("dirty", dirty)]
    else:
        groups = None

    if groups is not None:
        log_post = np.zeros(view.n_dags)
        for i in range(view.n_dags):
            log_post[i] = sum(local(tag, rows, node, view.parents[i][node])
                              for tag, rows in groups for node in range(view.k))
    else:
        table = np.empty((view.n_dags, scorer.n_subsets))
        for i in range(view.n_dags):
            clean_term = sum(local("clean", clean, node, view.parents[i][node])
                             for node in range(view.k))
            for subset in range(scorer.n_subsets):
                parents = scorer._dirty_parents(i, subset)
                table[i, subset] = clean_term + sum(
                    local("dirty", dirty, node, parents[node]) for node in range(view.k))
        shift = table.max(axis=1, keepdims=True)
        log_post = np.log(np.exp(table - shift).sum(axis=1)) + shift.ravel()

    log_post = log_post - log_post.max()
    weights = np.exp(log_post)
    return weights / weights.sum()


def episode_data(seed, clamp_prob, rounds=4):
    """Real environment data -- not `rng.normal`. A previous scorer passed 29 tests on
    independent columns and then returned Z=0 on the first real episode."""
    env = TwoAgentEnv(MAConfig(topology=T113, n_obs=600, n_int=100, budget=rounds),
                      seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    view = env.views["A"]
    b_actions = env.views["B"].actions
    clamp_private = b_actions.index((1, "clamp"))
    # When NOT clamping, B must avoid clamping its private node specifically -- a uniform
    # draw over its action list hits that ~1/8 of the time and silently produces clean
    # rows, which made an earlier version of this helper unable to build a
    # genuinely-no-clean-rows episode.
    other = [i for i, (target, mode) in enumerate(b_actions)
             if target != -1 and not (target == 1 and mode == "clamp")]
    for _ in range(rounds):
        b = clamp_private if rng.random() < clamp_prob else int(rng.choice(other))
        env.step(int(rng.integers(view.n_actions - 1)), b)
    return (env.samples[:, view.nodes], env.known["A"], env.clean["A"], view)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("clamp_prob", [0.0, 0.5, 1.0],
                         ids=["no-clamp", "some-clamp", "all-clamp"])
def test_fast_path_matches_the_reference(rule, clamp_prob):
    samples, known, clean, view = episode_data(seed=3, clamp_prob=clamp_prob)
    scorer = RegimeScorer(view, [view.pos[node] for node in view.shared])

    fast = scorer.log_posterior(samples, known, clean, rule)
    slow = reference_log_posterior(scorer, samples, known, clean, rule)

    # Compared over the WHOLE posterior, not just the mass on the truth.
    assert np.allclose(fast, slow, rtol=0, atol=1e-12), (
        f"max abs difference {np.abs(fast - slow).max():.3e}")


@pytest.mark.parametrize("rule", RULES)
def test_posterior_is_a_distribution(rule):
    samples, known, clean, view = episode_data(seed=11, clamp_prob=0.5)
    scorer = RegimeScorer(view, [view.pos[node] for node in view.shared])
    post = scorer.log_posterior(samples, known, clean, rule)
    assert post.shape == (view.n_dags,)
    assert np.isfinite(post).all()
    assert (post >= 0).all()
    assert post.sum() == pytest.approx(1.0)


def test_slot_table_covers_every_parent_set_exactly_once():
    """The packing maps (node, parent-set) onto k * 2^(k-1) slots. If two different parent
    sets ever collided, one hypothesis would be silently scored as another."""
    view = AgentView("A", T113)
    scorer = RegimeScorer(view, [view.pos[node] for node in view.shared])
    seen = {}
    for i in range(view.n_dags):
        for node in range(view.k):
            index = int(scorer.clean_slots[i, node])
            parents = tuple(sorted(view.parents[i][node]))
            if index in seen:
                assert seen[index] == (node, parents), (
                    f"slot {index} used by both {seen[index]} and {(node, parents)}")
            seen[index] = (node, parents)
    assert len(seen) <= scorer.n_slots


def test_with_no_clean_rows_the_three_simple_rules_agree():
    """POOLED, SUBSET and JOINT must all reduce to scoring everything once when the partner
    never clamps. JOINT_CONF deliberately does not -- see ma/score_regimes.py."""
    samples, known, clean, view = episode_data(seed=5, clamp_prob=0.0)
    assert not clean.any()
    scorer = RegimeScorer(view, [view.pos[node] for node in view.shared])
    pooled = scorer.log_posterior(samples, known, clean, POOLED)
    for rule in (SUBSET, JOINT):
        assert np.allclose(pooled, scorer.log_posterior(samples, known, clean, rule))
    joint_conf = scorer.log_posterior(samples, known, clean, JOINT_CONF)
    assert not np.allclose(pooled, joint_conf)


def test_joint_conf_marginalises_rather_than_maximising():
    """The confounded-subset dimension is summed out, not argmaxed. A max would make the
    posterior over-confident, and the difference only shows when several subsets are
    plausible."""
    samples, known, clean, view = episode_data(seed=7, clamp_prob=0.5)
    scorer = RegimeScorer(view, [view.pos[node] for node in view.shared])
    post = scorer.log_posterior(samples, known, clean, JOINT_CONF)

    clean_scores = np.zeros(scorer.n_slots)
    dirty_scores = np.zeros(scorer.n_slots)
    for index in range(scorer.n_slots):
        node = int(scorer.slot_node[index])
        parents = scorer.slot_parents[index]
        for target, rows in ((clean_scores, clean), (dirty_scores, ~clean)):
            keep = known[rows][:, node] < 0.5
            block = samples[rows][keep]
            target[index] = (0.0 if len(block) <= len(parents) + 2
                             else view.score.local_score(node, parents, block))
    table = (clean_scores[scorer.clean_slots].sum(axis=1)[:, None]
             + dirty_scores[scorer.dirty_slots].sum(axis=2))

    maxed = table.max(axis=1)
    maxed = np.exp(maxed - maxed.max())
    maxed /= maxed.sum()
    assert not np.allclose(post, maxed), "marginalisation collapsed to a maximum"
