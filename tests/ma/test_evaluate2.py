"""The success criterion itself, tested. The headline number comes from here.

`evaluate2` decides what counts as success, so a bug in it does not produce an error -- it
produces a plausible number that means something other than what is claimed. These tests
pin the three parts of the criterion against hand-built cases where the right answer is
known by inspection rather than by running the code.
"""
from __future__ import annotations

import numpy as np
import pytest

from ma.env2 import MA2Config, TwoAgentEnv2
from ma.evaluate2 import credit_set, evaluate_episode, union_graph
from ma.topology import Topology
from sa.graphs import is_acyclic, mec_signature


@pytest.fixture(scope="module")
def topology():
    return Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


@pytest.fixture(scope="module")
def env(topology):
    return TwoAgentEnv2(MA2Config(topology=topology, n_obs=200, n_int=50, budget=2))


def test_the_truth_is_always_in_its_own_credit_set(env):
    """Trivial, and exactly the kind of thing that silently breaks: if the criterion cannot
    credit the correct answer, every success rate is measured against nothing."""
    for name in ("A", "B"):
        window = env.windows[name]
        for seed in range(12):
            env.reset(seed=seed)
            truth = window.induced(env.true_adjacency)
            mask = credit_set(window, truth)
            from ma.baselines2 import _Window
            index = next(i for i, dag in enumerate(_Window.get(window.k).dags)
                         if np.array_equal(dag, truth))
            assert mask[index], "the true graph is not in its own credit set"


def test_credit_requires_private_incident_edges_to_be_exactly_right(env, topology):
    """Part 1 of the criterion is BOUNDARY-INCLUSIVE: every edge touching a private node
    must be oriented correctly, not merely present. At (1,1,3) that is 3 edges per agent,
    so this part is not vacuous at the starting topology."""
    from ma.baselines2 import _Window
    window = env.windows["A"]
    private = [window.pos[n] for n in window.private]
    assert len(private) == 1
    p = private[0]

    env.reset(seed=3)
    truth = window.induced(env.true_adjacency)
    mask = credit_set(window, truth)
    for index, dag in enumerate(_Window.get(window.k).dags):
        if not mask[index]:
            continue
        assert np.array_equal(dag[p, :], truth[p, :])
        assert np.array_equal(dag[:, p], truth[:, p])


def test_credit_allows_shared_reorientation_within_the_equivalence_class(env):
    """Part 2 relaxes SHARED-shared edges to CPDAG resolution. So the credit set must be
    able to contain graphs other than the truth -- otherwise the relaxation is not
    implemented and the criterion is silently stricter than specified."""
    window = env.windows["A"]
    sizes = []
    for seed in range(25):
        env.reset(seed=seed)
        truth = window.induced(env.true_adjacency)
        sizes.append(int(credit_set(window, truth).sum()))
    assert max(sizes) > 1, (
        "no episode admitted more than one correct answer, so the CPDAG relaxation is "
        "not doing anything")


def test_every_credited_graph_is_markov_equivalent_to_the_truth(env):
    from ma.baselines2 import _Window
    window = env.windows["B"]
    for seed in range(8):
        env.reset(seed=seed)
        truth = window.induced(env.true_adjacency)
        mask = credit_set(window, truth)
        target = mec_signature(truth)
        for index in np.flatnonzero(mask):
            assert mec_signature(_Window.get(window.k).dags[index]) == target


def test_union_of_two_correct_windows_reproduces_the_true_graph(env):
    """The federation claim in one assertion: cross-private edges are forbidden, so every
    permitted edge lies in one window or the other and nothing is lost by restriction."""
    from ma.baselines2 import _Window
    for seed in range(10):
        env.reset(seed=seed)
        indices = {}
        for name in ("A", "B"):
            window = env.windows[name]
            truth = window.induced(env.true_adjacency)
            indices[name] = next(
                i for i, dag in enumerate(_Window.get(window.k).dags)
                if np.array_equal(dag, truth))
        union = union_graph(env, indices)
        assert np.array_equal(union, np.asarray(env.true_adjacency).astype(np.int8))


def test_the_acyclicity_check_can_actually_fail(env, topology):
    """The check exists because two agents may orient a SHARED edge differently within the
    same equivalence class and union into a cycle. A check that cannot fire is decoration,
    so this constructs the failure it is supposed to catch."""
    d = topology.d
    x1, x2 = topology.exposed[0], topology.exposed[1]
    cyclic = np.zeros((d, d), dtype=np.int8)
    cyclic[x1, x2] = 1
    cyclic[x2, x1] = 1                       # A says x1->x2, B says x2->x1
    assert not is_acyclic(cyclic), (
        "a two-cycle was judged acyclic -- the global consistency check is inert")


def test_evaluate_episode_reports_every_part_of_the_criterion(env):
    env.reset(seed=41)
    env.step(0, 2)
    report = evaluate_episode(env)
    for key in ("per_agent", "private_and_shared_ok", "union_acyclic",
                "union_equivalent", "success"):
        assert key in report
    # success is the conjunction, so it can never exceed any of its own conditions.
    if report["success"]:
        assert report["union_acyclic"]
        assert report["union_equivalent"]
        assert all(report["private_and_shared_ok"].values())


def test_success_is_never_true_when_a_part_fails(env):
    """Guards against the conjunction being assembled with the wrong operator -- a mistake
    that would raise every reported number and never raise an error."""
    for seed in range(15):
        env.reset(seed=seed)
        env.step(1, 3)
        report = evaluate_episode(env)
        parts = (report["union_acyclic"], report["union_equivalent"],
                 *report["private_and_shared_ok"].values())
        assert report["success"] == all(parts)
