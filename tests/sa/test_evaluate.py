"""Tests for the measurement protocol.

The two most important tests here encode failures already made:

`test_gap_closed_cannot_be_gamed_by_failing` -- the first smoke run produced an agent that
scored gap-closed 2.04, apparently twice as good as greedy, while agreeing with the oracle
6% of the time. It solved easy episodes fast and let hard ones hit the budget, and the
solved-only average then excluded exactly the episodes it was bad at.

`test_reference_policies_anchor_the_metric` -- random must score exactly 0.0 and greedy
exactly 1.0 by definition. They did not, because stateful policies carried RNG state that
advanced between the reference run and the evaluation run.
"""
import numpy as np
import pytest

from sa.baselines import make_baselines
from sa.env import EnvConfig
from sa.evaluate import (
    EpisodeTrace,
    check_criteria,
    episode_costs,
    evaluate,
    gap_closed,
    informative_fraction,
    mean_interventions_when_solved,
    mean_regret,
    run_episodes,
    summarise_seeds,
    under_acting_rate,
)
from sa.graphs import build_graph_space


def trace(identified, n, passed=False, mec=2, regrets=(), informative=()):
    return EpisodeTrace(identified=identified, n_interventions=n, passed_early=passed,
                        mec_size=mec, is_singleton=(mec == 1),
                        regrets=list(regrets), informative=list(informative),
                        optimal=[r == 0 for r in regrets])


@pytest.fixture(scope="module")
def space3():
    return build_graph_space(3)


@pytest.fixture(scope="module")
def refs(space3):
    cfg = EnvConfig(d=3, budget=20)
    b = make_baselines(space3, seed=0)
    return (cfg,
            run_episodes(cfg, b["random"], 60, seed=99, space=space3),
            run_episodes(cfg, b["greedy_oracle"], 60, seed=99, space=space3),
            b)


# --- the anchors --------------------------------------------------------------------

def test_reference_policies_anchor_the_metric(refs, space3):
    """Random must be exactly 0.0 and greedy exactly 1.0 -- that is the metric's definition.

    They came out 0.233 and 1.067 before stateful policies were made resettable.
    """
    cfg, random_ref, greedy_ref, b = refs
    r = evaluate(cfg, b["random"], random_ref, greedy_ref, 60, seed=99, space=space3)
    g = evaluate(cfg, b["greedy_oracle"], random_ref, greedy_ref, 60, seed=99, space=space3)
    assert r["gap_closed"] == pytest.approx(0.0, abs=1e-9)
    assert g["gap_closed"] == pytest.approx(1.0, abs=1e-9)


def test_evaluating_the_same_policy_twice_is_identical(refs, space3):
    cfg, random_ref, greedy_ref, b = refs
    a = evaluate(cfg, b["random"], random_ref, greedy_ref, 40, seed=99, space=space3)
    c = evaluate(cfg, b["random"], random_ref, greedy_ref, 40, seed=99, space=space3)
    assert a["gap_closed"] == pytest.approx(c["gap_closed"])
    assert a["mean_cost"] == pytest.approx(c["mean_cost"])


# --- the loophole -------------------------------------------------------------------

def test_gap_closed_cannot_be_gamed_by_failing():
    """An agent that abandons hard episodes must NOT outscore one that solves them."""
    budget = 20
    # Reference policies both solve everything.
    random_ref = [trace(True, 4) for _ in range(20)]
    greedy_ref = [trace(True, 2) for _ in range(20)]
    # Quitter: solves the easy half in one step, never solves the rest.
    quitter = [trace(True, 1) for _ in range(10)] + [trace(False, budget) for _ in range(10)]
    # Honest: solves everything, slightly slower than greedy.
    honest = [trace(True, 3) for _ in range(20)]

    assert gap_closed(honest, random_ref, greedy_ref, budget) > \
        gap_closed(quitter, random_ref, greedy_ref, budget)
    # And the discarded metric is exactly where the illusion came from.
    assert mean_interventions_when_solved(quitter).mean() < \
        mean_interventions_when_solved(honest).mean()


def test_episode_costs_charge_failures_at_the_budget():
    costs = episode_costs([trace(True, 3), trace(False, 7)], budget=20)
    np.testing.assert_allclose(costs, [3.0, 20.0])


def test_gap_closed_is_nan_when_references_are_indistinguishable():
    same = [trace(True, 2) for _ in range(5)]
    assert np.isnan(gap_closed(same, same, same, budget=20))


# --- hard-fail criteria ----------------------------------------------------------------

def test_under_acting_counts_only_giving_up_not_running_out_of_budget():
    """Passing while unidentified is the agent's choice; exhausting the budget is not."""
    traces = [trace(False, 0, passed=True), trace(False, 20, passed=False),
              trace(True, 2, passed=False), trace(True, 1, passed=False)]
    assert under_acting_rate(traces) == pytest.approx(0.25)


def test_criteria_fail_on_low_solve_rate_even_with_good_gap():
    good = {"gap_closed": 0.95, "under_acting_rate": 0.0,
            "solve_rate": 0.70, "greedy_solve_rate": 0.99}
    verdict = check_criteria(good)
    assert verdict["checks"]["gap_closed"] is True
    assert verdict["checks"]["solve_rate"] is False
    assert verdict["passed"] is False


def test_criteria_fail_on_under_acting_even_with_good_gap():
    verdict = check_criteria({"gap_closed": 0.95, "under_acting_rate": 0.25,
                              "solve_rate": 0.99, "greedy_solve_rate": 0.99})
    assert verdict["checks"]["no_under_acting"] is False
    assert verdict["passed"] is False


def test_criteria_detect_deterministic_collapse():
    deterministic = {"gap_closed": 0.50, "under_acting_rate": 0.0,
                     "solve_rate": 0.99, "greedy_solve_rate": 0.99}
    sampled = {"gap_closed": 0.95}
    verdict = check_criteria(deterministic, sampled)
    assert verdict["checks"]["no_collapse"] is False


def test_collapse_check_is_none_without_a_sampled_pass():
    """Absent evidence must not read as a pass."""
    verdict = check_criteria({"gap_closed": 0.9, "under_acting_rate": 0.0,
                              "solve_rate": 0.99, "greedy_solve_rate": 0.99})
    assert verdict["checks"]["no_collapse"] is None


# --- oracle-derived diagnostics ---------------------------------------------------------

def test_regret_ignores_uninformative_steps():
    """Averaging over steps where every option ties is what produced the retracted
    99.4% agreement figure."""
    t = trace(True, 3, regrets=[0.0, 5.0, 0.0], informative=[False, True, False])
    assert mean_regret([t]) == pytest.approx(5.0)
    assert informative_fraction([t]) == pytest.approx(1 / 3)


def test_regret_is_nan_when_nothing_was_informative():
    t = trace(True, 2, regrets=[0.0, 0.0], informative=[False, False])
    assert np.isnan(mean_regret([t]))


# --- seed aggregation ---------------------------------------------------------------------

def test_summary_reports_the_minimum_not_the_mean():
    """A mean hides a lucky run -- the failure mode the previous project never caught."""
    per_seed = [{"gap_closed": 0.95, "passed": True}, {"gap_closed": 0.90, "passed": True},
                {"gap_closed": 0.20, "passed": False}]
    s = summarise_seeds(per_seed, min_passing=3)
    assert s["min_gap_closed"] == pytest.approx(0.20)
    assert s["passed"] is False


def test_summary_passes_when_enough_seeds_pass():
    per_seed = [{"gap_closed": 0.9, "passed": True}] * 4 + [{"gap_closed": 0.1, "passed": False}]
    assert summarise_seeds(per_seed, min_passing=4)["passed"] is True


# --- stratification ------------------------------------------------------------------------

def test_stratification_separates_easy_from_hard(refs, space3):
    cfg, random_ref, greedy_ref, b = refs
    m = evaluate(cfg, b["greedy_oracle"], random_ref, greedy_ref, 60, seed=99, space=space3)
    bands = m["by_mec_size"]
    assert set(bands) == {"mec_1", "mec_2-4", "mec_5-inf"}
    assert sum(v["n"] for v in bands.values()) == 60
