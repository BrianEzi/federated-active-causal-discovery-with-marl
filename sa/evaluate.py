"""The measurement protocol. Every reported number comes from here.

Criteria are fixed in docs/SA_PLAN.md and implemented here so they cannot drift between
what was agreed and what gets reported. All metrics are derived from one set of recorded
traces, so nothing needs a separate run and no two numbers can disagree about what
happened.

Primary metric is **gap closed**: `(random - agent) / (random - greedy)`, in interventions
to identify. 1.0 matches greedy, 0.0 is no better than random. Calibrated against an
epsilon-greedy oracle: 0.80 corresponds to choosing correctly ~70% of the time, 0.90 to
~80-90%.

Two evaluation passes per policy. The **deterministic** pass produces every pass/fail
number -- it is the deployment condition, and the one the previous project collapsed in.
The **sampled** pass exists only to detect a gap between the two.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig
from sa.oracle import InterventionOracle


@dataclass
class EpisodeTrace:
    identified: bool
    n_interventions: int
    passed_early: bool
    mec_size: int
    is_singleton: bool
    # Per-step oracle scoring of the action actually taken.
    regrets: List[float] = field(default_factory=list)
    informative: List[bool] = field(default_factory=list)
    optimal: List[bool] = field(default_factory=list)


def run_episodes(config: EnvConfig, policy: Callable, n_episodes: int = 300,
                 seed: int = 0, space=None,
                 oracle: Optional[InterventionOracle] = None) -> List[EpisodeTrace]:
    """Run `policy` and record everything the criteria need.

    The oracle scores each action *as it is taken*, against the posterior the agent
    actually held at that moment -- not against a reconstruction afterwards.
    """
    env = CausalDiscoveryEnv(config, space=space)
    if oracle is None:
        oracle = InterventionOracle(env.space)
    # Stateful policies carry an RNG that would otherwise advance between runs, making the
    # same policy score differently each time it is evaluated. Reset it so a reference run
    # and an evaluation run of the same policy are identical by construction.
    if hasattr(policy, "reset"):
        policy.reset()

    traces: List[EpisodeTrace] = []
    for i in range(n_episodes):
        result = env.reset(seed=seed * 100_000 + i)
        trace = EpisodeTrace(
            identified=result.identified,
            n_interventions=0,
            passed_early=False,
            mec_size=result.info["mec_size"],
            is_singleton=result.info["is_singleton"],
        )
        while not result.done:
            action = policy(env, result)
            if action != PASS_ACTION:
                scored = oracle.score_choice(action, result.posterior)
                trace.regrets.append(scored["regret"])
                trace.informative.append(bool(scored["informative"]))
                trace.optimal.append(bool(scored["is_optimal"]))
            result = env.step(action)

        trace.identified = result.identified
        trace.n_interventions = result.n_interventions
        # "Gave up": ended by passing while the graph was still unidentified. Distinct
        # from running out of budget, which is not the agent's choice.
        trace.passed_early = bool(result.info["passed"] and not result.identified)
        traces.append(trace)
    return traces


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def mean_interventions_when_solved(traces: List[EpisodeTrace]) -> np.ndarray:
    """Interventions used, over solved episodes only. DIAGNOSTIC -- not the primary metric.

    Kept for interpretability, but it must not drive pass/fail, because it is gameable by
    failing. Discovered on the very first smoke run: an agent solving 65% of episodes
    scored gap-closed 2.04 -- apparently twice as good as greedy -- while agreeing with the
    oracle only 6% of the time. It solved the easy episodes quickly and let the hard ones
    hit the budget, and the hard ones were then excluded from this average.
    """
    return np.array([t.n_interventions for t in traces if t.identified], dtype=float)


def episode_costs(traces: List[EpisodeTrace], budget: int) -> np.ndarray:
    """Cost of every episode: interventions used, or the full budget if never identified.

    This is what `gap_closed` runs on. Charging a failure at the budget is the standard
    censored-data treatment and it closes the loophole above -- an agent cannot improve its
    score by abandoning hard instances, because abandoning one costs the maximum.

    Note this makes the metric sensitive to solve rate as well as efficiency, which is
    correct: identifying the graph is the task, and being quick on the subset you happen to
    manage is not the same as being good at it.
    """
    return np.array([t.n_interventions if t.identified else budget for t in traces],
                    dtype=float)


def gap_closed(agent: List[EpisodeTrace], random_ref: List[EpisodeTrace],
               greedy_ref: List[EpisodeTrace], budget: int) -> float:
    """(random - agent) / (random - greedy), on per-episode cost including failures."""
    a = episode_costs(agent, budget).mean()
    r = episode_costs(random_ref, budget).mean()
    g = episode_costs(greedy_ref, budget).mean()
    denominator = r - g
    if abs(denominator) < 1e-9:
        return float("nan")  # the references are indistinguishable; the metric is undefined
    return float((r - a) / denominator)


def under_acting_rate(traces: List[EpisodeTrace]) -> float:
    """Fraction of episodes ended by passing while still unidentified -- giving up.

    A HARD FAIL above 10%, independent of gap closed. This is the NOOP collapse that ended
    the previous round, and it is invisible to the primary metric by construction.
    """
    return float(np.mean([t.passed_early for t in traces]))


def mean_regret(traces: List[EpisodeTrace]) -> float:
    """Mean oracle regret in nats, over INFORMATIVE steps only.

    Restricting to informative steps is essential, not a refinement. Where every target
    ties at zero information gain, any choice is trivially optimal and averaging those in
    measures nothing -- that is precisely what produced the retracted 99.4% agreement
    figure, which turned out to be 93-98% vacuous.
    """
    values = [r for t in traces for r, inf in zip(t.regrets, t.informative) if inf]
    return float(np.mean(values)) if values else float("nan")


def optimal_rate(traces: List[EpisodeTrace]) -> float:
    """Fraction of INFORMATIVE actions that matched the oracle's best set."""
    values = [o for t in traces for o, inf in zip(t.optimal, t.informative) if inf]
    return float(np.mean(values)) if values else float("nan")


def informative_fraction(traces: List[EpisodeTrace]) -> float:
    """What share of scored actions the oracle actually had a preference about.

    Reported so a high `optimal_rate` can never again be read without knowing how much of
    it was vacuous.
    """
    flat = [inf for t in traces for inf in t.informative]
    return float(np.mean(flat)) if flat else float("nan")


def stratify_by_mec_size(agent: List[EpisodeTrace], random_ref: List[EpisodeTrace],
                         greedy_ref: List[EpisodeTrace], budget: int,
                         bins=((1, 1), (2, 4), (5, 10 ** 9))) -> Dict:
    """Gap closed within equivalence-class size bands.

    Class size drives difficulty far more than graph size does (correlation 0.56 vs 0.29
    for edge count at d=4). A single average mixes episodes needing zero interventions with
    episodes needing three, so it cannot separate a capable agent from an easy draw. This
    also splits two distinct skills: knowing when not to act (singleton classes) and
    choosing well when you do (large classes).
    """
    out = {}
    for low, high in bins:
        def keep(ts):
            return [t for t in ts if low <= t.mec_size <= high]
        a, r, g = keep(agent), keep(random_ref), keep(greedy_ref)
        label = f"mec_{low}" if low == high else f"mec_{low}-{'inf' if high > 10 ** 8 else high}"
        if not a or not r or not g:
            out[label] = {"n": len(a), "gap_closed": float("nan")}
            continue
        out[label] = {
            "n": len(a),
            "gap_closed": gap_closed(a, r, g, budget),
            "solve_rate": float(np.mean([t.identified for t in a])),
            "mean_interventions": float(np.mean([t.n_interventions for t in a])),
        }
    return out


def evaluate(config: EnvConfig, agent_policy: Callable, random_ref: List[EpisodeTrace],
             greedy_ref: List[EpisodeTrace], n_episodes: int = 300, seed: int = 0,
             space=None, oracle: Optional[InterventionOracle] = None) -> Dict:
    """Full metric set for one policy against pre-computed references."""
    traces = run_episodes(config, agent_policy, n_episodes, seed, space, oracle)
    steps = mean_interventions_when_solved(traces)
    costs = episode_costs(traces, config.budget)
    return {
        "gap_closed": gap_closed(traces, random_ref, greedy_ref, config.budget),
        "solve_rate": float(np.mean([t.identified for t in traces])),
        "greedy_solve_rate": float(np.mean([t.identified for t in greedy_ref])),
        "mean_cost": float(costs.mean()),
        "cost_ci": bootstrap_ci(costs, seed=seed),
        "mean_interventions_when_solved": float(steps.mean()) if steps.size else float("nan"),
        "under_acting_rate": under_acting_rate(traces),
        "mean_regret": mean_regret(traces),
        "optimal_rate": optimal_rate(traces),
        "informative_fraction": informative_fraction(traces),
        "by_mec_size": stratify_by_mec_size(traces, random_ref, greedy_ref, config.budget),
        "_traces": traces,
    }


# --- the pinned criteria ------------------------------------------------------------

GAP_CLOSED_THRESHOLD = 0.80
UNDER_ACTING_THRESHOLD = 0.10
COLLAPSE_TOLERANCE = 0.10
# Added after the first smoke run, which exposed that failing episodes was a way to score
# well. `episode_costs` now charges failures at the budget, which closes the loophole in
# the primary metric; this is the belt-and-braces version, stated directly so a low solve
# rate can never be traded against apparent efficiency.
SOLVE_RATE_SHORTFALL = 0.05


def check_criteria(deterministic: Dict, sampled: Optional[Dict] = None) -> Dict:
    """Apply the criteria from docs/SA_PLAN.md to one seed's results.

    Returns per-criterion booleans plus an overall verdict. `sampled` is optional; without
    it the collapse check cannot run and is reported as None rather than silently passing.
    """
    gap = deterministic["gap_closed"]
    under = deterministic["under_acting_rate"]
    solve = deterministic["solve_rate"]
    greedy_solve = deterministic["greedy_solve_rate"]

    passes_gap = bool(gap >= GAP_CLOSED_THRESHOLD)
    passes_under = bool(under <= UNDER_ACTING_THRESHOLD)
    passes_solve = bool(solve >= greedy_solve - SOLVE_RATE_SHORTFALL)

    if sampled is None:
        passes_collapse = None
    else:
        passes_collapse = bool(gap >= sampled["gap_closed"] - COLLAPSE_TOLERANCE)

    checks = {
        "gap_closed": passes_gap,
        "no_under_acting": passes_under,
        "solve_rate": passes_solve,
        "no_collapse": passes_collapse,
    }
    return {
        "checks": checks,
        "passed": (passes_gap and passes_under and passes_solve
                   and (passes_collapse is not False)),
        "gap_closed": gap,
        "under_acting_rate": under,
        "solve_rate": solve,
        "greedy_solve_rate": greedy_solve,
        "sampled_gap_closed": None if sampled is None else sampled["gap_closed"],
    }


def summarise_seeds(per_seed: List[Dict], min_passing: int = 4) -> Dict:
    """Aggregate across training seeds.

    Reports the MINIMUM gap closed, not the mean. A mean hides a lucky run, which is the
    failure mode the previous project never caught -- and the criterion is about the
    technique being reliable, not about its best day.
    """
    gaps = np.array([s["gap_closed"] for s in per_seed], dtype=float)
    n_passed = int(sum(s["passed"] for s in per_seed))
    return {
        "n_seeds": len(per_seed),
        "n_passed": n_passed,
        "min_gap_closed": float(np.nanmin(gaps)) if gaps.size else float("nan"),
        "median_gap_closed": float(np.nanmedian(gaps)) if gaps.size else float("nan"),
        "max_gap_closed": float(np.nanmax(gaps)) if gaps.size else float("nan"),
        "passed": n_passed >= min_passing,
    }
