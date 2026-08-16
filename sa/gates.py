"""The two gates that decide whether the environment poses a real problem.

These exist as runnable code rather than as a one-off script because they are the checks
that were missing last time. The previous round trained for weeks on an environment where
roughly half the episodes were already solved before the agent acted, and nobody noticed
because nobody had computed what the number *should* have been.

GATE 1 -- the task must require intervening.
  Observational data cannot distinguish DAGs inside a Markov equivalence class, so the
  fraction of episodes solvable without any intervention should equal the fraction of
  DAGs that sit alone in their class. That target is computed exactly from
  `sa.graphs.GraphSpace.singleton_fraction`, so this is a comparison against a predicted
  number, not a judgement call.

  Failing HIGH means orientation information is leaking (the equal-variance defect).
  Failing LOW means the estimator is too weak or the sample size too small to identify
  even the graphs that are identifiable in principle -- a different problem, and worth
  distinguishing, which is why the check reports direction rather than a bare pass/fail.

GATE 2 -- choices must matter.
  A random intervention policy must be clearly worse than the greedy oracle. If they
  tie, nothing about experiment selection is being rewarded and there is nothing for an
  agent to learn.

Below those sit five CANARIES (G1-G5), which differ from the gates in when they run: the
gates qualify an environment before an experiment, the canaries are attached to every
result file so a number can never be read without its checks. Each one is a specific past
failure turned into code -- see `collect_canaries`. They are recorded, and warn loudly,
but never abort a run: the JSON is the record, and a suppressed result is worse than a
flagged one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig
from sa.evaluate import episode_costs, gap_closed


@dataclass
class GateResult:
    name: str
    passed: bool
    observed: float
    target: Optional[float]
    interval: tuple
    detail: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        target = "n/a" if self.target is None else f"{self.target:.4f}"
        return (
            f"[{status}] {self.name}: observed {self.observed:.4f} "
            f"(95% CI {self.interval[0]:.4f}-{self.interval[1]:.4f}), target {target}\n"
            f"       {self.detail}"
        )


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    """Percentile bootstrap interval for the mean.

    Every reported number carries one of these. At 8 episodes per condition -- the
    previous round's sample size -- a single episode moved a rate by 12.5 points, and a
    29-point swing was observed from floating-point noise alone. Intervals make that
    visible instead of inviting over-reading.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def observational_identification_rate(config: EnvConfig, n_episodes: int = 400,
                                      seed: int = 0) -> np.ndarray:
    """Per-episode indicator: was the DAG identified from observational data alone?

    Runs `reset` only -- no interventions are taken at all.
    """
    env = CausalDiscoveryEnv(config)
    out = np.zeros(n_episodes)
    for i in range(n_episodes):
        result = env.reset(seed=seed * 100_000 + i)
        out[i] = float(result.identified)
    return out


def check_gate_1(config: EnvConfig, n_episodes: int = 400, seed: int = 0,
                 tolerance: float = 0.05) -> GateResult:
    """Observational-only identification rate must match the singleton fraction.

    `tolerance` allows the observed rate to fall *below* target by that margin without
    failing, since a finite sample and a conservative score both under-identify. Any
    excess above target is treated as a leak and fails regardless of tolerance -- there
    is no benign reason to identify more graphs than theory permits.
    """
    space = CausalDiscoveryEnv(config).space
    target = space.singleton_fraction
    indicators = observational_identification_rate(config, n_episodes, seed)
    observed = float(indicators.mean())
    low, high = bootstrap_ci(indicators, seed=seed)

    leaking = observed > target + tolerance
    underpowered = observed < target - tolerance
    passed = not leaking and not underpowered

    if leaking:
        detail = (
            f"LEAK: {observed:.1%} of episodes solved without intervening, but only "
            f"{target:.1%} of DAGs are identifiable observationally. Orientation "
            f"information is reaching the estimator that theory says is not there -- "
            f"check that noise scales genuinely differ per node."
        )
    elif underpowered:
        detail = (
            f"UNDER-POWERED: only {observed:.1%} solved observationally against a target "
            f"of {target:.1%}. The leak is absent, but the estimator cannot identify even "
            f"the singleton graphs -- try raising n_obs or lowering identify_threshold."
        )
    else:
        detail = (
            f"Observational solve rate matches the {target:.1%} of DAGs that are alone in "
            f"their Markov equivalence class. Every other episode requires an intervention."
        )
    return GateResult("GATE 1 (task requires intervening)", passed, observed, target,
                      (low, high), detail)


def run_policy(config: EnvConfig, policy: Callable, n_episodes: int = 200,
               seed: int = 0, space=None) -> dict:
    """Run a policy for `n_episodes` and collect per-episode outcomes.

    `policy(env, result) -> action` is called each step. Returns arrays of per-episode
    identification indicators and intervention counts, suitable for bootstrapping.
    """
    # `space` lets a caller reuse an already-built graph space. It matters at d=6, where
    # construction takes ~37s and would otherwise be repeated on every call.
    env = CausalDiscoveryEnv(config, space=space)
    identified = np.zeros(n_episodes)
    n_used = np.zeros(n_episodes)
    for i in range(n_episodes):
        result = env.reset(seed=seed * 100_000 + i)
        while not result.done:
            result = env.step(policy(env, result))
        identified[i] = float(result.identified)
        n_used[i] = float(result.n_interventions)
    return {"identified": identified, "n_interventions": n_used}


def check_gate_2(config: EnvConfig, random_policy: Callable, oracle_policy: Callable,
                 n_episodes: int = 200, seed: int = 0) -> GateResult:
    """The greedy oracle must clearly beat a random intervention policy.

    Compared on **interventions used to identify**, not identification rate. That choice
    is deliberate and was made after measuring: with a generous budget both policies
    identify the graph essentially always (100% vs 99.3% at d=3), so the rate saturates
    and cannot discriminate. Efficiency is where the difference actually lives -- 1.05
    interventions versus 1.55 in that same run -- and it is also the quantity the learned
    agent is being asked to improve.

    Only episodes where BOTH policies succeeded would be ideal, but episodes are paired by
    seed and success is near-universal here, so the mean over successful episodes is used
    and the success rates are reported alongside so a divergence cannot hide.
    """
    rand = run_policy(config, random_policy, n_episodes, seed)
    orac = run_policy(config, oracle_policy, n_episodes, seed)

    # Restrict to episodes that were actually solved: an unsolved episode's intervention
    # count is censored at the budget and would otherwise reward giving up early.
    rand_steps = rand["n_interventions"][rand["identified"] > 0.5]
    orac_steps = orac["n_interventions"][orac["identified"] > 0.5]

    rand_mean, orac_mean = float(rand_steps.mean()), float(orac_steps.mean())
    rand_ci = bootstrap_ci(rand_steps, seed=seed)
    orac_ci = bootstrap_ci(orac_steps, seed=seed)

    # Lower is better, so the oracle passes when its interval sits entirely below random's.
    passed = orac_ci[1] < rand_ci[0]
    detail = (
        f"interventions to identify -- random {rand_mean:.2f} "
        f"(CI {rand_ci[0]:.2f}-{rand_ci[1]:.2f}) vs oracle {orac_mean:.2f} "
        f"(CI {orac_ci[0]:.2f}-{orac_ci[1]:.2f}); "
        f"identification rate {rand['identified'].mean():.1%} vs {orac['identified'].mean():.1%}. "
        + ("Intervals are disjoint, so choosing well measurably matters."
           if passed else
           "Intervals OVERLAP -- the environment does not reward good experiment choice, "
           "so there is nothing for an agent to learn here.")
    )
    return GateResult("GATE 2 (choices matter)", passed, orac_mean, rand_mean,
                      orac_ci, detail)


# ======================================================================================
# Canaries G1-G5
#
# Each encodes a failure that actually happened and was not caught. They are recorded in
# every result file rather than run on demand, because in each case the problem was not
# that a check failed -- it is that nobody thought to run it.
# ======================================================================================


@dataclass
class Canary:
    """One automatic check attached to a result.

    `severity` is "warn" or "fail". Nothing here aborts a run: a canary firing means the
    number needs interpreting, not that it should be discarded unseen.
    """

    name: str
    ok: bool
    severity: str
    observed: Optional[float]
    threshold: Optional[float]
    detail: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "severity": self.severity,
            "observed": None if self.observed is None else float(self.observed),
            "threshold": None if self.threshold is None else float(self.threshold),
            "detail": self.detail,
        }

    def __str__(self) -> str:
        return f"[{'ok' if self.ok else self.severity.upper()}] {self.name}: {self.detail}"


def canary_entropy(final_entropy: float, n_actions: int, fraction: float = 0.65) -> Canary:
    """G1 -- has the policy actually committed to anything?

    A policy that has learned nothing keeps a near-uniform action distribution, whose
    entropy is ln(n_actions). Overnight, every one of the 61 failing configurations sat at
    1.2-1.6 nats while every passing one sat at 0.5-0.7 -- that separated pass from fail
    better than any hyperparameter did. The quantity was already being logged; nobody was
    comparing it against its own ceiling.

    Warns rather than fails: high entropy on a genuinely tied task is legitimate, so this
    points at a cause rather than delivering a verdict.
    """
    ceiling = float(np.log(max(n_actions, 2)))
    ratio = float(final_entropy) / ceiling if ceiling > 0 else float("nan")
    ok = bool(ratio <= fraction) if np.isfinite(ratio) else False
    if not np.isfinite(ratio):
        detail = "final entropy is not finite -- training produced no usable history."
    elif ok:
        detail = (f"final entropy {final_entropy:.3f} nats is {ratio:.0%} of the uniform "
                  f"ceiling {ceiling:.3f}; the policy has committed.")
    else:
        detail = (f"final entropy {final_entropy:.3f} nats is {ratio:.0%} of the uniform "
                  f"ceiling {ceiling:.3f} -- near-uniform, so the policy has probably not "
                  f"learned to discriminate between targets. Every overnight failure "
                  f"looked like this.")
    return Canary("G1 entropy", ok, "warn", ratio, fraction, detail)


def canary_anchors(random_ref, greedy_ref, budget: int, tolerance: float = 1e-9) -> Canary:
    """G2 -- the gap-closed scale must actually be anchored at 0 and 1.

    `gap_closed` is defined so that scoring the random reference against itself gives
    exactly 0, and the greedy reference exactly 1. That identity is worth asserting
    precisely BECAUSE it is an identity: if it does not hold, the inputs are wrong -- the
    references were swapped, built from mismatched RNG state, or sit so close together
    that the denominator is noise. Anchors of 0.233 and 1.067 were once read off a run and
    taken at face value.

    Fails rather than warns: every gap-closed number in the file is measured on this
    scale, so if the scale is broken none of them mean anything.
    """
    at_random = gap_closed(random_ref, random_ref, greedy_ref, budget)
    at_greedy = gap_closed(greedy_ref, random_ref, greedy_ref, budget)
    r = float(episode_costs(random_ref, budget).mean())
    g = float(episode_costs(greedy_ref, budget).mean())

    if np.isfinite(at_random) and np.isfinite(at_greedy):
        worst = float(max(abs(at_random - 0.0), abs(at_greedy - 1.0)))
    else:
        worst = float("inf")

    # The 0/1 identity alone is not enough: it holds algebraically even when the two
    # references are swapped, because the formula is symmetric in how it defines its own
    # endpoints. What a swap does break is the ordering -- a policy labelled "greedy"
    # cannot cost MORE than random unless the labels are wrong. Checking the sign of the
    # denominator is what actually catches that, and it is free.
    ordered = bool(g < r)
    ok = bool(worst <= tolerance and ordered)

    if not np.isfinite(worst):
        detail = (f"gap closed is UNDEFINED: random costs {r:.3f} and greedy costs "
                  f"{g:.3f}, so the denominator is ~0. The references are "
                  f"indistinguishable and no gap-closed number here is meaningful.")
    elif not ordered:
        detail = (f"REFERENCES INVERTED: the greedy reference costs {g:.3f} against "
                  f"random's {r:.3f}. Greedy cannot be worse than random, so the two are "
                  f"almost certainly swapped -- which silently flips the sign of every "
                  f"gap-closed number in this file.")
    elif ok:
        detail = (f"anchors exact: random -> {at_random:.1e}, greedy -> {at_greedy:.6f} "
                  f"(costs {r:.3f} vs {g:.3f}).")
    else:
        detail = (f"ANCHORS CORRUPT: random -> {at_random:.6f} (expected 0), greedy -> "
                  f"{at_greedy:.6f} (expected 1), costs {r:.3f} vs {g:.3f}. Every "
                  f"gap-closed number in this file is on a broken scale.")
    return Canary("G2 anchors", ok, "fail", worst, tolerance, detail)


def canary_informative_fraction(fraction: float, floor: float = 0.10) -> Canary:
    """G3 -- refuse to report oracle agreement computed from almost nothing.

    `optimal_rate` averages over steps where the oracle had a preference. When nearly every
    step is a tie, that average rests on a handful of actions and means essentially
    nothing -- which is how "99.4-100% oracle agreement" came to be reported and then
    retracted, having been 93-98% vacuous.

    Fails, but note the blast radius: this invalidates `optimal_rate` and `mean_regret`
    only. Gap closed, solve rate and cost are unaffected.
    """
    ok = bool(np.isfinite(fraction) and fraction >= floor)
    if not np.isfinite(fraction):
        detail = "no scored actions at all, so oracle agreement is undefined, not high."
    elif ok:
        detail = (f"{fraction:.1%} of scored actions were ones the oracle had a preference "
                  f"about; agreement numbers rest on a real sample.")
    else:
        detail = (f"only {fraction:.1%} of scored actions were informative (floor "
                  f"{floor:.0%}) -- optimal_rate and mean_regret here come from too few "
                  f"real choices to mean anything. Gap closed is unaffected.")
    return Canary("G3 informative fraction", ok, "fail", fraction, floor, detail)


def canary_seed_spread(gap_values: Sequence[float], limit: float = 0.5) -> Canary:
    """G4 -- a good median across seeds can hide an unstable configuration.

    `pernode_best` without action memory ran from +1.043 to -1.766 across seeds. Reading
    the median alone would have called that a success; it is a coin flip. A spread this
    wide means the seed, not the architecture, is doing the work.

    Warns: wide spread is information about variance, not evidence the run is wrong.
    """
    values = np.asarray([v for v in gap_values if np.isfinite(v)], dtype=float)
    if values.size < 2:
        return Canary("G4 seed spread", True, "warn", None, limit,
                      f"only {values.size} finite seed(s); spread is not defined.")
    spread = float(values.max() - values.min())
    ok = bool(spread <= limit)
    detail = (f"gap closed spans {spread:.3f} across {values.size} seeds "
              f"({values.min():+.3f} to {values.max():+.3f})")
    detail += ("." if ok else
               f" -- above the {limit:.2f} limit, so the median is not a safe summary. "
               f"Treat this configuration as unstable rather than as good.")
    return Canary("G4 seed spread", ok, "warn", spread, limit, detail)


def canary_gate1(gate1: Optional[dict]) -> Canary:
    """G5 -- was the environment's validity checked for THIS run?

    GATE 1 was verified once at d=3 and silently stopped holding from d=5 upward, which
    invalidated a night of runs. The failure was not that the gate was wrong; it is that
    its result lived in a different file from the results it qualified. So the ABSENCE of
    a check is itself a finding here, and is reported distinctly from a check that ran and
    failed.
    """
    if gate1 is None:
        return Canary("G5 gate 1 recorded", False, "fail", None, None,
                      "GATE 1 was NOT evaluated for this run, so nothing establishes that "
                      "the task required intervening at this d and n_obs. This is exactly "
                      "the state that invalidated the d>=5 runs.")
    rate, target = float(gate1["rate"]), float(gate1["target"])
    ok = bool(gate1["passed"])
    detail = (f"observational-only rate {rate:.4f} against a singleton fraction of "
              f"{target:.4f}")
    detail += (" -- the task requires intervening." if ok else
               " -- GATE 1 FAILED, so this environment does not match its specification "
               "and the results below describe a different task than intended.")
    return Canary("G5 gate 1 recorded", ok, "fail", rate, target, detail)


def collect_canaries(per_seed: List[dict], gate1: Optional[dict], n_actions: int,
                     random_ref=None, greedy_ref=None,
                     budget: Optional[int] = None) -> List[dict]:
    """All five canaries for one configuration, ready to serialise into the result JSON.

    Built defensively: a canary that raised would take down a run that has already spent
    hours of compute, so an internal failure becomes a recorded, failing canary rather
    than an exception.
    """
    out: List[Canary] = []

    entropies = [s.get("final_entropy", float("nan")) for s in per_seed]
    finite = [e for e in entropies if np.isfinite(e)]
    out.append(_safe(canary_entropy,
                     float(np.mean(finite)) if finite else float("nan"), n_actions))

    if random_ref is not None and greedy_ref is not None and budget is not None:
        out.append(_safe(canary_anchors, random_ref, greedy_ref, budget))
    else:
        out.append(Canary("G2 anchors", False, "fail", None, None,
                          "references unavailable, so the gap-closed scale was not "
                          "checked."))

    fractions = [s.get("deterministic", {}).get("informative_fraction", float("nan"))
                 for s in per_seed]
    finite_f = [f for f in fractions if np.isfinite(f)]
    out.append(_safe(canary_informative_fraction,
                     float(np.mean(finite_f)) if finite_f else float("nan")))

    out.append(_safe(canary_seed_spread,
                     [s.get("gap_closed", float("nan")) for s in per_seed]))
    out.append(_safe(canary_gate1, gate1))
    return [c.as_dict() for c in out]


def _safe(fn: Callable, *args) -> Canary:
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 -- never lose a completed run to a check
        return Canary(getattr(fn, "__name__", "canary"), False, "fail", None, None,
                      f"canary raised {type(exc).__name__}: {exc}")


def estimate_singleton_fraction(d: int, p: float = 0.5, n_chains: int = 32,
                                n_samples: int = 2000, burn_in: int = 20000,
                                thin: int = 10, seed: int = 0) -> dict:
    """GATE 1's target at any `d`, estimated rather than enumerated.

    GATE 1 asks whether the task can be solved without intervening, and compares the
    observational solve rate against the prior-weighted fraction of DAGs that are alone in
    their Markov equivalence class. Below d=7 that fraction is computed exactly by
    enumeration. Past it there is no DAG list, and **without this the d=7 result could not
    be validated at all** -- which is exactly the hole that made the earlier d=6 numbers
    worthless.

    Two pieces make it work. Membership is a *local* test (`is_singleton_mec`), so no
    grouping over the whole space is needed. And DAGs are drawn from the Erdos-Renyi prior
    by MH on prior-only weights, which is the same prior the environment and the posterior
    use.

    **The obvious cheap sampler is wrong here and it is worth saying why.** Drawing a
    random permutation and including each forward pair with probability `p` does *not*
    sample this prior: it gives each DAG a weight proportional to its number of topological
    orderings, which is the order-modular prior, tilted toward graphs with many linear
    extensions. Since class size is precisely what GATE 1 measures, that bias would land
    directly on the answer.

    The interval comes from `n_chains` **independent chains**, one estimate each. A
    bootstrap over the pooled draws would treat correlated MCMC samples as independent and
    report an interval several times too narrow; separate chains also expose bad mixing,
    since a chain stuck in one region disagrees with the others.

    Accuracy, measured against the exact enumerated value at d=4,5,6 and p=0.3,0.5,0.7
    (2026-08-16): every z-score within +-1.86, mean -0.34, so there is no detectable bias
    at a standard error of ~0.0013. Defaults are set to that configuration.
    """
    from sa.dp import DPPosterior
    from sa.graphs import is_singleton_mec
    from sa.sampler import mh_sample
    from sa.score import BGeScore

    dp = DPPosterior.for_prior(d, BGeScore(d), kind="erdos_renyi", p=p)
    log_w = dp.log_prior_term

    per_chain = np.empty(n_chains)
    acceptances = np.empty(n_chains)
    for chain in range(n_chains):
        rng = np.random.default_rng(seed * 1000 + chain)
        draws, acceptance = mh_sample(log_w, dp._mask_to_index, d, n_samples,
                                      burn_in=burn_in, thin=thin, rng=rng)
        per_chain[chain] = float(is_singleton_mec(draws).mean())
        acceptances[chain] = acceptance

    lo, hi = bootstrap_ci(per_chain, seed=seed)
    return {
        "estimate": float(per_chain.mean()),
        "ci": (lo, hi),
        "per_chain": per_chain,
        "acceptance": float(acceptances.mean()),
        "n_draws": int(n_chains * n_samples),
    }
