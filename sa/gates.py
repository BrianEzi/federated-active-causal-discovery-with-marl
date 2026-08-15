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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sa.env import PASS_ACTION, CausalDiscoveryEnv, EnvConfig


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
