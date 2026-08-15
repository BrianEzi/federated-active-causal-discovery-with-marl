"""The acceptance test: does the oracle built on MH samples make the same CHOICES?

Everything so far validated pieces. The subset DP reproduces Z and edge marginals exactly.
Monte Carlo over posterior samples reproduces the oracle's choice to within 0.0009 nats of
regret -- but that used samples drawn by enumerating the posterior, which is the thing we
are trying to avoid. The MH sampler removes the enumeration, but was only checked against
EDGE MARGINALS, which are per-edge quantities.

The oracle needs the distribution over DESCENDANT SETS -- a joint property of the whole
graph. A chain can reproduce marginals correctly while getting joint structure wrong, so
that check does not transfer. This runs the real thing end to end:

    local score table  ->  MH samples  ->  descendants per sample  ->  entropy  ->  choice

and compares the choice against the exact oracle on the exact posterior.

No enumeration anywhere in the pipeline under test. Enumeration appears only to produce the
ground truth being compared against, which is available up to d=6.
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mh_sampler import mh_sample, parent_index_maps

from sa.env import CausalDiscoveryEnv, EnvConfig
from sa.graphs import build_graph_space
from sa.oracle import InterventionOracle
from sa.posterior import PosteriorEngine
from sa.score import BGeScore


def descendant_codes(adjacencies, d):
    """[N, d] -- each sampled DAG's reachability row per node, packed into an integer.

    Same transitive closure the exact oracle uses, run on a batch of sampled graphs rather
    than on every enumerated graph. Two graphs share a code at node i exactly when
    intervening on i cannot distinguish them.
    """
    reach = adjacencies.astype(bool).copy()
    for k in range(d):
        reach |= reach[:, :, k][:, :, None] & reach[:, k, :][:, None, :]
    bit = (1 << np.arange(d)).astype(np.int64)
    return reach.astype(np.int64) @ bit


def mc_oracle_scores(adjacencies, d):
    """[d] expected information gain per node, estimated from sampled graphs."""
    codes = descendant_codes(adjacencies, d)
    out = np.zeros(d)
    n = len(adjacencies)
    for node in range(d):
        _, counts = np.unique(codes[:, node], return_counts=True)
        mass = counts / n
        out[node] = float(-np.sum(mass * np.log(mass)))
    return out


def capture_episode_states(d, n_episodes, space, engine, seed=0):
    """Realistic (score table, exact posterior) pairs from actual episodes."""
    env = CausalDiscoveryEnv(EnvConfig(d=d, n_obs=1000, budget=8), space=space)
    rng = np.random.default_rng(seed)
    states = []
    for i in range(n_episodes):
        result = env.reset(seed=seed * 1000 + i)
        states.append((engine.local_score_table(env.samples, env.intervened),
                       env.posterior.copy()))
        while not result.done:
            result = env.step(int(rng.integers(0, d)))
            states.append((engine.local_score_table(env.samples, env.intervened),
                           env.posterior.copy()))
    return states


def run(d, n_episodes=12, n_samples=1000, thin=10, burn_in=3000, warm_start=True, seed=0):
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    oracle = InterventionOracle(space)
    states = capture_episode_states(d, n_episodes, space, engine, seed)

    _, lookup = parent_index_maps(d)
    rng = np.random.default_rng(seed + 7)

    hits, regrets, elapsed = 0, [], 0.0
    informative = 0
    previous = None
    for table, posterior in states:
        scores_exact, best = oracle.best_targets(posterior)
        # Only steps where the oracle genuinely has a preference. Where everything ties,
        # any choice is trivially correct and averaging those in measures nothing.
        if best.sum() >= d or scores_exact.max() <= 1e-9:
            previous = None
            continue
        informative += 1

        t0 = time.perf_counter()
        draws, _ = mh_sample(table, lookup, d, n_samples, burn_in, thin, rng,
                             adj=previous.copy() if (warm_start and previous is not None)
                             else None)
        elapsed += time.perf_counter() - t0
        # Warm start the next step's chain from this step's final graph: the posterior
        # changes only slightly between steps, so the chain should not need to find the
        # high-mass region again from empty.
        previous = draws[-1] if warm_start else None

        approx = mc_oracle_scores(draws, d)
        choice = int(np.argmax(approx))
        hits += bool(best[choice])
        regrets.append(float(scores_exact.max() - scores_exact[choice]))

    label = "warm" if warm_start else "cold"
    print(f"  d={d} {label:<5} n={n_samples:<5} agreement {hits/informative:>6.1%}   "
          f"mean regret {np.mean(regrets):.4f}   max {np.max(regrets):.4f}   "
          f"{elapsed/informative*1e3:6.0f} ms/step   ({informative} informative steps)")


if __name__ == "__main__":
    print("MH-sampled oracle vs exact oracle -- choices, not entropies")
    for d in (4, 5, 6):
        run(d, warm_start=True)
    print()
    for d in (5,):
        run(d, warm_start=False)
        run(d, n_samples=3000, warm_start=True)
