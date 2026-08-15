"""How many posterior samples does the oracle need to make the SAME CHOICE?

Reachability is not decomposable, so the subset DP cannot produce the descendant-set
distribution the oracle needs. The obvious alternative is Monte Carlo: sample DAGs from
the posterior, compute each one's descendants directly (O(d^3) per sample), and estimate
the distribution.

The question is not "how accurate is the estimated entropy" -- it is "does the oracle pick
the same target". Those are very different bars. Plug-in entropy from N samples is
downward-biased, but the bias is similar across nodes and largely cancels in a comparison,
so the choice can be right long before the entropies are.

Ground truth: the exact oracle on the exact posterior, available up to d=6.

Two agreement measures, because they answer different questions:
  strict  -- the MC argmax lies inside the exact tied-best SET (what the agent's own
             `optimal_rate` metric uses)
  regret  -- how much expected information gain is LOST by taking the MC choice instead of
             the best one. A disagreement between two near-identical targets costs nothing
             and should not be counted as a failure.
"""
import numpy as np

from sa.env import CausalDiscoveryEnv, EnvConfig
from sa.graphs import build_graph_space
from sa.oracle import InterventionOracle


def realistic_posteriors(d, n_episodes, space, seed=0):
    """Posteriors from actual episodes, at every step -- not synthetic Dirichlet draws.

    Sample efficiency depends on how concentrated the posterior is, and that changes a lot
    between step 0 and the end of an episode. Testing on synthetic posteriors would measure
    the wrong distribution.
    """
    env = CausalDiscoveryEnv(EnvConfig(d=d, n_obs=1000, budget=8), space=space)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_episodes):
        result = env.reset(seed=seed * 1000 + i)
        out.append(env.posterior.copy())
        while not result.done:
            result = env.step(int(rng.integers(0, d)))
            out.append(env.posterior.copy())
    return out


def mc_scores(oracle, posterior, n_samples, rng):
    """Oracle scores estimated from `n_samples` DAGs drawn from `posterior`."""
    idx = rng.choice(len(posterior), size=n_samples, p=posterior)
    d = oracle.space.d
    out = np.zeros(d)
    for node in range(d):
        counts = np.bincount(oracle.signatures[idx, node],
                             minlength=oracle.n_groups[node]).astype(float)
        mass = counts[counts > 0] / n_samples
        out[node] = float(-np.sum(mass * np.log(mass)))
    return out


def run(d, n_episodes=25, trials=8, seed=0):
    space = build_graph_space(d, fast=True)
    oracle = InterventionOracle(space)
    posteriors = realistic_posteriors(d, n_episodes, space, seed)
    rng = np.random.default_rng(seed + 99)

    exact = [oracle.best_targets(p) for p in posteriors]
    # Only posteriors where the oracle actually HAS a preference are informative; where
    # every target ties, any choice is trivially correct and averaging those in measures
    # nothing. This is the same vacuity trap that produced the retracted 99.4% figure.
    informative = [i for i, (s, best) in enumerate(exact)
                   if best.sum() < d and s.max() > 1e-9]
    print(f"d={d}: {len(posteriors)} posteriors, {len(informative)} informative "
          f"({len(informative)/len(posteriors):.0%})")

    print(f"  {'samples':>8} {'agreement':>10} {'mean regret':>12} {'max regret':>11}")
    for n_samples in (50, 100, 200, 500, 1000, 2000, 5000):
        hits, regrets = 0, []
        for _ in range(trials):
            for i in informative:
                scores_exact, best = exact[i]
                approx = mc_scores(oracle, posteriors[i], n_samples, rng)
                choice = int(np.argmax(approx))
                hits += bool(best[choice])
                regrets.append(float(scores_exact.max() - scores_exact[choice]))
        total = trials * len(informative)
        print(f"  {n_samples:>8} {hits/total:>9.1%} {np.mean(regrets):>12.4f} "
              f"{np.max(regrets):>11.4f}")


if __name__ == "__main__":
    for d in (4, 5, 6):
        run(d)
        print()
