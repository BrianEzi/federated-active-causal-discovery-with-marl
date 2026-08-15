"""How does the MH chain's cost scale with d, to hold oracle error fixed?

Only d <= 6 can be checked against ground truth, so the plan is: measure the chain length
needed at 4, 5 and 6, see how it grows, and extrapolate with the assumption stated rather
than hidden.

Two numbers per configuration:
  regret  -- information lost by taking the sampled oracle's choice instead of the exact
             oracle's. The bar is the IDEAL sampler (draws taken from the enumerated
             posterior), which scored 0.0010 / 0.0009 / 0.0009 at d=4/5/6 with 1000 draws.
             That is the floor Monte Carlo imposes; MH cannot beat it, only approach it.
  ms/step -- what it would cost inside an RL loop, where this runs once per environment
             step.
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mh_sampler import mh_sample, parent_index_maps
from verify_oracle_sampling import capture_episode_states, mc_oracle_scores

from sa.graphs import build_graph_space
from sa.oracle import InterventionOracle
from sa.posterior import PosteriorEngine
from sa.score import BGeScore

IDEAL = {4: 0.0010, 5: 0.0009, 6: 0.0009}


def run(d, n_episodes=10, seed=0):
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    oracle = InterventionOracle(space)
    states = capture_episode_states(d, n_episodes, space, engine, seed)
    _, lookup = parent_index_maps(d)

    usable = []
    for table, posterior in states:
        scores_exact, best = oracle.best_targets(posterior)
        if best.sum() < d and scores_exact.max() > 1e-9:
            usable.append((table, scores_exact, best))

    print(f"d={d}  ({len(usable)} informative steps, ideal-sampler regret "
          f"{IDEAL[d]:.4f})")
    for n_samples, burn, thin in ((1000, 3000, 10), (4000, 5000, 10), (16000, 10000, 5)):
        rng = np.random.default_rng(seed + 7)
        hits, regrets, elapsed = 0, [], 0.0
        previous = None
        for table, scores_exact, best in usable:
            t0 = time.perf_counter()
            draws, _ = mh_sample(table, lookup, d, n_samples, burn, thin, rng,
                                 adj=previous)
            elapsed += time.perf_counter() - t0
            previous = draws[-1]
            choice = int(np.argmax(mc_oracle_scores(draws, d)))
            hits += bool(best[choice])
            regrets.append(float(scores_exact.max() - scores_exact[choice]))
        mean_regret = float(np.mean(regrets))
        print(f"   n={n_samples:<6} burn={burn:<6} agreement {hits/len(usable):>6.1%}   "
              f"regret {mean_regret:.4f}  ({mean_regret/IDEAL[d]:>5.1f}x ideal)   "
              f"max {np.max(regrets):.4f}   {elapsed/len(usable)*1e3:7.0f} ms/step")


if __name__ == "__main__":
    for d in (4, 5, 6):
        run(d)
