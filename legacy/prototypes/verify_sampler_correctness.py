"""Is the sampler sampling the right distribution at all?

Everything so far measured samplers through the oracle, which conflates two questions.
This asks the direct one: draw many graphs, count how often each DISTINCT graph appears,
and compare against its exact posterior probability. At d=4 there are only 543 graphs, so
this is a complete comparison with nothing hidden.

If a sampler is correct, sampled frequency tracks exact posterior probability. If it does
not, no amount of extra sampling or better mixing will fix the oracle built on top.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from BROKEN_combined_sampler import combined_sample
from BROKEN_gibbs_sampler import parent_mask_table
from mh_sampler import mh_sample, parent_index_maps

from sa.graphs import build_graph_space
from sa.posterior import PosteriorEngine
from sa.score import BGeScore

d = 4
space = build_graph_space(d, fast=True)
engine = PosteriorEngine(space, BGeScore(d))
rng = np.random.default_rng(0)
data = rng.normal(size=(300, d))
intervened = np.zeros((300, d))

table = engine.local_score_table(data, intervened)
prior = np.full(space.n_dags, 1.0 / space.n_dags)
posterior = engine.posterior(data, intervened, prior)

# Map each enumerated DAG to a key so sampled graphs can be looked up.
keys = {tuple(np.asarray(g).astype(bool).ravel().tolist()): i
        for i, g in enumerate(space.dags)}

masks = parent_mask_table(engine.parent_sets, d)
_, lookup = parent_index_maps(d)

N = 20000
mh_draws, _ = mh_sample(table, lookup, d, N, 5000, 5, np.random.default_rng(1))
comb_draws, _, _ = combined_sample(table, engine.parent_sets, lookup, masks, d,
                                   N, 50, 1, np.random.default_rng(2))

for name, draws in (("MH", mh_draws), ("Gibbs+rev", comb_draws)):
    counts = np.zeros(space.n_dags)
    missing = 0
    for g in draws:
        k = tuple(g.astype(bool).ravel().tolist())
        if k in keys:
            counts[keys[k]] += 1
        else:
            missing += 1
    freq = counts / max(counts.sum(), 1)
    order = np.argsort(posterior)[::-1][:6]
    print(f"\n{name}:  {missing} draws not found in the enumerated space")
    print(f"  total variation distance = {0.5*np.abs(freq-posterior).sum():.4f}")
    print(f"  {'graph':>6} {'exact':>9} {'sampled':>9}")
    for i in order:
        print(f"  {i:>6} {posterior[i]:>9.4f} {freq[i]:>9.4f}")
