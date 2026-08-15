"""Sampling DAGs from the posterior WITHOUT enumerating them.

The Monte Carlo oracle needs ~1000 posterior samples per step. At d<=6 those can be drawn
by enumerating the posterior, which is exactly what we are trying to stop doing.

Metropolis-Hastings needs only score RATIOS, and the score decomposes per node, so a
single-edge move changes at most two nodes' local terms. No normalising constant, no
enumeration -- it needs only the local score table we already build.

Correctness is measured, not assumed: the sampler's edge marginals are compared against
the exact enumerated ones at d=4, 5 and 6, where ground truth exists.
"""
import time
from itertools import combinations

import numpy as np

from sa.graphs import build_graph_space
from sa.posterior import PosteriorEngine
from sa.score import BGeScore


def parent_index_maps(d):
    """(list of parent-set tuples per node, dict mapping frozenset -> index)."""
    sets, lookup = [], []
    for node in range(d):
        others = [k for k in range(d) if k != node]
        s = [c for r in range(len(others) + 1) for c in combinations(others, r)]
        sets.append(s)
        lookup.append({frozenset(c): i for i, c in enumerate(s)})
    return sets, lookup


def creates_cycle(adj, u, v):
    """Would adding u->v create a cycle? True iff u is already reachable from v."""
    d = adj.shape[0]
    seen = np.zeros(d, dtype=bool)
    stack = [v]
    while stack:
        x = stack.pop()
        if x == u:
            return True
        if seen[x]:
            continue
        seen[x] = True
        stack.extend(np.flatnonzero(adj[x]))
    return False


def mh_sample(table, lookup, d, n_samples, burn_in, thin, rng, adj=None):
    """Single-edge add / delete / reverse moves, accepted by exact local score ratio."""
    if adj is None:
        adj = np.zeros((d, d), dtype=bool)
    parents = [set() for _ in range(d)]
    for v in range(d):
        parents[v] = set(np.flatnonzero(adj[:, v]).tolist())

    def local(v):
        return table[v, lookup[v][frozenset(parents[v])]]

    samples, accepted, total = [], 0, 0
    pairs = [(u, v) for u in range(d) for v in range(d) if u != v]
    n_steps = burn_in + n_samples * thin
    for step in range(n_steps):
        u, v = pairs[rng.integers(len(pairs))]
        total += 1

        if adj[u, v]:                                   # delete u->v
            before = local(v)
            parents[v].discard(u)
            if rng.random() < np.exp(local(v) - before):
                adj[u, v] = False
                accepted += 1
            else:
                parents[v].add(u)
        elif adj[v, u]:                                 # reverse v->u  =>  u->v
            before = local(u) + local(v)
            parents[u].discard(v)
            parents[v].add(u)
            ok = not creates_cycle_after(adj, v, u, u, v)
            if ok and rng.random() < np.exp(local(u) + local(v) - before):
                adj[v, u] = False
                adj[u, v] = True
                accepted += 1
            else:
                parents[u].add(v)
                parents[v].discard(u)
        else:                                           # add u->v
            if not creates_cycle(adj, u, v):
                before = local(v)
                parents[v].add(u)
                if rng.random() < np.exp(local(v) - before):
                    adj[u, v] = True
                    accepted += 1
                else:
                    parents[v].discard(u)

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(adj.copy())
    return np.array(samples), accepted / max(total, 1)


def creates_cycle_after(adj, du, dv, au, av):
    """Cycle check for a reversal: drop du->dv, then test adding au->av."""
    tmp = adj.copy()
    tmp[du, dv] = False
    return creates_cycle(tmp, au, av)


def run(d, n_obs=500, n_samples=4000, seed=0):
    space = build_graph_space(d, fast=True)
    engine = PosteriorEngine(space, BGeScore(d))
    rng = np.random.default_rng(seed)
    samples_data = rng.normal(size=(n_obs, d))
    intervened = np.zeros((n_obs, d))
    intervened[: n_obs // 5, 1] = 1.0

    table = engine.local_score_table(samples_data, intervened)
    post = engine.posterior(samples_data, intervened,
                            np.full(space.n_dags, 1.0 / space.n_dags))
    exact = engine.edge_marginals(post)

    _, lookup = parent_index_maps(d)
    t0 = time.perf_counter()
    draws, acc = mh_sample(table, lookup, d, n_samples, burn_in=5000, thin=10,
                           rng=np.random.default_rng(seed + 1))
    elapsed = time.perf_counter() - t0
    approx = draws.mean(axis=0)

    print(f"d={d}  max|edge marginal error|={np.abs(exact-approx).max():.4f}  "
          f"mean={np.abs(exact-approx).mean():.4f}  accept={acc:.2f}  "
          f"{n_samples} draws in {elapsed:.1f}s")


if __name__ == "__main__":
    for d in (4, 5, 6):
        run(d)
