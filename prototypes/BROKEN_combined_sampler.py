"""BROKEN -- DO NOT USE. Systematically wrong; see prototypes/README.md.

Kept because docs/SA_EXPERIMENT_LOG.md cites measurements taken from it, and the root
cause was never found. Verified wrong against the exact posterior: total variation
0.0684 / 0.3888 / 0.1227 with 0 / 1 / 3 interventions at d=4, against the MH sampler's
0.0217 / 0.0085 / 0.0037 on identical data.
"""

"""Gibbs sweeps PLUS atomic edge reversals.

Diagnosis that produced this. Parent-set Gibbs alone froze: 3 distinct graphs in 500 draws,
against an exact posterior with effective support of 9.1 graphs whose top two entries are
tied at 0.3328 each. Tied entries mean Markov-equivalent DAGs, and moving between them
requires FLIPPING an edge -- which changes the parent sets of BOTH endpoints at once.

A single-node Gibbs update cannot do that. Dropping u from v's parents and adding v to u's
parents are two separate updates, and the state in between is exactly the low-probability
valley. So Gibbs, despite never rejecting anything, cannot cross precisely the gap that
matters. This is why it did WORSE than the single-edge MH chain, which had an atomic
reversal move all along.

The combination keeps both: Gibbs sweeps explore parent sets efficiently within a
structure, and atomic reversals move between Markov-equivalent structures. Both kernels
leave the same target invariant, so alternating them is valid.
"""
import numpy as np

from BROKEN_gibbs_sampler import _reachable_from


def _local(table, lookup, parents, v):
    return table[v, lookup[v][frozenset(parents)]]


def combined_sample(table, parent_sets, lookup, masks, d, n_samples,
                    burn_sweeps, thin_sweeps, rng, adj=None, n_rev=None):
    if adj is None:
        adj = np.zeros((d, d), dtype=bool)
    else:
        adj = adj.copy()
    if n_rev is None:
        n_rev = d * (d - 1)

    valid_len = [len(s) for s in parent_sets]
    parents = [set(np.flatnonzero(adj[:, v]).tolist()) for v in range(d)]
    samples = np.empty((n_samples, d, d), dtype=bool)
    kept, rev_accepted, rev_tried = 0, 0, 0

    for sweep in range(burn_sweeps + n_samples * thin_sweeps):
        # --- Gibbs sweep over parent sets -------------------------------------------
        for v in rng.permutation(d):
            adj[:, v] = False
            forbidden = _reachable_from(adj, v, d) | (1 << v)
            m = masks[v, : valid_len[v]]
            ok = (m & forbidden) == 0
            scores = table[v, : valid_len[v]][ok]
            w = np.exp(scores - scores.max())
            w /= w.sum()
            chosen = int(m[ok][rng.choice(len(w), p=w)])
            parents[v] = set()
            bits = chosen
            while bits:
                low = bits & -bits
                p = low.bit_length() - 1
                adj[p, v] = True
                parents[v].add(p)
                bits ^= low

        # --- atomic edge reversals ---------------------------------------------------
        # The move Gibbs structurally cannot make. Reversing u->v changes both endpoints'
        # parent sets simultaneously, so it crosses between Markov-equivalent graphs
        # without passing through the valley where the edge is absent.
        for _ in range(n_rev):
            edges = np.argwhere(adj)
            if not len(edges):
                break
            u, v = edges[rng.integers(len(edges))]
            u, v = int(u), int(v)
            before = _local(table, lookup, parents[u], u) + \
                _local(table, lookup, parents[v], v)

            adj[u, v] = False
            parents[v].discard(u)
            # v must not already reach u, or v->u closes a cycle.
            if (_reachable_from(adj, v, d) >> u) & 1:
                adj[u, v] = True
                parents[v].add(u)
                continue
            rev_tried += 1
            adj[v, u] = True
            parents[u].add(v)
            after = _local(table, lookup, parents[u], u) + \
                _local(table, lookup, parents[v], v)
            if rng.random() < np.exp(min(after - before, 700.0)):
                rev_accepted += 1
            else:
                adj[v, u] = False
                parents[u].discard(v)
                adj[u, v] = True
                parents[v].add(u)

        if sweep >= burn_sweeps and (sweep - burn_sweeps) % thin_sweeps == 0:
            if kept < n_samples:
                samples[kept] = adj
                kept += 1
    return samples[:kept], adj, rev_accepted / max(rev_tried, 1)
