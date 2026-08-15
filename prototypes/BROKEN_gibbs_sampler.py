"""BROKEN -- DO NOT USE. Systematically wrong; see prototypes/README.md.

Kept because docs/SA_EXPERIMENT_LOG.md cites measurements taken from it, and the root
cause was never found. Verified wrong against the exact posterior: total variation
0.0684 / 0.3888 / 0.1227 with 0 / 1 / 3 interventions at d=4, against the MH sampler's
0.0217 / 0.0085 / 0.0037 on identical data.
"""

"""Parent-set Gibbs sampling: resample a whole parent set at once, exactly.

Why the single-edge chain mixed badly. To turn u->v into v->u it must pass through an
intermediate state with the edge absent, and that state is usually much less probable than
either endpoint. The chain is therefore stuck between two good configurations separated by
a bad one, which is exactly what 6-12% acceptance looks like.

The fix exploits something specific to our setup: we ALREADY compute the local score of
every parent set for every node -- that is the score table, d * 2^(d-1) entries. So instead
of proposing one edge and accepting or rejecting, we can sample a node's ENTIRE parent set
directly from its exact conditional distribution:

    p(parents of v | everything else) proportional to exp(local_score(v, Pa)) * [acyclic]

Because the score decomposes per node, no other node's term depends on v's parents, so this
conditional is exact. Sampling from it is a Gibbs step: always accepted, no MH ratio, and a
single move can cross the gap the edge chain got stuck in.

Acyclicity is the only coupling. With v's incoming edges removed, adding a parent p is safe
exactly when p is not reachable FROM v -- otherwise p -> v closes a loop. So the allowed
parents are the non-descendants, and we restrict the conditional to parent sets inside that
set. This keeps the chain on DAGs by construction rather than by rejection.
"""
import numpy as np


def parent_mask_table(parent_sets, d):
    """[d, n_parent_sets] bitmask of each parent set, for fast subset filtering."""
    n = max(len(s) for s in parent_sets)
    masks = np.zeros((d, n), dtype=np.int64)
    for node in range(d):
        for i, parents in enumerate(parent_sets[node]):
            m = 0
            for p in parents:
                m |= 1 << p
            masks[node, i] = m
    return masks


def _reachable_from(adj, v, d):
    """Bitmask of nodes reachable from v (v excluded), on the CURRENT graph."""
    seen = 0
    stack = [v]
    visited = np.zeros(d, dtype=bool)
    while stack:
        x = stack.pop()
        if visited[x]:
            continue
        visited[x] = True
        if x != v:
            seen |= 1 << x
        stack.extend(np.flatnonzero(adj[x]).tolist())
    return seen


def gibbs_sample(table, parent_sets, masks, d, n_samples, burn_in_sweeps, thin_sweeps,
                 rng, adj=None):
    """Return [n_samples, d, d] boolean adjacency matrices drawn by parent-set Gibbs."""
    if adj is None:
        adj = np.zeros((d, d), dtype=bool)
    else:
        adj = adj.copy()

    n_sets = masks.shape[1]
    valid_len = [len(s) for s in parent_sets]
    samples = np.empty((n_samples, d, d), dtype=bool)
    kept = 0
    sweeps = burn_in_sweeps + n_samples * thin_sweeps

    for sweep in range(sweeps):
        for v in rng.permutation(d):
            adj[:, v] = False                       # detach v from its parents
            forbidden = _reachable_from(adj, v, d) | (1 << v)
            allowed = ~forbidden

            m = masks[v, : valid_len[v]]
            ok = (m & ~allowed) == 0                # parent set inside the allowed nodes
            scores = table[v, : valid_len[v]][ok]
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()

            chosen = int(masks[v, : valid_len[v]][ok][rng.choice(len(weights), p=weights)])
            bits = chosen
            while bits:
                low = bits & -bits
                adj[low.bit_length() - 1, v] = True
                bits ^= low

        if sweep >= burn_in_sweeps and (sweep - burn_in_sweeps) % thin_sweeps == 0:
            if kept < n_samples:
                samples[kept] = adj
                kept += 1
    return samples[:kept], adj
