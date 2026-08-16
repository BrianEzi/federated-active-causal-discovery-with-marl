"""Sampling DAGs from the posterior, for the things the DP cannot reach.

The subset DP in `sa/dp.py` gives the partition function and every edge marginal exactly,
which covers the belief state. It cannot give **reachability**: whether node `u` is an
ancestor of node `v` is a property of a whole path, not of any node's parent set, so it
does not decompose and the recurrence has no way to express it. The greedy
information-gain oracle needs exactly that -- it groups hypotheses by which nodes are
downstream of the intervention target.

So the oracle needs samples. Metropolis-Hastings supplies them from the same local score
table the DP uses: a single-edge move changes at most two nodes' local terms, so the
acceptance ratio is two table lookups, with no normalising constant and no enumeration.

**Verified against ground truth directly.** Its sampled edge marginals are compared with
the exact enumerated ones, and its oracle *choices* with the exact oracle's, at d = 4, 5
and 6. That order matters: on 2026-08-15 a parent-set Gibbs sampler was measured only
through the oracle for three rounds, which conflated a correctness bug with slow mixing.
The direct check settled it in one run (total variation 0.068/0.389/0.123 for Gibbs against
0.022/0.009/0.004 for this sampler), and the Gibbs version was discarded.

Move set is add / delete / reverse. Reversal is not redundant with add-then-delete: an
edge flip changes two nodes' parent sets at once, and the intermediate state (both or
neither) can be far lower probability than either endpoint, so a sampler without it gets
stuck between two orientations of the same skeleton -- which is precisely the ambiguity
this whole project is about resolving.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _reachable(adjacency: np.ndarray, start: int, target: int) -> bool:
    """Is `target` reachable from `start` following edges forward?"""
    d = adjacency.shape[0]
    seen = np.zeros(d, dtype=bool)
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if seen[node]:
            continue
        seen[node] = True
        stack.extend(np.flatnonzero(adjacency[node]).tolist())
    return False


def mh_sample(log_w: np.ndarray, mask_to_index: np.ndarray, d: int,
              n_samples: int, burn_in: int = 5000, thin: int = 10,
              rng: Optional[np.random.Generator] = None,
              init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Draw `n_samples` DAGs from the distribution `P(G) ~ prod_i exp(log_w[i, Pa_i])`.

    `log_w` is the `[d, 2^(d-1)]` table from `DPPosterior.log_weights` -- score *and*
    prior, so passing a prior-only table (`DPPosterior.log_prior_term`) samples from the
    prior instead, which is what the GATE 1 estimate needs.

    Returns `(draws [n_samples, d, d] bool, acceptance_rate)`. The acceptance rate is
    returned rather than logged because a rate near 0 or 1 is the usual sign that the chain
    is not exploring, and it should be visible at the call site.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    adjacency = (np.zeros((d, d), dtype=bool) if init is None
                 else np.asarray(init, dtype=bool).copy())

    # Parent sets carried as bitmasks so a local score is one array lookup.
    parent_mask = np.zeros(d, dtype=np.int64)
    for v in range(d):
        parent_mask[v] = int(np.dot(adjacency[:, v], 1 << np.arange(d)))

    def local(v: int) -> float:
        return log_w[v, mask_to_index[v, parent_mask[v]]]

    pairs = [(u, v) for u in range(d) for v in range(d) if u != v]
    n_pairs = len(pairs)
    draws = np.empty((n_samples, d, d), dtype=bool)
    kept = 0
    accepted = 0
    proposed = 0

    n_steps = burn_in + n_samples * thin
    for step in range(n_steps):
        u, v = pairs[rng.integers(n_pairs)]
        proposed += 1

        if adjacency[u, v]:                                    # delete u -> v
            before = local(v)
            parent_mask[v] &= ~(1 << u)
            if np.log(rng.random()) < local(v) - before:
                adjacency[u, v] = False
                accepted += 1
            else:
                parent_mask[v] |= 1 << u

        elif adjacency[v, u]:                                  # reverse v -> u
            # Legality first: dropping v->u and adding u->v is a cycle exactly when v is
            # still reachable from u without that edge.
            adjacency[v, u] = False
            legal = not _reachable(adjacency, v, u)
            if legal:
                before = local(u) + local(v)
                parent_mask[u] &= ~(1 << v)
                parent_mask[v] |= 1 << u
                if np.log(rng.random()) < local(u) + local(v) - before:
                    adjacency[u, v] = True
                    accepted += 1
                else:
                    parent_mask[u] |= 1 << v
                    parent_mask[v] &= ~(1 << u)
                    adjacency[v, u] = True
            else:
                adjacency[v, u] = True

        else:                                                  # add u -> v
            if not _reachable(adjacency, v, u):
                before = local(v)
                parent_mask[v] |= 1 << u
                if np.log(rng.random()) < local(v) - before:
                    adjacency[u, v] = True
                    accepted += 1
                else:
                    parent_mask[v] &= ~(1 << u)

        if step >= burn_in and (step - burn_in) % thin == 0 and kept < n_samples:
            draws[kept] = adjacency
            kept += 1

    return draws[:kept], accepted / max(proposed, 1)


def descendant_codes(draws: np.ndarray) -> np.ndarray:
    """[n_draws, d] integer code of each node's descendant set, per sampled DAG.

    Two DAGs are indistinguishable by `do(X_i)` exactly when their codes agree at `i`, so
    the code is the sufficient statistic the oracle groups by. Computed by blocked
    Floyd-Warshall over all draws at once -- a Python loop over samples costs more than the
    sampling did.
    """
    n, d, _ = draws.shape
    reach = np.asarray(draws, dtype=bool).copy()
    for k in range(d):
        reach |= reach[:, :, k][:, :, None] & reach[:, k, :][:, None, :]
    return reach.astype(np.int64) @ (1 << np.arange(d)).astype(np.int64)
