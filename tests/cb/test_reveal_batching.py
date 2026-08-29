"""`estimated_reveal_all` must be EXACTLY `estimated_reveal` per node, and much faster.

The batching exists because `FisherZ` depends only on (data, intervened, alpha, foreign) and
`ancestral_evidence()` / `pair_power()` each return the full [k, k] matrix, while both
callers kept one row and rebuilt the whole thing for the next node. Equivalence is the whole
safety argument for the change, so it is pinned here rather than assumed.
"""
import time

import numpy as np
import pytest

from cb.versionspace import estimated_reveal, estimated_reveal_all


def _data(k=8, rows=200, n_int=4, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(rows, k))
    intervened = np.zeros((rows, k), dtype=float)
    for i, node in enumerate(range(n_int)):
        lo = 40 + i * 30
        intervened[lo:lo + 30, node] = 1.0
        data[lo:lo + 30, node] = 0.0
        # a real effect, so the tests have something to find
        data[lo:lo + 30, (node + 1) % k] *= 0.2
    return data, intervened


@pytest.mark.parametrize("k,n_int", [(6, 3), (8, 4), (12, 5)])
def test_batched_matches_one_at_a_time(k, n_int):
    data, intervened = _data(k=k, n_int=n_int)
    nodes = tuple(range(n_int))
    batched = estimated_reveal_all(data, intervened, nodes, k)
    for x in nodes:
        assert batched[x] == estimated_reveal(data, intervened, x, k), f"node {x} differs"


def test_batched_returns_exactly_the_requested_nodes():
    data, intervened = _data()
    out = estimated_reveal_all(data, intervened, (1, 3), 8)
    assert set(out) == {1, 3}


def test_row_length_excludes_the_node_itself():
    data, intervened = _data(k=8)
    evidence, powered = estimated_reveal_all(data, intervened, (2,), 8)[2]
    assert len(evidence) == 7 and len(powered) == 7


def test_batching_is_faster_than_the_loop_it_replaces():
    """The point of the change. Not a tight bound -- a loose one that would catch a regression
    to per-node construction, which is what the callers used to do."""
    data, intervened = _data(k=12, n_int=6, rows=400)
    nodes = tuple(range(6))

    start = time.perf_counter()
    for _ in range(3):
        {x: estimated_reveal(data, intervened, x, 12) for x in nodes}
    loop = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(3):
        estimated_reveal_all(data, intervened, nodes, 12)
    batched = time.perf_counter() - start

    assert batched < loop / 2, f"batched {batched:.3f}s vs loop {loop:.3f}s -- expected >2x"
