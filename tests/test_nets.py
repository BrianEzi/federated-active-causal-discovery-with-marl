"""Per-node network tests, salvaged 2026-08-23 when sa/ was dissolved.

The rest of `tests/sa/test_policy.py` exercised the single-agent PPO agent and its
environment, both of which are gone. These are the architecture tests, which apply
to `ma/nets.PerNodeActorCritic` unchanged -- the class moved verbatim.

`test_pernode_is_permutation_equivariant` is the load-bearing one: an earlier version
pooled neighbours in index order and was equivariant only under permutations that
happened to preserve that order, which is to say not equivariant. The test caught it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ma.graphs import build_graph_space
from ma.nets import PerNodeActorCritic


@pytest.fixture(scope="module")
def space3():
    """The d=3 graph space. Carried over with the salvaged tests; the original
    lived in the deleted tests/sa/conftest.py."""
    return build_graph_space(3)

def _flat_from_matrix(matrix, budget, counts=None):
    """Pack a [d, d] marginal matrix into the flat observation layout."""
    d = matrix.shape[0]
    off = ~np.eye(d, dtype=bool)
    parts = [matrix[off], np.array([budget])]
    if counts is not None:
        parts.append(counts)
    return np.concatenate(parts).astype(np.float32)


# --- the permutation-equivariant architecture -------------------------------------------

def _flat_from_matrix(matrix, budget, counts=None):
    """Pack a [d, d] marginal matrix into the flat observation layout."""
    d = matrix.shape[0]
    off = ~np.eye(d, dtype=bool)
    parts = [matrix[off], np.array([budget])]
    if counts is not None:
        parts.append(counts)
    return np.concatenate(parts).astype(np.float32)


def test_pernode_rebuilds_the_marginal_matrix_correctly():
    """Node i's neighbour pairs must be (i->j, j->i) for its own row and column."""
    d = 4
    net = PerNodeActorCritic(d, hidden=8, include_counts=False, allow_pass=True)
    matrix = np.arange(d * d, dtype=np.float32).reshape(d, d)
    np.fill_diagonal(matrix, 0.0)
    obs = torch.as_tensor(_flat_from_matrix(matrix, 0.5)).unsqueeze(0)

    pairs = net._neighbour_pairs(obs)[0]
    assert pairs.shape == (d, d - 1, 2)
    for i in range(d):
        others = np.arange(d)[np.arange(d) != i]
        np.testing.assert_allclose(pairs[i, :, 0].numpy(), matrix[i, others], atol=1e-5)
        np.testing.assert_allclose(pairs[i, :, 1].numpy(), matrix[others, i], atol=1e-5)


def test_pernode_is_permutation_equivariant():
    """Relabel the nodes and the logits must permute with them.

    This is the property the whole architecture exists for, and it is true of the oracle:
    node identity carries no information, only structure does. The flat MLP cannot express
    it, which is why it must learn each node's scorer separately.
    """
    d, perm = 4, np.array([2, 0, 3, 1])
    torch.manual_seed(0)
    net = PerNodeActorCritic(d, hidden=16, include_counts=False, allow_pass=True)
    rng = np.random.default_rng(0)

    matrix = rng.random((d, d)).astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    permuted = matrix[np.ix_(perm, perm)]

    with torch.no_grad():
        base, base_value = net(torch.as_tensor(_flat_from_matrix(matrix, 0.5)))
        other, other_value = net(torch.as_tensor(_flat_from_matrix(permuted, 0.5)))

    # Node logits permute; the pass logit (last) and the value are invariant.
    np.testing.assert_allclose(other[:d].numpy(), base[:d].numpy()[perm], atol=1e-5)
    assert float(other[d]) == pytest.approx(float(base[d]), abs=1e-5)
    assert float(other_value) == pytest.approx(float(base_value), abs=1e-5)


def test_pernode_parameter_count_does_not_grow_with_d():
    """One shared scorer serves every node, so the same model form carries to d=6."""
    sizes = [sum(p.numel() for p in
                 PerNodeActorCritic(d, hidden=32).parameters()) for d in (4, 5, 6)]
    # Only the input width (2(d-1)+1) changes, so growth is linear and small -- not the
    # quadratic blow-up of a dense layer over d(d-1) inputs mapping to d outputs.
    assert sizes[2] - sizes[1] == sizes[1] - sizes[0]


# REMOVED 2026-08-23 with the sa/ dissolution: test_pernode_shapes_match_the_flat_network
# and test_pernode_rejects_the_posterior_observation both drove `PPOAgent`, the
# single-agent trainer, which no longer exists. They tested the AGENT's wiring rather
# than the network, and the network's own shape contract is covered above.


# =======================================================================================
# THE POOLING SUBGRADIENT, pinned deliberately (30 Aug 2026).
#
# Max-pooling over neighbours moved from `t.max(dim).values` to `torch.amax(t, dim)`,
# which is ~7x faster on this build and was 3.3 s of a 43 s profiled training run. The
# FORWARD is bit-identical. The BACKWARD is not, and only at exact ties:
#
#   max   sends the whole gradient to the LOWEST-INDEXED maximum
#   amax  splits it evenly among all tied maxima
#
# Both are valid subgradients of a function that has no derivative there. It is not a
# correctness fix -- parameter gradients were checked to be permutation-equivariant under
# either rule, because tied outputs here come from tied inputs. But ties are COMMON in
# this environment rather than measure-zero: under the uniform prior every neighbour pair
# carries the same marginals, so every episode starts fully tied. Splitting is the
# symmetric choice, so it is the one taken; this test stops it being reverted by accident
# and stops it changing again unnoticed.
#
# Measured cost of the switch on a 120-episode probe: training entropy differs in the
# 10th decimal after one update; every reported arm metric identical.
# =======================================================================================
import torch


def test_neighbour_pooling_splits_the_gradient_across_tied_maxima():
    tied = torch.zeros(1, 4, 3, 5, requires_grad=True)

    torch.amax(tied, dim=2).sum().backward()
    split = tied.grad[0, 0, :, 0]
    assert torch.allclose(split, torch.full((3,), 1.0 / 3.0)), (
        f"expected an even split over three tied neighbours, got {split.tolist()}")

    tied.grad = None
    tied.max(dim=2).values.sum().backward()
    first_index = tied.grad[0, 0, :, 0]
    assert first_index.tolist() == [1.0, 0.0, 0.0], (
        "the rule this replaced should still route everything to the lowest index -- if "
        "torch changed that, the note above needs rewriting rather than this assertion")


def test_the_two_pooling_rules_agree_exactly_on_the_forward_pass():
    """The values are identical, so no reported number moves; only the gradient rule does."""
    torch.manual_seed(0)
    for shape in [(1, 12, 11, 64), (7, 4, 3, 5), (1, 2, 2, 2)]:
        x = torch.randn(*shape)
        assert torch.equal(torch.amax(x, dim=2), x.max(dim=2).values)
        # And with deliberate ties, which is the case that actually arises here.
        x[..., 0] = x[..., 1]
        assert torch.equal(torch.amax(x, dim=2), x.max(dim=2).values)
