"""Depth in the per-node scorer.

The load-bearing test here is the first one: `layers=1` must reproduce the network that
produced the d=4, d=5 and d=6 results *exactly*, not merely have the same shape. If adding
the parameter perturbs the RNG draw, every one of those results silently becomes
unreproducible, and the failure would look like ordinary seed variance rather than like a
bug.
"""
import numpy as np
import pytest
import torch

from sa.policy import PerNodeActorCritic


def _net(d=5, hidden=64, seed=0, **kwargs):
    torch.manual_seed(seed)
    return PerNodeActorCritic(d, hidden, **kwargs)


# --------------------------------------------------------------------------------------
# layers=1 must be the original network, bit for bit
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [4, 5, 6])
@pytest.mark.parametrize("include_counts", [False, True])
def test_layers_one_has_identical_parameters_to_the_default(d, include_counts):
    """Same seed, same draws: the extra modules must not exist to consume RNG."""
    explicit = _net(d=d, seed=7, include_counts=include_counts, layers=1)
    default = _net(d=d, seed=7, include_counts=include_counts)

    a, b = explicit.state_dict(), default.state_dict()
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), f"parameter {key} differs"


@pytest.mark.parametrize("d", [4, 5])
def test_layers_one_produces_identical_outputs(d):
    net_a, net_b = _net(d=d, seed=3, layers=1), _net(d=d, seed=3)
    obs = torch.randn(8, d * (d - 1) + 1)

    logits_a, value_a = net_a(obs)
    logits_b, value_b = net_b(obs)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(value_a, value_b)


def test_layers_one_adds_nothing_to_the_state_dict():
    """An empty ModuleList must not introduce keys that would break checkpoint loading."""
    net = _net(layers=1)
    assert not [k for k in net.state_dict() if k.startswith("rounds")]


def test_deeper_networks_do_add_parameters():
    """Guard against the parameter being silently ignored -- which would make the whole
    depth probe measure nothing at all, twice, and agree with itself."""
    counts = [sum(p.numel() for p in _net(layers=k).parameters()) for k in (1, 2, 3)]
    assert counts[0] < counts[1] < counts[2]
    assert len(_net(layers=3).rounds) == 2


def test_layers_must_be_at_least_one():
    with pytest.raises(ValueError, match="layers must be"):
        _net(layers=0)


# --------------------------------------------------------------------------------------
# Added depth must not break the property the architecture exists for
# --------------------------------------------------------------------------------------

def _permute_observation(obs, perm, d):
    """Relabel nodes in a flat edge-marginal observation."""
    mask = ~np.eye(d, dtype=bool)
    matrix = np.zeros((d, d))
    matrix[mask] = obs[: d * (d - 1)]
    matrix = matrix[np.ix_(perm, perm)]
    return np.concatenate([matrix[mask], obs[d * (d - 1):]])


@pytest.mark.parametrize("layers", [1, 2, 3])
@pytest.mark.parametrize("d", [4, 5])
def test_equivariance_survives_extra_rounds(layers, d):
    """Relabel the nodes and the logits must permute with them.

    An earlier version of this class pooled neighbours in index order and was equivariant
    only under permutations that happened to preserve that order -- which is to say, not
    equivariant. The extra rounds gather neighbour embeddings by index too, so the same
    mistake is available again and is checked for again here.
    """
    net = _net(d=d, seed=1, layers=layers).eval()
    rng = np.random.default_rng(d * 10 + layers)
    obs = rng.random(d * (d - 1) + 1)
    perm = rng.permutation(d)

    with torch.no_grad():
        base, base_value = net(torch.tensor(obs, dtype=torch.float32))
        permuted, permuted_value = net(
            torch.tensor(_permute_observation(obs, perm, d), dtype=torch.float32))

    # Node logits permute; the pass logit (last) and the value are invariant.
    assert np.allclose(permuted[:d].numpy(), base[:d].numpy()[perm], atol=1e-5)
    assert np.allclose(permuted[d].item(), base[d].item(), atol=1e-5)
    assert np.allclose(permuted_value.item(), base_value.item(), atol=1e-5)


@pytest.mark.parametrize("layers", [2, 3])
def test_deeper_networks_still_run_batched_and_single(layers):
    d = 5
    net = _net(d=d, seed=2, layers=layers)

    single_logits, single_value = net(torch.randn(d * (d - 1) + 1))
    assert single_logits.shape == (d + 1,) and single_value.shape == ()

    batch_logits, batch_value = net(torch.randn(4, d * (d - 1) + 1))
    assert batch_logits.shape == (4, d + 1) and batch_value.shape == (4,)


@pytest.mark.parametrize("layers", [2, 3])
def test_deeper_networks_are_trainable(layers):
    """Every added parameter must receive gradient -- an unreachable round would make the
    probe report 'depth does not help' when what it measured was a disconnected module."""
    net = _net(d=5, seed=4, layers=layers)
    logits, value = net(torch.randn(6, 5 * 4 + 1))
    (logits.sum() + value.sum()).backward()

    ungrad = [name for name, p in net.named_parameters()
              if p.grad is None or torch.all(p.grad == 0)]
    assert not [n for n in ungrad if n.startswith("rounds")], ungrad


def test_extra_rounds_change_the_function():
    """Depth must actually alter the computation, not just add dead weight."""
    d = 5
    obs = torch.randn(4, d * (d - 1) + 1)
    shallow, deep = _net(d=d, seed=5, layers=1), _net(d=d, seed=5, layers=2)
    with torch.no_grad():
        assert not torch.allclose(shallow(obs)[0], deep(obs)[0])
