"""One set of weights, many federation shapes.

The claim under test is narrow and checkable: `PortableRoleActorCritic` has NO learned
tensor whose shape depends on the window size or the agent count, so a checkpoint trained
on one shape loads into another and acts. Everything else in this file exists to stop that
claim being true by accident -- a net that ignored its input would also "work" everywhere.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from ma.env import ROUND_ROBIN, VARY, MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO, PPOConfig, PortableRoleActorCritic
from ma.topology import federated_topology


def _env(n_agents, private_size, budget, arch="gnn_portable", n_shared=3, seed=0):
    topology = federated_topology(n_agents, private_size, n_shared)
    return TwoAgentEnv(MAConfig(
        topology=topology, n_obs=60, n_int=20, budget=budget, disclose_regime=True,
        turn_order=ROUND_ROBIN, action_modes=(VARY,), belief_backend="version_space",
        policy_arch=arch, episode_mix="confounded", reward_criterion="claims",
        claim_bar=1.0, per_agent_reward=True, observe_belief_channels=True,
        observe_partner_counts=True), seed=seed)


SHAPES = ((3, 1, 6), (4, 1, 8), (4, 2, 10), (5, 1, 10))


def test_no_learned_tensor_has_a_window_shaped_dimension():
    """The property the portability rests on, asserted directly rather than inferred from
    a load succeeding: every learned width is per-node or per-pair."""
    sizes = set()
    for n_agents, private_size, budget in SHAPES:
        env = _env(n_agents, private_size, budget)
        net = PortableRoleActorCritic(env.windows[0], env.topology.n_agents - 1)
        shapes = {name: tuple(tensor.shape)
                  for name, tensor in net.state_dict().items()}
        sizes.add(tuple(sorted(shapes.items())))
        # The role buffer is derived from the window and must not be saved.
        assert "role" not in net.state_dict()
    assert len(sizes) == 1, "the saved parameter shapes differ between federation shapes"


def test_one_checkpoint_loads_into_every_shape_and_acts_legally():
    trained = _env(*SHAPES[0])
    learner = IndependentPPO(trained, PPOConfig(total_episodes=8, episodes_per_update=8))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "portable.pt"
        learner.save(path)
        for shape in SHAPES:
            env = _env(*shape)
            restored = IndependentPPO.load(path, env)
            result = env.reset(seed=3)
            for agent in env.topology.agents:
                action = restored.policy(agent)(env, result)
                assert 0 <= action < env.n_actions(agent)


def test_the_shared_network_is_one_object_not_n_copies():
    env = _env(*SHAPES[1])
    learner = IndependentPPO(env, PPOConfig(total_episodes=8, episodes_per_update=8))
    nets = [learner.nets[a] for a in env.topology.agents]
    assert all(net is nets[0] for net in nets)
    assert learner.shared_net is nets[0]


def test_a_pooled_partner_block_is_permutation_invariant_over_partners():
    """Pooling is the mechanism that makes the width fixed; invariance is what it BUYS, and
    it is the property that would silently disappear if the pooling were replaced by a
    reshape that happens to have the right size."""
    env = _env(4, 1, 8)
    net = PortableRoleActorCritic(env.windows[0], env.topology.n_agents - 1)
    net.eval()
    env.reset(seed=11)
    obs = torch.as_tensor(env.observation(0), dtype=torch.float32)

    k, n_others, n_shared = net.d, net.n_others, net.n_shared
    start = k * (k - 1) + 1
    permuted = obs.clone()
    # Reverse partner order inside the disclosed block; a permutation-invariant policy is
    # unmoved by it.
    block = permuted[start:start + n_others * n_shared].view(n_others, n_shared)
    permuted[start:start + n_others * n_shared] = block.flip(0).reshape(-1)

    with torch.no_grad():
        base_logits, base_value = net(obs)
        swapped_logits, swapped_value = net(permuted)
    assert torch.allclose(base_logits, swapped_logits, atol=1e-6)
    assert torch.allclose(base_value, swapped_value, atol=1e-6)


def test_the_policy_actually_reads_its_observation():
    """A network that ignored its input would pass every test above. This one fails if the
    logits do not move when the belief does."""
    env = _env(4, 1, 8)
    net = PortableRoleActorCritic(env.windows[0], env.topology.n_agents - 1)
    net.eval()
    env.reset(seed=11)
    obs = torch.as_tensor(env.observation(0), dtype=torch.float32)
    disturbed = obs.clone()
    disturbed[: net.d * (net.d - 1)] = 1.0 - disturbed[: net.d * (net.d - 1)]
    with torch.no_grad():
        a, _ = net(obs)
        b, _ = net(disturbed)
    assert not torch.allclose(a, b, atol=1e-6)


def test_rebinding_refuses_a_different_observation_layout():
    """The encoder slices positionally, so a layout mismatch would mis-read every feature.
    It must raise rather than produce plausible numbers."""
    wide = _env(4, 1, 8)
    net = PortableRoleActorCritic(wide.windows[0], 3)
    narrow = TwoAgentEnv(MAConfig(
        topology=federated_topology(4, 1, 3), n_obs=60, n_int=20, budget=8,
        disclose_regime=True, turn_order=ROUND_ROBIN, action_modes=(VARY,),
        belief_backend="version_space", policy_arch="gnn_portable",
        episode_mix="confounded", reward_criterion="claims", claim_bar=1.0,
        observe_belief_channels=True, observe_partner_counts=False), seed=0)
    with pytest.raises(ValueError, match="partner-count flag"):
        net.rebind(narrow.windows[0], 3)


def test_binding_is_refused_for_window_shaped_architectures():
    env = _env(4, 1, 8, arch="gnn")
    learner = IndependentPPO(env, PPOConfig(total_episodes=8, episodes_per_update=8))
    with pytest.raises(ValueError, match="gnn_portable"):
        learner.bind(_env(5, 1, 10, arch="gnn"))


def test_the_shared_update_pools_every_agents_experience_into_one_step():
    """Stepping once per agent on shared parameters would make the effective learning rate
    scale with the agent count. Pinned by parameter identity before and after."""
    env = _env(4, 1, 8)
    learner = IndependentPPO(env, PPOConfig(total_episodes=16, episodes_per_update=16,
                                            seed=0))
    before = [p.detach().clone() for p in learner.shared_net.parameters()]
    batch = learner.collect(2, 0, mask_pass=False)
    sizes = {a: len(batch["buffers"][a]["reward"]) for a in env.topology.agents}
    learner.update(batch["buffers"])
    after = [p.detach().clone() for p in learner.shared_net.parameters()]
    assert any(not torch.allclose(x, y) for x, y in zip(before, after))
    # Every agent contributed: the per-agent buffers are non-empty and equal in length
    # (round-robin gives each the same number of turns to be queried).
    assert all(size > 0 for size in sizes.values())
    assert len(set(sizes.values())) == 1
