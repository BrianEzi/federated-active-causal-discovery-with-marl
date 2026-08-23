"""The role-aware GNN wrapper -- Phase 3 gates.

The load-bearing property is EQUIVARIANCE WITHIN ROLES: swap two shared authority nodes
and the action logits must swap with them, exactly; a shared node and a private node must
NOT be interchangeable, or the network cannot express "clamp my own private node". Both
directions are asserted, because a wrapper that ignored the role features entirely would
pass the first and fail the second -- full equivariance is the failure mode here.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ma.env import CLAMP, VARY, MAConfig, MODES, ROUND_ROBIN, TwoAgentEnv
from ma.policy import IndependentPPO, PPOConfig, RolePerNodeActorCritic
from ma.topology import two_agent

TOPO = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))


def _env(**kwargs):
    kwargs.setdefault("belief_backend", "constraint")
    kwargs.setdefault("cb_n_boot", 4)
    kwargs.setdefault("action_modes", (VARY,))
    kwargs.setdefault("policy_arch", "gnn")
    config = MAConfig(topology=TOPO, n_obs=120, n_int=40, budget=3,
                      turn_order=ROUND_ROBIN, **kwargs)
    return TwoAgentEnv(config, seed=0)


def test_logits_match_the_action_space():
    env = _env()
    for agent in env.topology.agents:
        net = RolePerNodeActorCritic(env.windows[agent], env.topology.n_agents - 1)
        obs = torch.as_tensor(env.observation(agent), dtype=torch.float32)
        logits, value = net(obs)
        assert logits.shape == (env.n_actions(agent),)
        assert value.shape == ()
        assert torch.isfinite(logits).all() and torch.isfinite(value)


def _swap_obs(env, agent, obs, node_u, node_v):
    """The observation as it would look with window positions of u and v relabelled."""
    window = env.windows[agent]
    k = window.k
    pu, pv = window.pos[node_u], window.pos[node_v]
    perm = list(range(k))
    perm[pu], perm[pv] = pv, pu

    m = k * (k - 1)
    off = ~np.eye(k, dtype=bool)
    matrix = np.zeros((k, k))
    matrix[off] = obs[:m]
    matrix = matrix[np.ix_(perm, perm)]

    out = obs.copy()
    out[:m] = matrix[off]
    # disclosed block is indexed by SHARED order; swap the two shared slots per partner.
    su, sv = window.shared.index(node_u), window.shared.index(node_v)
    base = m + 1
    n_shared = len(window.shared)
    for other in range(env.topology.n_agents - 1):
        i, j = base + other * n_shared + su, base + other * n_shared + sv
        out[i], out[j] = out[j], out[i]
    return out


def test_swapping_two_shared_nodes_swaps_their_logits():
    env = _env()
    agent = 0
    window = env.windows[agent]
    net = RolePerNodeActorCritic(window, env.topology.n_agents - 1)
    obs = env.observation(agent)
    obs[: window.k * (window.k - 1)] = np.random.default_rng(1).uniform(
        0.05, 0.95, window.k * (window.k - 1))

    node_u, node_v = 2, 3          # both shared, both authority
    swapped = _swap_obs(env, agent, obs, node_u, node_v)
    with torch.no_grad():
        logits_a, _ = net(torch.as_tensor(obs, dtype=torch.float32))
        logits_b, _ = net(torch.as_tensor(swapped, dtype=torch.float32))

    au = window.authority.index(node_u)
    av = window.authority.index(node_v)
    assert torch.allclose(logits_a[au], logits_b[av], atol=1e-5)
    assert torch.allclose(logits_a[av], logits_b[au], atol=1e-5)
    others = [i for i in range(len(logits_a)) if i not in (au, av)]
    assert torch.allclose(logits_a[others], logits_b[others], atol=1e-5)


def test_a_private_and_a_shared_node_are_not_interchangeable():
    """Full equivariance is the FAILURE mode: without effective role features the net
    cannot prefer its own private node, which is the headline behaviour."""
    env = _env()
    agent = 0
    window = env.windows[agent]
    torch.manual_seed(3)
    net = RolePerNodeActorCritic(window, env.topology.n_agents - 1)

    # An observation symmetric between node 0 (private) and node 2 (shared): identical
    # marginal rows/columns, no disclosure, no signals. Any logit difference between them
    # is then attributable to the ROLE features alone.
    obs = np.zeros(env.obs_size(agent), dtype=np.float32)
    k = window.k
    m = k * (k - 1)
    obs[:m] = 0.5
    with torch.no_grad():
        logits, _ = net(torch.as_tensor(obs))
    a0 = window.authority.index(0)
    a2 = window.authority.index(2)
    assert not torch.allclose(logits[a0], logits[a2], atol=1e-6), \
        "role features had no effect: private and shared nodes are indistinguishable"


def test_two_modes_are_refused():
    config = MAConfig(topology=TOPO, n_obs=120, n_int=40, budget=3,
                      action_modes=MODES, belief_backend="constraint", cb_n_boot=4,
                      policy_arch="gnn")
    env = TwoAgentEnv(config, seed=0)
    with pytest.raises(NotImplementedError, match="single intervention mode"):
        RolePerNodeActorCritic(env.windows[0], env.topology.n_agents - 1)


def test_gnn_ppo_trains_and_roundtrips(tmp_path):
    env = _env()
    learner = IndependentPPO(env, PPOConfig(total_episodes=4, episodes_per_update=2,
                                            hidden=32, gnn_layers=2, seed=0))
    history = learner.train()
    assert len(history) == 2
    assert all(np.isfinite(rec["entropy"]) for rec in history)

    path = tmp_path / "gnn.pt"
    learner.save(path)
    loaded = IndependentPPO.load(path, env)
    obs = torch.as_tensor(env.observation(0), dtype=torch.float32)
    with torch.no_grad():
        a, _ = learner.nets[0](obs)
        b, _ = loaded.nets[0](obs)
    assert torch.allclose(a, b)

    # Cross-architecture and cross-backend loads are refused, not warned about.
    mlp_env = _env(policy_arch="mlp")
    with pytest.raises(ValueError, match="policy_arch"):
        IndependentPPO.load(path, mlp_env)
    exact_env = TwoAgentEnv(MAConfig(topology=TOPO, n_obs=120, n_int=40, budget=3,
                                     action_modes=(CLAMP,), policy_arch="gnn"), seed=0)
    with pytest.raises(ValueError, match="belief_backend"):
        IndependentPPO.load(path, exact_env)
