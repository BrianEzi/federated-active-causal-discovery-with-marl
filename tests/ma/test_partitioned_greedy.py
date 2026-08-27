"""The coordinated control baseline.

`PartitionedGreedyAgent` exists to answer one objection: the learned policy's advantage is
almost entirely on the joint criterion and the mechanism is division of labour, so comparing
it against a baseline that never divides anything rewards it for a job its opponent was not
attempting. This baseline does the same myopic targeting with the shared surface divided by
a positional convention.

Two properties carry the argument and both are silent when broken: the partition must be
CONSISTENT across agents (computed independently, no communication, no gaps or collisions),
and the fallback must mean the agent is never worse off for having a share.
"""
from __future__ import annotations

import pytest

from ma.baselines import PartitionedGreedyAgent, make_baselines
from ma.env import ROUND_ROBIN, VARY, MAConfig, TwoAgentEnv
from ma.topology import federated_topology


def _env(n_agents=4, private=2, shared=4, **kw):
    topology = federated_topology(n_agents, private_size=private, n_shared=shared)
    return TwoAgentEnv(MAConfig(
        topology=topology, n_obs=60, n_int=20, budget=12, disclose_regime=True,
        turn_order=ROUND_ROBIN, action_modes=(VARY,), belief_backend="factored",
        episode_mix="confounded", reward_criterion="claims", claim_bar=1.0, **kw), seed=0)


def test_the_partition_is_a_partition_no_gaps_and_no_collisions():
    """Every shared node is owned by exactly one agent, and each computes it alone."""
    env = _env(n_agents=4, shared=10)
    owners = {}
    for agent in env.topology.agents:
        mine = PartitionedGreedyAgent(agent, 4)._mine(env.windows[agent])
        for node in mine & set(env.topology.exposed):
            assert node not in owners, f"shared node {node} claimed by two agents"
            owners[node] = agent
    assert set(owners) == set(env.topology.exposed)


def test_every_agent_keeps_all_of_its_own_private_nodes():
    """The partition divides the SHARED surface only -- a private node is nobody else's."""
    env = _env(n_agents=4, private=3, shared=6)
    for agent in env.topology.agents:
        window = env.windows[agent]
        assert set(window.private) <= PartitionedGreedyAgent(agent, 4)._mine(window)


def test_agents_compute_the_same_partition_without_communicating():
    """Agent 0's view of who owns what must match agent 3's, or the division silently
    collides. Both read `topology.exposed`, which is one global list, so this holds by
    construction -- pinned because a per-agent ordering would break it invisibly."""
    env = _env(n_agents=4, shared=8)
    def owner_of(observer):
        window = env.windows[observer]
        return {node: index % 4 for index, node in enumerate(window.shared)}
    reference = owner_of(0)
    for agent in env.topology.agents:
        assert owner_of(agent) == reference


def test_it_targets_inside_its_share_while_its_share_is_open():
    env = _env(n_agents=4, shared=4)
    env.reset(seed=4)
    for agent in env.topology.agents:
        window = env.windows[agent]
        action = PartitionedGreedyAgent(agent, 4)(env, None)
        node = window.actions[action][0]
        if action != window.pass_index:
            assert node in PartitionedGreedyAgent(agent, 4)._mine(window)


def test_it_falls_back_rather_than_passing_when_its_own_share_is_settled():
    """The partition may only REDIRECT effort, never waste it. With one shared node each and
    that node already settled, the agent must still act on whatever else is open."""
    env = _env(n_agents=4, shared=4)
    env.reset(seed=4)
    agent = 0
    window = env.windows[agent]
    baseline = PartitionedGreedyAgent(agent, 4)
    mine = baseline._mine(window)
    # Settle every pair touching this agent's own share, leaving other nodes open.
    belief = window.belief.last
    for node in mine:
        position = window.pos[node]
        for other in range(window.k):
            if other == position:
                continue
            belief.adjacency[position, other] = belief.adjacency[other, position] = 0.0
    action = baseline(env, None)
    if action != window.pass_index:
        assert window.actions[action][0] in set(window.authority)


def test_it_is_offered_by_make_baselines_and_is_built_lazily():
    env = _env(n_agents=4, shared=4)
    baselines = make_baselines(env, 0, seed=0)
    assert "greedy_partitioned" in baselines
    assert isinstance(baselines["greedy_partitioned"], PartitionedGreedyAgent)
    assert baselines["greedy_partitioned"] is baselines["greedy_partitioned"]


def _duplicate_coverage(build, n_agents=4, private=6, shared=6, episodes=20, budget=16):
    env = _env(n_agents=n_agents, private=private, shared=shared)
    env.config.budget = budget
    agents = list(env.topology.agents)
    policies = {a: build(a) for a in agents}
    total = 0.0
    for episode in range(episodes):
        result = env.reset(seed=500 + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in agents})
        total += env.duplicate_coverage()
    return total / episodes


def test_the_partitioned_agent_really_does_duplicate_less_than_the_plain_rule():
    """The property the baseline exists to have, pinned at a MATERIAL margin.

    The first version of this class asserted only `<`, passed on a small configuration, and
    was then measured in the actual comparison at duplicate coverage 0.167 against the plain
    rule's 0.169 -- and at k=12 it was WORSE, 0.150 against 0.126. It was not a coordinated
    control at all, so "the learned policy beats it" answered nothing. A weak inequality on a
    favourable configuration is exactly how that got through, so this asks for a real gap on
    the configuration the comparison actually runs at.
    """
    from ma.baselines import UncertaintyGreedyAgent

    plain = _duplicate_coverage(lambda a: UncertaintyGreedyAgent(a, seed=0, bar=1.0))
    divided = _duplicate_coverage(lambda a: PartitionedGreedyAgent(a, 4, seed=0, bar=1.0))
    assert divided < 0.75 * plain, {"plain": plain, "partitioned": divided}


def test_it_breaks_ties_towards_the_least_touched_node():
    """The tie-break reads DISCLOSED partner counts -- the same channel the learned policy
    sees -- so it must actually consult them. With two equally informative targets and one
    already hit by a partner, the untouched one is taken."""
    env = _env(n_agents=4, private=2, shared=4, observe_partner_counts=True)
    env.reset(seed=4)
    agent, window = 1, None
    window = env.windows[1]
    baseline = PartitionedGreedyAgent(agent, 4, seed=0, bar=1.0)
    node = window.shared[0]
    before = baseline._touches(env, window, node)
    env.partner_counts[agent][0, 0] += 3.0
    assert baseline._touches(env, window, node) == before + 3.0
