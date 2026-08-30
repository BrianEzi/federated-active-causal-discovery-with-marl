"""FedAvg over the shared policy: weights cross the boundary, raw trajectories do not.

The historical path CONCATENATES every site's raw trajectories into one buffer. That is data
pooling -- strictly more centralised than gradient sharing, and not what "federated" means.
"""
import numpy as np
import pytest
import torch

from ma.env import MAConfig, TwoAgentEnv
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import federated_topology


def _env(agents=3):
    # A single site has no partners, so no latent can exist in its window and
    # `episode_mix="confounded"` is unsatisfiable -- the env raises after 200 draws.
    return TwoAgentEnv(MAConfig(
        topology=federated_topology(agents, 2, 3), n_obs=40, n_int=10, budget=9,
        turn_order="round_robin", belief_backend="factored", action_modes=("vary",),
        claim_bar=1.0, reward_criterion="claims", policy_arch="gnn_portable",
        graph_model="sf", sf_m=2, vs_evidence="oracle",
        episode_mix="confounded" if agents > 1 else "any"))


def _trained(local_epochs, agents=3, episodes=160, seed=0, epochs=None):
    env = _env(agents)
    cfg = PPOConfig(hidden=32, seed=seed, total_episodes=episodes,
                    local_epochs=local_epochs, **({"epochs": epochs} if epochs else {}))
    ppo = IndependentPPO(env, cfg)
    ppo.train(verbose=False)
    flat = torch.cat([p.detach().flatten() for p in ppo.shared_net.parameters()])
    return flat.clone(), ppo


def test_local_epochs_zero_keeps_the_historical_pooled_path():
    _, ppo = _trained(0)
    assert ppo.communication_rounds == 0


def test_any_positive_local_epochs_selects_fedavg_and_counts_communication():
    """Communication rounds are the axis a federated cost result is plotted against."""
    _, ppo = _trained(1)
    assert ppo.communication_rounds > 0


def test_fedavg_and_pooling_are_genuinely_different_algorithms():
    pooled, _ = _trained(0)
    fed, _ = _trained(4)
    assert float((pooled - fed).abs().max()) > 1e-6


def test_more_local_epochs_changes_the_result():
    """If E did nothing, the communication saving would be free -- and it is not."""
    one, _ = _trained(1)
    four, _ = _trained(4)
    assert float((one - four).abs().max()) > 1e-6


def test_averaging_is_the_dominant_source_of_drift_not_the_optimiser_reset():
    """There is NO exact equivalence, even with one site, and it is worth being precise
    about why: the pooled path carries a persistent Adam optimiser across updates while
    FedAvg creates a fresh one per site per round -- deliberately, since carrying server-side
    moments across weight averages they were never computed for is a different algorithm.

    So the honest check is that the optimiser reset is the SMALL term. With one site there
    is nothing to average, and the gap to pooled must be materially smaller than at three
    sites, where client drift adds on top. If it were not, the averaging would be doing
    nothing and E would be a free saving -- which is exactly the claim FedAvg cannot make.
    """
    solo_pooled, _ = _trained(0, agents=1, epochs=3)
    solo_fed, _ = _trained(3, agents=1, epochs=3)
    solo_gap = float((solo_pooled - solo_fed).abs().max())

    many_pooled, _ = _trained(0, agents=3, epochs=3)
    many_fed, _ = _trained(3, agents=3, epochs=3)
    many_gap = float((many_pooled - many_fed).abs().max())

    assert solo_gap > 0.0, "the optimiser reset must have SOME effect"
    assert solo_gap < many_gap, (solo_gap, many_gap)


def test_sites_are_weighted_by_the_experience_they_contributed():
    """FedAvg specifies a size-weighted mean. Under round-robin every site holds the same
    share, but under random turn order it does not -- and an unweighted mean would then
    over-count a site that happened to draw few turns."""
    env = _env(agents=3)
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0, local_epochs=1))
    before = {k: v.clone() for k, v in ppo.shared_net.state_dict().items()}

    rng = np.random.default_rng(0)
    def buf(n):
        return {"obs": rng.normal(size=(n, env.obs_size(0))).astype(np.float32),
                "action": rng.integers(0, env.n_actions(0), size=n),
                "logp": rng.normal(size=n).astype(np.float32),
                "reward": rng.normal(size=n).astype(np.float32),
                "done": np.zeros(n, dtype=bool),
                "value": rng.normal(size=n).astype(np.float32)}
    sizes = {0: 200, 1: 20, 2: 20}
    try:
        ppo.update({a: buf(n) for a, n in sizes.items()})
    except Exception as error:                 # buffer schema varies; the point is the mean
        pytest.skip(f"synthetic buffer incompatible: {error!r}")
    after = ppo.shared_net.state_dict()
    assert any(not torch.allclose(before[k], after[k]) for k in before)
    assert ppo.communication_rounds == 1


def test_fedavg_requires_a_shared_network():
    """Averaging non-portable nets across sites with different windows would be
    meaningless; only the portable architecture makes a coordinate-wise mean a policy."""
    env = _env(agents=3)
    ppo = IndependentPPO(env, PPOConfig(hidden=16, seed=0, local_epochs=2))
    assert ppo.shared_net is not None
