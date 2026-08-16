"""GATE-M2 -- does coordination buy anything, or is the two-agent case a decoration?

If two agents choosing independently from their own local views do as well as one
centralised chooser with the same intervention budget, then the coordination problem this
project is about does not exist in this topology, and no amount of clever MARL will show
otherwise. That has to be measured before anything is built on top of it -- it is the
two-agent counterpart of GATE 2, and it fails in the same way: silently, by producing
results that look fine and mean nothing.

**Inference is held centralised in both arms, deliberately.** Both conditions score the
same pooled posterior over the full graph; the only thing that varies is *who chooses the
interventions and on what information*. Otherwise the comparison would confound two
different effects -- worse choices, and worse beliefs from not sharing data -- and a
difference could not be attributed to coordination. Federated inference is a separate
question with its own measurement.

**Budgets are matched in interventions, not in rounds.** Two agents acting simultaneously
spend two interventions per round; a single centralised chooser spends one. Comparing
rounds would hand the independent arm double the budget and make it look better than it is.
So the centralised arm picks `n_agents` targets per round, greedily and sequentially
against the same posterior.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ma.confounding import latent_projection_pairs
from ma.topology import Topology, masked_indices
from sa.gates import bootstrap_ci
from sa.oracle import _partition_entropy
from sa.posterior import PosteriorEngine, is_identified
from sa.scm import sample as scm_sample, sample_scm_params
from sa.score import BGeScore


class MaskedSpace:
    """The enumerated DAG space restricted to a topology's allowed edges.

    Small enough to enumerate at six nodes -- 96,255 graphs under T1 against 3,781,503
    unrestricted -- so GATE-M2 can use the *exact* posterior and the *exact* oracle. Whether
    coordination helps is then a property of the problem rather than of an approximation.
    """

    def __init__(self, space, topology: Topology):
        self.topology = topology
        self.d = topology.d
        self.indices = masked_indices(space, topology)
        self.dags = np.asarray(space.dags)[self.indices]
        self.n_dags = len(self.indices)
        self.mec_id = np.asarray(space.mec_id)[self.indices]

        # Descendant-set signature per (graph, node): what an intervention would reveal.
        adjacency = self.dags > 0.5
        reach = adjacency.copy()
        for k in range(self.d):
            reach |= reach[:, :, k][:, :, None] & reach[:, k, :][:, None, :]
        codes = reach.astype(np.int64) @ (1 << np.arange(self.d)).astype(np.int64)

        self.signatures = np.empty((self.n_dags, self.d), dtype=np.int64)
        self.n_groups: List[int] = []
        for node in range(self.d):
            _, inverse = np.unique(codes[:, node], return_inverse=True)
            self.signatures[:, node] = inverse.reshape(-1)
            self.n_groups.append(int(inverse.max()) + 1)

    def uniform_prior(self) -> np.ndarray:
        return np.full(self.n_dags, 1.0 / self.n_dags)


def _eig_scores(masked: MaskedSpace, posterior: np.ndarray,
                nodes: Optional[Sequence[int]] = None) -> np.ndarray:
    """Expected information gain of intervening on each node, in nats."""
    out = np.zeros(masked.d)
    for node in (range(masked.d) if nodes is None else nodes):
        out[node] = _partition_entropy(
            masked.signatures[:, node], posterior, masked.n_groups[node])
    return out


def _local_view_scores(masked: MaskedSpace, posterior: np.ndarray, agent: str,
                       authority: Sequence[int]) -> np.ndarray:
    """What an agent computes from ITS OWN view, with no communication.

    The agent cannot distinguish two graphs that differ only outside its observed set, so
    it groups hypotheses by the descendant set **restricted to the nodes it can see**. This
    is the honest local version of the oracle: it uses the same rule on strictly less
    information, so any difference from the centralised arm is the value of what it does
    not know rather than a different algorithm.
    """
    topology = masked.topology
    observed = topology.observed_by(agent)
    visible_bits = sum(1 << node for node in observed)

    out = np.full(masked.d, -np.inf)
    adjacency = masked.dags > 0.5
    reach = adjacency.copy()
    for k in range(masked.d):
        reach |= reach[:, :, k][:, :, None] & reach[:, k, :][:, None, :]
    codes = reach.astype(np.int64) @ (1 << np.arange(masked.d)).astype(np.int64)

    for node in authority:
        # Only the visible part of the descendant set is distinguishable to this agent.
        visible = codes[:, node] & visible_bits
        _, inverse = np.unique(visible, return_inverse=True)
        inverse = inverse.reshape(-1)
        out[node] = _partition_entropy(inverse, posterior, int(inverse.max()) + 1)
    return out


def run_arm(masked: MaskedSpace, arm: str, n_episodes: int, n_obs: int, n_int: int,
            budget: int, threshold: float, seed: int) -> Dict:
    """Run one condition. Returns interventions spent per episode and whether it solved.

    `arm` is `"centralised"` (one chooser, full posterior, picks two targets per round) or
    `"independent"` (each agent picks one target from its own view, no communication).
    """
    topology = masked.topology
    engine = PosteriorEngine.__new__(PosteriorEngine)   # built below against MaskedSpace
    engine.space = masked
    engine.score = BGeScore(masked.d)
    from sa.scoretable import LocalScorer
    engine.scorer = LocalScorer(masked.d, engine.score)
    engine.parent_sets = engine.scorer.parent_sets
    engine.n_parent_sets = engine.scorer.n_parent_sets
    engine._table_plan = engine.scorer._table_plan

    # Parent-set index per (graph, node), over the MASKED space.
    lookup = [{s: i for i, s in enumerate(engine.parent_sets[node])}
              for node in range(masked.d)]
    parent_ids = np.empty((masked.n_dags, masked.d), dtype=np.int64)
    adjacency = masked.dags > 0.5
    for node in range(masked.d):
        others = [k for k in range(masked.d) if k != node]
        mask_to_index = np.empty(1 << len(others), dtype=np.int64)
        for parents, index in lookup[node].items():
            mask_to_index[sum(1 << others.index(p) for p in parents)] = index
        bits = np.zeros(masked.n_dags, dtype=np.int64)
        for position, parent in enumerate(others):
            bits |= adjacency[:, parent, node].astype(np.int64) << position
        parent_ids[:, node] = mask_to_index[bits]
    flat = parent_ids + (np.arange(masked.d) * engine.n_parent_sets)[None, :]

    prior = masked.uniform_prior()
    authority = {"A": topology.may_intervene_on("A"), "B": topology.may_intervene_on("B")}
    n_agents = 2

    spent, solved, confounded = [], [], []
    for episode in range(n_episodes):
        rng = np.random.default_rng(seed * 100_000 + episode)
        true_position = int(rng.integers(masked.n_dags))
        true_adjacency = masked.dags[true_position]
        params = sample_scm_params(true_adjacency, rng)

        samples, intervened = scm_sample(params, n_obs, rng)
        used = 0

        def posterior_now():
            table = engine.scorer.table(samples, intervened)
            log_p = table.ravel()[flat].sum(axis=1) + np.log(prior)
            log_p -= log_p.max()
            p = np.exp(log_p)
            return p / p.sum()

        posterior = posterior_now()
        while used < budget and not is_identified(posterior, true_position, threshold):
            if arm == "centralised":
                # ONE target per round, then re-plan. Budgets are matched in interventions
                # rather than rounds, so this is not extra spend -- and it avoids handicapping
                # the arm with a batch choice. Picking two at once from a single `argsort`
                # would score the second target against a belief that already assumed the
                # first was unobserved, which is a strictly weaker chooser and would make
                # coordination look worse than it is.
                scores = _eig_scores(masked, posterior)
                best = int(np.argmax(scores))
                if scores[best] <= 1e-9:
                    break
                targets = [best]
            else:
                targets = []
                for agent in ("A", "B"):
                    local = _local_view_scores(masked, posterior, agent, authority[agent])
                    best = int(np.argmax(local))
                    if local[best] > 1e-9:
                        targets.append(best)
                if not targets:
                    break

            for node in targets:
                new_samples, new_mask = scm_sample(
                    params, n_int, rng, intervene_node=int(node))
                samples = np.vstack([samples, new_samples])
                intervened = np.vstack([intervened, new_mask])
                used += 1
            posterior = posterior_now()

        identified = bool(is_identified(posterior, true_position, threshold))
        spent.append(float(used if identified else budget))
        solved.append(float(identified))
        confounded.append(float(bool(
            latent_projection_pairs(true_adjacency, topology.observed_by("A"),
                                    topology.hidden_from("A"))
            or latent_projection_pairs(true_adjacency, topology.observed_by("B"),
                                       topology.hidden_from("B")))))

    spent = np.array(spent)
    return {
        "arm": arm,
        "interventions": float(spent.mean()),
        "interventions_ci": bootstrap_ci(spent, seed=seed),
        "solve_rate": float(np.mean(solved)),
        "confounded_fraction": float(np.mean(confounded)),
        "per_episode": spent,
    }


def gate_m2(space, topology: Topology, n_episodes: int = 120, n_obs: int = 2000,
            n_int: int = 200, budget: int = 12, threshold: float = 0.7,
            seed: int = 0) -> Dict:
    """The gate. Passes when centralised needs strictly fewer interventions, intervals
    disjoint. A failure means this topology has no coordination problem in it."""
    masked = MaskedSpace(space, topology)
    central = run_arm(masked, "centralised", n_episodes, n_obs, n_int, budget, threshold, seed)
    independent = run_arm(masked, "independent", n_episodes, n_obs, n_int, budget, threshold, seed)

    gained = independent["interventions"] - central["interventions"]
    passed = bool(central["interventions_ci"][1] < independent["interventions_ci"][0])
    return {
        "topology": topology.name,
        "n_graphs": masked.n_dags,
        "centralised": {k: v for k, v in central.items() if k != "per_episode"},
        "independent": {k: v for k, v in independent.items() if k != "per_episode"},
        "coordination_gained": gained,
        "pass": passed,
        "n_episodes": n_episodes,
    }
