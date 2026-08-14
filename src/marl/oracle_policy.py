"""Information-optimal oracle intervention policy, and agreement scoring against it.

Motivation (see docs/INVESTIGATION_GRAPH_HEAD_REGRESSION.md): every metric used so far
to judge the agents -- SHD, reached-SHD0, target diversity -- either conflates policy
quality with estimator quality, or measures diversity without regard to whether that
diversity was *useful*. The UCB experiment made this concrete: target diversity went from
~10% to 100% while reached-SHD0 went to 0%, i.e. the agent explored more and achieved
less, and no existing metric distinguished "explored usefully" from "explored blindly."

This module provides the missing reference point. Because this environment's hypothesis
space is small and fully enumerable (8 candidate topologies, see
src/generators.py::get_all_4node_topologies), the information-optimal next intervention
is exactly computable rather than merely approximable: for a given posterior over those
8 hypotheses, intervening on node i is valuable to the extent that the hypotheses
*disagree* about what intervening on i would produce. If every plausible hypothesis
predicts the same outcome for do(X_i), that intervention cannot discriminate between them
and is worthless no matter how "unexplored" node i is.

Scoring rule (`expected_discrimination`): under a hard intervention on node i, each
hypothesis h predicts a specific set of descendants of i whose distributions shift.
Represent that as h's reachability vector from i, r_h(i) in {0,1}^d. Two hypotheses are
distinguishable by do(X_i) exactly when r_h(i) != r_h'(i). The discrimination score for
node i is the posterior-weighted probability that a randomly drawn pair of hypotheses
disagrees:

    score(i) = sum_{h,h'} P(h) P(h') * [r_h(i) != r_h'(i)]

which is 1 - sum_g P(g)^2 over the *groups* g of hypotheses sharing an identical
reachability vector -- a Gini/Simpson diversity index over the partition that do(X_i)
induces on the hypothesis space. It is 0 when all plausible hypotheses agree (useless
intervention) and maximal when the intervention splits the posterior into many equally
likely, mutually distinguishable groups. This is a standard, well-founded
experiment-design criterion, not an ad-hoc heuristic, and it needs no simulation: pure
combinatorics over the known candidate set plus the posterior.

Deliberate scope notes (stated rather than silently assumed):
- Uses reachability (transitive descendants), not just direct children, because a hard
  intervention's distributional effect propagates through the whole downstream subgraph.
- Assumes the `hard` intervention regime (the project default), consistent with
  src/marl/bayes_optimal_estimator.py. Under `soft_shift` the reachability set is
  unchanged but effect magnitudes differ; the *partition* this scoring depends on is the
  same, so the ranking is still meaningful, though the "distinguishable in principle"
  reading is cleanest under hard interventions.
- This is a single-step (myopic/greedy) optimum, not a full sequential-design optimum.
  Computing the latter would require search over intervention sequences; the greedy
  criterion is the standard tractable choice and is what "what should the agent do *now*"
  actually asks.
"""
import numpy as np
from typing import Dict, Tuple


def compute_reachability(adjacency: np.ndarray) -> np.ndarray:
    """[d, d] boolean matrix: reach[i, j] is True iff j is reachable from i via a
    directed path of length >= 1 (i.e. j is a strict descendant of i)."""
    d = adjacency.shape[0]
    reach = (np.asarray(adjacency) > 0.5).astype(bool).copy()
    # Transitive closure (Floyd-Warshall style; d is tiny so this is trivially cheap).
    for k in range(d):
        reach = reach | (reach[:, k][:, None] & reach[k, :][None, :])
    return reach


def expected_discrimination(posterior: np.ndarray, candidate_adjacencies: np.ndarray) -> np.ndarray:
    """Per-node information value [d]: the posterior-weighted probability that two
    randomly drawn hypotheses are distinguishable by intervening on that node.

    See the module docstring for the derivation. Returns 0 for a node all plausible
    hypotheses agree about (no discriminating power), higher for nodes that split the
    posterior into many equally-likely distinguishable groups.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    candidate_adjacencies = np.asarray(candidate_adjacencies)
    H, d, _ = candidate_adjacencies.shape

    reach_per_hyp = np.stack([compute_reachability(candidate_adjacencies[h]) for h in range(H)])  # [H, d, d]

    scores = np.zeros(d)
    for node in range(d):
        # Group hypotheses by their reachability signature from `node`.
        group_mass: Dict[bytes, float] = {}
        for h in range(H):
            sig = reach_per_hyp[h, node].tobytes()
            group_mass[sig] = group_mass.get(sig, 0.0) + posterior[h]
        # Gini/Simpson: probability two independent draws land in *different* groups.
        scores[node] = 1.0 - sum(m * m for m in group_mass.values())
    return scores


def oracle_best_targets(
    posterior: np.ndarray,
    candidate_adjacencies: np.ndarray,
    valid_mask: np.ndarray = None,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (scores [d], best_targets [d] boolean).

    `best_targets` marks every node achieving the maximum score (a set, not a single
    argmax, because ties are common and genuinely equivalent -- scoring an agent wrong
    for picking a tied-optimal node would be a metric artifact, not a real error).
    `valid_mask` [d]: if given, restricts consideration to nodes the agent was actually
    allowed to target, so agreement measures the agent's choice among *its own* legal
    options rather than penalizing it for not picking a node it could never have chosen.
    """
    scores = expected_discrimination(posterior, candidate_adjacencies)
    considered = scores.copy()
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask) > 0.5
        considered = np.where(valid_mask, considered, -np.inf)
    best_value = np.max(considered)
    best_targets = considered >= best_value - tol
    return scores, best_targets


def score_agent_choice(
    chosen_target: int,
    posterior: np.ndarray,
    candidate_adjacencies: np.ndarray,
    valid_mask: np.ndarray = None,
) -> Dict[str, float]:
    """Scores one agent's chosen intervention target against the oracle (target-level only)."""
    scores, best_targets = oracle_best_targets(posterior, candidate_adjacencies, valid_mask)
    best_score = float(np.max(np.where(np.asarray(valid_mask) > 0.5, scores, -np.inf))) if valid_mask is not None else float(np.max(scores))
    chosen_score = float(scores[chosen_target])
    if best_score <= 1e-12:
        normalized = 1.0
        is_opt = 1.0
    else:
        is_opt = float(bool(best_targets[chosen_target]))
        normalized = chosen_score / best_score
    return {
        "is_optimal": is_opt,
        "score": chosen_score,
        "best_score": best_score,
        "regret": max(0.0, best_score - chosen_score),
        "normalized_score": normalized,
    }


def score_agent_action(
    category: int,
    target: int,
    posterior: np.ndarray,
    candidate_adjacencies: np.ndarray,
    valid_mask: np.ndarray = None,
) -> Dict[str, float]:
    """Scores a full action (category + target) against the information-optimal oracle.

    Handles the stopping decision (NOOP vs INTERVENE):
    - When uncertainty is resolved (best_score <= 1e-12), NOOP is optimal (1.0) and INTERVENE is suboptimal (0.0).
    - When information remains (best_score > 1e-12), NOOP is suboptimal (0.0) and INTERVENE is scored by target quality.
    """
    from src.types import ActionCategory
    scores, best_targets = oracle_best_targets(posterior, candidate_adjacencies, valid_mask)
    best_score = float(np.max(np.where(np.asarray(valid_mask) > 0.5, scores, -np.inf))) if valid_mask is not None else float(np.max(scores))

    if best_score <= 1e-12:
        # Uncertainty is resolved: NOOP is the only optimal action
        if category == int(ActionCategory.NOOP):
            return {
                "is_optimal": 1.0,
                "score": 0.0,
                "best_score": 0.0,
                "regret": 0.0,
                "normalized_score": 1.0,
                "action_type": "noop_correct"
            }
        else:
            return {
                "is_optimal": 0.0,
                "score": 0.0,
                "best_score": 0.0,
                "regret": 0.5,  # Penalize wasting budget when solved
                "normalized_score": 0.0,
                "action_type": "intervene_unnecessary"
            }
    else:
        # Information to gain: INTERVENE on optimal target is required
        if category == int(ActionCategory.NOOP):
            return {
                "is_optimal": 0.0,
                "score": 0.0,
                "best_score": best_score,
                "regret": best_score,
                "normalized_score": 0.0,
                "action_type": "noop_premature"
            }
        else:
            chosen_score = float(scores[target])
            is_opt = float(bool(best_targets[target]))
            normalized = chosen_score / best_score
            return {
                "is_optimal": is_opt,
                "score": chosen_score,
                "best_score": best_score,
                "regret": max(0.0, best_score - chosen_score),
                "normalized_score": normalized,
                "action_type": "intervene_optimal" if is_opt else "intervene_suboptimal"
            }

