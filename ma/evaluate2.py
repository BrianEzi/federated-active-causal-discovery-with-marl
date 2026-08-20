"""PHASE 6 -- the three-part success criterion [U14].

    1. PRIVATE   each agent recovers its own private substructure as a DAG.
    2. SHARED    each agent recovers the shared structure to CPDAG resolution -- orientation
                 within an equivalence class is not required where it is not identifiable.
    3. GLOBAL    the union of the two agents' recovered structures resolves to the true
                 global graph, and is ACYCLIC.

WHY THE ACYCLICITY CHECK IS NOT REDUNDANT. `ma/env.py` comments that global identification
"needs nothing extra", because two fully correct induced DAGs union to the true graph. That
argument is sound for FULL DAG recovery. But criterion 2 relaxes the shared part to CPDAG,
and two agents that orient a shared edge differently within the same equivalence class can
union into a cycle. The check earns its place precisely because of the relaxation.

WHY THE JOINT OBJECT, NOT THE MARGINALS. ~10% of posterior mass can sit on a wrong skeleton
while every edge marginal looks correct, so a marginal-based criterion would pass on graphs
that are jointly wrong. Everything here reads posterior mass over sets of DAGs.

The credit set for criterion 2 is the DAGs that (a) agree with the truth exactly on every
edge touching a private node, and (b) are Markov equivalent to the truth. Interventions on
private nodes make (a) identifiable; (b) is the honest limit of what shared data can pin
down.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ma.baselines2 import _Window, enumerated_posterior
from ma.belief_dp import JOINT_CONF
from ma.env2 import AGENTS, TwoAgentEnv2
from sa.graphs import is_acyclic, mec_signature


def credit_set(window, truth: np.ndarray) -> np.ndarray:
    """Boolean mask over the window's DAGs: which count as a correct answer.

    A DAG qualifies when it matches the truth on every private-incident edge AND lies in
    the truth's Markov equivalence class.
    """
    space = _Window.get(window.k)
    private = [window.pos[node] for node in window.private]

    # Equivalence membership is one array comparison against the precomputed partition,
    # not 543 signature computations per call.
    mask = space.mec_id == space.id_of(truth)
    if not mask.any():
        return mask
    # Private-incident edges must match EXACTLY -- both the node's row and its column, so
    # orientation counts, not just adjacency. This is [U14] part 1, and it is boundary
    # inclusive: at (1,1,3) it pins 3 edges per agent.
    for p in private:
        same_row = (space.dags[:, p, :] == truth[p, :]).all(axis=1)
        same_col = (space.dags[:, :, p] == truth[:, p]).all(axis=1)
        mask &= same_row & same_col
    return mask


def credit_candidates(window, truth: np.ndarray) -> np.ndarray:
    """The credit set, built WITHOUT enumerating the window.

    `credit_set` returns a mask over all 543 window DAGs, which is fine for reporting but
    puts window enumeration on the path of anything that runs per step. It is also
    unnecessary: criterion 1 pins every edge INCIDENT TO A PRIVATE NODE to the truth, so
    the only freedom left is the SHARED-SHARED subgraph.

    So enumerate the shared subgraph alone -- 25 DAGs at |X| = 3 against 543 for the
    window -- graft each onto the truth's fixed private-incident structure, and keep the
    ones that are acyclic and Markov equivalent to the truth.

    Cost is exponential in |X| and constant in the window size, which is the same axis the
    confounding enumeration already costs and the axis the federation boundary keeps small
    by design. The window may grow to the k the subset DP reaches without this term moving.
    """
    from sa.graphs import build_graph_space

    shared = [window.pos[node] for node in window.shared]
    truth = np.asarray(truth) > 0.5
    space = build_graph_space(len(shared))
    target = mec_signature(truth)

    out = []
    for sub in np.asarray(space.dags, dtype=bool):
        candidate = truth.copy()
        # Replace only the shared-shared block; everything touching a private node is
        # pinned by criterion 1 and must stay exactly as the truth has it.
        for a, u in enumerate(shared):
            for b, v in enumerate(shared):
                if u != v:
                    candidate[u, v] = sub[a, b]
        if not is_acyclic(candidate.astype(np.int8)):
            continue
        if mec_signature(candidate) == target:
            out.append(candidate.copy())
    return np.asarray(out)


def agent_report(env: TwoAgentEnv2, name: str) -> Dict[str, float]:
    """Posterior mass on each of the criteria, for one agent."""
    window = env.windows[name]
    truth = window.induced(env.true_adjacency)
    clean = (env.clean[name] if env.config.disclose_regime
             else np.zeros(len(env.samples), dtype=bool))
    posterior = enumerated_posterior(
        window, env.samples[:, window.nodes], env.known[name], clean,
        env.config.score_rule)
    space = _Window.get(window.k)

    exact = (space.dags == truth).all(axis=(1, 2))
    # `mec_signature(truth)` was being recomputed inside a 543-iteration comprehension.
    equivalent = space.mec_id == space.id_of(truth)
    credit = credit_set(window, truth)

    if env.config.score_rule == JOINT_CONF:
        # THE POSTERIOR IS INDEXED BY THE AUGMENTED GRAPH, NOT THE CAUSAL ONE.
        #
        # Under joint_conf a hypothesis is (DAG H, confounding set P) with P's edges
        # present in H, so `posterior[credit].sum()` compares H against the true CAUSAL
        # graph. On a confounded episode the truth contains no confounding edge, so it
        # matches only under the empty assignment -- the hypothesis that refuses to model
        # the confounding -- and the reported success rate was EXACTLY 0.000 on every
        # confounded episode. The metric could not score the case the design exists for.
        candidates = space.dags[credit]
        pairs = env._confounded_positions(name)
        mass_credit = window.belief.joint_conf_set_probability(
            env.samples[:, window.nodes], env.known[name], clean, candidates, pairs)
        # MAP over CAUSAL graphs, restricted to the credit set. Only used for the union
        # check, and if the agent is not credited then success is already false, so the
        # restriction costs nothing. Credit sets are small -- they range over the shared
        # subgraph only -- so this is a handful of lookups.
        best, map_index = -1.0, int(np.argmax(posterior))
        for local, index in enumerate(np.flatnonzero(credit)):
            value = window.belief.joint_conf_dag_probability(
                env.samples[:, window.nodes], env.known[name], clean,
                space.dags[index], pairs)
            if value > best:
                best, map_index = value, int(index)
    else:
        mass_credit = float(posterior[credit].sum())
        map_index = int(np.argmax(posterior))

    return {
        "mass_exact": float(posterior[exact].sum()),
        "mass_equivalent": float(posterior[equivalent].sum()),
        "mass_credit": float(mass_credit),
        "map_index": map_index,
    }


def union_graph(env: TwoAgentEnv2, map_indices: Dict[str, int]) -> np.ndarray:
    """Stitch both agents' MAP window graphs into a global adjacency.

    Shared edges are claimed by both agents. Disagreement is resolved by OR, which is the
    permissive choice: it can only create cycles, never hide them, so the acyclicity check
    below sees the worst case rather than a tidied-up one.
    """
    d = env.topology.d
    union = np.zeros((d, d), dtype=np.int8)
    for name in AGENTS:
        window = env.windows[name]
        dag = _Window.get(window.k).dags[map_indices[name]]
        for i, u in enumerate(window.nodes):
            for j, v in enumerate(window.nodes):
                if dag[i, j]:
                    union[u, v] = 1
    return union


def evaluate_episode(env: TwoAgentEnv2) -> Dict[str, object]:
    """Every criterion for one finished episode."""
    threshold = env.config.identify_threshold
    reports = {name: agent_report(env, name) for name in AGENTS}
    map_indices = {name: reports[name]["map_index"] for name in AGENTS}
    union = union_graph(env, map_indices)

    acyclic = bool(is_acyclic(union))
    matches_truth = bool(np.array_equal(union, np.asarray(env.true_adjacency)))
    # Equivalence of the GLOBAL graph, the honest version of "resolves to the true graph"
    # once the shared part is only pinned to CPDAG.
    globally_equivalent = acyclic and (
        mec_signature(union) == mec_signature(np.asarray(env.true_adjacency)))

    return {
        "per_agent": reports,
        "private_and_shared_ok": {
            name: reports[name]["mass_credit"] >= threshold for name in AGENTS},
        "exact_ok": {name: reports[name]["mass_exact"] >= threshold for name in AGENTS},
        "union_acyclic": acyclic,
        "union_matches_truth": matches_truth,
        "union_equivalent": globally_equivalent,
        # All three parts of [U14], which is the number to report.
        "success": bool(all(reports[n]["mass_credit"] >= threshold for n in AGENTS)
                        and acyclic and globally_equivalent),
    }


def run_arm(env: TwoAgentEnv2, policies: Dict[str, object], episodes: int,
            seed: int = 0) -> Dict[str, object]:
    """Play one (policy pair) over seeded episodes and score every criterion."""
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)

    rows: List[Dict[str, object]] = []
    clamps = moves = 0
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            actions = {n: policies[n](env, result) for n in AGENTS}
            for name, index in actions.items():
                node, mode = env.windows[name].actions[index]
                if node == -1:
                    continue
                moves += 1
                clamps += (mode == "clamp")
            result = env.step(actions["A"], actions["B"])
        row = evaluate_episode(env)
        row["threshold_identified"] = result.info["both_identified"]
        row["steps"] = max(result.n_interventions.values())
        rows.append(row)

    def rate(key) -> float:
        return float(np.mean([bool(r[key]) for r in rows]))

    return {
        "episodes": episodes,
        "success": rate("success"),
        "success_ci": bootstrap_ci([float(r["success"]) for r in rows], seed=seed),
        "threshold_identified": rate("threshold_identified"),
        "union_acyclic": rate("union_acyclic"),
        "union_equivalent": rate("union_equivalent"),
        "union_matches_truth": rate("union_matches_truth"),
        "private_and_shared_A": float(np.mean(
            [r["private_and_shared_ok"]["A"] for r in rows])),
        "private_and_shared_B": float(np.mean(
            [r["private_and_shared_ok"]["B"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "clamp_fraction": float(clamps / moves) if moves else float("nan"),
    }


def bootstrap_ci(values: Sequence[float], seed: int = 0, draws: int = 2000) -> List[float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
