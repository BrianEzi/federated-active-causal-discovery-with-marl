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

from ma.baselines import _Window, enumerated_posterior
from crosscheck.belief_dp import JOINT_CONF
from ma.env import TwoAgentEnv
from ma.graphs import is_acyclic, mec_signature


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
    from ma.graphs import build_graph_space

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


def agent_report(env: TwoAgentEnv, agent: int) -> Dict[str, float]:
    """Posterior mass on each of the criteria, for one agent.

    Under the constraint backend the analogues are replicate fractions, not masses
    (`cb/backend.py` gives the criterion): strict recovery for `mass_exact`, consistency
    alone (adjacency + confounding right, orientations sound, nothing required) for
    `mass_equivalent`, private-pinned for `mass_credit`. `map_index` has no analogue --
    nothing enumerates the window -- and is -1.
    """
    if env.config.belief_backend == "constraint":
        window = env.windows[agent]
        mag = env._true_mag(agent)
        private_positions = [window.pos[n] for n in window.private]
        return {
            "mass_exact": float(window.belief.credit_fraction(mag, strict=True)),
            "mass_equivalent": float(window.belief.credit_fraction(mag)),
            "mass_credit": float(window.belief.credit_fraction(mag, private_positions)),
            "map_index": -1,
        }

    window = env.windows[agent]
    truth = window.induced(env.true_adjacency)
    clean = (env.clean[agent] if env.config.disclose_regime
             else np.zeros(len(env.samples), dtype=bool))
    posterior = enumerated_posterior(
        window, env.samples[:, window.nodes], env.known[agent], clean,
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
        pairs = env._confounded_positions(agent)
        mass_credit = window.belief.joint_conf_set_probability(
            env.samples[:, window.nodes], env.known[agent], clean, candidates, pairs)
        # MAP over CAUSAL graphs, restricted to the credit set. Only used for the union
        # check, and if the agent is not credited then success is already false, so the
        # restriction costs nothing. Credit sets are small -- they range over the shared
        # subgraph only -- so this is a handful of lookups.
        best, map_index = -1.0, int(np.argmax(posterior))
        for local, index in enumerate(np.flatnonzero(credit)):
            value = window.belief.joint_conf_dag_probability(
                env.samples[:, window.nodes], env.known[agent], clean,
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


def union_graph(env: TwoAgentEnv, map_indices: Dict[int, int]) -> np.ndarray:
    """Stitch all agents' MAP window graphs into a global adjacency.

    Shared edges are claimed by multiple agents. Disagreement is resolved by OR, which is the
    permissive choice: it can only create cycles, never hide them, so the acyclicity check
    below sees the worst case rather than a tidied-up one.
    """
    d = env.topology.d
    union = np.zeros((d, d), dtype=np.int8)
    for agent in env.topology.agents:
        window = env.windows[agent]
        dag = _Window.get(window.k).dags[map_indices[agent]]
        for i, u in enumerate(window.nodes):
            for j, v in enumerate(window.nodes):
                if dag[i, j]:
                    union[u, v] = 1
    return union


def evaluate_episode(env: TwoAgentEnv) -> Dict[str, object]:
    """Every criterion for one finished episode."""
    threshold = env.config.identify_threshold
    reports = {agent: agent_report(env, agent) for agent in env.topology.agents}
    map_indices = {agent: reports[agent]["map_index"] for agent in env.topology.agents}
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
            agent: reports[agent]["mass_credit"] >= threshold for agent in env.topology.agents},
        "exact_ok": {agent: reports[agent]["mass_exact"] >= threshold for agent in env.topology.agents},
        "union_acyclic": acyclic,
        "union_matches_truth": matches_truth,
        "union_equivalent": globally_equivalent,
        # All three parts of [U14], which is the number to report.
        "success": bool(all(reports[a]["mass_credit"] >= threshold for a in env.topology.agents)
                        and acyclic and globally_equivalent),
    }


def run_arm(env: TwoAgentEnv, policies: Dict[int, object], episodes: int,
            seed: int = 0) -> Dict[str, object]:
    """Play policy set over seeded episodes and score every criterion."""
    for policy in policies.values():
        if hasattr(policy, "reset"):
            policy.reset(seed)

    rows: List[Dict[str, object]] = []
    clamps = moves = 0
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            actions = {a: policies[a](env, result) for a in env.topology.agents}
            result = env.step(actions)
            # Tallied AFTER the step, from what the environment actually applied. Counting
            # the submitted actions instead double-counts under turn-taking, where the
            # inactive agent's move is discarded by the protocol.
            for node, mode in env.last_chosen.values():
                if node == -1:
                    continue
                moves += 1
                clamps += (mode == "clamp")
        row = evaluate_episode(env)
        info = result.info
        row["threshold_identified"] = info["both_identified"]
        row["steps"] = max(result.n_interventions.values())
        row["rounds"] = info["rounds"]
        # Per agent, never a max. An idle agent is invisible inside `steps`, and free-riding
        # is precisely what the shared round budget was introduced to make measurable.
        row["interventions"] = dict(info["interventions"])
        row["forfeits"] = dict(info["forfeits"])
        # Clamps split by target REGION. Only a clamp on one's OWN PRIVATE node de-confounds
        # for a partner; a clamp on a shared node does nothing for them. The aggregate clamp
        # fraction cannot separate altruism from self-interest, so it is not evidence of
        # cooperation on its own.
        row["clamps_private"] = dict(info["clamps_private"])
        row["clamps_shared"] = dict(info["clamps_shared"])
        row["done_bit"] = dict(info["done_bit"])
        row["connected"] = bool(info["connected"])
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
        # `mean_steps` is per-agent INTERVENTIONS and `mean_rounds` is episode LENGTH.
        # They coincide under simultaneous play and differ by ~2x under turn-taking, so
        # both are reported and neither may be quoted as the other across protocols.
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_rounds": float(np.mean([r["rounds"] for r in rows])),
        **_per_agent_block(env, rows),
        "clamp_fraction": float(clamps / moves) if moves else float("nan"),
    }


def _per_agent_block(env: TwoAgentEnv, rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Per-agent behaviour, the connectedness split, and the free-rider index.

    `free_rider_index` is `min(interventions) / max(interventions)` across agents: 1.0 when
    the agents pull their weight evenly, 0.0 when one agent did nothing at all. Episodes in
    which NOBODY acted are excluded -- the ratio is undefined there and would otherwise read
    as perfect cooperation.

    Every headline is also reported split by CONNECTED. A disconnected graph gives the agents
    independent subproblems -- no cross-boundary path, so no confounding and nothing to
    coordinate about -- and pooling those episodes with connected ones dilutes exactly the
    effect this project exists to measure.
    """
    def mean_over(key: str, agent: int) -> float:
        return float(np.mean([r[key][agent] for r in rows]))

    ratios = []
    for r in rows:
        counts = [r["interventions"][a] for a in env.topology.agents]
        if max(counts) > 0:
            ratios.append(min(counts) / max(counts))

    out: Dict[str, object] = {
        "interventions_per_agent": {a: mean_over("interventions", a) for a in env.topology.agents},
        "forfeits_per_agent": {a: mean_over("forfeits", a) for a in env.topology.agents},
        "clamps_private_per_agent": {a: mean_over("clamps_private", a) for a in env.topology.agents},
        "clamps_shared_per_agent": {a: mean_over("clamps_shared", a) for a in env.topology.agents},
        "done_bit_per_agent": {a: mean_over("done_bit", a) for a in env.topology.agents},
        "free_rider_index": float(np.mean(ratios)) if ratios else float("nan"),
        "never_acted_episodes": float(np.mean(
            [max(r["interventions"][a] for a in env.topology.agents) == 0 for r in rows])),
        "connected_fraction": float(np.mean([r["connected"] for r in rows])),
    }
    for label, want in (("connected", True), ("disconnected", False)):
        subset = [r for r in rows if r["connected"] is want]
        out["success_%s" % label] = (
            float(np.mean([float(r["success"]) for r in subset])) if subset else float("nan"))
        out["episodes_%s" % label] = len(subset)
    return out


def bootstrap_ci(values: Sequence[float], seed: int = 0, draws: int = 2000) -> List[float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
