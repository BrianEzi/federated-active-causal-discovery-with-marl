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

from itertools import combinations

import numpy as np

from ma.baselines import _Window, enumerated_posterior
from crosscheck.belief_dp import JOINT_CONF
from ma.env import ATTRIBUTED, CLAIM_BACKENDS, RANDOM_TURN, TwoAgentEnv
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
    # CLAIM_BACKENDS, not a literal list. This was hard-coded twice and both copies were
    # missed when the attributed backend landed, so a run trained for 25 minutes and then
    # died in its own report -- the checkpoint survived, the evaluation did not. Any backend
    # that scores CLAIMS has no enumerable posterior, so this is the correct predicate and
    # it cannot fall out of date.
    if env.config.belief_backend in CLAIM_BACKENDS:
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


NONE, FWD, BACK, BI = 0, 1, 2, 3
_MARK_NAMES = ("none", "->", "<-", "<->")


def _pair_masses(belief, u: int, v: int) -> np.ndarray:
    """(NONE, FWD, BACK, BI) mass for one pair, from the frequency matrices every claim
    backend exposes. Backend-agnostic on purpose: on the factored path the frequencies are
    uniform over the surviving marks, so `mass > 0` recovers the survivor set EXACTLY, and
    on the bootstrap path it means "some replicate still supports this mark", which is the
    same notion one level down."""
    return np.array([1.0 - float(np.asarray(belief.adjacency)[u, v]),
                     float(np.asarray(belief.directed)[u, v]),
                     float(np.asarray(belief.directed)[v, u]),
                     float(np.asarray(belief.bidirected)[u, v])])


def _true_index(mag: np.ndarray, u: int, v: int) -> int:
    from ma.projection import BIDIRECTED as _BI, DIRECTED as _DIR
    if mag[u, v] == _BI:
        return BI
    if mag[u, v] == _DIR:
        return FWD
    if mag[v, u] == _DIR:
        return BACK
    return NONE


def pooled_global_belief(env: TwoAgentEnv, tol: float = 1e-9) -> Dict[tuple, dict]:
    """Assemble ONE global belief from the sites, by intersecting their surviving marks.

    WHY THIS REPLACES `union_graph`. That function reads `_Window.get(k).dags[map_index]`,
    so it ENUMERATES: it dies above k=5 and the entire factored ladder is out of its reach.
    `_constraint_union` is the non-enumerating fallback but it is majority-vote BINARY
    ADJACENCY -- no orientations, no bidirected marks -- and neither runs on the `claims`
    criterion every ladder run uses. So in practice the project had no working global-graph
    metric at the sizes it actually reports.

    WHY INTERSECTION IS THE RIGHT POOLING, and why it is sound. Each site's mark set for a
    pair contains the truth (`cb/factored.py`: every update is individually sound, so the
    belief stays unsure rather than settling wrongly). A family of sets that each contain the
    truth has an intersection that contains it too, so the pooled belief is at least as tight
    as any single site's and never excludes the true mark. That is exactly federated
    aggregation of a graph, which is what the causal-discovery literature federates -- not
    the parameters of a policy network.

    CROSS-PRIVATE PAIRS ARE ABSENT AND THAT IS NOT A GAP. `Topology.allowed_edges` permits an
    edge only where one agent observes both endpoints, so a pair spanning two private blocks
    cannot exist and no site holds a belief about it. They are roughly half of all pairs and
    are guaranteed true non-edges, so including them would add exactly zero error and dilute
    every difference by a constant.

    THE ONE SUBTLETY, reported rather than hidden. A shared pair's true mark is a LATENT
    PROJECTION into a window, and different windows project out different private blocks, so
    two sites can legitimately hold DIFFERENT true marks for the same pair. Those pairs are
    flagged `mark_disagreement` and scored against each site's own truth, averaged -- there
    is no single global answer to score against.

    Returns `{(global_u, global_v): {...}}`, one entry per covered pair.
    """
    # SCOPE. This reads the `adjacency` / `directed` / `bidirected` frequency matrices, which
    # are the CLAIM_BACKENDS interface. The exact DP belief (`WindowBeliefDP`) has no `.last`
    # and no per-pair mark set, so there is nothing to intersect -- the caller gets an empty
    # map and `global_graph_report` reports nan rather than a fabricated zero.
    if env.config.belief_backend not in CLAIM_BACKENDS:
        return {}
    seen: Dict[tuple, dict] = {}
    for agent, window in env.windows.items():
        belief = getattr(window.belief, "last", None)
        if belief is None or not hasattr(belief, "adjacency"):
            continue
        mag = np.asarray(env._true_mag(agent))
        for u, v in combinations(range(window.k), 2):
            key = tuple(sorted((window.nodes[u], window.nodes[v])))
            masses = _pair_masses(belief, u, v)
            entry = seen.setdefault(key, {"marks": [], "truth": [], "soft": []})
            entry["marks"].append(frozenset(np.flatnonzero(masses > tol).tolist()))
            entry["truth"].append(_true_index(mag, u, v))
            entry["soft"].append(1.0 - float(masses[_true_index(mag, u, v)]))

    out: Dict[tuple, dict] = {}
    for key, entry in seen.items():
        truths = set(entry["truth"])
        pooled = frozenset.intersection(*entry["marks"]) if entry["marks"] else frozenset()
        disagree = len(truths) > 1
        if disagree or not pooled or entry["truth"][0] not in pooled:
            # No single truth to score against, or the sites contradict each other. Fall
            # back to the per-site mean rather than inventing a pooled verdict.
            soft = float(np.mean(entry["soft"]))
            hard = int(soft > 0.5)
            contradiction = bool(not disagree and (not pooled
                                                   or entry["truth"][0] not in pooled))
        else:
            soft = 1.0 - 1.0 / len(pooled)
            hard = int(len(pooled) > 1 or next(iter(pooled)) != entry["truth"][0])
            contradiction = False
        out[key] = {"soft": soft, "hard": hard, "sites": len(entry["marks"]),
                    "resolved": bool(not disagree and len(pooled) == 1),
                    "mark_disagreement": disagree, "contradiction": contradiction}
    return out


def global_graph_report(env: TwoAgentEnv) -> Dict[str, float]:
    """Headline numbers for the pooled global graph -- the object a federated causal
    discovery paper reports on. Each covered pair counts ONCE, unlike the per-window average
    in `scripts/shd.py`, which counts a shared pair once per agent."""
    pooled = pooled_global_belief(env)
    if not pooled:
        return {"global_soft_shd": float("nan"), "global_hard_shd": float("nan"),
                "global_resolved_fraction": float("nan"), "global_pairs": 0,
                "global_mark_disagreement": float("nan"),
                "global_contradiction": float("nan")}
    n = len(pooled)
    return {
        "global_soft_shd": float(sum(p["soft"] for p in pooled.values()) / n),
        "global_hard_shd": float(sum(p["hard"] for p in pooled.values()) / n),
        "global_resolved_fraction": float(sum(p["resolved"] for p in pooled.values()) / n),
        "global_pairs": n,
        # Shared pairs whose true mark differs between windows, because each window projects
        # out a different set of private blocks. Small but real: 2.7-6.7% on the ladder.
        "global_mark_disagreement": float(
            sum(p["mark_disagreement"] for p in pooled.values()) / n),
        # Sites whose surviving mark sets have empty intersection, or exclude the truth.
        # Non-zero means a site's belief was unsound, not that the graph is hard.
        "global_contradiction": float(sum(p["contradiction"] for p in pooled.values()) / n),
    }


def _constraint_union(env: TwoAgentEnv) -> np.ndarray:
    """The constraint-side union: majority-vote directed edges, OR-stitched.

    `union_graph` reads `space.dags[map_index]`, and the constraint backend has no MAP
    index -- it reports -1, which numpy would silently read as THE LAST DAG in the
    enumeration. This function exists so that -1 is never used as an index.
    """
    d = env.topology.d
    union = np.zeros((d, d), dtype=np.int8)
    for agent in env.topology.agents:
        window = env.windows[agent]
        majority = env.marginals[agent] >= 0.5
        for i, u in enumerate(window.nodes):
            for j, v in enumerate(window.nodes):
                if majority[i, j]:
                    union[u, v] = 1
    return union


def evaluate_episode(env: TwoAgentEnv) -> Dict[str, object]:
    """Every criterion for one finished episode."""
    threshold = env.config.identify_threshold
    reports = {agent: agent_report(env, agent) for agent in env.topology.agents}
    # CLAIM_BACKENDS, not a literal list. This was hard-coded twice and both copies were
    # missed when the attributed backend landed, so a run trained for 25 minutes and then
    # died in its own report -- the checkpoint survived, the evaluation did not. Any backend
    # that scores CLAIMS has no enumerable posterior, so this is the correct predicate and
    # it cannot fall out of date.
    if env.config.belief_backend in CLAIM_BACKENDS:
        union = _constraint_union(env)
    else:
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
        # The pooled global graph -- see `pooled_global_belief`. These SUPERSEDE the three
        # union_* fields above, which enumerate and are therefore unavailable above k=5.
        # Kept side by side so older comparisons stay reproducible.
        **global_graph_report(env),
        # All three parts of [U14], which is the number to report -- EXCEPT under the
        # claims criterion, where success is the claims verdict itself. BUG 9, found
        # 2026-08-24 when a probe's "success" (4%) contradicted the direct decomposition
        # (43% of agent-windows identified): this expression was still scoring the
        # superseded per-replicate conjunction through `mass_credit` on the constraint
        # path, so every claims-era probe under-reported. The criterion the env pays is
        # the criterion evaluation must report.
        "success": (_claims_success(env) if env.config.reward_criterion == "claims"
                    else bool(all(reports[a]["mass_credit"] >= threshold
                                  for a in env.topology.agents)
                              and acyclic and globally_equivalent)),
    }


def _claims_success(env: TwoAgentEnv) -> bool:
    """Whether every window is identified, on EXACTLY the criterion `_result` pays for.

    MUST MIRROR `TwoAgentEnv._result`. It did not (found 2026-08-27, fixed 2026-08-28):

      * it called `score_window` with the DEFAULT `require_all_types` instead of the
        configured `claims_require_all_types`, so a run that relaxed the criterion was
        graded on the strict one;
      * on the ATTRIBUTED backend it scored `confounding_claims` -- the very claims that
        backend replaces -- and never looked at attribution at all.

    WHAT THAT ACTUALLY COST, measured rather than assumed. The old criterion is strictly
    WEAKER per window: over 714 windows at the `attr3a` configuration it credited 23 the env
    would not, and zero the other way, so `new implies old` and the old rule over-credits
    about 3% of windows. But it did NOT move a single EPISODE verdict in 604 sampled states
    across two configurations -- a joint verdict needs every window, and the over-credited
    window was never the last one blocking. So the `success` field in
    `results/attr_scale/*.json` is not shown to be wrong; the criterion was simply more
    permissive than the one training paid for, and with fewer windows or a stronger policy
    that difference would reach the joint verdict.

    `tests/ma/test_claims_success_mirrors_env.py` fails if the two drift apart again,
    whichever side moves. It checks the WINDOW level as well as the episode level, because
    the episode level is where the difference is currently unreachable.
    """
    from cb.attribution import score_groups
    from cb.claims import score_window

    cfg = env.config
    attributed = cfg.belief_backend == ATTRIBUTED
    for agent, window in env.windows.items():
        score = score_window(window.belief.last, env._true_mag(agent),
                             [window.pos[n] for n in window.private],
                             bar=cfg.claim_bar,
                             require_all_types=cfg.claims_require_all_types,
                             confounding_claims=not attributed)
        if not score.identified:
            return False
        if attributed and not score_groups(window.belief.last, window.belief.true_groups,
                                           bar=cfg.claim_bar)["identified"]:
            return False
    return True


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
        # Coordination mechanism, and the structural floor it has to be read against. These
        # live on `info`, not on `evaluate_episode`, and were never copied across -- so they
        # existed per episode and reached no result file.
        # Time to identification, per agent, ALREADY CENSORED. `info["identified_round"]`
        # holds None where a window was never identified, and None propagates to nan through
        # the survival summary -- so the censor must be applied here, by the function that
        # owns its meaning, rather than left to the consumer.
        row["identified_round"] = dict(
            env.rounds_to_identification(censor=env.config.budget + 1))
        row["duplicate_coverage"] = float(info["duplicate_coverage"])
        row["duplicate_coverage_floor"] = float(info.get("duplicate_coverage_floor", 0.0))
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
        # The pooled global graph. Computed per episode by `evaluate_episode` and previously
        # dropped here, so the metric that REPLACES the union_* fields reached no result
        # file. nan-safe because non-claim backends report nan by design.
        **{key: float(np.nanmean([r[key] for r in rows]))
           for key in ("global_soft_shd", "global_hard_shd", "global_resolved_fraction",
                       "global_mark_disagreement", "global_contradiction")
           if key in rows[0]},
        "global_pairs": int(rows[0].get("global_pairs", 0)),
        # Duplicate coverage AND its floor: past `len(exposed)` shared interventions
        # duplication is forced, so the raw number is not comparable across arms that spend
        # differently on the shared surface.
        "duplicate_coverage": float(np.mean([r["duplicate_coverage"] for r in rows])),
        "duplicate_coverage_floor": float(
            np.mean([r["duplicate_coverage_floor"] for r in rows])),
        "duplicate_coverage_excess": float(np.mean(
            [r["duplicate_coverage"] - r["duplicate_coverage_floor"] for r in rows])),
        **_per_agent_block(env, rows),
        "clamp_fraction": float(clamps / moves) if moves else float("nan"),
    }


def run_arm_paths(env: TwoAgentEnv, policies: Dict[int, object], episodes: int,
                  seed: int = 0, paths: int = 3) -> Dict[str, object]:
    """`run_arm` over several independent POLICY SAMPLE PATHS, not just several episodes.

    WHY THIS EXISTS. `IndependentPPO.__init__` seeded the global torch stream, and `load`
    went through it, so every evaluation of a checkpoint replayed ONE fixed sample path.
    Repeating the evaluation returned the identical number, and every confidence interval
    this project has reported therefore excluded policy stochasticity entirely -- the one
    source of variance an evaluation of a stochastic policy is supposed to capture. The
    reseed is now suppressed on `load`; this is the protocol that exploits the fix.

    THE GRAPHS ARE HELD FIXED ACROSS PATHS, deliberately. `run_arm` replays
    `seed * 100_000 + episode`, so every path sees the SAME episodes and the spread between
    paths is policy stochasticity alone. Graph variance is already reported by the paired
    per-episode vectors; conflating the two would double-count it and inflate every
    interval.

    Returns the pooled per-path means plus `path_sd` and `path_ci`. Quote the pooled mean
    with `path_ci`; a single-path number is a point estimate with no error bar at all.
    """
    import torch
    scalar_keys = None
    per_path: List[Dict[str, float]] = []
    for path in range(max(1, paths)):
        # A DISTINCT sample path per repeat. Same episodes, different policy draws.
        torch.manual_seed(seed * 1_000 + path)
        for policy in policies.values():
            if hasattr(policy, "reset"):
                policy.reset(seed)
        row = run_arm(env, policies, episodes, seed=seed)
        if scalar_keys is None:
            scalar_keys = [k for k, v in row.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)]
        per_path.append({k: float(row[k]) for k in scalar_keys})

    out: Dict[str, object] = {"paths": len(per_path), "episodes": episodes}
    for key in scalar_keys or []:
        values = np.array([row[key] for row in per_path], dtype=float)
        out[key] = float(np.nanmean(values))
        if len(values) > 1:
            sd = float(np.nanstd(values, ddof=1))
            out[f"{key}__path_sd"] = sd
            out[f"{key}__path_ci"] = 1.96 * sd / np.sqrt(len(values))
    out["per_path"] = per_path
    return out


def survival_summary(rounds: Sequence[float], censor: int) -> Dict[str, float]:
    """Summarise time-to-identification WITHOUT averaging across censored observations.

    WHY A PLAIN MEAN IS WRONG HERE. `rounds_to_identification` right-censors at `budget + 1`
    for every window that was never identified, and censoring on this task is severe -- most
    episodes at the harder rungs never identify at all. Averaging then reports the censoring
    horizon rather than the policy: an arm that solves 10% of episodes in 2 rounds and an arm
    that solves none both land near budget+1, and pushing the budget up moves the "mean" for
    every arm at once. `docs/METRICS.md` flags this; this is the fix.

    Three numbers, none of which pretends the censored observations were finite:

      `censored_fraction`     how much of the sample never identified. Report it BESIDE the
                              others -- the other two are uninterpretable without it.
      `median_rounds`         the smallest round by which at least half identified, or nan
                              when fewer than half ever do. Undefined is the honest answer;
                              a number would be extrapolation.
      `restricted_mean`       mean of min(round, censor) -- the restricted mean survival
                              time at the censoring horizon. Always defined, bounded by
                              `censor`, and comparable across arms at the SAME budget only.
    """
    values = np.asarray(list(rounds), dtype=float)
    if values.size == 0:
        return {"censored_fraction": float("nan"), "median_rounds": float("nan"),
                "restricted_mean": float("nan"), "n": 0}
    censored = values >= censor
    finished = np.sort(values[~censored])
    half = 0.5 * values.size
    median = float(finished[int(np.ceil(half)) - 1]) if finished.size >= half else float("nan")
    return {"censored_fraction": float(censored.mean()),
            "median_rounds": median,
            "restricted_mean": float(np.minimum(values, censor).mean()),
            "n": int(values.size)}


def _mean_where(rows, keep) -> float:
    """Mean `success` over the episodes `keep` selects; nan when it selects none."""
    picked = [float(bool(r["success"])) for r in rows if keep(r)]
    return float(np.mean(picked)) if picked else float("nan")


def _evenness_null(env: TwoAgentEnv, draws: int = 4000) -> float:
    """`effort_evenness` the PROTOCOL alone would produce, with no policy involved.

    Round-robin hands every agent exactly `budget // n` moves, so the null is 1.0 and any
    shortfall is behaviour (an agent that passed). Random turn order draws the actor
    uniformly each round, so the counts are multinomial and the ratio is small for
    arithmetic reasons alone -- which is what makes the raw index useless for comparing the
    two protocols. Simultaneous play gives every agent a move every round, so the null is
    again 1.0. Estimated by simulation rather than in closed form: E[min/max] of a
    multinomial has no tidy expression and this costs microseconds.
    """
    order = env.config.turn_order
    if order != RANDOM_TURN:
        return 1.0
    n = env.topology.n_agents
    rng = np.random.default_rng(0)
    counts = rng.multinomial(env.config.budget, [1.0 / n] * n, size=draws)
    lo, hi = counts.min(axis=1), counts.max(axis=1)
    live = hi > 0
    return float(np.mean(lo[live] / hi[live])) if live.any() else float("nan")


def _per_agent_block(env: TwoAgentEnv, rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Per-agent behaviour, the connectedness split, and the free-rider index.

    `free_rider_index` is `min(interventions) / max(interventions)` across agents: 1.0 when
    the agents pull their weight evenly, 0.0 when one agent did nothing at all. Episodes in
    which NOBODY acted are excluded -- the ratio is undefined there and would otherwise read
    as perfect cooperation.

    THE NAME IS BACKWARDS AND THE NUMBER IS PROTOCOL-CONFOUNDED. It is reported unchanged so
    that older result files stay readable, but three fields are added beside it:

      `effort_evenness`       the same quantity under a name that says which way is good.
      `effort_evenness_null`  what the PROTOCOL alone produces, with no policy at all. Under
                              `random` turn order the actor is drawn uniformly, so per-agent
                              move counts are Multinomial(budget, 1/n) and the ratio is small
                              by arithmetic: measured 0.140 at 8 agents / budget 24 against a
                              pure-sampling expectation of 0.158. Under round-robin every
                              agent is handed exactly budget/n moves, so the null is 1.0.
                              Quote `effort_evenness / effort_evenness_null`, never the raw
                              number, whenever turn orders are compared.
      `some_agent_never_acted`  the fraction of episodes in which at least ONE agent got no
                              move. This is the field that matters and it did not exist:
                              `never_acted_episodes` tests `max(...) == 0`, i.e. whether
                              NOBODY acted, which is almost never true and hides the failure
                              mode entirely. At 8 agents with budget 24 under random turns,
                              29.9% of episodes leave some agent with zero moves -- and an
                              agent that never acts can never settle its own private nodes,
                              because `Topology.allowed_edges` forbids cross-private edges.
                              Those episodes are unwinnable for ANY policy, so joint success
                              must be read against this. See `docs/FINDINGS_TURN_ORDER_2026_08_29.md`.

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
        # Same quantity, named so the direction is unambiguous: HIGHER IS MORE EVEN.
        "effort_evenness": float(np.mean(ratios)) if ratios else float("nan"),
        "effort_evenness_null": _evenness_null(env),
        # `never_acted_episodes` asks whether NOBODY acted (max == 0). Kept for old files.
        "never_acted_episodes": float(np.mean(
            [max(r["interventions"][a] for a in env.topology.agents) == 0 for r in rows])),
        # The one that matters: did SOME agent get no move at all? Those episodes cannot be
        # jointly solved by any policy, because no partner can reach another's private nodes.
        "some_agent_never_acted": float(np.mean(
            [min(r["interventions"][a] for a in env.topology.agents) == 0 for r in rows])),
        # Joint success restricted to episodes every agent could in principle have solved.
        "success_feasible": _mean_where(
            rows, lambda r: min(r["interventions"][a] for a in env.topology.agents) > 0),
        "connected_fraction": float(np.mean([r["connected"] for r in rows])),
        # Time-to-identification, summarised as survival rather than averaged. See
        # `survival_summary` for why a plain mean reports the censoring horizon.
        "time_to_identification": {
            a: survival_summary(
                [r["identified_round"][a] for r in rows if a in r.get("identified_round", {})],
                censor=env.config.budget + 1)
            for a in env.topology.agents},
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
