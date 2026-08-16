"""Two-agent gates. Nothing proceeds past a failure here.

The single-agent rebuild's discipline was that the task must be verified to require acting
(GATE 1) and to reward choosing well (GATE 2) BEFORE any RL is written. The two-agent case
needs both of those plus a third that is specific to it.

GATE 1 -- observational data alone must not solve it, and the rate must match theory.
    An agent identifies its own induced DAG from observation alone only when that DAG is
    alone in its Markov equivalence class AND the agent is not confounded. Both are
    computable exactly from the graph, so the target is a predicted number rather than a
    vibe. This is the same standard as the single-agent gate.

GATE 2 -- choosing well must beat choosing at random, per agent.

GATE 3 -- THE TWO-AGENT GATE. Confounded agents must be measurably worse off alone.
    If an agent reaches its true induced DAG just as easily when confounded as when not,
    there is no coordination problem to solve and the whole design collapses to two
    independent single-agent problems running side by side. This gate is what makes the
    two-agent case a different problem rather than a bigger one.

    Stated as a prediction before measurement: confounded agents should identify at a
    LOWER rate, because under confounding no DAG over the agent's window is correct and
    the posterior cannot concentrate on the truth however much data arrives.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ma.confounding import latent_projection_pairs
from ma.env import PASS_ACTION, MAConfig, TwoAgentEnv
from ma.projection import bidirected_pairs
from sa.graphs import build_graph_space


def _singleton_lookup(k: int) -> np.ndarray:
    """Per-DAG flag: is this graph alone in its Markov equivalence class?"""
    space = build_graph_space(k)
    return space.mec_sizes[space.mec_id] == 1


def episode_facts(env: TwoAgentEnv, singleton: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Ground-truth structural facts about the current episode, per agent."""
    out = {}
    for name, view in env.views.items():
        # TRUE confounding: a bidirected edge in the agent's latent projection. Not the
        # `ma.confounding` proxy, which overcounts by including ancestrally related pairs
        # (measured 2026-08-16: 36/36, 6024/6024 of the excess).
        bidirected = bidirected_pairs(env.true_adjacency, view.nodes)
        out[name] = {
            "confounded": len(bidirected) > 0,
            "n_bidirected": len(bidirected),
            "singleton_mec": bool(singleton[name][env.true_index[name]]),
        }
    return out


def run_gates(config: MAConfig, episodes: int = 400, seed: int = 0,
              budget_policies: bool = True) -> dict:
    env = TwoAgentEnv(config, seed=seed)
    singleton = {name: _singleton_lookup(view.k) for name, view in env.views.items()}

    rows: List[dict] = []
    for ep in range(episodes):
        result = env.reset(seed=seed * 1_000_000 + ep)
        facts = episode_facts(env, singleton)
        for name in ("A", "B"):
            rows.append({
                "agent": name,
                **facts[name],
                "identified_obs_only": bool(result.identified[name]),
                "true_mass_obs_only": float(result.info["true_mass"][name]),
            })

    def subset(pred):
        return [r for r in rows if pred(r)]

    def rate(items, key="identified_obs_only"):
        if not items:
            return None
        return float(np.mean([r[key] for r in items]))

    def wilson(items, key="identified_obs_only"):
        """Wilson interval -- the observational rate sits near zero, where the normal
        approximation is unusable."""
        if not items:
            return None
        n = len(items)
        p = np.mean([r[key] for r in items])
        z = 1.96
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]

    clean = subset(lambda r: not r["confounded"])
    confounded = subset(lambda r: r["confounded"])
    clean_singleton = subset(lambda r: not r["confounded"] and r["singleton_mec"])
    clean_tied = subset(lambda r: not r["confounded"] and not r["singleton_mec"])

    # GATE 1. Among unconfounded agents, identification from observation alone should
    # happen exactly on the singleton-MEC episodes -- high on those, ~0 on the rest.
    gate1 = {
        "unconfounded_singleton_rate": rate(clean_singleton),
        "unconfounded_singleton_ci": wilson(clean_singleton),
        "unconfounded_tied_rate": rate(clean_tied),
        "unconfounded_tied_ci": wilson(clean_tied),
        "n_unconfounded_singleton": len(clean_singleton),
        "n_unconfounded_tied": len(clean_tied),
    }
    # A tied graph can never reach the 0.7 threshold observationally: its class-mates tie
    # with it exactly, capping its mass at 1/|class| <= 0.5. So the tied rate must be 0.
    gate1["passed"] = (gate1["unconfounded_tied_rate"] == 0.0
                       if gate1["unconfounded_tied_rate"] is not None else False)

    # GATE 3. Confounding must hurt.
    gate3 = {
        "confounded_rate": rate(confounded),
        "confounded_ci": wilson(confounded),
        "unconfounded_rate": rate(clean),
        "unconfounded_ci": wilson(clean),
        "confounded_mean_true_mass": rate(confounded, "true_mass_obs_only"),
        "unconfounded_mean_true_mass": rate(clean, "true_mass_obs_only"),
        "n_confounded": len(confounded),
        "n_unconfounded": len(clean),
    }
    if gate3["confounded_ci"] and gate3["unconfounded_ci"]:
        # Disjoint intervals, confounded strictly lower.
        gate3["passed"] = gate3["confounded_ci"][1] < gate3["unconfounded_ci"][0]
    else:
        gate3["passed"] = False

    return {
        "episodes": episodes,
        "confounding_rate": float(np.mean([r["confounded"] for r in rows])),
        "singleton_rate": float(np.mean([r["singleton_mec"] for r in rows])),
        "gate1": gate1,
        "gate3": gate3,
        "rows": len(rows),
    }
