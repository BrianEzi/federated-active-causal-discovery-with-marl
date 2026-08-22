"""What fraction of confounding is detectable from OBSERVATIONAL data, at infinite sample?

This is a STRUCTURAL ceiling, not a statistical one, and the distinction is the point.
Two independent limits stop an agent identifying confounding from its own observations:

  statistical   finite data, noisy tests. Fixable with more samples.
  structural    the observed conditional-independence pattern is reproducible by some DAG
                over the observed variables ALONE, so no latent is required to explain it.
                NO amount of data fixes this.

Only the second is measured here, and it upper-bounds any observational-only method
whatsoever -- including our own BGe posterior, which uses exactly this evidence implicitly.
Worth having BEFORE spending cluster time on an observational-only training ablation, since
it says what that ablation could achieve even in the best case.

THE TEST. For a sampled true DAG, take one agent's window (its observed nodes) and compute
the conditional independencies the true graph implies among those nodes -- d-separation in
the FULL graph, hidden nodes included, restricted to queries whose variables and
conditioning sets are all observed. That is the marginal CI structure the agent can see.
Then ask whether ANY DAG on the observed nodes alone induces exactly that CI structure.

  some DAG matches   -> a latent is NOT required; the pattern is explainable without one,
                        so confounding is structurally invisible here
  none matches       -> a latent IS required; confounding is detectable in principle

This is equivalent to asking whether FCI would be forced to place a bidirected edge, but is
computed by direct enumeration rather than by implementing FCI's orientation rules, so it
needs no appeal to the completeness results (Zhang 2008) to be trusted -- it is a definition
check, not an algorithm.

Cost is controlled by precomputing every DAG's CI signature ONCE for the window size, then
per episode computing the true signature and testing set membership.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from itertools import combinations

import numpy as np

from ma.projection import bidirected_pairs
from ma.topology import Topology, two_agent
from sa.graphs import build_graph_space
from sa.priors import connectivity_prior_p


def ancestors(adjacency: np.ndarray, nodes) -> set:
    """All ancestors of `nodes`, inclusive."""
    d = adjacency.shape[0]
    seen = set(nodes)
    frontier = list(nodes)
    while frontier:
        v = frontier.pop()
        for u in range(d):
            if adjacency[u, v] and u not in seen:
                seen.add(u)
                frontier.append(u)
    return seen


def d_separated(adjacency: np.ndarray, x: int, y: int, z: frozenset) -> bool:
    """Standard moralisation test: ancestral subgraph, moralise, drop Z, check connectivity."""
    d = adjacency.shape[0]
    keep = ancestors(adjacency, set([x, y]) | set(z))

    # Moral graph over `keep`: edges, plus parents of a common child married.
    undirected = {v: set() for v in keep}
    for u in keep:
        for v in keep:
            if adjacency[u, v]:
                undirected[u].add(v)
                undirected[v].add(u)
    for v in keep:
        parents = [u for u in keep if adjacency[u, v]]
        for a, b in combinations(parents, 2):
            undirected[a].add(b)
            undirected[b].add(a)

    blocked = set(z)
    if x in blocked or y in blocked:
        return True
    frontier, seen = [x], {x}
    while frontier:
        v = frontier.pop()
        if v == y:
            return False
        for w in undirected[v]:
            if w not in seen and w not in blocked:
                seen.add(w)
                frontier.append(w)
    return True


def ci_signature(adjacency: np.ndarray, observed) -> frozenset:
    """The full set of (x, y, Z) conditional independencies among `observed`.

    Positions are indices INTO `observed`, so signatures from a full graph and from a
    window-sized graph are directly comparable.
    """
    obs = list(observed)
    out = set()
    for i, j in combinations(range(len(obs)), 2):
        others = [t for t in range(len(obs)) if t not in (i, j)]
        for size in range(len(others) + 1):
            for combo in combinations(others, size):
                z = frozenset(obs[t] for t in combo)
                if d_separated(adjacency, obs[i], obs[j], z):
                    out.add((i, j, frozenset(combo)))
    return frozenset(out)


def observable_signatures(k: int) -> set:
    """Every CI signature achievable by a DAG on `k` nodes with NO latents."""
    space = build_graph_space(k, fast=True)
    return {ci_signature(np.asarray(space.dags[i]), range(k))
            for i in range(space.n_dags)}


def run(topology: Topology, episodes: int, seed: int, prior_p: float) -> dict:
    achievable = {}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(episodes):
        adjacency = topology.sample_dag(rng, p=prior_p)
        for agent in topology.agents:
            observed = topology.observed_by(agent)
            k = len(observed)
            if k not in achievable:
                achievable[k] = observable_signatures(k)
            confounded = bidirected_pairs(adjacency, observed)
            signature = ci_signature(adjacency, observed)
            # Does any latent-free DAG on the window reproduce what the agent sees?
            explainable = signature in achievable[k]
            rows.append({"agent": int(agent), "confounded": bool(confounded),
                         "n_confounded_pairs": len(confounded),
                         "needs_latent": (not explainable)})

    conf = [r for r in rows if r["confounded"]]
    detect = [r for r in conf if r["needs_latent"]]
    unconf = [r for r in rows if not r["confounded"]]
    false_pos = [r for r in unconf if r["needs_latent"]]
    return {
        "topology": topology.name, "d": topology.d, "prior_p": prior_p,
        "episodes": episodes, "windows_scored": len(rows),
        "confounded_fraction": len(conf) / len(rows) if rows else float("nan"),
        "ceiling_detectable_given_confounded": len(detect) / len(conf) if conf else float("nan"),
        "n_confounded": len(conf), "n_detectable": len(detect),
        "unconfounded_but_needs_latent": len(false_pos) / len(unconf) if unconf else float("nan"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/structural_ceiling.json")
    args = ap.parse_args(argv)

    shapes = [
        two_agent("T1_1-1-3_current", (0,), (1,), (2, 3, 4)),
        two_agent("T1_2-2-2", (0, 1), (2, 3), (4, 5)),
    ]

    out_rows = []
    header = "%-24s %2s %7s %10s %12s %10s" % (
        "topology", "d", "p", "conf%", "CEILING%", "n_conf")
    print(header); print("-" * len(header))
    for topology in shapes:
        for label, p in (("old", 0.5), ("new", connectivity_prior_p(topology.d))):
            row = run(topology, args.episodes, args.seed, p)
            row["p_label"] = label
            out_rows.append(row)
            print("%-24s %2d %7.3f %9.1f%% %11.1f%% %10d" % (
                topology.name, row["d"], p, 100 * row["confounded_fraction"],
                100 * row["ceiling_detectable_given_confounded"], row["n_confounded"]))
    print()
    print("CEILING% = of windows that ARE confounded, the share where no latent-free DAG")
    print("reproduces the observed CI pattern, i.e. the most any observational-only method")
    print("could detect at infinite data. The rest are structurally invisible.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_rows, indent=1))
    print("\nwrote %s" % out)
    return out_rows


if __name__ == "__main__":
    main()
