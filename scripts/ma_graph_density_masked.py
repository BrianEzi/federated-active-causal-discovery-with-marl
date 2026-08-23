"""Connectivity under the FEDERATION MASK, which is what the environment actually draws.

`scripts/sa_graph_density.py` measured plain Erdos-Renyi with no mask, and that is what the
"92-99% connected" figure for `2 ln(d)/d` refers to. The environment never draws such a
graph: `Topology.allowed_edges` forbids every pair no single agent observes, which removes
10-27% of the possible edges depending on shape. Removing edges can only reduce
connectivity, so the unmasked number is an upper bound rather than an estimate.

Run:  PYTHONPATH=. python scripts/ma_graph_density_masked.py
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from ma.env import _is_connected
from ma.topology import Topology, two_agent
from ma.priors import connectivity_prior_p


def shapes():
    """The current setting, plus the rungs of the scaling ladder."""
    return [
        ("2 agents, 1-1-3 (current)", two_agent("t113", (0,), (1,), (2, 3, 4))),
        ("2 agents, 2-2-2", two_agent("t222", (0, 1), (2, 3), (4, 5))),
        ("rung 1: 3 agents, 1 each, 3 shared",
         Topology("r1", private=((0,), (1,), (2,)), exposed=(3, 4, 5))),
        ("rung 2: 3 agents, 2 each, 3 shared",
         Topology("r2", private=((0, 1), (2, 3), (4, 5)), exposed=(6, 7, 8))),
        ("rung 3: 5 agents, 1 each, 5 shared",
         Topology("r3", private=tuple((i,) for i in range(5)), exposed=(5, 6, 7, 8, 9))),
    ]


def unmasked_connected(d: int, p: float, draws: int, rng) -> float:
    """Plain ER, for the comparison. Mirrors scripts/sa_graph_density.py exactly."""
    connected = 0
    for _ in range(draws):
        order = rng.permutation(d)
        adjacency = np.zeros((d, d), dtype=np.int8)
        for i in range(d):
            for j in range(i + 1, d):
                if rng.random() < p:
                    adjacency[order[i], order[j]] = 1
        connected += _is_connected(adjacency)
    return connected / draws


def measure(topology: Topology, draws: int, rng) -> dict:
    p = connectivity_prior_p(topology.d)
    connected, degrees = 0, []
    for _ in range(draws):
        adjacency = topology.sample_dag(rng, p=p)
        connected += _is_connected(adjacency)
        degrees.append(adjacency.sum() * 2.0 / topology.d)
    allowed = topology.allowed_edges()
    return {
        "name": topology.name, "d": topology.d, "n_agents": topology.n_agents,
        "prior_p": float(p),
        "allowed_edge_fraction": float(allowed.sum() / (topology.d * (topology.d - 1))),
        "connected_masked": connected / draws,
        "connected_unmasked": unmasked_connected(topology.d, p, draws, rng),
        "mean_degree": float(np.mean(degrees)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/graph_density_masked.json")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    rows = []
    header = "%-36s %2s %6s %9s %8s %10s %7s" % (
        "shape", "d", "p", "allowed%", "conn%", "unmasked%", "degree")
    print(header)
    print("-" * len(header))
    for label, topology in shapes():
        row = measure(topology, args.draws, rng)
        row["label"] = label
        rows.append(row)
        print("%-36s %2d %6.3f %8.1f%% %7.1f%% %9.1f%% %7.2f" % (
            label, row["d"], row["prior_p"], 100 * row["allowed_edge_fraction"],
            100 * row["connected_masked"], 100 * row["connected_unmasked"],
            row["mean_degree"]))

    worst = min(rows, key=lambda r: r["connected_masked"])
    print("\nworst masked connectivity: %.1f%% at %s (unmasked would be %.1f%%)" % (
        100 * worst["connected_masked"], worst["label"],
        100 * worst["connected_unmasked"]))
    print("Connectivity is reported per episode and every headline is split by it, so this "
          "costs\nstatistical power on the connected half rather than correctness.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"draws": args.draws, "seed": args.seed, "rows": rows},
                              indent=1))
    print("\nwrote %s" % out)
    return rows


if __name__ == "__main__":
    main()
