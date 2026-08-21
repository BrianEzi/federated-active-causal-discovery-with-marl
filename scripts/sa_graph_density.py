"""What edge probability should the prior use, as `d` grows?

`prior_p = 0.5` is fixed and does not scale. At `d = 5` it gives an expected degree of 2.0,
which lands in the literature's sparse regime by accident; at `d = 30` it gives **14.5**
against a literature norm of 2-4. One connected blob is neither realistic nor informative.

Two DIFFERENT thresholds get conflated in this area and the distinction decides the answer:

  GIANT COMPONENT (percolation)   `p_c = 1/d`, expected degree 1. Above this a single
                                  component contains a constant fraction of the nodes.
                                  This is the regime `p_e = Theta(1/d)` that Chevalley,
                                  Mehrjou & Schwab (arXiv:2511.02536) identify as the
                                  literature's sparse setting.

  FULL CONNECTIVITY               `p = ln(d)/d`, expected degree `ln d`. Above this EVERY
                                  node is in one component, with no isolated stragglers.

**We want the second.** A disconnected graph splits the agents into independent subproblems:
no path crosses the private/shared boundary, so there is no latent confounding and nothing to
coordinate about. Those episodes cannot test what this project is building.

So the honest prescription is to scale with the CONNECTIVITY threshold and say so, citing the
percolation regime as the sparser alternative we deliberately did not choose. The percolation
framing itself is ours, not a citation.

Measured here rather than asserted: for each `d`, the fraction of sampled DAGs that are a
single component, at several candidate rules.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from ma.env import _is_connected


def sample_dag(d: int, p: float, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(d)
    adjacency = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                adjacency[order[i], order[j]] = True
    return adjacency


def measure(d: int, p: float, draws: int, rng: np.random.Generator) -> dict:
    connected, degrees = 0, []
    for _ in range(draws):
        a = sample_dag(d, p, rng)
        connected += _is_connected(a)
        degrees.append(a.sum() * 2.0 / d)          # mean total degree, in+out
    return {"p": float(p), "connected_fraction": connected / draws,
            "mean_degree": float(np.mean(degrees))}


RULES = {
    "fixed_0.5": lambda d: 0.5,
    "percolation_1/d": lambda d: 1.0 / d,
    "er2_2/(d-1)": lambda d: 2.0 / (d - 1),
    "connectivity_ln(d)/d": lambda d: math.log(d) / d,
    "er4_4/(d-1)": lambda d: 4.0 / (d - 1),
    "2ln(d)/d": lambda d: 2.0 * math.log(d) / d,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", default="5,8,10,15,20,30")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/graph_density.json")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    dims = [int(x) for x in args.dims.split(",")]
    report = {"args": vars(args), "by_rule": {}}

    header = "rule".ljust(22) + "".join(f"d={d}".rjust(16) for d in dims)
    print(header)
    print("-" * len(header))
    for name, rule in RULES.items():
        row, cells = {}, ""
        for d in dims:
            p = min(max(rule(d), 1e-6), 0.999)
            m = measure(d, p, args.draws, rng)
            row[str(d)] = m
            cells += f"{m['connected_fraction']:.2f}/{m['mean_degree']:.1f}".rjust(16)
        report["by_rule"][name] = row
        print(name.ljust(22) + cells)
    print()
    print("cells are  connected-fraction / mean-degree")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    return report


if __name__ == "__main__":
    main()
