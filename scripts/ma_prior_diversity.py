"""How DIVERSE is the set of graphs the prior actually generates -- not just how connected.

Connectivity going up when `p` goes up is not a finding, it's arithmetic. What matters for
generalisation is whether training sees a broad spread of structures or the same handful of
shapes over and over. This measures that directly, across every shape on the scaling
ladder, not just the topology currently wired into `scripts/ma_train.py`.

**Pure Monte Carlo -- deliberately, after a wrong attempt at something faster.** A first
version tried to get the induced distribution in closed form by reweighting the enumerated
DAG space by `p^|E| (1-p)^(nonedges)`. That is wrong: `sample_dag` draws a random
topological order and includes each allowed forward pair independently, and the true
induced probability of a specific DAG is proportional to `p^|E| (1-p)^(nonedges)` TIMES the
number of topological orderings consistent with that DAG -- a highly sparse, near-empty DAG
has far more compatible orderings than a near-total-order one, so the reweighting without
that factor was silently biased toward dense structures. Caught by cross-checking against a
400k-draw Monte Carlo before anything was reported: the edge-count histograms disagreed by
up to 10 points of probability mass at the mode. Counting topological orderings exactly is
possible for small d but adds real complexity for no benefit here, since direct simulation
already IS the exact generative process the environment uses -- there is nothing to
approximate. Large N and a convergence check are the honest way to get this number right.

Reported per shape, DAG-level and MEC-level (the level the belief actually distinguishes):
entropy (nats), effective support = exp(entropy), and top-1 / top-5 mass concentration.
Convergence is checked by re-running at 2x draws and confirming entropy is stable.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

import numpy as np

from ma.topology import Topology, two_agent
from ma.graphs import mec_signature
from ma.priors import connectivity_prior_p


def shapes():
    return [
        two_agent("T1_2-2-2", (0, 1), (2, 3), (4, 5)),
        two_agent("T1_1-1-3_current", (0,), (1,), (2, 3, 4)),
        Topology("rung1_3agents_1each", private=((0,), (1,), (2,)), exposed=(3, 4, 5)),
        Topology("rung2_3agents_2each", private=((0, 1), (2, 3), (4, 5)), exposed=(6, 7, 8)),
        Topology("rung3_5agents_1each", private=tuple((i,) for i in range(5)),
                 exposed=(5, 6, 7, 8, 9)),
    ]


def entropy_and_support(counts: np.ndarray) -> tuple:
    """Shannon entropy in nats, effective support `exp(H)`, and top-1/top-5 mass."""
    p = counts / counts.sum()
    p_nz = p[p > 0]
    h = float(-(p_nz * np.log(p_nz)).sum())
    ordered = np.sort(p)[::-1]
    return h, float(np.exp(h)), float(ordered[0]), float(ordered[:5].sum())


def sample(topology: Topology, p: float, draws: int, rng) -> dict:
    dag_counts = Counter()
    mec_counts = Counter()
    for _ in range(draws):
        adjacency = topology.sample_dag(rng, p=p)
        dag_counts[adjacency.tobytes()] += 1
        mec_counts[mec_signature(adjacency)] += 1

    dag_h, dag_supp, dag_top1, dag_top5 = entropy_and_support(
        np.array(list(dag_counts.values()), dtype=float))
    mec_h, mec_supp, mec_top1, mec_top5 = entropy_and_support(
        np.array(list(mec_counts.values()), dtype=float))

    return {
        "draws": draws,
        "n_distinct_dags_seen": len(dag_counts), "n_distinct_mecs_seen": len(mec_counts),
        "dag_entropy_nats": dag_h, "dag_effective_support": dag_supp,
        "dag_top1_mass": dag_top1, "dag_top5_mass": dag_top5,
        "mec_entropy_nats": mec_h, "mec_effective_support": mec_supp,
        "mec_top1_mass": mec_top1, "mec_top5_mass": mec_top5,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/prior_diversity.json")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    rows = []
    header = ("%-24s %2s %6s %6s  %8s   %8s %10s %8s %8s   %8s %10s %8s %8s" % (
        "shape", "d", "old_p", "new_p", "n_dags", "dag_H", "dag_eff", "dag_top1", "dag_top5",
        "n_mec", "mec_H", "mec_eff", "mec_top1"))
    print(header); print("-" * len(header))

    for topology in shapes():
        p_old, p_new = 0.5, connectivity_prior_p(topology.d)
        for p_label, p in [("old", p_old), ("new", p_new)]:
            r = sample(topology, p, args.draws, rng)
            # convergence check: half the draws, same seed continuation, should give a
            # visibly close entropy -- if it does not, `draws` is too small to trust.
            r_half = sample(topology, p, args.draws // 2, np.random.default_rng(rng.integers(1 << 31)))
            r.update(name=topology.name, d=topology.d, p_label=p_label, p=p,
                     dag_entropy_at_half_draws=r_half["dag_entropy_nats"])
            rows.append(r)
            print("%-24s %2d %6.3f %6.3f  %8d   %10.3f %8.1f %8.3f %8.3f   %8d %10.3f %8.1f %8.3f" % (
                topology.name, topology.d, p_old, p_new, r["n_distinct_dags_seen"],
                r["dag_entropy_nats"], r["dag_effective_support"], r["dag_top1_mass"],
                r["dag_top5_mass"], r["n_distinct_mecs_seen"], r["mec_entropy_nats"],
                r["mec_effective_support"], r["mec_top1_mass"]))
        print()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print("wrote %s" % out)
    return rows


if __name__ == "__main__":
    main()
