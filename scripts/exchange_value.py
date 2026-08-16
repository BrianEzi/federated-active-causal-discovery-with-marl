"""What do the `|X|^2` disclosed bits actually buy?

MA_DESIGN section 5 established that local acyclicity plus agreement on `X` does NOT imply
global acyclicity, and that the exact remedy is for each agent to disclose, for every
ordered pair of shared nodes, one bit: *does a directed path exist on my side?* At most
`|X|^2` bits, naming no private variable.

I originally wrote that check as a safety net -- catch a jointly impossible graph before
declaring victory. That framing is wrong, and this script measures the right one. If both
agents have recovered their true induced DAGs the union is acyclic by construction, so the
check can never fire at the truth. Its real job is PRUNING: it deletes joint hypotheses
that are individually plausible but jointly cyclic, and moves posterior mass onto the
survivors.

That makes it quantifiable in the same units as the disclosure. Bits in, bits out.

This is a purely structural measurement -- no data, no scores, no sampling. Uniform prior
over legal joint hypotheses, so the numbers are properties of the topology alone and
cannot be blamed on an estimator.

PRE-REGISTERED PREDICTION, before the numbers exist:
    The exchange should return well under one bit at (1,1,2) -- with |X|=2 there is only
    one shared pair, so there is very little room for the two agents' induced orders to
    conflict. At (1,1,3) it should be several times larger, because three shared pairs
    give three chances to disagree, and a cycle needs only one. If the yield at (1,1,2)
    is near zero that is a further argument for starting at (1,1,3), independent of the
    confounding-rate argument.

    I do NOT expect yield to approach the |X|^2 ceiling. The bound is worst-case; most
    disclosed bits will be redundant with what agreement on X already implies.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from ma.projection import ancestor_matrix
from ma.topology import Topology
from sa.graphs import build_graph_space


def induced_order_on_shared(adjacency: np.ndarray, observed, shared) -> int:
    """The agent's disclosure, packed into an integer bitmask.

    Bit for ordered pair (i, j) of shared nodes is set iff the agent's graph contains a
    directed path from i to j -- possibly routed through the agent's private nodes, which
    is exactly the part the other agent cannot see and cannot reconstruct.
    """
    anc = ancestor_matrix(adjacency)
    bits = 0
    k = len(shared)
    for a in range(k):
        for b in range(k):
            if a != b and anc[shared[a], shared[b]]:
                bits |= 1 << (a * k + b)
    return bits


def jointly_acyclic(bits_a: int, bits_b: int, k: int) -> bool:
    """Is the union of the two induced orders on `X` acyclic?

    Section 5: any cycle in the global union must decompose into segments lying wholly in
    one agent's graph, each entering and leaving through `X`, so it projects onto a cycle
    here. Checking this relation is therefore necessary AND sufficient.
    """
    reach = np.zeros((k, k), dtype=bool)
    union = bits_a | bits_b
    for a in range(k):
        for b in range(k):
            if a != b and (union >> (a * k + b)) & 1:
                reach[a, b] = True
    closed = reach.copy()
    for m in range(k):                       # transitive closure, then look for a loop
        closed |= np.outer(closed[:, m], closed[m, :])
    return not closed.diagonal().any()


def measure(topology: Topology) -> dict:
    """Enumerate every legal joint hypothesis and score the exchange."""
    shared = list(topology.exposed)
    k = len(shared)
    obs = {"A": list(topology.observed_by("A")), "B": list(topology.observed_by("B"))}

    # Each agent's hypothesis space is DAGs over its OWN window. Enumerated in local
    # index space, then mapped back to global node ids.
    local = {}
    for agent in ("A", "B"):
        nodes = obs[agent]
        space = build_graph_space(len(nodes))
        pos = {node: i for i, node in enumerate(nodes)}
        shared_pos = [pos[s] for s in shared]
        entries = []
        for adjacency in space.dags:
            adjacency = np.asarray(adjacency, dtype=np.int8)
            # Signature of the induced subgraph on X -- what the two agents must agree on.
            sig = tuple(int(adjacency[shared_pos[a], shared_pos[b]])
                        for a in range(k) for b in range(k) if a != b)
            entries.append((sig, induced_order_on_shared(adjacency, nodes, shared_pos)))
        local[agent] = entries

    # Group by the shared-subgraph signature: only pairs agreeing on X are legal joint
    # hypotheses at all, so agreement is applied BEFORE the exchange is scored. Otherwise
    # the exchange would be credited with pruning that agreement already did.
    by_sig = {}
    for sig, bits in local["A"]:
        by_sig.setdefault(sig, [[], []])[0].append(bits)
    for sig, bits in local["B"]:
        if sig in by_sig:
            by_sig[sig][1].append(bits)

    total = surviving = 0
    for sig, (as_, bs_) in by_sig.items():
        if not bs_:
            continue
        for ba in as_:
            for bb in bs_:
                total += 1
                if jointly_acyclic(ba, bb, k):
                    surviving += 1

    if total == 0:
        raise RuntimeError("no legal joint hypotheses -- topology or signature is wrong")

    pruned = total - surviving
    # Uniform prior, so entropy is log2 of the count and the reduction is exactly the
    # log-ratio. This is the honest "bits out" figure.
    bits_gained = float(np.log2(total) - np.log2(surviving)) if surviving else float("inf")

    return {
        "topology": topology.name,
        "shared": shared,
        "n_agent_hypotheses": {"A": len(local["A"]), "B": len(local["B"])},
        "joint_agreeing_on_X": total,
        "jointly_acyclic": surviving,
        "pruned": pruned,
        "pruned_fraction": pruned / total,
        "bits_disclosed_worst_case": k * k,
        "bits_gained": bits_gained,
        "yield_per_disclosed_bit": bits_gained / (k * k),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ma/exchange_value.json")
    args = ap.parse_args()

    topologies = [
        Topology("(1,1,2)", a_private=(0,), b_private=(1,), exposed=(2, 3)),
        Topology("(1,1,3)", a_private=(0,), b_private=(1,), exposed=(2, 3, 4)),
        Topology("(2,2,2)", a_private=(0, 1), b_private=(2, 3), exposed=(4, 5)),
    ]

    rows = []
    for topology in topologies:
        row = measure(topology)
        rows.append(row)
        print(f"{row['topology']}: joint={row['joint_agreeing_on_X']:>9}"
              f"  pruned={row['pruned_fraction']:>6.1%}"
              f"  bits_gained={row['bits_gained']:.3f}"
              f"  of {row['bits_disclosed_worst_case']} disclosed"
              f"  ({row['yield_per_disclosed_bit']:.3f}/bit)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
