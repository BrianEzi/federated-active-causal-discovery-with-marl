"""How far does the disclosure design actually carry, end to end?

Asked 2026-08-23: disclosure is O(|shared|^2) numbers, but the number of shared PAIRS grows
quadratically, so what breaks first on the way to d=30?

Three costs, measured rather than reasoned about:

  A. assignment enumeration   3^C(|X|,2), the joint_conf mixture. PRE-EXISTING -- this cost
                              is paid today, with or without disclosure.
  B. sender-side q            C(|X|,2) x (2^|private| - 1) partition calls. Inclusion-
                              exclusion, because "some private node parents both" is a
                              UNION over private nodes and unions do not factorise.
  C. the subset DP itself     O(k 2^k) per call, k = window size.

The point of measuring rather than counting is that the constant matters: 729 assignments at
|X|=4 is a different proposition depending on whether one belief update is 20 ms or 2 s, and
an RL run performs one per agent per round for tens of thousands of rounds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from itertools import combinations

import numpy as np

from ma.belief_dp import WindowBeliefDP, JOINT_CONF


def time_belief_update(k: int, n_shared: int, n_rows: int, repeats: int = 3) -> dict:
    """One joint_conf belief update on a window of `k` nodes with `n_shared` shared."""
    shared_positions = list(range(k - n_shared, k))
    rng = np.random.default_rng(0)

    build_start = time.perf_counter()
    belief = WindowBeliefDP(k, shared_positions)
    build_s = time.perf_counter() - build_start

    samples = rng.normal(size=(n_rows, k))
    known = np.zeros((n_rows, k))
    # Two regimes, so the clean/dirty split is exercised rather than short-circuited.
    clean = np.concatenate([np.ones(n_rows // 2), np.zeros(n_rows - n_rows // 2)])

    best = float("inf")
    for _ in range(repeats):
        # The cache is keyed on the data shape, so it must be cleared between repeats or
        # every run after the first measures a dictionary lookup.
        belief._assign_key = None
        belief._assign_cache = None
        belief._table_key = None
        belief._table_cache = None
        start = time.perf_counter()
        belief.edge_marginals(samples, known, clean, JOINT_CONF)
        best = min(best, time.perf_counter() - start)

    return {"k": k, "n_shared": n_shared, "n_rows": n_rows,
            "pairs": len(belief.pairs), "assignments": len(belief.assignments),
            "build_s": build_s, "update_s": best}


def sender_cost(n_shared: int, n_private: int) -> dict:
    """Partition calls to produce one agent's full disclosure vector.

    Inclusion-exclusion over private nodes: the event is a UNION ("SOME private node is a
    common parent"), and a union of `m` events needs `2^m - 1` terms exactly. That is
    exponential in the PRIVATE set, not the window -- which is survivable precisely because
    more agents means smaller private sets.
    """
    pairs = n_shared * (n_shared - 1) // 2
    per_pair = (2 ** n_private) - 1
    return {"n_shared": n_shared, "n_private": n_private,
            "pairs": pairs, "ie_terms_per_pair": per_pair,
            "partition_calls": pairs * per_pair}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-rows", type=int, default=200)
    ap.add_argument("--max-shared", type=int, default=5)
    ap.add_argument("--out", default="results/disclosure_scaling.json")
    args = ap.parse_args(argv)

    print("A + C. One joint_conf belief update, %d rows\n" % args.n_rows)
    header = "%3s %7s %6s %13s %10s %11s" % (
        "k", "shared", "pairs", "assignments", "build s", "update s")
    print(header); print("-" * len(header))

    rows = []
    for n_shared in range(2, args.max_shared + 1):
        k = n_shared + 1              # one private node, the current topology's shape
        row = time_belief_update(k, n_shared, args.n_rows)
        rows.append(row)
        print("%3d %7d %6d %13d %10.3f %11.3f" % (
            row["k"], row["n_shared"], row["pairs"], row["assignments"],
            row["build_s"], row["update_s"]))

    print("\nB. Sender-side disclosure, partition calls per belief update\n")
    header = "%7s %8s %6s %14s %16s" % (
        "shared", "private", "pairs", "IE terms/pair", "partition calls")
    print(header); print("-" * len(header))
    sender_rows = []
    for n_shared, n_private in ((3, 1), (4, 1), (5, 2), (8, 3), (10, 4), (10, 6)):
        row = sender_cost(n_shared, n_private)
        sender_rows.append(row)
        print("%7d %8d %6d %14d %16d" % (
            row["n_shared"], row["n_private"], row["pairs"],
            row["ie_terms_per_pair"], row["partition_calls"]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"belief": rows, "sender": sender_rows}, indent=1))
    print("\nwrote %s" % out)
    return rows, sender_rows


if __name__ == "__main__":
    main()
