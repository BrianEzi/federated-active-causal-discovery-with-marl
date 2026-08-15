"""V1 -- Graph enumeration, Markov equivalence classes, singleton fractions.

Verifies the combinatorial baselines quoted throughout Part I:

  * |DAG(d)| against Robinson's recurrence, computed independently here.
  * |MEC(d)| against OEIS A007984.
  * The singleton fraction (fraction of DAGs alone in their equivalence class),
    which is GATE 1's exact target in Part VI.
  * That the Verma-Pearl signature (skeleton, v-structures) is constant on the
    classes produced by the vectorised bit-code path in sa/graphs.py.

Run:  python textbook/verification/v1_graph_space.py
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np

from sa.graphs import (N_DAGS, N_MECS, build_graph_space, enumerate_dags,
                       enumerate_dags_fast, mec_signature, descendants)


def robinson_counts(dmax: int) -> list:
    """a(d) = sum_{k=1}^{d} (-1)^{k+1} C(d,k) 2^{k(d-k)} a(d-k),  a(0) = 1.

    Robinson (1973). This is the *sink-based inclusion-exclusion* recurrence: choose
    the non-empty set of k sinks (nodes with no outgoing edge), which may be joined
    from any of the remaining d-k nodes in 2^{k(d-k)} ways; the alternating sign
    corrects for the overcount of graphs with more than k sinks.
    """
    a = [1]
    for d in range(1, dmax + 1):
        total = 0
        for k in range(1, d + 1):
            total += (-1) ** (k + 1) * comb(d, k) * 2 ** (k * (d - k)) * a[d - k]
        a.append(total)
    return a


def main() -> None:
    print("=" * 72)
    print("V1  Graph space, Markov equivalence classes, singleton fractions")
    print("=" * 72)

    # -- 1. Robinson's recurrence vs. brute-force enumeration -------------------
    rob = robinson_counts(7)
    print("\n[1] Labelled DAG counts")
    print(f"    Robinson recurrence a(0..7) = {rob}")
    for d in range(1, 6):
        brute = len(enumerate_dags(d))
        assert brute == rob[d] == N_DAGS[d], (d, brute, rob[d], N_DAGS[d])
        print(f"    d={d}: enumerated {brute:>9,} == Robinson {rob[d]:>9,}  OK")
    for d in (6, 7):
        assert rob[d] == N_DAGS.get(d, rob[d])
        print(f"    d={d}: Robinson {rob[d]:>13,}  (enumeration not attempted)")

    # -- 2. Vectorised enumeration is bit-identical to the reference -----------
    print("\n[2] enumerate_dags_fast == enumerate_dags (order included)")
    for d in range(1, 6):
        assert np.array_equal(enumerate_dags(d), enumerate_dags_fast(d)), d
        print(f"    d={d}: identical arrays  OK")

    # -- 3. MEC counts and signature invariance --------------------------------
    print("\n[3] Markov equivalence classes")
    for d in range(1, 6):
        space = build_graph_space(d)
        assert space.n_mecs == N_MECS[d], (d, space.n_mecs, N_MECS[d])
        # Verma-Pearl signature must be constant within, and distinct between, classes.
        sigs = {}
        for idx in range(space.n_dags):
            sigs.setdefault(int(space.mec_id[idx]), set()).add(
                mec_signature(space.dags[idx]))
        assert all(len(s) == 1 for s in sigs.values()), d
        distinct = {next(iter(s)) for s in sigs.values()}
        assert len(distinct) == space.n_mecs, d
        print(f"    d={d}: {space.n_dags:>6,} DAGs in {space.n_mecs:>5,} classes "
              f"(A007984 = {N_MECS[d]:>5,})  signature constant  OK")

    # -- 4. Singleton fractions (GATE 1 target) --------------------------------
    print("\n[4] Singleton fraction = fraction identifiable WITHOUT intervening")
    for d in range(1, 6):
        space = build_graph_space(d)
        n_singleton = int(np.sum(space.mec_sizes == 1))
        frac = space.singleton_fraction
        largest = int(space.mec_sizes.max())
        print(f"    d={d}: singleton classes {n_singleton:>5,}/{space.n_mecs:<6,}"
              f"  P(singleton DAG) = {frac:.4f}"
              f"  largest class = {largest}")

    # -- 5. MEC size distribution at d=3 (used in the Part I worked example) ---
    print("\n[5] d=3 equivalence-class census")
    space = build_graph_space(3)
    for cls in range(space.n_mecs):
        members = np.flatnonzero(space.mec_id == cls)
        a0 = space.dags[members[0]]
        sk, vs = mec_signature(a0)
        sk_txt = "{" + ",".join(
            f"{min(e)}-{max(e)}" for e in sorted(map(sorted, sk))) + "}"
        vs_txt = "{" + ",".join(f"{i}->{k}<-{j}" for i, k, j in sorted(vs)) + "}"
        print(f"    class {cls}: size {len(members)}  skeleton {sk_txt:<14} "
              f"v-structures {vs_txt}")

    # -- 6. Deterministic reachability, used by the oracle in Part III ---------
    print("\n[6] Descendant sets are a deterministic function of the DAG")
    space = build_graph_space(3)
    reach = np.array([descendants(a) for a in space.dags])
    # Two DAGs are indistinguishable by do(X_i) exactly when reach[:, i] agrees.
    for node in range(3):
        codes = reach[:, node, :] @ (1 << np.arange(3))
        print(f"    node {node}: {len(np.unique(codes))} distinct descendant sets "
              f"across {space.n_dags} DAGs")

    print("\nALL V1 CHECKS PASSED")


if __name__ == "__main__":
    main()
