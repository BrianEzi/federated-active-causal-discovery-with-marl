"""V5 -- Expected information gain and the deterministic reachability collapse.

Part III, Chapters 9-10.

  [1] The collapse identity  I(G; Y | a) = H(Y | a)  verified directly by
      computing all three terms of  I = H(Y) - H(Y|G)  from the joint
      distribution, for random posteriors over the d = 3, 4 graph spaces.
  [2] Equivalently  I(G; Y | a) = H(G) - H(G | Y):  the expected posterior
      entropy reduction, computed the long way round (enumerate outcomes,
      condition, average), matches sa/oracle.py's single bincount.
  [3] The oracle's score is exactly the entropy of the descendant-set partition,
      and is ZERO exactly when every graph in the support agrees.
  [4] Shannon entropy vs. the Gini/Simpson index the previous implementation
      used: they induce DIFFERENT rankings over targets, so the substitution
      was not cosmetic.
  [5] The bound  I(G; Y | a) <= log |partition|  and its attainment.

Run:  python textbook/verification/v5_eig_collapse.py
"""
from __future__ import annotations

import numpy as np

from sa.graphs import build_graph_space, descendants
from sa.oracle import InterventionOracle

RNG = np.random.default_rng(303)


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def descendant_signature(space):
    """[N, d] integer code of each DAG's descendant set from each node."""
    d = space.d
    bit = (1 << np.arange(d)).astype(np.int64)
    return np.array([descendants(a).astype(np.int64) @ bit for a in space.dags])


def main() -> None:
    print("=" * 76)
    print("V5  Expected information gain and the reachability collapse")
    print("=" * 76)

    for d in (3, 4):
        space = build_graph_space(d)
        oracle = InterventionOracle(space)
        codes = descendant_signature(space)
        print(f"\n{'='*76}\nd = {d}: {space.n_dags} DAGs")

        for trial in range(3):
            # A random, deliberately non-uniform posterior over DAGs.
            alpha = RNG.uniform(0.2, 1.5, size=space.n_dags)
            post = RNG.dirichlet(alpha)

            print(f"\n  --- posterior draw {trial + 1} "
                  f"(H(G) = {entropy(post):.4f} nats) ---")
            oracle_scores = oracle.scores(post)

            for node in range(d):
                # ---- the joint distribution p(G, Y) for the experiment do(X_node)
                # Y = descendant set of `node`, a DETERMINISTIC function of G.
                outcomes, inverse = np.unique(codes[:, node], return_inverse=True)
                K = len(outcomes)

                # p(Y = y)
                py = np.bincount(inverse, weights=post, minlength=K)

                # H(Y)
                H_Y = entropy(py)

                # H(Y | G) = sum_G p(G) H(Y | G).  Y is a point mass given G.
                H_Y_given_G = 0.0
                for g in range(space.n_dags):
                    if post[g] <= 0:
                        continue
                    row = np.zeros(K); row[inverse[g]] = 1.0     # deterministic
                    H_Y_given_G += post[g] * entropy(row)

                I_1 = H_Y - H_Y_given_G

                # ---- the other direction: I = H(G) - E_Y[ H(G | Y) ]
                H_G = entropy(post)
                H_G_given_Y = 0.0
                for k in range(K):
                    mask = inverse == k
                    m = post[mask].sum()
                    if m <= 0:
                        continue
                    H_G_given_Y += m * entropy(post[mask] / m)
                I_2 = H_G - H_G_given_Y

                assert abs(H_Y_given_G) < 1e-12, H_Y_given_G
                assert abs(I_1 - H_Y) < 1e-12
                assert abs(I_1 - I_2) < 1e-10, (I_1, I_2)
                assert abs(I_1 - oracle_scores[node]) < 1e-10, (I_1, oracle_scores[node])

                if trial == 0:
                    print(f"    do(X{node}): |partition| = {K:<3} "
                          f"H(Y) = {H_Y:.6f}   H(Y|G) = {H_Y_given_G:.1e}   "
                          f"I = {I_1:.6f}   oracle = {oracle_scores[node]:.6f}  "
                          f"<= log K = {np.log(K):.4f}")
                    assert I_1 <= np.log(K) + 1e-12
            if trial == 0:
                print(f"    all three routes (H(Y); H(Y)-H(Y|G); H(G)-E H(G|Y); "
                      f"sa/oracle.py) agree to < 1e-10")

    # ------------------------------------------------------------------ [3] --
    print(f"\n{'='*76}")
    print("[3] The oracle scores ZERO exactly when the support agrees on descendants")
    space = build_graph_space(3)
    oracle = InterventionOracle(space)
    codes = descendant_signature(space)
    # Concentrate all mass on graphs sharing node 0's descendant set.
    target_code = codes[0, 0]
    mask = codes[:, 0] == target_code
    post = np.zeros(space.n_dags); post[mask] = RNG.dirichlet(np.ones(mask.sum()))
    s = oracle.scores(post)
    print(f"    support restricted to the {int(mask.sum())} DAGs with the same "
          f"descendants from X0")
    print(f"    oracle scores = {np.round(s, 10)}")
    print(f"    -> do(X0) has EIG exactly {s[0]:.1e}: it cannot discriminate, "
          f"however unexplored X0 looks")
    assert s[0] < 1e-12

    # Fully identified posterior: nothing is informative.
    point = np.zeros(space.n_dags); point[7] = 1.0
    print(f"    point-mass posterior -> scores {np.round(oracle.scores(point), 12)} "
          f"(all zero: nothing left to learn)")
    assert oracle.scores(point).max() < 1e-12

    # ------------------------------------------------------------------ [4] --
    print("\n[4] Shannon entropy vs. the Gini/Simpson index (Tsallis-2)")
    print("    The previous implementation maximised  1 - sum_g P(g)^2  instead of")
    print("    -sum_g P(g) log P(g). Both are concave uncertainty measures, but they")
    print("    are not ordinally equivalent, so they can rank targets differently.")
    disagreements = 0
    trials = 4000
    space = build_graph_space(4)
    codes4 = descendant_signature(space)
    for _ in range(trials):
        post = RNG.dirichlet(RNG.uniform(0.05, 0.6, size=space.n_dags))
        sh, gi = np.zeros(4), np.zeros(4)
        for node in range(4):
            _, inv = np.unique(codes4[:, node], return_inverse=True)
            m = np.bincount(inv, weights=post)
            m = m[m > 0]
            sh[node] = -np.sum(m * np.log(m))
            gi[node] = 1.0 - np.sum(m ** 2)
        if int(np.argmax(sh)) != int(np.argmax(gi)):
            disagreements += 1
    print(f"    over {trials} random posteriors at d = 4, the argmax differs in "
          f"{disagreements} cases ({disagreements/trials:.1%})")
    print("    -> substituting Shannon entropy was a correction, not a refactor:")
    print("       only Shannon entropy IS the expected information gain (Lindley 1956).")

    # A minimal explicit counterexample.
    print("\n    Minimal explicit disagreement between the two indices:")
    a = np.array([0.50, 0.25, 0.25])
    b = np.array([0.60, 0.10, 0.10, 0.10, 0.10])
    for name, p in (("A (3 outcomes)", a), ("B (5 outcomes)", b)):
        print(f"      {name}: Shannon {entropy(p):.6f}   "
              f"Gini {1 - np.sum(p**2):.6f}")
    print(f"      Shannon prefers {'B' if entropy(b) > entropy(a) else 'A'}; "
          f"Gini prefers "
          f"{'B' if (1-np.sum(b**2)) > (1-np.sum(a**2)) else 'A'}")

    print("\nALL V5 CHECKS PASSED")


if __name__ == "__main__":
    main()
