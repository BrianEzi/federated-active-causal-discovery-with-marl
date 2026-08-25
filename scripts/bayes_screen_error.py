"""How wrong is the assignment screen? Measured against full enumeration.

`WindowBeliefDP` above `MAX_EAGER_ASSIGNMENTS` no longer scores every confounding
assignment. It ranks them by an additive per-pair surrogate costing `1 + 2 * n_pairs`
partition calls, keeps the best `screen_keep`, and scores THOSE exactly. The surrogate is
mean-field -- it ignores interaction between simultaneously-declared pairs -- so the
question is not whether it is approximate (it is) but whether the shortlist ever omits an
assignment that mattered.

That is answerable exactly at |X| = 3 and 4, where full enumeration still runs. Three
numbers, all on REAL SCM DRAWS rather than `rng.normal`, because a flat posterior over
structureless data would make any shortlist look equally good:

  mass_kept   posterior mass of the kept assignments, against all of them. The headline.
  linf        largest absolute difference in any edge marginal. What the policy sees.
  rank_recall fraction of the exact top-`keep` that the screen also chose.

The metric-reachability question is separate and lives in the test suite: the TRUE
assignment must survive the screen, or `joint_conf_dag_probability` reads zero for reasons
that have nothing to do with the agent.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ma.belief_dp import WindowBeliefDP
from sa.scm import sample, sample_multi, sample_scm_params


def episode(k: int, seed: int):
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((k, k), dtype=int)
    for u in range(k):
        for v in range(u + 1, k):
            if rng.random() < 2 * np.log(k) / k:
                adjacency[u, v] = 1
    params = sample_scm_params(adjacency, rng)
    obs, obs_known = sample(params, 800, rng)
    blocks, knowns = [obs], [obs_known]
    for node in range(min(k, 3)):
        d, m = sample_multi(params, 100, rng, intervene_nodes={node: 0.0})
        blocks.append(d); knowns.append(m)
    samples, known = np.vstack(blocks), np.vstack(knowns)
    clean = np.zeros(len(samples))
    return samples, known, clean


def compare(k: int, n_shared: int, keep: int, seeds: int = 5) -> dict:
    shared = list(range(k - n_shared, k))
    rows = []
    for seed in range(seeds):
        samples, known, clean = episode(k, seed)

        exact = WindowBeliefDP(k, shared)                       # eager
        screened = WindowBeliefDP(k, shared, screen_keep=keep, max_eager=0)

        t0 = time.perf_counter()
        prep_e = [(a, z) for a, _, z in exact.prepared_assignments(samples, known, clean)
                  if np.isfinite(z)]
        m_exact = exact.joint_conf_marginals(samples, known, clean)
        t_exact = time.perf_counter() - t0

        t0 = time.perf_counter()
        prep_s = [(a, z) for a, _, z in screened.prepared_assignments(samples, known, clean)
                  if np.isfinite(z)]
        m_screen = screened.joint_conf_marginals(samples, known, clean)
        t_screen = time.perf_counter() - t0

        all_z = np.asarray([z for _, z in prep_e])
        w = np.exp(all_z - all_z.max()); w /= w.sum()
        by_assignment = {a: wi for (a, _), wi in zip(prep_e, w)}
        kept = {a for a, _ in prep_s}
        mass_kept = float(sum(by_assignment.get(a, 0.0) for a in kept))

        top_exact = {a for a, _ in sorted(prep_e, key=lambda t: -t[1])[:keep]}
        recall = len(top_exact & kept) / max(len(top_exact), 1)

        rows.append({"mass_kept": mass_kept,
                     "linf": float(np.max(np.abs(m_exact - m_screen))),
                     "rank_recall": recall,
                     "n_exact": len(prep_e), "n_screened": len(prep_s),
                     "t_exact": t_exact, "t_screen": t_screen})

    agg = {key: float(np.mean([r[key] for r in rows])) for key in
           ("mass_kept", "linf", "rank_recall", "t_exact", "t_screen")}
    agg.update({"k": k, "n_shared": n_shared, "keep": keep,
                "mass_kept_min": float(np.min([r["mass_kept"] for r in rows])),
                "linf_max": float(np.max([r["linf"] for r in rows])),
                "n_exact": rows[0]["n_exact"], "n_screened": rows[0]["n_screened"]})
    return agg


def main() -> None:
    out = []
    for k, n_shared, keep in ((4, 3, 8), (5, 4, 16), (6, 4, 32), (7, 4, 32), (8, 4, 64)):
        row = compare(k, n_shared, keep)
        out.append(row)
        print(f"k={row['k']} |X|={row['n_shared']} keep={row['keep']:3d}  "
              f"exact={row['n_exact']:4d}->kept={row['n_screened']:3d}  "
              f"mass={row['mass_kept']:.6f} (min {row['mass_kept_min']:.6f})  "
              f"Linf={row['linf']:.2e} (max {row['linf_max']:.2e})  "
              f"recall={row['rank_recall']:.2f}  "
              f"{row['t_exact']:.2f}s -> {row['t_screen']:.2f}s "
              f"({row['t_exact']/max(row['t_screen'],1e-9):.1f}x)", flush=True)
    dest = Path("results/bayes_screen_error.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
