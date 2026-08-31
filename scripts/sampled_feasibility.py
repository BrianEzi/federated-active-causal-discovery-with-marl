"""Is there ANY signal under sampled evidence? Gate the sampled sweep on this.

WHY. Measured 31 Aug: greedy scores 0.000 under sampled evidence at n_obs 60, 200 AND 1000.
`oracle_cover` refuses under sampling by construction -- the belief is not a function of the
intervened SET alone, so no set is sufficient with certainty and the required cover does not
exist -- so there is no ceiling arm either. If greedy is at zero and there is no ceiling, the
sampled sweep could produce 60 runs with nothing separating any arm from any other.

That is a ~126 core-hour commitment, and it is exactly the failure the beta-feasibility gate
caught in the oracle sweep: budgets so tight that nothing could succeed, which read as a
scaling result rather than as a broken configuration.

WHAT IS SWEPT, and why these axes. Under sampled evidence the belief prunes by
`estimated_reveal`, a test for whether an intervened node's effect is visible in the data. So
the binding quantities are the SAMPLE SIZES that test runs on and the ROUNDS available:

    n_int    interventional rows per round -- the direct power of the ancestry test, and the
             prime suspect at the default of 20
    n_obs    observational rows -- the baseline the test compares against
    beta     budget as a multiple of the required cover, so more rounds means more evidence

Reported against RANDOM as well as greedy: greedy above zero is necessary, but greedy above
RANDOM is what says the belief is responding to targeting rather than to volume of data.

    .venv/bin/python scripts/sampled_feasibility.py --episodes 30
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ma.baselines import make_baselines                              # noqa: E402
from ma.env import MAConfig, TwoAgentEnv                             # noqa: E402
from ma.evaluate import run_arm                                      # noqa: E402
from ma.topology import federated_topology                           # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private", type=int, default=4)
    ap.add_argument("--shared", type=int, default=4)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n_int", default="20,100,400")
    ap.add_argument("--n_obs", default="60,400")
    ap.add_argument("--budgets", default="35,105")
    ap.add_argument("--out", default="results/sampled_feasibility.json")
    args = ap.parse_args(argv)

    rows = []
    print(f"{'n_int':>6s} {'n_obs':>6s} {'budget':>7s} {'greedy':>8s} {'random':>8s} "
          f"{'gap':>7s} {'greedy SHD':>11s} {'random SHD':>11s}")
    for n_int, n_obs, budget in itertools.product(
            [int(x) for x in args.n_int.split(",")],
            [int(x) for x in args.n_obs.split(",")],
            [int(x) for x in args.budgets.split(",")]):
        cfg = MAConfig(topology=federated_topology(args.agents, args.private, args.shared),
                       n_obs=n_obs, n_int=n_int, budget=budget, turn_order="round_robin",
                       belief_backend="factored", action_modes=("vary",), claim_bar=1.0,
                       reward_criterion="claims", policy_arch="gnn_portable",
                       graph_model="sf", sf_m=2, episode_mix="confounded",
                       vs_evidence="sampled", per_agent_reward=True)
        env = TwoAgentEnv(cfg)
        out = {}
        for label in ("greedy_uncertainty", "random_vary"):
            policies = {a: make_baselines(env, a, 0)[label] for a in env.topology.agents}
            out[label] = run_arm(env, policies, args.episodes, seed=0)
        g, r = out["greedy_uncertainty"], out["random_vary"]
        row = {"n_int": n_int, "n_obs": n_obs, "budget": budget,
               "greedy": g["success"], "random": r["success"],
               "greedy_shd": g["global_soft_shd"], "random_shd": r["global_soft_shd"]}
        rows.append(row)
        print(f"{n_int:6d} {n_obs:6d} {budget:7d} {g['success']:8.3f} {r['success']:8.3f} "
              f"{g['success']-r['success']:+7.3f} {g['global_soft_shd']:11.4f} "
              f"{r['global_soft_shd']:11.4f}")

    best = max(rows, key=lambda x: x["greedy"] - x["random"])
    print(f"\nbest separation: n_int={best['n_int']} n_obs={best['n_obs']} "
          f"budget={best['budget']} -> greedy {best['greedy']:.3f} vs random "
          f"{best['random']:.3f}")
    if best["greedy"] <= 0.0:
        print("\nGATE FAILS: greedy never rises above zero at any setting tried. The sampled")
        print("sweep would produce runs with no separable arms. Widen the sweep or reconsider")
        print("what the sampled results can claim BEFORE committing ~126 core-hours.")
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=1))
    print(f"wrote {path}")
    return 0 if best["greedy"] > 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
