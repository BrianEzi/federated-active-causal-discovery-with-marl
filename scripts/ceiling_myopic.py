"""The myopic-global arm of the centralisation ceiling, and the comparability check it needs.

THE WORK ORDER asked for a single controller -- K=1, Z_1 = V, d=30, the federation's pooled
budget -- measured myopically first because it is cheap and brackets the question. It also
asked that the 1-agent topology be sanity-checked against `ma/topology` before the number is
trusted. It does not survive that check as specified, and this script is the measurement of
why rather than an assertion of it.

`Topology.allowed_edges` is the JOINTLY-VISIBLE rule: an edge may exist only where some single
agent observes both endpoints. So the partition does not merely decide who acts -- it decides
which graphs exist at all. At the principal cell (4 agents, 6 private each, 6 exposed, d=30):

    federation   438 directed edges allowed   219 undirected pairs covered
    1x24+6       870 directed edges allowed   435 undirected pairs covered

A single controller that sees everything forbids nothing, so it faces a graph family with
almost twice as many possible edges AND is scored over almost twice as many pairs. Its hard
SHD is a per-pair rate over a different denominator, drawn from a different generator. It is
not a ceiling on the federation's problem; it is a different problem.

AND THERE IS A SECOND BLOCKER THAT SUBSUMES THE FIRST. The ladder trains and scores on
`episode_mix=confounded`. Confounding here is a bidirected pair in some agent's projected MAG
-- a common cause OUTSIDE that agent's window. A controller whose window is the whole graph has
nothing outside it, so no pair can be bidirected and no episode can be confounded. Measured
over 2,000 draws at the principal cell:

    federation 4x6+6   1821/2000 confounded   91.05%
    single 1x24+6         0/2000 confounded    0.00%

The environment does not fail quietly on this: `_sample_mixed_dag` gives up after 200 draws and
raises. So the arm as specified cannot produce a single qualifying episode. Centralisation does
not make the federated problem easier -- it DISSOLVES it. That is the same reason topology T3
was rejected in August: removing the boundary removes the confounding the design exists to
study.

This script therefore measures what can be measured -- the federation's own myopic reference --
and reports the reachability test as the finding.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ma.baselines import RandomAgent, UncertaintyGreedyAgent      # noqa: E402
from ma.env import MAConfig, TwoAgentEnv                          # noqa: E402
from ma.evaluate import global_graph_report                       # noqa: E402
from ma.topology import federated_topology                        # noqa: E402


# The federation ladder's exact settings, from results/central12k/run_ladder12k.sh. Anything
# that differs here would make the ceiling incomparable for a second, avoidable reason.
LADDER = dict(n_obs=60, n_int=20, budget=50, turn_order="round_robin",
              belief_backend="factored", graph_model="sf", sf_m=2, claim_bar=1.0,
              episode_mix="confounded", vs_evidence="oracle", reward_criterion="claims")


def build(topology, **over):
    cfg = dict(LADDER)
    cfg.update(over)
    return TwoAgentEnv(MAConfig(topology=topology, action_modes=("vary",), **cfg))


def play(env, policies, episodes: int, seed: int):
    hard, soft, pairs = [], [], []
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        while not result.done:
            result = env.step({a: policies[a](env, result) for a in env.topology.agents})
        r = global_graph_report(env)
        hard.append(r["global_hard_shd"])
        soft.append(r["global_soft_shd"])
        pairs.append(r["global_pairs"])
    return np.array(hard), np.array(soft), np.array(pairs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reach_draws", type=int, default=2000,
                    help="graph draws for the confounding-reachability test")
    ap.add_argument("--out", default="results/ceiling/myopic_global.json")
    args = ap.parse_args(argv)

    arms = {
        "federation_4x6+6": federated_topology(4, 6, 6),
        "single_1x24+6": federated_topology(1, 24, 6),
    }
    payload = {"episodes": args.episodes, "seed": args.seed, "ladder_config": LADDER,
               "reachability": {}, "arms": {}}

    # REACHABILITY FIRST. A metric that cannot be earned on a regime is worse than a missing
    # one, so establish that the regime EXISTS for each topology before measuring anything on
    # it. This is the check that 529 passing tests did not make in August.
    from ma.projection import bidirected_pairs
    print("Can episode_mix='confounded' be drawn at all?")
    for label, topology in arms.items():
        rng = np.random.default_rng(args.seed)
        windows = [tuple(sorted(set(topology.private[a]) | set(topology.exposed)))
                   for a in topology.agents]
        # ONE graph per draw, tested against every window. Written as a loop because the
        # obvious comprehension puts `sample_dag` inside the `any(... for w in windows)`
        # generator, which draws a FRESH graph per window and short-circuits on the first
        # confounded one -- that read 99.85% for the federation where the true rate is 91.05%,
        # because it was asking "is any of four independent graphs confounded somewhere".
        n = 0
        for _ in range(args.reach_draws):
            g = topology.sample_dag(rng, p=0.227, model="sf", m=2)
            if any(bidirected_pairs(g, w) for w in windows):
                n += 1
        payload["reachability"][label] = {"draws": args.reach_draws, "confounded": n,
                                          "rate": n / args.reach_draws,
                                          "windows": [len(w) for w in windows]}
        print(f"  {label:20s} windows={[len(w) for w in windows]}  "
              f"{n}/{args.reach_draws} = {n / args.reach_draws:.3%}")
        if n == 0:
            print(f"  ^ UNREACHABLE. The ladder's episode regime cannot be generated on this "
                  f"topology, so no number measured here would answer the ladder's question.")

    for label, topology in arms.items():
        if payload["reachability"][label]["confounded"] == 0:
            print(f"\n=== {label} === SKIPPED: episode_mix='confounded' is unreachable")
            payload["arms"][label] = {"skipped": "episode_mix=confounded unreachable"}
            continue
        allowed = topology.allowed_edges()
        und = int(((allowed | allowed.T)[np.triu_indices(topology.d, 1)]).sum())
        env = build(topology)
        greedy = {a: UncertaintyGreedyAgent(a, args.seed, bar=1.0) for a in topology.agents}
        rnd = {a: RandomAgent(a, args.seed, allow_clamp=False) for a in topology.agents}
        gh, gs, gp = play(env, greedy, args.episodes, args.seed)
        rh, rs, rp = play(env, rnd, args.episodes, args.seed)
        payload["arms"][label] = {
            "d": topology.d, "n_agents": topology.n_agents,
            "directed_edges_allowed": int(allowed.sum()), "undirected_pairs_allowed": und,
            "myopic_hard": float(gh.mean()), "myopic_hard_se": float(gh.std(ddof=1) / np.sqrt(len(gh))),
            "myopic_soft": float(gs.mean()), "random_hard": float(rh.mean()),
            "mean_global_pairs_scored": float(gp.mean()),
        }
        print(f"\n=== {label} ===")
        print(f"  d={topology.d}  agents={topology.n_agents}  budget={LADDER['budget']}")
        print(f"  allowed: {int(allowed.sum())} directed, {und} undirected pairs")
        print(f"  pairs actually scored per episode: {gp.mean():.1f}")
        print(f"  myopic hard SHD {gh.mean():.5f} +/- {gh.std(ddof=1)/np.sqrt(len(gh)):.5f}")
        print(f"  random hard SHD {rh.mean():.5f}")

    print("\n" + "=" * 78)
    print("THE CEILING AS SPECIFIED CANNOT BE MEASURED, and that is the finding.")
    r = payload["reachability"]
    print(f"  confounded episodes: federation {r['federation_4x6+6']['rate']:.2%}, "
          f"single controller {r['single_1x24+6']['rate']:.2%}")
    print("  Confounding is a bidirected pair in an agent's projected MAG -- a common cause")
    print("  outside its window. A controller that sees everything has no outside, so the")
    print("  regime the ladder trains and scores on is definitionally empty for it.")
    print("  Separately, and now moot: the single controller's edge mask allows 870 directed")
    print("  edges against the federation's 438, and it would be scored over 435 pairs")
    print("  against 219, so even an unconfounded comparison would use a different")
    print("  denominator on a different graph family.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
