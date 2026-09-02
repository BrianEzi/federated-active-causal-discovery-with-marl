"""Is the learned policy's higher SHD concentrated on pairs it is NOT rewarded for?

THE DIVERGENCE THIS EXPLAINS. At k=20/k=30 with full replication the learned policy completes
more windows than greedy on 6 of 6 seeds (+0.083 and +0.125 on success) while its mean hard
SHD is no better and sometimes worse. Those two facts are only contradictory if the policy is
supposed to be optimising the graph uniformly. It is not: `reward_criterion="claims"` with
`per_agent_reward` scores an agent on the claims touching its OWN PRIVATE nodes, while
`global_hard_shd` scores every covered pair in the pooled graph, most of which are
shared-shared and carry no reward.

So the hypothesis is REWARD ALIGNMENT, not failure: the policy should beat greedy on
private-incident pairs (what it is paid for) and lose on shared-shared pairs (what it is not).
Greedy targets uncertainty uniformly and should show the opposite profile.

If the split comes out flat, the hypothesis is wrong and the divergence needs another
explanation -- which is worth knowing before it reaches a thesis.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ma.baselines import RandomAgent, UncertaintyGreedyAgent
from ma.evaluate import pooled_global_belief
from ma.policy import IndependentPPO
from scripts.rescore_from_config import env_from_config


def classify(env, pair):
    """private-incident (some endpoint is somebody's private node) vs shared-shared."""
    exposed = set(env.topology.exposed)
    u, v = pair
    return "shared_shared" if (u in exposed and v in exposed) else "private_incident"


def play(env, policies, episodes, seed):
    # Seed the torch RNG. The learned arm samples its actions from a Categorical, so without
    # this the same checkpoint draws a different action sequence on every invocation and the
    # measurement is not reproducible -- greedy and random carry their own seeded generators
    # and are unaffected, which is what makes the defect invisible in a spot-check.
    # Same fix as scripts/global_shd_paired.py; numbers produced before 2 Sep 22:00 should
    # not be expected to match.
    import torch
    torch.manual_seed(seed)
    for p in policies.values():
        if hasattr(p, "reset"): p.reset(seed)
    acc = collections.defaultdict(lambda: [0.0, 0])
    for ep in range(episodes):
        r = env.reset(seed=seed * 100_000 + ep)
        while not r.done:
            r = env.step({a: policies[a](env, r) for a in env.topology.agents})
        for pair, rec in pooled_global_belief(env).items():
            cls = classify(env, pair)
            acc[cls][0] += rec["hard"]; acc[cls][1] += 1
    return {k: (v[0] / max(v[1], 1), v[1]) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", default="results/shd_by_class.json")
    args = ap.parse_args()
    payload = []
    for path in args.results:
        path = pathlib.Path(path)
        rep = json.loads(path.read_text()); cfg = rep["config"]
        seed = rep.get("seed", 0)
        env = env_from_config(cfg, seed=seed)
        ck = path.with_name(path.stem + "_best.pt")
        if not ck.exists():
            print(f"!! {path.stem}: no checkpoint, skipped"); continue
        ppo = IndependentPPO.load(str(ck), env)
        arms = {"learned": ppo.policies(deterministic=False),
                "greedy": {a: UncertaintyGreedyAgent(a, seed, bar=1.0) for a in env.topology.agents},
                "random": {a: RandomAgent(a, seed, allow_clamp=False) for a in env.topology.agents}}
        print(f"\n=== {path.stem} ({args.episodes} episodes) ===")
        print(f"{'arm':10s} {'private-incident':>17s} {'shared-shared':>14s}   pairs/ep p/s")
        row = {"source": str(path), "seed": seed, "arms": {}}
        for label, pol in arms.items():
            r = play(env, pol, args.episodes, seed)
            pi = r.get("private_incident", (float('nan'), 0))
            ss = r.get("shared_shared", (float('nan'), 0))
            row["arms"][label] = {"private_incident": pi[0], "shared_shared": ss[0],
                                  "n_private": pi[1], "n_shared": ss[1]}
            print(f"{label:10s} {pi[0]:17.5f} {ss[0]:14.5f}   "
                  f"{pi[1]//args.episodes}/{ss[1]//args.episodes}")
        payload.append(row)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
