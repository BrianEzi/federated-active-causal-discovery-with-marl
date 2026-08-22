"""Does tb_both's advantage over tb_clamp come from `vary` acting as an INSTRUMENT?

tb_both beats tb_clamp by +0.021 (CI [+0.001, +0.042], 20 seeds) -- small, significant, and
so far unexplained. One candidate explanation: an intervention that VARIES a node (rather
than clamping it to a constant) leaves variance in it, which makes it usable as an
instrumental variable for relationships among the agent's other nodes. A clamp has zero
variance and is useless as an instrument, so clamp-only would forfeit exactly this.

The across-seed correlation is suggestive but underpowered: corr(vary_fraction, advantage)
= +0.34, CI [-0.29, +0.81] over 20 seeds -- the right sign, far too wide to call.

THIS IS THE HIGH-POWER VERSION. `ma/evaluate.py::run_arm` seeds each episode as
`seed*100_000 + episode`, so a given seed produces the SAME graphs for any policy. Both
checkpoints are therefore replayed over identical episodes, and the advantage is split by
whether the instrument structure is actually PRESENT in that episode's true graph.

  IV structure present, for agent `a` with private node `p` and shared set `S`:
      exists X, Y in S, X != Y, with   p -> X          (relevance: p moves X)
                                 and   p not adjacent Y (exclusion: no direct p-Y edge)

  prediction, if the IV account is right:
      advantage(tb_both - tb_clamp) should be LARGER on episodes where the structure holds
      than on episodes where it does not.

A null result here is informative too: it would say the +0.021 is something other than
instrument value, and clamp-only costs whatever it costs for a different reason.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

from ma.env import CLAMP, MAConfig, MODES, ROUND_ROBIN, TwoAgentEnv
from ma.evaluate import evaluate_episode
from ma.policy import IndependentPPO, PPOConfig
from ma.topology import two_agent


def iv_available(adjacency: np.ndarray, private_nodes, shared) -> bool:
    """Is there an instrument-shaped configuration for this agent's private node?"""
    for p in private_nodes:
        movable = [x for x in shared if adjacency[p, x]]
        excluded = [y for y in shared
                    if not adjacency[p, y] and not adjacency[y, p]]
        for x in movable:
            if any(y != x for y in excluded):
                return True
    return False


def replay(checkpoint: pathlib.Path, env: TwoAgentEnv, episodes: int, seed: int) -> list:
    """Run a saved policy pair over the SAME seeded episodes run_arm would produce."""
    learner = IndependentPPO.load(checkpoint, env)
    policies = learner.policies(deterministic=False)
    rows = []
    for episode in range(episodes):
        result = env.reset(seed=seed * 100_000 + episode)
        adjacency = env.true_adjacency.copy()
        varies = clamps = 0
        while not result.done:
            actions = {a: policies[a](env, result) for a in env.topology.agents}
            result = env.step(actions)
            for node, mode in env.last_chosen.values():
                if node == -1:
                    continue
                if mode == CLAMP:
                    clamps += 1
                else:
                    varies += 1
        scored = evaluate_episode(env)
        iv = any(iv_available(adjacency, env.topology.private[a],
                              env.windows[a].shared)
                 for a in env.topology.agents)
        rows.append({"episode": episode, "success": bool(scored["success"]),
                     "iv_structure": bool(iv), "varies": varies, "clamps": clamps})
    return rows


def build_env(action_modes) -> TwoAgentEnv:
    topology = two_agent(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    return TwoAgentEnv(MAConfig(
        topology=topology, n_obs=1000, n_int=100, budget=10, disclose_regime=True,
        turn_order=ROUND_ROBIN, action_modes=action_modes, prior_p=0.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--out", default="results/iv_decomposition.json")
    args = ap.parse_args(argv)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = sorted(int(pathlib.Path(f).stem.rsplit("_s", 1)[1])
                       for f in glob.glob("results/ma_fixed/tb_both_s*.pt"))

    per_seed = []
    for seed in seeds:
        both_ckpt = pathlib.Path("results/ma_fixed/tb_both_s%d.pt" % seed)
        clamp_ckpt = pathlib.Path("results/ma_fixed/tb_clamp_s%d.pt" % seed)
        if not (both_ckpt.exists() and clamp_ckpt.exists()):
            print("seed %d: missing checkpoint, skipped" % seed)
            continue
        both = replay(both_ckpt, build_env(MODES), args.episodes, seed)
        clamp = replay(clamp_ckpt, build_env((CLAMP,)), args.episodes, seed)
        # Episodes are seeded identically for both policies, so index i is the same graph.
        for b, c in zip(both, clamp):
            assert b["episode"] == c["episode"]
        per_seed.append({"seed": seed, "both": both, "clamp": clamp})
        n_iv = sum(r["iv_structure"] for r in both)
        print("seed %2d done  IV-structure episodes %3d/%3d  both %.3f  clamp %.3f" % (
            seed, n_iv, len(both),
            np.mean([r["success"] for r in both]),
            np.mean([r["success"] for r in clamp])), flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(per_seed, indent=1))
    print("\nwrote %s" % out)
    return per_seed


if __name__ == "__main__":
    main()
