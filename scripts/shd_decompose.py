"""What is the pooled hard SHD actually counting?

RAISED BY BRIAN, 6 Sep, after the budget axis showed SHD and joint recovery telling
different stories. Reading `ma/evaluate.py::pooled_global_belief`, a pooled pair scores
`hard = 1` in THREE distinct situations, and the chapter's phrase "SHD on committed marks"
only describes the first:

  WRONG        the intersection is a single mark and it is not the true one.
  AMBIGUOUS    the intersection still holds more than one mark -- `len(pooled) > 1` scores
               as a hard error even though nothing was committed to.
  DISAGREEMENT the sites hold different TRUE marks (a shared pair is a latent projection and
               different windows project out different private blocks), so there is no single
               answer; the fallback scores the per-site mean and thresholds it at 0.5.

If the second and third dominate, the metric is measuring residual UNCERTAINTY rather than
committed ERROR, and the name misdescribes it. This script decomposes the measured value.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from itertools import combinations

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ma.baselines import make_baselines                                  # noqa: E402
from ma.evaluate import CLAIM_BACKENDS, _true_index, _pair_masses        # noqa: E402
from ma.policy import IndependentPPO                                     # noqa: E402
from scripts.rescore_from_config import env_from_config                  # noqa: E402


def decompose(env):
    """One episode: hard-SHD contributions split by cause."""
    seen = {}
    for agent, window in env.windows.items():
        belief = getattr(window.belief, "last", None)
        if belief is None or not hasattr(belief, "adjacency"):
            continue
        mag = np.asarray(env._true_mag(agent))
        for u, v in combinations(range(window.k), 2):
            key = tuple(sorted((window.nodes[u], window.nodes[v])))
            masses = _pair_masses(belief, u, v)
            e = seen.setdefault(key, {"marks": [], "truth": [], "soft": []})
            e["marks"].append(frozenset(np.flatnonzero(masses > 1e-9).tolist()))
            e["truth"].append(_true_index(mag, u, v))
            e["soft"].append(1.0 - float(masses[_true_index(mag, u, v)]))
    tally = dict(pairs=0, wrong=0, ambiguous=0, disagreement=0, clean=0,
                 hard_from_wrong=0, hard_from_ambig=0, hard_from_disagree=0)
    for e in seen.values():
        tally["pairs"] += 1
        truths = set(e["truth"])
        pooled = frozenset.intersection(*e["marks"])
        if len(truths) > 1 or not pooled or e["truth"][0] not in pooled:
            tally["disagreement"] += 1
            tally["hard_from_disagree"] += int(float(np.mean(e["soft"])) > 0.5)
        elif len(pooled) > 1:
            tally["ambiguous"] += 1
            tally["hard_from_ambig"] += 1
        elif next(iter(pooled)) != e["truth"][0]:
            tally["wrong"] += 1
            tally["hard_from_wrong"] += 1
        else:
            tally["clean"] += 1
    return tally


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--arm", default="learned", choices=["learned", "greedy_uncertainty"])
    args = ap.parse_args(argv)

    report = json.loads(pathlib.Path(args.result).read_text())
    config, seed = report["config"], report.get("seed", 0)
    if config["belief_backend"] not in CLAIM_BACKENDS:
        raise SystemExit("pooling is defined only for the claim backends")
    env = env_from_config(config, seed=seed)
    if args.arm == "learned":
        ck = pathlib.Path(args.result).with_name(
            pathlib.Path(args.result).stem + "_best.pt")
        import torch
        torch.manual_seed(seed)
        pol = IndependentPPO.load(str(ck), env).policies(deterministic=False)
    else:
        b = {a: make_baselines(env, a, seed) for a in env.topology.agents}
        pol = {a: b[a]["greedy_uncertainty"] for a in env.topology.agents}

    total = dict(pairs=0, wrong=0, ambiguous=0, disagreement=0, clean=0,
                 hard_from_wrong=0, hard_from_ambig=0, hard_from_disagree=0)
    for ep in range(args.episodes):
        r = env.reset(seed=seed * 100_000 + ep)
        while not r.done:
            r = env.step({a: pol[a](env, r) for a in env.topology.agents})
        t = decompose(env)
        if t:
            for k in total:
                total[k] += t[k]
    n = max(total["pairs"], 1)
    print(f"{args.arm}, {args.episodes} episodes, {total['pairs']} pooled pair-instances")
    for k in ("clean", "wrong", "ambiguous", "disagreement"):
        print(f"  {k:14s} {total[k]:7d}   {total[k]/n:.5f} of pairs")
    hw, ha, hd = (total["hard_from_wrong"], total["hard_from_ambig"],
                  total["hard_from_disagree"])
    print(f"  ACTUAL hard=1 assignments: wrong {hw}, ambiguous {ha}, disagreement {hd}")
    print(f"  reported global_hard_shd  = {(hw+ha+hd)/n:.5f}")
    if hw + ha + hd:
        print(f"  share from genuine error  = {hw/(hw+ha+hd):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
