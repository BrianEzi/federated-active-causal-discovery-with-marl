"""I(S;A)/H(A) per agent -- did this policy condition on its observation, or not?

WHY IT EXISTS AGAIN. `mi_check2.py` produced the numbers quoted throughout
`docs/PLAN_2026_08_28.md` and `docs/FINDINGS_2026_08_27.md` and is not in the repository --
another scratchpad tool that did not survive, exactly as `scripts/attr_score.py` was lost and
had to be rebuilt. Every "did this rung train" question needs it, including for the seeds
being added now.

WHAT IT MEASURES, and why this estimator and not a plug-in one.

    I(S;A)/H(A) = [ H(A) - E_s H(pi(.|s)) ] / H(A)

H(A) is the entropy of the MARGINAL action distribution over states visited; E_s H(pi(.|s))
is the mean entropy of the policy's own conditional distribution, read off the logits. Both
terms are exact given the policy -- no discretisation of the observation, which is continuous
and never repeats, so a plug-in estimator over observed (s, a) pairs would see one sample per
state and report zero conditional entropy for ANY policy.

The scale is the one the existing numbers are on:
  * a DETERMINISTIC policy has E_s H = 0 and scores 1.000. That is why greedy is "1.000 by
    construction".
  * a policy that ignores its observation has pi(.|s) constant, so H(A) = E_s H, and scores
    0.000 -- the near-fixed-mixture signature.

READ IT AS A FLOOR, NOT A QUALITY MEASURE. Near zero voids a number outright; above the
floor it measures commitment to the TRAINING objective, so it must never be compared across
arms trained on different objectives. That mistake was made on 2026-08-27 and withdrawn.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np
import torch

from ma.policy import IndependentPPO
from scripts.rescore_from_config import env_from_config


def _entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    return float(-(p * np.log(p)).sum())


def measure(result_path: pathlib.Path, episodes: int, seed: int = None) -> dict:
    report = json.loads(result_path.read_text())
    use_seed = seed if seed is not None else report.get("seed", 0)
    env = env_from_config(report["config"], seed=use_seed)
    checkpoint = result_path.with_suffix(".pt")
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint beside {result_path.name}")
    ppo = IndependentPPO.load(str(checkpoint), env)

    # Per agent: the conditional distributions at every visited state. Collected by PLAYING
    # the policy, so the states are the ones it actually reaches -- the marginal over a
    # uniform state sample would answer a question no training run asks.
    conditionals: Dict[int, List[np.ndarray]] = {a: [] for a in env.topology.agents}
    policies = ppo.policies(deterministic=False)
    for episode in range(episodes):
        result = env.reset(seed=use_seed * 100_000 + episode)
        while not result.done:
            for agent in env.topology.agents:
                # Exactly the tensor `IndependentPPO.policy` builds, so the distribution
                # measured here is the one the policy actually acts on.
                with torch.no_grad():
                    logits, _ = ppo.nets[agent](
                        torch.as_tensor(env.observation(agent), dtype=torch.float32))
                conditionals[agent].append(torch.softmax(logits, dim=-1).numpy())
            result = env.step({a: policies[a](env, result)
                               for a in env.topology.agents})

    out = {"source": str(result_path), "episodes": episodes, "seed": use_seed,
           "per_agent": {}, "final_entropy": report.get("final_entropy")}
    ratios = []
    for agent, rows in conditionals.items():
        stacked = np.asarray(rows)
        marginal = stacked.mean(axis=0)
        h_marginal = _entropy(marginal)
        h_conditional = float(np.mean([_entropy(row) for row in stacked]))
        # Clamped: MI is non-negative by definition, and a near-uniform policy makes
        # h_conditional and h_marginal agree to within float error, which can produce a
        # spurious -2e-08. Mirrors `ma/checkpoints.py::mi_ratio` -- the two estimators must
        # not disagree, since one ranks checkpoints and the other certifies them.
        ratio = (0.0 if h_marginal <= 0
                 else max(0.0, (h_marginal - h_conditional) / h_marginal))
        out["per_agent"][str(agent)] = {"h_marginal": h_marginal,
                                        "h_conditional": h_conditional,
                                        "mi_ratio": ratio,
                                        "states": len(rows)}
        ratios.append(ratio)
    out["mi_ratio_mean"] = float(np.mean(ratios))
    out["mi_ratio_min"] = float(np.min(ratios))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--floor", type=float, default=0.15,
                    help="below this the run is treated as not having trained")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    payload = []
    print(f"{'run':22s} {'mean':>7s} {'min':>7s} {'entropy':>8s}  per agent")
    for path in args.results:
        row = measure(pathlib.Path(path), args.episodes, args.seed)
        payload.append(row)
        per = " ".join(f"{v['mi_ratio']:.3f}" for v in row["per_agent"].values())
        flag = "" if row["mi_ratio_mean"] >= args.floor else "   <-- BELOW FLOOR"
        entropy = row["final_entropy"]
        print(f"{pathlib.Path(path).stem:22s} {row['mi_ratio_mean']:7.3f} "
              f"{row['mi_ratio_min']:7.3f} "
              f"{entropy if entropy is not None else float('nan'):8.3f}  {per}{flag}")
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
