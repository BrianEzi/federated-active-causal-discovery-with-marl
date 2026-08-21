"""Re-run GATE 3 alone, under the CORRECTED success criterion.

GATE 3 asks whether coordination is necessary AND available: on CONFOUNDED episodes, a pair
that cannot clamp must do worse than a pair that can. The gap is the headroom a learned
policy competes for.

WHY THIS RE-RUN EXISTS. The recorded GATE 3 result (never-clamp 0.000 vs mixed-clamp 0.184,
`results/ma2/gates_withbit_v7.json`, 2026-08-20 01:12) was measured BEFORE the identification
criterion was corrected at 12:11/12:35. That gate scores confounded episodes only -- exactly
the regime the old criterion could not credit -- so its headroom cannot be taken at face
value, in either direction:

  * if the old 0.000 was partly a measurement artefact, the corrected never-clamp rate rises
    and the headroom SHRINKS;
  * the old criterion also demanded the exact true DAG with the exact confounding set, which
    is strictly harder than [U14], so both arms may rise together.

Either way the number has to be re-earned rather than inherited. Skipped episodes cost only
a reset, so a large `--episodes` buys confounded sample size cheaply.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from ma.baselines import RandomAgent
from ma.env import AGENTS, MAConfig, TwoAgentEnv
from ma.topology import Topology
from scripts.ma_gates2 import play


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--n_obs", type=int, default=1000)
    ap.add_argument("--n_int", type=int, default=100)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disclose_regime", action="store_true")
    ap.add_argument("--out", default="results/ma_fixed/gate3_recheck.json")
    args = ap.parse_args(argv)

    topology = Topology(name="T1_1_1_3", a_private=(0,), b_private=(1,), exposed=(2, 3, 4))
    env = TwoAgentEnv(MAConfig(
        topology=topology, n_obs=args.n_obs, n_int=args.n_int, budget=args.budget,
        disclose_regime=args.disclose_regime))
    assert env.config.reward_criterion == "u14", env.config.reward_criterion

    started = time.time()
    never = {n: RandomAgent(n, seed=args.seed + 2, allow_clamp=False) for n in AGENTS}
    mixed = {n: RandomAgent(n, seed=args.seed + 3, allow_clamp=True) for n in AGENTS}
    g_never = play(env, never, args.episodes, args.seed, only=True)
    print(f"never-clamp  n={g_never['n']}  rate {g_never['rate']:.3f}", flush=True)
    g_mixed = play(env, mixed, args.episodes, args.seed, only=True)
    print(f"mixed-clamp  n={g_mixed['n']}  rate {g_mixed['rate']:.3f}", flush=True)

    headroom = g_mixed["rate"] - g_never["rate"]
    report = {
        "criterion": "u14 (corrected)", "budget": args.budget,
        "episodes_attempted": args.episodes, "disclose_regime": args.disclose_regime,
        "never_clamp": g_never, "mixed_clamp": g_mixed, "headroom": headroom,
        # Non-overlapping intervals, not just a positive difference.
        "passed": bool(headroom > 0 and g_mixed["ci"][0] > g_never["ci"][1]),
        "prior_run": {"source": "results/ma2/gates_withbit_v7.json",
                      "criterion": "pre-correction", "never_clamp": 0.0,
                      "mixed_clamp": 0.18421052631578946, "n": 38},
        "seconds": time.time() - started,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"GATE 3 (corrected): headroom {headroom:+.3f} -> "
          f"{'PASS' if report['passed'] else 'FAIL'}  [{report['seconds']:.0f}s] -> {out}")
    return report


if __name__ == "__main__":
    main()
