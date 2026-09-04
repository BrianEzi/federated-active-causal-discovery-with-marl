"""Does the sampled-evidence engine's ANSWER quality degrade as n_int grows?

WHY THIS EXISTS. results/nint_curve/ (4 Sep) shows structural error rising with sample
volume above n_int~100-1000 in every arm including random -- the opposite of the
consistency Brian expected, where n_int -> infinity should converge the sampled test to
the oracle. If the engine's per-query answers get WORSE with n, the U belongs to the
evidence engine, not the policies. This measures that directly, with no learned policy in
the loop: greedy agents play seeded episodes, and at episode end each backend's stored
evidence state (`FactoredBackend._detected`) is scored against that window's own truth
(`self.truth`, set at reset in every regime).

Two failure channels, tallied separately because they prune the truth differently:
  false_detect   detected ancestry that is not true (test miscalibration; prunes truth
                 directly -- the "expensive direction" the alpha=0.001 comment guards)
  powered_miss   true ancestry, pair declared POWERED, not detected. consistent_with_
                 evidence() then prunes any candidate claiming it -- the truth included.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                      # noqa: E402

from cb.versionspace import reveal                                      # noqa: E402
from ma.baselines import make_baselines                                 # noqa: E402
from scripts.rescore_from_config import env_from_config                 # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="run JSON supplying the cell config")
    ap.add_argument("--n_ints", default="200,10000")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disclose_regime", action="store_true",
                    help="the one-bit foreign-intervention disclosure of MAConfig: rows a "
                         "partner privately intervened in are masked out of the contrasts. "
                         "The hypothesis this flag tests: the growing false-detect rate is "
                         "the documented price of privacy (cb/citest.py, 'bug 5 wearing a "
                         "mask'), not test miscalibration.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    config = json.loads(pathlib.Path(args.result).read_text())["config"]
    payload = {}
    for n in (int(v) for v in args.n_ints.split(",")):
        env = env_from_config(dict(config, vs_evidence="sampled", n_int=n,
                                   disclose_regime=args.disclose_regime), seed=args.seed)
        builders = {a: make_baselines(env, a, args.seed) for a in env.topology.agents}
        policies = {a: builders[a]["greedy_uncertainty"] for a in env.topology.agents}
        tally = dict(true_detect=0, false_detect=0, powered_miss=0,
                     silent_miss=0, queries=0,
                     false_detect_fullgraph_true=0, false_detect_spurious=0)
        for episode in range(args.episodes):
            result = env.reset(seed=args.seed * 100_000 + episode)
            while not result.done:
                result = env.step({a: policies[a](env, result)
                                   for a in env.topology.agents})
            for a in env.topology.agents:
                be = env.windows[a].belief
                detected = getattr(be, "_detected", None)
                if not detected:
                    continue
                # FULL-GRAPH ancestry, for the split that decides what a "false" detect
                # is. The window truth is a PROJECTION: x can be a genuine full-graph
                # ancestor of y through hidden out-of-window nodes with no directed path in
                # the window MAG. A detection there is a real interventional effect that the
                # projected oracle never asserts -- a representation mismatch, not a
                # statistical false positive. The two need different fixes, so they are
                # tallied apart.
                adj = env.params.adjacency
                import numpy as _np
                reach = _np.linalg.matrix_power(
                    (adj + _np.eye(adj.shape[0])).astype(bool).astype(int),
                    adj.shape[0] - 1) > 0
                win = env.windows[a]
                for x, (ancestry, powered) in detected.items():
                    true_anc = reveal(be.truth, be.k, x)
                    gx = win.nodes[x]
                    others = [y for y in range(be.k) if y != x]
                    for (det, pw, tr), y in zip(zip(ancestry, powered, true_anc), others):
                        gy = win.nodes[y]
                        tally["queries"] += 1
                        if det and tr:
                            tally["true_detect"] += 1
                        elif det and not tr:
                            if reach[gx, gy]:
                                tally["false_detect_fullgraph_true"] += 1
                            else:
                                tally["false_detect_spurious"] += 1
                            tally["false_detect"] += 1
                        elif tr and pw:
                            tally["powered_miss"] += 1
                        elif tr:
                            tally["silent_miss"] += 1
        q = max(tally["queries"], 1)
        payload[n] = {**tally,
                      "false_detect_rate": tally["false_detect"] / q,
                      "powered_miss_rate": tally["powered_miss"] / q}
        print(f"n_int={n:6d}  queries {tally['queries']:6d}  "
              f"false_detect {tally['false_detect']:5d} ({tally['false_detect']/q:.4f}) "
              f"[fullgraph-true {tally['false_detect_fullgraph_true']}, "
              f"spurious {tally['false_detect_spurious']}]  "
              f"powered_miss {tally['powered_miss']:5d}  "
              f"true_detect {tally['true_detect']:5d}  silent_miss {tally['silent_miss']:5d}",
              flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config_from": args.result, "seed": args.seed,
                               "episodes": args.episodes, "per_n": payload}, indent=1))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
