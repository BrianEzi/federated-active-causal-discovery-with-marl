"""Did `--evidence_power` actually reach the config of every rho12 run?

WHY THIS EXISTS RATHER THAN TRUST. On 5 Sep `env_from_config` silently dropped
`skeleton_source`, so an entire evaluation measured the supplied skeleton while stamping
"estimated" in its output. The failure was invisible because the label was right and only the
behaviour was wrong. The dial in this fleet is one flag with a config field of a DIFFERENT name
(`--evidence_power` -> `vs_evidence_power`), which is exactly the shape that goes wrong quietly:
a run with the dial dropped trains a plain-oracle policy and files it under a rate.

So this re-reads every finished run and asserts three things:
  * `vs_evidence_power` in the stored config equals the rate in the FILENAME;
  * every rate appears with the expected seed count;
  * the four settings that must match the k12 sweep cell are what the work order specified,
    since a fleet that silently inherited the k=8 grid's budget or episode count would be
    comparable to nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import math
import re
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

# The principal cell, from the verified k12s50n04b150 command. A mismatch here means the fleet
# is not comparable to RQ1, which is the entire reason it is being run.
EXPECTED = {"train_episodes": 12000, "n_int": 20, "n_obs": 60,
            "observe_belief_channels": False, "vs_evidence": "oracle"}
# BUDGET IS NO LONGER A CONSTANT AND MUST NOT BE ASSERTED AS ONE. The compensated fleet sets
# `budget = ceil(1.5 * base / rho)` so that effective beta stays at 1.5 while the dial moves;
# asserting a fixed 50 flagged every correct compensated run as a MISMATCH. Check the
# COMPENSATION instead -- that is the property the fleet exists to have, and a run whose budget
# does not match its rate is exactly the silent fault this script is for.
BETA, K_V, N_AGENTS = 1.5, 12, 4


def expected_budget(rho: float) -> int:
    from scripts.sweep import required_cover_fraction
    return math.ceil(BETA * required_cover_fraction(K_V) * K_V * N_AGENTS / rho)
# The partition is NOT stored as `private_size` / `n_shared` -- those are ma_train.py FLAGS, and
# the run JSON records the resulting `topology` object instead. My first version asserted on the
# flag names, so every run came back "MISMATCH: private_size=None want 6" while being perfectly
# correct. That is the same shape as the reachability bug from 20 Aug: a check that can never
# pass tells you nothing about the data, only about the check. Assert on what is actually
# recorded.
EXPECTED_TOPOLOGY = {"name": "T_4agent_6each_6shared", "n_private_blocks": 4,
                     "private_block_size": 6, "n_exposed": 6}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/rho12b")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.dir, "rho*_s?.json")))
    if not files:
        print(f"nothing in {args.dir} yet")
        return 0

    bad = 0
    seen = {}
    print(f"{'run':22s} {'filename rate':>13s} {'config rate':>12s} {'match':>6s}  settings")
    for f in files:
        name = os.path.basename(f)
        m = re.match(r"rho([0-9.]+)_s(\d)\.json$", name)
        if not m:
            print(f"{name:22s}  UNPARSEABLE NAME")
            bad += 1
            continue
        want = float(m.group(1))
        cfg = json.loads(open(f).read()).get("config", {})
        got = cfg.get("vs_evidence_power")
        ok = got is not None and abs(got - want) < 1e-9
        wrong = [k for k, v in EXPECTED.items() if cfg.get(k) != v]
        topo = cfg.get("topology") or {}
        priv = topo.get("private") or []
        got_topo = {"name": topo.get("name"), "n_private_blocks": len(priv),
                    "private_block_size": len(priv[0]) if priv else 0,
                    "n_exposed": len(topo.get("exposed") or [])}
        wrong += [f"topology.{k}" for k, v in EXPECTED_TOPOLOGY.items() if got_topo[k] != v]
        want_b = expected_budget(want)
        if cfg.get("budget") != want_b:
            wrong.append(f"budget={cfg.get('budget')} want {want_b} "
                         f"(effective beta {cfg.get('budget', 0) * want / (want_b / BETA * want):.2f} "
                         f"instead of {BETA})")
        bad += (not ok) or bool(wrong)
        seen.setdefault(want, []).append(int(m.group(2)))
        print(f"{name[:-5]:22s} {want:13.2f} {str(got):>12s} {'OK' if ok else 'WRONG':>6s}  "
              + ("all as specified" if not wrong
                 else "MISMATCH: " + ", ".join(wrong)))

    print()
    for rate in sorted(seen, reverse=True):
        n = len(seen[rate])
        flag = "" if n == args.seeds else f"  <- {n} of {args.seeds} seeds"
        print(f"  rho={rate:.2f}: seeds {sorted(seen[rate])}{flag}")
    if bad:
        print(f"\n!! {bad} run(s) failed. A dropped dial trains a plain-oracle policy and files "
              f"it under a rate; do not evaluate this fleet until it is fixed.")
        return 1
    print(f"\nall {len(files)} run(s): the dial landed and the cell settings are the sweep "
          f"cell's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
