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
import re

# The principal cell, from the verified k12s50n04b150 command. A mismatch here means the fleet
# is not comparable to RQ1, which is the entire reason it is being run.
EXPECTED = {"budget": 50, "train_episodes": 12000, "n_int": 20, "n_obs": 60,
            "private_size": 6, "n_shared": 6, "n_agents": 4,
            "observe_belief_channels": False, "vs_evidence": "oracle"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/rho12")
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
        bad += (not ok) or bool(wrong)
        seen.setdefault(want, []).append(int(m.group(2)))
        print(f"{name[:-5]:22s} {want:13.2f} {str(got):>12s} {'OK' if ok else 'WRONG':>6s}  "
              + ("all as specified" if not wrong
                 else "MISMATCH: " + ", ".join(f"{k}={cfg.get(k)!r} want {EXPECTED[k]!r}"
                                               for k in wrong)))

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
