"""The federation ladder as an EQUIVALENCE BOUND rather than an absence of evidence.

WHY. Section 4.3.1 reported "no seed separates beyond two standard errors", which is absence
of evidence and the easiest thing in the chapter for a reviewer to dismiss. The fix is a
two-one-sided-tests (TOST) bound: declare a margin that would matter, then show the paired
difference sits significantly INSIDE it. That converts "we could not find a cost" into "any
cost is smaller than X, at this confidence".

THE MARGIN IS NOT ARBITRARY and must not be chosen after seeing the answer. The yardstick used
here is the myopic arm's own error on the same episodes: the quantity the whole chapter treats
as the reference a learned policy is measured against. A federation cost smaller than the gap
to the myopic baseline is a cost that could not change any conclusion the chapter draws.

BOTH CHECKPOINT CONVENTIONS ARE REPORTED, and this is not optional. Measured 6 Sep over twelve
seeds, the selected checkpoint favours the federated arm and the final update favours the
pooled one -- they disagree in SIGN. Quoting one alone would be selecting the direction.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pairs(convention: str):
    """Per-seed (federated, pooled, myopic) means for one checkpoint convention."""
    out = {}
    for arm, key in (("A", "fed"), ("E", "pool")):
        q = ROOT / f"results/rerows/ladder12k_{arm}_{convention}.json"
        if q.exists():
            for e in json.loads(q.read_text()):
                out.setdefault(e["seed"], {})[key] = e["means"]["learned"]["hard"]
                out[e["seed"]]["myopic"] = e["means"]["greedy"]["hard"]
    for f in sorted(glob.glob(str(ROOT / f"results/central12k/scored/v2_k12_?_s*_{convention}.json"))):
        stem = pathlib.Path(f).stem
        arm = stem.split("_")[2]
        seed = int(stem.split("_s")[1].split("_")[0])
        e = json.loads(pathlib.Path(f).read_text())
        e = e[0] if isinstance(e, list) else e
        key = "fed" if arm == "A" else "pool"
        out.setdefault(seed, {})[key] = e["means"]["learned"]["hard"]
        out[seed]["myopic"] = e["means"]["greedy"]["hard"]
    return {s: v for s, v in sorted(out.items()) if "fed" in v and "pool" in v}


def tost(d: np.ndarray, margin: float, alpha: float = 0.05):
    """Two one-sided tests. Equivalence is declared when the (1-2a) CI lies inside +/-margin."""
    from scipy import stats
    n = len(d)
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    t_lo = (mean + margin) / se
    t_hi = (margin - mean) / se
    p = max(stats.t.sf(t_lo, n - 1), stats.t.sf(t_hi, n - 1))
    crit = stats.t.ppf(1 - alpha, n - 1)
    return dict(n=n, mean=mean, se=se, p=float(p), equivalent=bool(p < alpha),
                ci_lo=mean - crit * se, ci_hi=mean + crit * se)


def main() -> int:
    report = {}
    for conv in ("best", "final"):
        rows = _pairs(conv)
        if len(rows) < 6:
            print(f"!! {conv}: only {len(rows)} seeds, skipping")
            continue
        d = np.array([rows[s]["fed"] - rows[s]["pool"] for s in rows])
        myopic = float(np.mean([rows[s]["myopic"] for s in rows]))
        res = tost(d, margin=myopic)
        report[conv] = {**res, "margin": myopic, "seeds": sorted(rows)}
        print(f"\n=== {conv} checkpoint, {res['n']} seeds ===")
        print(f"  paired federated - pooled : {res['mean']:+.6f} +/- {res['se']:.6f}")
        print(f"  90% CI                    : [{res['ci_lo']:+.6f}, {res['ci_hi']:+.6f}]")
        print(f"  margin (myopic on the same episodes): {myopic:.6f}")
        print(f"  TOST p = {res['p']:.2e}  ->  "
              f"{'EQUIVALENT within the margin' if res['equivalent'] else 'NOT equivalent'}")
        print(f"  bound: any cost is at most {max(abs(res['ci_lo']), abs(res['ci_hi']))/myopic:.0%}"
              f" of the gap to the myopic rule")
    if len(report) == 2:
        a, b = report["best"]["mean"], report["final"]["mean"]
        print(f"\nCONVENTIONS DISAGREE IN SIGN: best {a:+.6f}, final {b:+.6f}. "
              "Both must be reported; quoting one selects the direction.")
    out = ROOT / "results/ladder_equivalence.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
