"""The answer-rate dose-response curve: transfer performance against the partial-oracle rate.

RQ2's third part. A policy trained under a partial oracle -- oracle ancestry answers withheld
with probability 1 - rho -- is evaluated under genuine sampled evidence it never saw. This
reports transfer hard SHD against rho, with the greedy baseline as the reference line.

THE FALSIFICATION, fixed before any of these numbers existed (docs/AGENT_B_INBOX.md, 2 Sep
03:00). The curve is a finding only if transfer quality varies SYSTEMATICALLY with rho and the
rho=1.00 (plain oracle) end is worst. A flat curve, or a lone spike at the calibrated optimum
with noise elsewhere, is a NULL: it would mean the transfer win came from the observation
features rather than the answer-rate dial, which is the attribution gap section 5 of
FINDINGS_TRANSFER_2026_09_02.md leaves open. This script prints the verdict either way and
does not editorialise.

Two summaries, because they answer different questions:

  * per-seed paired deltas -- learned minus greedy on identical episodes, with the paired SE,
    which is the quantity a claim about one cell rests on;
  * the across-seed mean per rate with its own SE over seeds, which is what the curve plots.
    Seed-to-seed spread is the larger term at these sample sizes, so a rate is only really
    "better" if it separates on the SECOND of these, not just the first.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np


def load(directory: str):
    cells = {}
    for path in sorted(glob.glob(os.path.join(directory, "xfer_rho*.json"))):
        m = re.search(r"xfer_rho([0-9.]+)_s(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        rho, seed = float(m.group(1)), int(m.group(2))
        payload = json.loads(open(path).read())
        entry = payload[0] if isinstance(payload, list) else payload
        cells[(rho, seed)] = entry
    return cells


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/power/rho")
    ap.add_argument("--out", default="results/power/rho/CURVE.json")
    args = ap.parse_args(argv)

    cells = load(args.dir)
    if not cells:
        print(f"no xfer_rho*.json in {args.dir} yet")
        return

    rates = sorted({r for r, _ in cells}, reverse=True)
    print(f"{'rho':>6s} {'n':>2s} {'learned':>9s} {'greedy':>9s} "
          f"{'delta':>10s} {'+/-seedSE':>10s} {'verdict':>12s}")
    curve = []
    for rho in rates:
        entries = [(s, e) for (r, s), e in cells.items() if r == rho]
        deltas, learned, greedy = [], [], []
        for _seed, e in entries:
            d = e["paired"].get("learned-greedy")
            if d is None:
                continue
            deltas.append(d["delta"])
            learned.append(e["means"]["learned"]["hard"])
            greedy.append(e["means"]["greedy"]["hard"])
        if not deltas:
            continue
        n = len(deltas)
        mean_delta = float(np.mean(deltas))
        # SE OVER SEEDS, not the within-cell paired SE. With three seeds this is a crude
        # estimate, and it is reported rather than hidden because it is the term that decides
        # whether two rates are actually different.
        seed_se = float(np.std(deltas, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        verdict = "n/a"
        if n > 1:
            verdict = ("beats greedy" if mean_delta + 2 * seed_se < 0 else
                       "loses" if mean_delta - 2 * seed_se > 0 else "tied")
        print(f"{rho:6.2f} {n:2d} {np.mean(learned):9.5f} {np.mean(greedy):9.5f} "
              f"{mean_delta:+10.5f} {seed_se:10.5f} {verdict:>12s}")
        curve.append({"rho": rho, "n_seeds": n, "mean_learned_hard": float(np.mean(learned)),
                      "mean_greedy_hard": float(np.mean(greedy)),
                      "mean_delta": mean_delta, "seed_se": seed_se,
                      "per_seed_delta": deltas, "verdict": verdict})

    print("\nper-seed paired deltas (learned - greedy, hard SHD):")
    for rho in rates:
        row = " ".join(f"s{s}:{cells[(rho, s)]['paired']['learned-greedy']['delta']:+.5f}"
                       for (r, s) in sorted(cells) if r == rho
                       and cells[(r, s)]["paired"].get("learned-greedy"))
        print(f"  rho={rho:.2f}  {row}")

    # -- the falsification, applied ----------------------------------------------------------
    # SEED GUARD, added 2 Sep after this fired DOSE-RESPONSE SUPPORTED on two single-seed
    # rates. With one seed a rate has no `seed_se` at all (nan), so `nanmean` silently took
    # the SE of the ONLY multi-seed rate and compared a spread built from single points
    # against it. That is a positive verdict manufactured from missing data, and it is
    # exactly what the pre-registered falsification exists to stop. A rate now has to carry
    # at least MIN_SEEDS seeds to enter the verdict at all, and the verdict is withheld
    # entirely until enough rates qualify.
    MIN_SEEDS = 2
    ready = [c for c in curve if c["n_seeds"] >= MIN_SEEDS]
    pending = [c for c in curve if c["n_seeds"] < MIN_SEEDS]
    if pending:
        print(f"\n{len(pending)} rate(s) below {MIN_SEEDS} seeds and excluded from the "
              f"verdict: " + ", ".join(f"rho={c['rho']:.2f} (n={c['n_seeds']})"
                                       for c in pending))
    if len(ready) < 3:
        print(f"VERDICT WITHHELD -- only {len(ready)} rate(s) have >= {MIN_SEEDS} seeds; "
              f"need 3 to judge a curve. Nothing here is a finding yet.")
    else:
        curve_v = ready
        plain = next((c for c in curve_v if c["rho"] == 1.0), None)
        best = min(curve_v, key=lambda c: c["mean_delta"])
        spread = max(c["mean_delta"] for c in curve_v) - min(c["mean_delta"] for c in curve_v)
        typical_se = float(np.nanmean([c["seed_se"] for c in curve_v]))
        print(f"\nspread across rates {spread:.5f}, typical seed SE {typical_se:.5f}")
        if plain is None:
            print("VERDICT: cannot apply -- no rho=1.00 control in the directory")
        elif spread < 2 * typical_se:
            print("VERDICT: NULL -- the curve is flat within seed noise. The transfer win is "
                  "NOT attributable to the answer-rate dial; the observation features are the "
                  "remaining candidate. Report as a null (see module docstring).")
        elif plain["mean_delta"] == max(c["mean_delta"] for c in curve_v):
            print(f"VERDICT: DOSE-RESPONSE SUPPORTED -- rho=1.00 (plain oracle) is the worst "
                  f"at {plain['mean_delta']:+.5f}, best is rho={best['rho']:.2f} at "
                  f"{best['mean_delta']:+.5f}, spread {spread:.5f} exceeds 2x seed SE.")
        else:
            print(f"VERDICT: MIXED -- spread {spread:.5f} exceeds noise, but rho=1.00 is not "
                  f"the worst rate, which the dose-response story requires. Best is "
                  f"rho={best['rho']:.2f}. Report the shape honestly rather than as support.")

    with open(args.out, "w") as f:
        json.dump({"curve": curve}, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
