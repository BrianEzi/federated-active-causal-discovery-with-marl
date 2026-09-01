"""A closed-form predictor of federated attribution performance.

THE MODEL. Measured over partner counts and shared fractions, attribution decomposes into two
independent factors:

    attribution  ~=  P(resolve | single-pair group)  x  share of groups that are single-pair

The FIRST is set by COVERAGE, and only weakly by the partner count -- a correction made
1 Sep after the matched-budget control. With one partner ownership is forced (the hypothesis
space is a singleton) and everything settled resolves. From two partners onward the rate is
NEARLY CONSTANT at ~0.76 provided the budget covers the window: 0.798, 0.765, 0.718 at 2, 3
and 7 partners with rounds-per-agent held at 15.

An earlier version of this docstring claimed the rate collapsed to 5% at seven partners
through exponential hypothesis-space growth. That 5% was budget starvation -- the sweep held
the budget fixed at 60 rounds however many agents shared it. At budget 120 the same cell
recovers to 72%.

The SECOND is pure graph combinatorics -- how many latent groups explain exactly one pair --
and is computable from the topology with NO SIMULATION AT ALL.

Groups explaining two or more pairs contribute nothing beyond one partner, because separating
a clique from several smaller latents needs a PARTIAL response, which needs the owner to probe
its private variables one at a time, which no policy here does.

WHY THIS IS WORTH HAVING. It turns "attribution gets hard with more sites" into a number you
can compute before running anything: given a site count and a contended fraction, this says
what share of the latent structure is recoverable. It is also falsifiable per cell, which is
the point -- the residual column is the test.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/attr_ceiling.json")
    args = ap.parse_args()

    rows = json.loads(pathlib.Path(args.results).read_text())
    header = (f"{'cell':>14s} {'partners':>8s} {'P(res|1pair)':>12s} {'share 1pair':>11s} "
              f"{'predicted':>9s} {'measured':>8s} {'residual':>8s}")
    print(header); print("-" * len(header))
    worst = 0.0
    for r in rows:
        by = r["by_size"]
        two = by.get("2", {})
        n_two = sum(two.values())
        p_resolve = two.get("right", 0) / n_two if n_two else float("nan")
        sizes = r["true_group_sizes"]
        total = sum(sizes.values())
        share = sizes.get("2", 0) / total if total else float("nan")
        predicted = p_resolve * share
        residual = r["measured"] - predicted
        worst = max(worst, abs(residual))
        print(f"{r['cell']:>14s} {r['n_agents'] - 1:8d} {p_resolve:12.3f} {share:11.3f} "
              f"{predicted:9.3f} {r['measured']:8.3f} {residual:+8.3f}")
    print(f"\nlargest absolute residual: {worst:.3f}")
    print("Groups explaining two or more pairs are assumed to contribute ZERO, which holds at")
    print("every partner count above one. At ONE partner they do contribute and the model")
    print("under-predicts by design -- that cell is the exception that identifies the cause.")


if __name__ == "__main__":
    main()
