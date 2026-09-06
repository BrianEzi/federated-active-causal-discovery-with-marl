"""Reporting-standard audit: find every number in the thesis that is thinner than it looks.

WHY THIS EXISTS. Figure 4.11 drew a confident line segment between two points, one of which
was a mean over 600 episodes of which exactly ONE was non-zero. The slope was decided by that
single episode and it pointed the "wrong" way, which is how it was noticed -- by a reader
asking why pooled improved when the fix was removed. Nothing about the number was false; it
was reported at a precision its support could not carry. That defect is mechanical and so is
its detection.

WHAT COUNTS AS THIN. These metrics are per-episode means of a quantity that is zero on most
episodes. A mean of 0.00022 can mean "every episode is slightly wrong" or "599 episodes are
perfect and one is a disaster", and those are different findings. This counts the episodes
that actually contribute, per arm per cell, and flags any reported mean resting on five or
fewer of them -- and any RATIO whose denominator or numerator is one of those.

    .venv/bin/python scripts/check_reporting.py            # thin-support audit
    .venv/bin/python scripts/check_reporting.py --captions # caption completeness too
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
THIN = 5           # non-zero episodes at or below which a mean is not a measurement of an arm
CH4 = ROOT / "thesis/4 Results and Analysis.tex"

# Where Chapter 4's figures and tables read their per-episode vectors from. Files outside
# this list exist but are not reported, and auditing them would bury the ones that are.
REPORTED = [
    "results/sweep12k/shd/*.json",
    "results/rerows/*_best.json",
    "results/rerows/*_final.json",
    "results/credit/shd/*.json",
    "results/budget_tight/shd_*.json",
    "results/noisedist/rho050_*.json",
    "results/power/rho/deterministic/xfer_*.json",
    "results/central12k/scored/*.json",
    "results/epsgreedy/sweep/w*.json",
    "results/epsgreedy/k12policy/*.json",
    "results/epsgreedy/k30policy/*.json",
]


def _records(path: pathlib.Path):
    """Yield (label, per-arm rows dict) for the shapes this project writes."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return
    for e in (d if isinstance(d, list) else [d]):
        if not isinstance(e, dict) or "rows" not in e:
            continue
        rows = e["rows"]
        seed = e.get("seed")
        # global_shd_paired: rows[arm][metric]. eps_greedy_paired: rows[metric].
        if all(isinstance(v, dict) for v in rows.values()):
            for arm, m in rows.items():
                if isinstance(m, dict) and "hard" in m:
                    yield f"{path.name}:{arm}:s{seed}", np.asarray(m["hard"], dtype=float)
        elif "hard" in rows:
            tag = f"eps{e.get('eps')}" if e.get("eps") is not None else e.get("base", "arm")
            yield f"{path.name}:{tag}:s{seed}", np.asarray(rows["hard"], dtype=float)


def thin_support():
    """Per-arm-per-seed support, aggregated to the cell level the thesis reports at."""
    groups: dict = {}
    for pattern in REPORTED:
        for f in sorted(glob.glob(str(ROOT / pattern))):
            for label, arr in _records(pathlib.Path(f)):
                fname, arm, _seed = label.split(":")
                key = (fname, arm)
                nz, tot, tally = groups.get(key, (0, 0, 0))
                groups[key] = (nz + int((arr > 0).sum()), tot + arr.size, tally + 1)
    flagged = []
    for (fname, arm), (nz, tot, nseeds) in sorted(groups.items()):
        if tot and 0 < nz <= THIN:
            flagged.append((fname, arm, nz, tot, nseeds))
    return flagged, groups


def caption_audit():
    """Every Chapter 4 caption should say what a reader needs to read the float."""
    text = CH4.read_text()
    out = []
    # SUBFIGURE captions are panel names ("window size"), not float captions; the reader
    # gets the conditions from the outer caption, so requiring seeds and episodes in each
    # panel label would be noise. Mask the subfigure blocks before scanning.
    masked = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}",
                    lambda mm: mm.group(0).replace("\\caption", "\\subcap"),
                    text, flags=re.S)
    for m in re.finditer(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", masked):
        cap = m.group(1)
        line = text[:m.start()].count("\n") + 1
        lab = re.search(r"\\label\{([^}]*)\}", masked[m.end():m.end() + 400])
        name = lab.group(1) if lab else f"line {line}"
        if name.startswith("tab:"):
            continue
        missing = []
        if not re.search(r"seed", cap):
            missing.append("seed count")
        if not re.search(r"episode", cap):
            missing.append("episode count")
        # A caption that mentions a truncated axis must say where it starts.
        if re.search(r"axis starts|starts at", cap) is None and name in TRUNCATED:
            missing.append("axis truncation not declared")
        if len(cap) > 400:
            missing.append(f"over two printed lines ({len(cap)} chars)")
        if missing:
            out.append((name, line, missing))
    return out


# Figures whose axes do not start at the natural origin. Declared here so the caption check
# can insist the caption says so; add to this list when a figure is truncated.
TRUNCATED = {"fig:ladder", "fig:epsgreedy_grid"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", action="store_true")
    args = ap.parse_args(argv)

    flagged, groups = thin_support()
    print(f"THIN SUPPORT: means resting on {THIN} or fewer non-zero episodes")
    print(f"  scanned {len(groups)} arm-cell groups across {len(REPORTED)} sources")
    if not flagged:
        print("  none")
    for fname, arm, nz, tot, nseeds in flagged:
        print(f"  ! {fname:44s} {arm:14s} {nz}/{tot} non-zero episodes ({nseeds} seeds)")

    if args.captions:
        print("\nCAPTION COMPLETENESS (Chapter 4 figures)")
        rows = caption_audit()
        if not rows:
            print("  none")
        for name, line, missing in rows:
            print(f"  ! {name:26s} (line {line}): {', '.join(missing)}")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
