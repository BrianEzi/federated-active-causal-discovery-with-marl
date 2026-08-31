"""The results chapter, generated from whatever the sweep has produced so far.

DEGRADES GRACEFULLY BY DESIGN. It reads what is on disk and reports it; a cell that has not
finished is absent rather than guessed at, and a partial sweep produces a partial table. So
it can be run repeatedly while runs land, and the same command produces the final version.

EVERY ROW CARRIES THE THREE FIELDS. Every wrong claim on this project has come from one of
the MI gate, the evidence mode or the evaluation policy being left implicit, so no number is
printed without them. Cells below the MI floor are flagged rather than quietly averaged in:
an untrained policy's score says nothing about the task, and averaging it with trained ones
is how an artefact becomes a trend.

    .venv/bin/python scripts/sweep_report.py --dir results/sweep/oracle
    .venv/bin/python scripts/sweep_report.py --dir results/sweep/oracle --figures
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.sweep import build_cells                                # noqa: E402

MI_FLOOR = 0.15


def load(directory: pathlib.Path):
    """One row per (cell, seed), with everything a claim needs attached to it."""
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name in ("calibration.json", "throughput.json"):
            continue
        try:
            d = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if "arms" not in d or "learned" not in d.get("arms", {}):
            continue
        arms, cfg = d["arms"], d.get("config", {})
        checkpoints = d.get("checkpoints") or {}
        name = path.stem.rsplit("_s", 1)[0]
        rows.append({
            "cell": name, "seed": d.get("seed"),
            "k": cfg.get("k"), "sigma": cfg.get("sigma_contended"),
            "n": cfg.get("n_agents"), "budget": cfg.get("budget"),
            "n_int": cfg.get("n_int"), "evidence": cfg.get("vs_evidence"),
            "credit": cfg.get("ppo_turn_aware_credit", cfg.get("turn_aware_credit")),
            "local_epochs": cfg.get("ppo_local_epochs", cfg.get("local_epochs")),
            "learned": arms["learned"]["success"],
            "greedy": arms.get("greedy_uncertainty", {}).get("success"),
            "partitioned": arms.get("greedy_partitioned", {}).get("success"),
            "ceiling": arms.get("oracle_cover", {}).get("success"),
            "steps": arms["learned"]["mean_steps"],
            "greedy_steps": arms.get("greedy_uncertainty", {}).get("mean_steps"),
            "ceiling_steps": arms.get("oracle_cover", {}).get("mean_steps"),
            "shd": arms["learned"].get("global_soft_shd"),
            "mi": checkpoints.get("best_mi_ratio"),
            "entropy": (d.get("history") or [{}])[-1].get("entropy"),
        })
    return rows


def _agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def table(rows, axis_cells, title, key):
    """One axis, one table. Cells below the MI floor are marked, not dropped."""
    by_cell = collections.defaultdict(list)
    for r in rows:
        if r["cell"] in axis_cells:
            by_cell[r["cell"]].append(r)
    if not by_cell:
        return
    print(f"\n### {title}")
    print(f"{'cell':22s} {key:>6s} {'seeds':>5s} {'learned':>15s} {'greedy':>7s} "
          f"{'gap':>15s} {'ceiling':>8s} {'MI':>6s} {'steps L/G/C':>17s}  gate")
    for cell in sorted(by_cell, key=lambda c: axis_cells[c]):
        got = sorted(by_cell[cell], key=lambda r: r["seed"] or 0)
        lm, ls = _agg([r["learned"] for r in got])
        gm, _ = _agg([r["greedy"] for r in got])
        cm, _ = _agg([r["ceiling"] for r in got])
        mim, _ = _agg([r["mi"] for r in got])
        gaps = [r["learned"] - r["greedy"] for r in got if r["greedy"] is not None]
        gm2, gs2 = _agg(gaps)
        sl, _ = _agg([r["steps"] for r in got])
        sg, _ = _agg([r["greedy_steps"] for r in got])
        sc, _ = _agg([r["ceiling_steps"] for r in got])
        below = [r["seed"] for r in got if (r["mi"] or 0) < MI_FLOOR]
        flag = f"MI<{MI_FLOOR} seeds {below}" if below else "ok"
        print(f"{cell:22s} {axis_cells[cell]:6} {len(got):5d} "
              f"{lm:7.3f}+-{ls:5.3f} {gm:7.3f} {gm2:+7.3f}+-{gs2:5.3f} {cm:8.3f} "
              f"{mim:6.3f} {sl:5.2f}/{sg:5.2f}/{sc:5.2f}  {flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/sweep/oracle")
    ap.add_argument("--evidence", default="oracle")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--fig_dir", default="results/figures")
    args = ap.parse_args(argv)

    directory = pathlib.Path(args.dir)
    rows = load(directory)
    if not rows:
        print(f"no finished runs in {directory}")
        return 1

    cells = build_cells(evidence=args.evidence)
    by_name = {c.name: c for c in cells}
    done = {r["cell"] for r in rows}
    print(f"# Sweep results — {args.dir}")
    print(f"\n{len(done)} of {len(cells)} cells have at least one seed; "
          f"{len(rows)} runs total.")
    missing = sorted(set(by_name) - done)
    if missing:
        print(f"still missing ({len(missing)}): {', '.join(missing)}")

    sample = rows[0]
    print(f"\nCONFIG (constant across cells): evidence={sample['evidence']}, "
          f"turn_aware_credit={sample['credit']}, local_epochs={sample['local_epochs']}, "
          f"n_int={sample['n_int']}, evaluation=sampled policy over 200 episodes.")
    print(f"MI floor {MI_FLOOR}; a cell below it is an untrained policy and its score says "
          f"nothing about the task.")

    for axis, key in (("k", "k"), ("sigma", "sigma"), ("n", "n"), ("beta", "beta"),
                      ("n_int", "n_int"), ("sigma_x_n", "sigma")):
        members = {c.name: getattr(c, key if key != "sigma" else "sigma")
                   for c in cells if c.axis in (axis, "baseline")}
        if axis == "sigma_x_n":
            members = {c.name: c.sigma for c in cells if c.axis == "sigma_x_n"}
        if members:
            table(rows, members, f"{axis} axis", key)

    trained = [r for r in rows if (r["mi"] or 0) >= MI_FLOOR and r["greedy"] is not None]
    if trained:
        gaps = [r["learned"] - r["greedy"] for r in trained]
        m, sd = _agg(gaps)
        beats = sum(1 for g in gaps if g > 0)
        print(f"\n### Headline, over the {len(trained)} runs that clear the MI gate")
        print(f"  learned - greedy   {m:+.3f} +- {sd:.3f}   (beats greedy in "
              f"{beats}/{len(trained)} runs)")
        ceil = [r["ceiling"] - r["learned"] for r in trained if r["ceiling"] is not None]
        if ceil:
            cm, csd = _agg(ceil)
            print(f"  ceiling - learned  {cm:+.3f} +- {csd:.3f}   "
                  f"(headroom left to the optimal arm)")
        excluded = len(rows) - len(trained)
        if excluded:
            print(f"  EXCLUDED: {excluded} run(s) below the MI floor, listed per-axis above.")

    if args.figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = pathlib.Path(args.fig_dir); out.mkdir(parents=True, exist_ok=True)
        for axis, key in (("k", "k"), ("n", "n"), ("beta", "beta"), ("sigma", "sigma")):
            members = {c.name: getattr(c, key) for c in cells
                       if c.axis in (axis, "baseline")}
            pts = collections.defaultdict(list)
            for r in rows:
                if r["cell"] in members:
                    pts[members[r["cell"]]].append(r)
            if len(pts) < 2:
                continue
            xs = sorted(pts)
            fig, ax = plt.subplots(figsize=(6, 3.8))
            for label, field, colour in (("optimal (A5)", "ceiling", "#4d8b5f"),
                                         ("learned", "learned", "#2f6f9f"),
                                         ("greedy", "greedy", "#b5651d")):
                ys = [_agg([r[field] for r in pts[x]])[0] for x in xs]
                es = [_agg([r[field] for r in pts[x]])[1] for x in xs]
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label, color=colour)
            ax.set_xlabel(key); ax.set_ylabel("joint identification rate")
            ax.set_ylim(-0.03, 1.05); ax.legend(frameon=False, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{key} axis — {sample['evidence']} evidence", fontsize=11)
            fig.tight_layout(); fig.savefig(out / f"axis_{key}.png", dpi=160)
            plt.close(fig)
            print(f"  wrote {out / f'axis_{key}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
