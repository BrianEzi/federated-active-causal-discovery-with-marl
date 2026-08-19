"""Build the two-agent results report as a self-contained HTML artifact.

Reads whatever exists and degrades gracefully -- a partial night still renders. Sources:
  results/ma_night/*.json      per-seed training + evaluation (local overnight run)
  results/ma_train/*.json      the same from the Myriad array, if it has landed
  results/ma2/gates_*.json     gate history, including the superseded runs
Learning curves come from each report's `history`, or are parsed from the run log when the
run predates history being saved.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List, Optional

import numpy as np

W, H = 560, 300
PAD_L, PAD_R, PAD_T, PAD_B = 48, 14, 16, 40

LEARNED = "#1F7A8C"
RANDOM = "#B3446C"
GREEDY = "#7A6FF0"
MUTED = "#8A9099"
GOOD = "#2E7D5B"
BAD = "#C0392B"


def load(pattern: str) -> List[dict]:
    out = []
    for path in sorted(pathlib.Path().glob(pattern)):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def parse_history_from_log(path: pathlib.Path) -> Dict[int, List[dict]]:
    """Recover learning curves from stdout for runs that predate `history` being saved."""
    if not path.exists():
        return {}
    seed, out = None, {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"with-bit seed (\d+)", line)
        if m:
            seed = int(m.group(1))
            out.setdefault(seed, [])
            continue
        m = re.search(r"update\s+(\d+)\s+entropy\s+([\d.]+)\s+solve\s+([\d.]+)", line)
        if m and seed is not None:
            out[seed].append({"update": int(m.group(1)), "entropy": float(m.group(2)),
                              "solve_rate": float(m.group(3))})
    return out


# -- svg ---------------------------------------------------------------------------------


def _sx(i: int, n: int) -> float:
    return PAD_L if n <= 1 else PAD_L + i * (W - PAD_L - PAD_R) / (n - 1)


def _sy(v: float, lo: float, hi: float) -> float:
    span = max(hi - lo, 1e-9)
    return PAD_T + (1 - (v - lo) / span) * (H - PAD_T - PAD_B)


def line_chart(series, xlabels, ylo=0.0, yhi=1.0, ylabel="") -> str:
    n = max((len(v) for _, _, v in series), default=0)
    if n == 0:
        return "<p class='empty'>no data yet</p>"
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for frac in (0, .25, .5, .75, 1):
        y = _sy(ylo + frac * (yhi - ylo), ylo, yhi)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{y+4:.1f}" class="tick ty">'
                     f'{ylo + frac*(yhi-ylo):.2f}</text>')
    step = max(1, len(xlabels) // 8)
    for i, lab in enumerate(xlabels):
        if i % step == 0:
            parts.append(f'<text x="{_sx(i, len(xlabels)):.1f}" y="{H-PAD_B+18}" '
                         f'class="tick tx">{lab}</text>')
    for label, colour, values in series:
        pts = " ".join(f"{_sx(i, n):.1f},{_sy(v, ylo, yhi):.1f}"
                       for i, v in enumerate(values))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                     f'stroke-width="2.2" stroke-linejoin="round"/>')
    parts.append(f'<text x="{PAD_L}" y="{H-8}" class="axis">{ylabel}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bars_with_ci(rows, title="") -> str:
    """rows: (label, value, lo, hi, colour)."""
    if not rows:
        return "<p class='empty'>no data yet</p>"
    h = 46 * len(rows) + 30
    parts = [f'<svg viewBox="0 0 {W} {h}" class="chart" role="img">']
    left, right = 132, W - 26
    for i, (label, value, lo, hi, colour) in enumerate(rows):
        y = 24 + i * 46
        parts.append(f'<text x="{left-10}" y="{y+15}" class="blabel">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{right-left}" height="22" '
                     f'rx="4" class="track"/>')
        w = max(2.0, (right - left) * float(np.clip(value, 0, 1)))
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="22" rx="4" '
                     f'fill="{colour}"/>')
        if hi > lo:
            x1 = left + (right-left) * float(np.clip(lo, 0, 1))
            x2 = left + (right-left) * float(np.clip(hi, 0, 1))
            parts.append(f'<line x1="{x1:.1f}" y1="{y+11}" x2="{x2:.1f}" y2="{y+11}" '
                         f'class="ci"/>')
        parts.append(f'<text x="{min(w+left+8, right-4):.1f}" y="{y+16}" '
                     f'class="bval">{value:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def dots(values, labels, threshold=None) -> str:
    """Per-seed strip plot -- shows spread, which a mean hides."""
    if not values:
        return "<p class='empty'>no data yet</p>"
    h = 150
    parts = [f'<svg viewBox="0 0 {W} {h}" class="chart" role="img">']
    for frac in (0, .5, 1):
        y = 24 + (1-frac) * 80
        parts.append(f'<line x1="{PAD_L}" y1="{y}" x2="{W-PAD_R}" y2="{y}" class="grid"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{y+4}" class="tick ty">{frac:.1f}</text>')
    if threshold is not None:
        y = 24 + (1-threshold) * 80
        parts.append(f'<line x1="{PAD_L}" y1="{y}" x2="{W-PAD_R}" y2="{y}" class="thresh"/>')
    for i, (v, lab) in enumerate(zip(values, labels)):
        x = _sx(i, max(len(values), 2))
        y = 24 + (1 - float(np.clip(v, 0, 1))) * 80
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{LEARNED}"/>')
        parts.append(f'<text x="{x:.1f}" y="{h-14}" class="tick tx">{lab}</text>')
    med = float(np.median(values))
    y = 24 + (1-med) * 80
    parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                 f'class="median"/>')
    parts.append(f'<text x="{W-PAD_R}" y="{y-6:.1f}" class="medlabel">median {med:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# -- assembly ----------------------------------------------------------------------------


def gate_rows(gates: dict) -> str:
    if not gates:
        return "<p class='empty'>no gate run loaded</p>"
    out = []
    g1 = gates.get("gate1", {})
    out.append(("GATE 1 — the task must require intervening",
                f"{g1.get('rate', float('nan')):.4f} against a predicted "
                f"{g1.get('target', {}).get('estimate', float('nan')):.4f}",
                bool(g1.get("passed"))))
    g2 = gates.get("gate2", {})
    out.append(("GATE 2 — choices must matter",
                f"greedy {g2.get('greedy', {}).get('rate', float('nan')):.3f} vs random "
                f"{g2.get('random_clamp', {}).get('rate', float('nan')):.3f}",
                bool(g2.get("passed"))))
    g3 = gates.get("gate3", {})
    out.append(("GATE 3 — coordination must be necessary and available",
                f"never-clamp {g3.get('never_clamp', {}).get('rate', float('nan')):.3f} vs "
                f"clamping {g3.get('forced_clamp', {}).get('rate', float('nan')):.3f}, "
                f"headroom {g3.get('headroom', float('nan')):+.3f}",
                bool(g3.get("passed"))))
    rows = []
    for name, detail, ok in out:
        cls = "pass" if ok else "fail"
        mark = "PASS" if ok else "FAIL"
        rows.append(f'<tr><td>{name}</td><td class="num">{detail}</td>'
                    f'<td class="{cls}">{mark}</td></tr>')
    return ("<table><thead><tr><th>Gate</th><th>Measured</th><th></th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def build(seeds: List[dict], gates: dict, log_history: dict) -> dict:
    """Everything the template needs, computed once."""
    seeds = [s for s in seeds if s.get("arms", {}).get("learned")]
    seeds.sort(key=lambda s: s.get("seed", 0))
    ctx: Dict[str, object] = {}

    learned = [s["arms"]["learned"]["success"] for s in seeds]
    labels = [f"s{s.get('seed')}" for s in seeds]
    ctx["n_seeds"] = len(seeds)
    ctx["seed_dots"] = dots(learned, labels, threshold=None)
    ctx["median"] = float(np.median(learned)) if learned else float("nan")
    ctx["spread"] = (float(np.std(learned)) if len(learned) > 1 else float("nan"))
    ctx["collapsed"] = sum(1 for s in seeds if s.get("collapsed"))

    arms = ("learned", "random_clamp", "random_vary", "greedy", "pass")
    colours = {"learned": LEARNED, "random_clamp": RANDOM, "random_vary": MUTED,
               "greedy": GREEDY, "pass": MUTED}
    rows = []
    for arm in arms:
        vals = [s["arms"][arm]["success"] for s in seeds if arm in s.get("arms", {})]
        if not vals:
            continue
        lo = float(np.percentile(vals, 25)) if len(vals) > 1 else vals[0]
        hi = float(np.percentile(vals, 75)) if len(vals) > 1 else vals[0]
        rows.append((arm.replace("_", " "), float(np.mean(vals)), lo, hi, colours[arm]))
    ctx["arm_bars"] = bars_with_ci(rows)
    ctx["arm_rows"] = rows

    clamp = [(f"s{s.get('seed')}", s["arms"]["learned"].get("clamp_fraction", float("nan")))
             for s in seeds]
    ctx["clamp_bars"] = bars_with_ci(
        [(lab, v if np.isfinite(v) else 0.0, 0, 0, LEARNED) for lab, v in clamp])

    histories = []
    for s in seeds:
        hist = s.get("history") or log_history.get(s.get("seed"), [])
        if hist:
            histories.append((s.get("seed"), hist))
    if histories:
        length = min(len(h) for _, h in histories)
        solve = np.mean([[p["solve_rate"] for p in h[:length]] for _, h in histories],
                        axis=0)
        entropy = np.mean([[p["entropy"] for p in h[:length]] for _, h in histories],
                          axis=0)
        xs = [str(h[i]["update"]) for i in range(length)]
        ctx["curve_solve"] = line_chart([("mean solve rate", LEARNED, list(solve))], xs,
                                        ylabel="PPO update")
        ctx["curve_entropy"] = line_chart([("policy entropy", GREEDY, list(entropy))], xs,
                                          ylo=0, yhi=float(max(entropy)) * 1.05,
                                          ylabel="PPO update")
        ctx["n_updates"] = length
    else:
        ctx["curve_solve"] = "<p class='empty'>no learning curves recorded</p>"
        ctx["curve_entropy"] = ""
        ctx["n_updates"] = 0

    ctx["gates"] = gate_rows(gates)
    first = [s.get("first_success_episode") for s in seeds
             if s.get("first_success_episode") is not None]
    ctx["first_success"] = (f"{int(np.median(first))}" if first else "n/a")
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/ma_night/report.html")
    ap.add_argument("--log", default=None, help="run log to recover curves from")
    args = ap.parse_args()

    seeds = load("results/ma_night/withbit_s*.json") + load("results/ma_train/withbit_s*.json")
    gates = {}
    for candidate in ("results/ma2/gates_withbit_v6.json",
                      "results/ma2/gates_withbit_v5.json"):
        path = pathlib.Path(candidate)
        if path.exists():
            gates = json.loads(path.read_text(encoding="utf-8"))
            break
    log_history = parse_history_from_log(pathlib.Path(args.log)) if args.log else {}

    ctx = build(seeds, gates, log_history)
    template = pathlib.Path("scripts/ma_report_template.html").read_text(encoding="utf-8")
    html = template
    for key, value in ctx.items():
        html = html.replace("{{" + key + "}}", str(value))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({ctx['n_seeds']} seeds)")


if __name__ == "__main__":
    main()
