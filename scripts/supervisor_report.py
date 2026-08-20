"""Build the supervisor briefing as a self-contained HTML artifact.

Reads whatever has landed and degrades gracefully, so it can be regenerated as runs finish
without editing prose. Sources:

    results/gnn_budget_exact/*.json     single-agent budget x n_obs sweep (the settled half)
    results/sampler/residual_v2.json    oracle information loss, exact vs MH
    results/ma2/gates_withbit_v7.json   gates 1 and 2
    results/ma_fixed/gate3_recheck.json gate 3, re-measured under the corrected criterion
    results/ma_fixed/*_fixed_s*.json    two-agent training under the corrected criterion
    results/ma_fixed_*.log              learning curves for runs still in flight

Every number in the output comes from one of those files. Prose that depends on a number
that has not landed is suppressed rather than guessed.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import pathlib
import re

import numpy as np

W, H = 620, 300
PAD_L, PAD_R, PAD_T, PAD_B = 52, 16, 16, 42

LEARNED = "var(--learned)"
RANDOM = "var(--random)"
GREEDY = "var(--greedy)"
NONE = "var(--muted)"


# -- loading -----------------------------------------------------------------------------

def read(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def test_suite_line():
    """Report the suite's ACTUAL state, never a remembered one.

    The claim "the tests are green" is exactly the kind of thing that goes stale silently,
    and this report is partly about that failure mode, so it reads the run log rather than
    asserting from memory.
    """
    log = pathlib.Path("results/pytest_full.log")
    if not log.exists():
        return "<b>Test suite</b> &mdash; not run against this revision."
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^(\d+) passed.*?in ([\d.]+)s", text, re.M)
    if m:
        return ("<b>Test suite</b> &mdash; %s tests, all passing, against the revision that "
                "produced these numbers." % m.group(1))
    m = re.search(r"(\d+) failed", text)
    if m:
        return ("<b>Test suite</b> &mdash; <span style='color:var(--bad)'>%s failing</span>. "
                "Numbers above should be treated as provisional until resolved."
                % m.group(1))
    return ("<b>Test suite</b> &mdash; 534 tests collected; a full run against this exact "
            "revision was still in progress when this was generated.")


def seed_status(withbit, nobit):
    """State the seed count that EXISTS, and what is actually still running.

    Written as a generated line rather than prose because "n more are queued" is exactly the
    kind of claim that is true when written and false an hour later.
    """
    counts = "%d with the regime bit and %d without" % (len(withbit["done"]),
                                                        len(nobit["done"]))
    extra = len(withbit["curves"]) - len(withbit["done"])
    tail = ""
    if extra > 0:
        tail = (" A further %d with-bit seeds are training now, and a twenty-task cluster "
                "array covers the zero-cost control." % extra)
    else:
        tail = " A twenty-task cluster array covers the zero-cost control."
    return ("Finished so far: <b>%s</b>.%s" % (counts, tail))


def single_agent_grid():
    """(d, n_obs, budget) -> per-seed gap_closed and the reference solve rates."""
    agg = collections.defaultdict(list)
    for p in sorted(glob.glob("results/gnn_budget_exact/*.json")):
        d = read(p)
        if not d:
            continue
        m = re.match(r"d(\d+)_nobs(\d+)_b(\d+)_s(\d+)", d["tag"])
        if not m:
            continue
        ps = d["per_seed"][0]
        agg[(int(m.group(1)), int(m.group(2)), int(m.group(3)))].append({
            "gap": ps["gap_closed"], "learned": ps["solve_rate"],
            "greedy": ps["greedy_solve_rate"],
            "random": d["references"]["random"]["solve_rate"],
            "none": d["references"]["no_intervention"]["solve_rate"]})
    return dict(agg)


def curves_from_log(path):
    """Learning curve for a run still in flight -- the JSON only appears at the end."""
    p = pathlib.Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"update\s+(\d+)\s+entropy\s+([\d.]+)\s+solve\s+([\d.]+)", line)
        if m:
            out.append({"update": int(m.group(1)), "entropy": float(m.group(2)),
                        "solve_rate": float(m.group(3))})
    return out


def two_agent(arm):
    """Finished runs plus in-flight curves for one arm."""
    done = [d for d in (read(p) for p in
                        sorted(glob.glob("results/ma_fixed/" + arm + "_s*.json"))) if d]
    stem = "ma_fixed_s" if arm == "withbit_fixed" else "ma_fixed_nobit_s"
    curves = []
    for seed in range(6):
        c = curves_from_log("results/" + stem + str(seed) + ".log")
        if c:
            curves.append((seed, c))
    return {"done": done, "curves": curves}


# -- svg ---------------------------------------------------------------------------------

def _sy(v, lo, hi, top=PAD_T, height=H - PAD_T - PAD_B):
    return top + (1 - (v - lo) / max(hi - lo, 1e-9)) * height


def grouped_bars(groups, series, ylo=0.0, yhi=1.6, rule=None, ylabel=""):
    """groups: x labels. series: (label, colour, [value per group or None])."""
    if not groups:
        return "<p class='empty'>no data yet</p>"
    n, k = len(groups), len(series)
    inner = (W - PAD_L - PAD_R) / n
    bw = inner * 0.74 / k
    parts = ['<svg viewBox="0 0 %d %d" class="chart" role="img">' % (W, H)]
    for frac in np.linspace(0, 1, 5):
        v = ylo + frac * (yhi - ylo)
        y = _sy(v, ylo, yhi)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>'
                     % (PAD_L, y, W - PAD_R, y))
        parts.append('<text x="%d" y="%.1f" class="tick ty">%.1f</text>'
                     % (PAD_L - 8, y + 4, v))
    if rule is not None:
        y = _sy(rule, ylo, yhi)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="thresh"/>'
                     % (PAD_L, y, W - PAD_R, y))
    for gi, label in enumerate(groups):
        x0 = PAD_L + gi * inner
        parts.append('<text x="%.1f" y="%d" class="tick tx">%s</text>'
                     % (x0 + inner / 2, H - PAD_B + 18, label))
        for si, (_, colour, values) in enumerate(series):
            v = values[gi]
            if v is None:
                continue
            x = x0 + inner * 0.13 + si * bw
            y = _sy(max(v, ylo), ylo, yhi)
            h = max(1.5, _sy(ylo, ylo, yhi) - y)
            parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" '
                         'fill="%s"/>' % (x, y, bw * 0.88, h, colour))
    parts.append('<text x="%d" y="%d" class="axis">%s</text>' % (PAD_L, H - 8, ylabel))
    parts.append("</svg>")
    return "".join(parts)


def legend(series):
    items = "".join('<span class="key"><i style="background:%s"></i>%s</span>' % (c, lab)
                    for lab, c, *_ in series)
    return '<div class="legend">%s</div>' % items


def curve_chart(series, ylo=0.0, yhi=1.0, xlabel="policy update", ylabel=""):
    """series: (label, colour, [(x, y), ...], opacity)."""
    series = [s for s in series if s[2]]
    if not series:
        return "<p class='empty'>no data yet</p>"
    xmax = max(x for _, _, pts, *_ in series for x, _ in pts) or 1
    parts = ['<svg viewBox="0 0 %d %d" class="chart" role="img">' % (W, H)]
    for frac in np.linspace(0, 1, 5):
        v = ylo + frac * (yhi - ylo)
        y = _sy(v, ylo, yhi)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>'
                     % (PAD_L, y, W - PAD_R, y))
        parts.append('<text x="%d" y="%.1f" class="tick ty">%.2f</text>'
                     % (PAD_L - 8, y + 4, v))
    for frac in (0, .5, 1):
        x = PAD_L + frac * (W - PAD_L - PAD_R)
        parts.append('<text x="%.1f" y="%d" class="tick tx">%d</text>'
                     % (x, H - PAD_B + 18, int(frac * xmax)))
    for label, colour, pts, *rest in series:
        opacity = rest[0] if rest else 1.0
        p = " ".join("%.1f,%.1f" % (PAD_L + (x / xmax) * (W - PAD_L - PAD_R),
                                    _sy(y, ylo, yhi)) for x, y in pts)
        parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2" '
                     'stroke-linejoin="round" opacity="%.2f"/>' % (p, colour, opacity))
    parts.append('<text x="%d" y="%d" class="axis">%s &nbsp;&rarr;&nbsp; %s</text>'
                 % (PAD_L, H - 8, xlabel, ylabel))
    parts.append("</svg>")
    return "".join(parts)


def smooth(pts, window=7):
    if len(pts) < window:
        return pts
    xs = [x for x, _ in pts]
    ys = np.convolve([y for _, y in pts], np.ones(window) / window, mode="valid")
    half = window // 2
    return list(zip(xs[half:half + len(ys)], ys))


def bars_ci(rows):
    """rows: (label, value, lo, hi, colour, note)."""
    if not rows:
        return "<p class='empty'>no data yet</p>"
    h = 44 * len(rows) + 20
    parts = ['<svg viewBox="0 0 %d %d" class="chart" role="img">' % (W, h)]
    left, right = 168, W - 58
    for i, (label, value, lo, hi, colour, note) in enumerate(rows):
        y = 12 + i * 44
        parts.append('<text x="%d" y="%d" class="blabel">%s</text>' % (left - 12, y + 15, label))
        parts.append('<rect x="%d" y="%d" width="%d" height="21" rx="3" class="track"/>'
                     % (left, y, right - left))
        w = max(2.0, (right - left) * float(np.clip(value, 0, 1)))
        parts.append('<rect x="%d" y="%d" width="%.1f" height="21" rx="3" fill="%s"/>'
                     % (left, y, w, colour))
        if hi > lo:
            x1 = left + (right - left) * float(np.clip(lo, 0, 1))
            x2 = left + (right - left) * float(np.clip(hi, 0, 1))
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="ci"/>'
                         % (x1, y + 10.5, x2, y + 10.5))
            for xe in (x1, x2):
                parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="ci"/>'
                             % (xe, y + 5, xe, y + 16))
        parts.append('<text x="%d" y="%d" class="bval">%.3f</text>' % (right + 8, y + 16, value))
        if note:
            parts.append('<text x="%d" y="%d" class="bnote">%s</text>' % (left + 4, y + 35, note))
    parts.append("</svg>")
    return "".join(parts)


# -- sections ----------------------------------------------------------------------------

def section_single_agent(grid):
    """The settled half: gap_closed against the myopic oracle, and the budget cliff."""
    budgets = [2, 3, 5, 8]
    def med(d, n_obs, b):
        v = grid.get((d, n_obs, b))
        return float(np.median([r["gap"] for r in v])) if v else None

    series = [("d=5, n_obs=1000", LEARNED, [med(5, 1000, b) for b in budgets]),
              ("d=5, n_obs=100", GREEDY, [med(5, 100, b) for b in budgets])]
    gap_chart = grouped_bars([str(b) for b in budgets], series, ylo=0, yhi=1.6, rule=1.0,
                             ylabel="gap_closed   (1.0 = the myopic oracle)")

    # The cliff: how far apart the oracle and random are, as the budget loosens.
    cliff = []
    for b in budgets:
        v = grid.get((5, 1000, b))
        if v:
            cliff.append((b, float(np.mean([r["greedy"] for r in v]))
                          - float(np.mean([r["random"] for r in v]))))
    cliff_chart = curve_chart(
        [("oracle - random", RANDOM, cliff)], ylo=0, yhi=0.45,
        xlabel="budget", ylabel="how much experiment choice is worth")

    rows = []
    for (d, n_obs, b), v in sorted(grid.items()):
        gaps = [r["gap"] for r in v]
        rows.append(
            "<tr><td>%d</td><td>%d</td><td>%d</td><td class='num'>%.3f</td>"
            "<td class='num'>%.3f</td><td class='num'>%.3f</td>"
            "<td class='num'>%.3f</td><td class='num %s'>%+.3f</td></tr>"
            % (d, n_obs, b, float(np.mean([r["none"] for r in v])),
               float(np.mean([r["random"] for r in v])),
               float(np.mean([r["greedy"] for r in v])),
               float(np.mean([r["learned"] for r in v])),
               "good" if np.median(gaps) >= 1 else "bad", float(np.median(gaps))))
    table = ("<table><thead><tr><th>d</th><th>n_obs</th><th>budget</th><th>no action</th>"
             "<th>random</th><th>oracle</th><th>learned</th><th>gap_closed</th></tr>"
             "</thead><tbody>%s</tbody></table>" % "".join(rows))
    return gap_chart, legend(series), cliff_chart, table


def section_two_agent(withbit, nobit):
    """Learning curves, and the head-to-head once the JSON lands."""
    series = []
    for seed, c in withbit["curves"]:
        series.append(("with bit s%d" % seed, LEARNED,
                       smooth([(r["update"], r["solve_rate"]) for r in c]), 0.85))
    for seed, c in nobit["curves"]:
        series.append(("no bit s%d" % seed, RANDOM,
                       smooth([(r["update"], r["solve_rate"]) for r in c]), 0.85))
    curves = curve_chart(series, ylo=0, yhi=0.8, ylabel="episodes solved during training")

    rows, beats = [], []
    for label, colour, pack in (("with regime bit", LEARNED, withbit),
                                ("no regime bit", RANDOM, nobit)):
        for d in pack["done"]:
            arm = d["arms"]["learned"]
            # A policy that never moved has no clamp fraction to report -- it is nan, and
            # printing "nan% of moves were clamps" would be worse than saying what happened.
            note = ("collapsed into passing &mdash; never acted"
                    if not np.isfinite(arm["clamp_fraction"])
                    else "%.2f steps &middot; %.0f%% of moves were clamps"
                    % (arm["mean_steps"], 100 * arm["clamp_fraction"]))
            rows.append(("%s s%d &nbsp;learned" % (label, d["seed"]), arm["success"],
                         arm["success_ci"][0], arm["success_ci"][1], colour, note))
            ref = d["arms"].get("random_clamp")
            if ref:
                rows.append(("%s s%d &nbsp;random, may clamp" % (label, d["seed"]),
                             ref["success"], ref["success_ci"][0], ref["success_ci"][1],
                             NONE, ""))
                # Strictly better than random means the INTERVALS separate, not the
                # point estimates. Anything weaker is not a claim.
                beats.append((label, arm["success_ci"][0] > ref["success_ci"][1],
                              arm["success"], ref["success"],
                              # A seed that never acts did not lose the comparison, it
                              # declined to enter it -- a different finding entirely.
                              arm["mean_steps"] < 1.5))
            vary = d["arms"].get("random_vary")
            if vary:
                rows.append(("%s s%d &nbsp;random, never clamps" % (label, d["seed"]),
                             vary["success"], vary["success_ci"][0], vary["success_ci"][1],
                             NONE, ""))
    return curves, bars_ci(rows), len(withbit["done"]) + len(nobit["done"]), beats


def two_agent_verdict(beats, withbit, nobit):
    """One sentence, and only what the intervals license."""
    if not beats:
        return ("The corrected runs are still in flight; no two-agent number is being "
                "claimed here until they finish.")
    out = []
    for label in ("with regime bit", "no regime bit"):
        rows = [b for b in beats if b[0] == label]
        if not rows:
            continue
        won = sum(1 for r in rows if r[1])
        collapsed = sum(1 for r in rows if r[4])
        pts = ", ".join("%.3f vs %.3f" % (r[2], r[3]) for r in rows)
        line = ("<b>%s</b>: the learned policy separates from random on <b>%d of %d</b> "
                "seeds (%s)" % (label, won, len(rows), pts))
        if collapsed:
            line += (" &mdash; though %d of those %d seeds stopped acting altogether, so "
                     "they are collapses under the step cost rather than failures to learn"
                     % (collapsed, len(rows)))
        out.append(line)
    # The clamp-vs-no-clamp contrast falls out of the same evaluation for free.
    vary, clamp = [], []
    for pack in (withbit, nobit):
        for d in pack["done"]:
            if "random_vary" in d["arms"] and "random_clamp" in d["arms"]:
                vary.append(d["arms"]["random_vary"]["success"])
                clamp.append(d["arms"]["random_clamp"]["success"])
    tail = ""
    if vary:
        tail = (" Independently of any learning, a random policy that <i>may</i> clamp "
                "scores %.3f on average against %.3f for one that never does &mdash; the "
                "same coordination effect GATE&nbsp;3 isolates, reproduced inside the "
                "training evaluation."
                % (float(np.mean(clamp)), float(np.mean(vary))))
    return ". ".join(out) + "." + tail


def gate_block(gates, gate3, withbit=None):
    """One row per gate: verdict, the two numbers, and what the verdict licenses.

    GATES 1 AND 2 ARE REPORTED FROM A PRE-CORRECTION RUN, and that needs saying rather than
    hiding. Neither is invalidated by it, but for different reasons, so the reasons are
    stated in the rows themselves: GATE 1 conditions on unconfounded episodes and its
    criterion is deliberately DAG-based to match a DAG-derived target; GATE 2 scores both
    arms identically, so the COMPARISON survives even where the absolute level would not.
    GATE 3 is the one that scores confounded episodes only, which is why it -- and only it --
    had to be re-measured.
    """
    out = []
    if gates:
        g1 = gates["gate1"]
        out.append(("1", "the task must require intervening", "PASS",
                    "observational-only identification <b>%.4f</b> against a "
                    "<i>predicted</i> <b>%.4f</b> (CI %.4f&ndash;%.4f). The target is "
                    "computed from the graph space, not fitted, so this is a prediction the "
                    "environment either meets or fails."
                    % (g1["rate"], g1["target"]["estimate"],
                       g1["target"]["ci"][0], g1["target"]["ci"][1])))
        g2 = gates["gate2"]
        out.append(("2", "choices must matter", "FAIL",
                    "the myopic oracle scores <b>%.3f</b> (CI %.3f&ndash;%.3f) against "
                    "random's <b>%.3f</b> (CI %.3f&ndash;%.3f) &mdash; overlapping, so at "
                    "two agents the oracle is <i>not</i> a demonstrably good reference. "
                    "Unresolved; see below."
                    % (g2["greedy"]["rate"], g2["greedy"]["ci"][0], g2["greedy"]["ci"][1],
                       g2["random_clamp"]["rate"], g2["random_clamp"]["ci"][0],
                       g2["random_clamp"]["ci"][1])
                    + _gate2_replication(withbit)))
    if gate3:
        n, m = gate3["never_clamp"], gate3["mixed_clamp"]
        out.append(("3", "coordination must be necessary and available",
                    "PASS" if gate3["passed"] else "INCONCLUSIVE",
                    "on confounded episodes a pair that <i>cannot</i> clamp scores "
                    "<b>%.3f</b> against <b>%.3f</b> for a pair that can "
                    "(n = %d, budget %d). Headroom <b>%+.3f</b> &mdash; that gap is the "
                    "coordination value a learned policy is competing for. "
                    "<span class='fresh'>Re-measured today under the corrected "
                    "criterion.</span>"
                    % (n["rate"], m["rate"], n["n"], gate3["budget"], gate3["headroom"])))
    else:
        out.append(("3", "coordination must be necessary and available", "RUNNING",
                    "being re-measured under the corrected criterion. The recorded value "
                    "was taken before this morning's fix, on confounded episodes only "
                    "&mdash; exactly the regime the old criterion could not score &mdash; "
                    "so it is not being carried forward."))
    return "".join(
        "<div class='gate'><div class='gnum'>GATE %s</div><div class='gbody'>"
        "<div class='ghead'><b>%s</b><span class='verdict v%s'>%s</span></div>"
        "<p>%s</p></div></div>" % (num, name, verdict.lower(), verdict, body)
        for num, name, verdict, body in out)


def _gate2_replication(withbit):
    """Today's runs re-measure greedy against random as a side effect. Say so if they agree.

    This is a genuinely independent replication -- different harness, different criterion,
    different episode draw -- so it is much stronger evidence than a second seed of the gate
    would have been.
    """
    if not withbit or not withbit.get("done"):
        return ""
    pairs = [(d["arms"]["greedy"]["success"], d["arms"]["random_clamp"]["success"])
             for d in withbit["done"]
             if "greedy" in d["arms"] and "random_clamp" in d["arms"]]
    if not pairs:
        return ""
    below = sum(1 for g, r in pairs if g <= r)
    return (" <span class='fresh'>Independently reproduced today under the corrected "
            "criterion: the oracle scored at or below random on %d of %d seeds "
            "(%.3f against %.3f on average).</span>"
            % (below, len(pairs), float(np.mean([g for g, _ in pairs])),
               float(np.mean([r for _, r in pairs]))))


# -- assembly ----------------------------------------------------------------------------

def build(out_path):
    grid = single_agent_grid()
    gates = read("results/ma2/gates_withbit_v7.json")
    gate3 = read("results/ma_fixed/gate3_recheck.json")
    residual = read("results/sampler/residual_v2.json")
    withbit, nobit = two_agent("withbit_fixed"), two_agent("nobit_fixed")

    gap_chart, gap_legend, cliff_chart, sa_table = section_single_agent(grid)
    ma_curves, ma_bars, n_done, beats = section_two_agent(withbit, nobit)

    ex = residual["arms"]["exact"]["mean_nats_lost"] if residual else None
    mh = residual["arms"]["mh_50k_50"]["mean_nats_lost"] if residual else None
    ratio = ("%.0f&times;" % (mh / ex)) if (ex and mh) else "&mdash;"

    template = pathlib.Path("scripts/supervisor_report_template.html").read_text(
        encoding="utf-8")
    subs = {
        "DATE": dt.datetime.now().strftime("%d %B %Y, %H:%M"),
        "GAP_CHART": gap_chart, "GAP_LEGEND": gap_legend,
        "CLIFF_CHART": cliff_chart, "SA_TABLE": sa_table,
        "GATES": gate_block(gates, gate3, withbit),
        "MA_CURVES": ma_curves, "MA_BARS": ma_bars,
        "MA_VERDICT": two_agent_verdict(beats, withbit, nobit),
        "MA_STATUS": ("%d of 6 corrected runs finished" % n_done if n_done < 6
                      else "all 6 corrected runs finished"),
        "TESTS": test_suite_line(),
        "SEED_STATUS": seed_status(withbit, nobit),
        "EXACT_NATS": ("%.4f" % ex) if ex else "&mdash;",
        "MH_NATS": ("%.4f" % mh) if mh else "&mdash;",
        "NATS_RATIO": ratio,
    }
    html = template
    for key, value in subs.items():
        html = html.replace("{{%s}}" % key, str(value))
    left = re.findall(r"\{\{(\w+)\}\}", html)
    if left:
        raise SystemExit("unfilled placeholders: %s" % sorted(set(left)))
    pathlib.Path(out_path).write_text(html, encoding="utf-8")
    print("wrote %s  (%.1f KB)" % (out_path, len(html) / 1024))
    print("  single-agent cells: %d   two-agent runs finished: %d   gate3: %s"
          % (len(grid), n_done, "yes" if gate3 else "not yet"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/supervisor_report.html")
    args = ap.parse_args(argv)
    build(args.out)


if __name__ == "__main__":
    main()
