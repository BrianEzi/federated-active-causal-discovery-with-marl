"""Render results/budget/budget_sweep.json as a self-contained HTML report.

Charts are emitted as inline SVG -- no libraries, no external requests, so the page works
as a published artifact under a strict CSP.
"""
from __future__ import annotations

import json
import pathlib
from typing import List, Sequence, Tuple

GREEDY = "#0E6E8C"
RANDOM = "#A3355E"
GAP = "#6D28D9"
GRID = "var(--rule)"

W, H = 560, 300
PAD_L, PAD_R, PAD_T, PAD_B = 46, 14, 16, 38


def _x(i: int, n: int) -> float:
    """Budgets are unevenly spaced (1..8, 10, 12, 16, 20), so plot by rank and label with
    the true value. Even spacing keeps the low-budget region -- where everything happens --
    from being squashed into the left margin."""
    if n == 1:
        return PAD_L
    return PAD_L + i * (W - PAD_L - PAD_R) / (n - 1)


def _y(v: float) -> float:
    return PAD_T + (1.0 - v) * (H - PAD_T - PAD_B)


def line_chart(budgets: Sequence[int], series: List[Tuple[str, str, Sequence[float]]],
               ymax: float = 1.0, ylabel: str = "solve rate") -> str:
    n = len(budgets)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart">']
    parts.append(f'<title>{ylabel} against intervention budget</title>')

    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = _y(frac * ymax)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" class="tick tick-y">'
                     f'{frac * ymax:.2f}</text>')

    for i, b in enumerate(budgets):
        if b in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20):
            parts.append(f'<text x="{_x(i, n):.1f}" y="{H - PAD_B + 18}" '
                         f'class="tick tick-x">{b}</text>')

    for label, color, values in series:
        pts = " ".join(f"{_x(i, n):.1f},{_y(min(v, ymax)):.1f}"
                       for i, v in enumerate(values))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="2.4" stroke-linejoin="round"/>')
        for i, v in enumerate(values):
            parts.append(f'<circle cx="{_x(i, n):.1f}" cy="{_y(min(v, ymax)):.1f}" '
                         f'r="2.6" fill="{color}"/>')

    parts.append(f'<text x="{PAD_L}" y="{H - 6}" class="axis-label">'
                 f'intervention budget</text>')
    parts.append("</svg>")
    return "".join(parts)


def legend(entries: Sequence[Tuple[str, str]]) -> str:
    items = "".join(
        f'<span class="key"><i style="background:{c}"></i>{l}</span>' for l, c in entries)
    return f'<div class="legend">{items}</div>'


def panel(title: str, subtitle: str, chart: str, keys) -> str:
    return (f'<figure class="panel"><figcaption><h4>{title}</h4>'
            f'<p>{subtitle}</p></figcaption>{legend(keys)}{chart}</figure>')


def rates(curve, key="solve_rate"):
    return [row[key] for row in curve]


def budgets_of(curve):
    return [row["budget"] for row in curve]


def setup_diagram() -> str:
    """The vertical partition, drawn once so the reader does not have to hold it in mind."""
    return '''<svg viewBox="0 0 560 210" role="img" class="chart diagram">
<title>Vertical partition with overlap: A's private set, the shared set, B's private set</title>
<rect x="14" y="26" width="196" height="150" rx="10" class="win win-a"/>
<rect x="196" y="26" width="350" height="150" rx="10" class="win win-b"/>
<text x="44" y="18" class="wlabel" fill="#0E6E8C">agent A sees</text>
<text x="430" y="18" class="wlabel" fill="#A3355E">agent B sees</text>
<circle cx="66" cy="101" r="21" class="node priv-a"/>
<text x="66" y="106" class="nlabel">Z<tspan class="sub">A</tspan></text>
<circle cx="278" cy="66" r="21" class="node shared"/>
<text x="278" y="71" class="nlabel">X<tspan class="sub">1</tspan></text>
<circle cx="278" cy="146" r="21" class="node shared"/>
<text x="278" y="151" class="nlabel">X<tspan class="sub">2</tspan></text>
<circle cx="372" cy="106" r="21" class="node shared"/>
<text x="372" y="111" class="nlabel">X<tspan class="sub">3</tspan></text>
<circle cx="492" cy="101" r="21" class="node priv-b"/>
<text x="492" y="106" class="nlabel">Z<tspan class="sub">B</tspan></text>
<path d="M 471 92 C 400 40 340 46 299 60" class="edge"/>
<path d="M 471 112 C 400 160 340 152 299 143" class="edge"/>
<path d="M 278 87 L 278 125" class="edge confound" stroke-dasharray="5 4"/>
<text x="236" y="110" class="clabel">looks
</text>
<text x="196" y="196" class="caption">Z_B is a common cause of X1 and X2 — invisible to A,
so A sees an unexplained dependence between two shared variables.</text>
</svg>'''


def main() -> None:
    data = json.loads(pathlib.Path("results/budget/budget_sweep.json").read_text())
    sa = {(e["d"], e["n_obs"]): e for e in data["single_agent"]}
    ma = {e["n_obs"]: e for e in data["two_agent"]}

    sa_panels = []
    for (d, n_obs) in ((5, 20000), (5, 100), (7, 20000), (7, 100)):
        e = sa[(d, n_obs)]
        b = budgets_of(e["arms"]["greedy"]["curve"])
        chart = line_chart(b, [
            ("greedy", GREEDY, rates(e["arms"]["greedy"]["curve"])),
            ("random", RANDOM, rates(e["arms"]["random"]["curve"])),
        ])
        sub = ("ample observational data" if n_obs >= 20000
               else "scarce observational data")
        sa_panels.append(panel(f"d = {d}, n_obs = {n_obs:,}", sub, chart,
                               [("greedy", GREEDY), ("random", RANDOM)]))

    gap_series = []
    for (d, n_obs), color, dash in (((5, 20000), GREEDY, ""), ((5, 100), "#3F9AB5", ""),
                                    ((7, 20000), GAP, ""), ((7, 100), RANDOM, "")):
        e = sa[(d, n_obs)]
        g = [a - r for a, r in zip(rates(e["arms"]["greedy"]["curve"]),
                                   rates(e["arms"]["random"]["curve"]))]
        gap_series.append((f"d={d}, n_obs={n_obs:,}", color, g))
    b = budgets_of(sa[(5, 20000)]["arms"]["greedy"]["curve"])
    gap_chart = line_chart(b, gap_series, ymax=0.6, ylabel="greedy minus random")

    ma_panels = []
    for n_obs in (20000, 100):
        e = ma[n_obs]
        for key, title in (("curve", "all episodes"),
                           ("curve_confounded", "confounded episodes only")):
            b = budgets_of(e["arms"]["greedy"][key])
            chart = line_chart(b, [
                ("greedy", GREEDY, rates(e["arms"]["greedy"][key])),
                ("random", RANDOM, rates(e["arms"]["random"][key])),
            ])
            ma_panels.append(panel(f"n_obs = {n_obs:,} — {title}",
                                   f"clamp rate: greedy "
                                   f"{e['arms']['greedy']['clamp_fraction']:.3f}, "
                                   f"random {e['arms']['random']['clamp_fraction']:.3f}",
                                   chart, [("greedy", GREEDY), ("random", RANDOM)]))

    html = TEMPLATE.format(
        setup=setup_diagram(),
        sa_panels="".join(sa_panels),
        gap_chart=gap_chart,
        ma_panels="".join(ma_panels),
    )
    out = pathlib.Path("results/budget/budget_report.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


TEMPLATE = """<title>The Budget Cliff</title>
<style>
:root {{
  --ground: #F6F6F4;  --surface: #FFFFFF;  --ink: #16181C;  --muted: #5C6169;
  --rule: #DEDEDA;    --greedy: #0E6E8C;   --random: #A3355E;  --accent: #6D28D9;
  --warn-bg: #FBF3E6; --warn-line: #C08A2E;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #131417; --surface: #1A1C21; --ink: #E9E9E6; --muted: #9AA0A8;
    --rule: #2C2F35;   --greedy: #3FA8C4;  --random: #DE6A96; --accent: #A78BFA;
    --warn-bg: #241E12; --warn-line: #C89A45;
  }}
}}
:root[data-theme="dark"] {{
  --ground: #131417; --surface: #1A1C21; --ink: #E9E9E6; --muted: #9AA0A8;
  --rule: #2C2F35;   --greedy: #3FA8C4;  --random: #DE6A96; --accent: #A78BFA;
  --warn-bg: #241E12; --warn-line: #C89A45;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--ground); color: var(--ink);
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 17px; line-height: 1.62; margin: 0; padding: 0 20px 6rem;
}}
.wrap {{ max-width: 46rem; margin: 0 auto; }}
header {{ padding: 4.5rem 0 2.5rem; border-bottom: 1px solid var(--rule); }}
.eyebrow {{
  font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 1rem;
}}
h1 {{
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: clamp(2.1rem, 5vw, 3rem); font-weight: 700; letter-spacing: -0.025em;
  line-height: 1.08; margin: 0 0 1rem; text-wrap: balance;
}}
.standfirst {{ font-size: 1.16rem; color: var(--muted); margin: 0; text-wrap: pretty; }}
h2 {{
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 1.5rem; font-weight: 650; letter-spacing: -0.018em;
  margin: 3.6rem 0 0.4rem; text-wrap: balance;
}}
h3 {{
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 1.08rem; font-weight: 650; margin: 2.2rem 0 0.3rem;
}}
h4 {{
  font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
  font-size: 12px; font-weight: 600; letter-spacing: 0.05em; margin: 0 0 0.15rem;
}}
p {{ margin: 0.85rem 0; }}
strong {{ font-weight: 700; }}
code {{
  font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
  font-size: 0.88em; background: var(--surface); border: 1px solid var(--rule);
  border-radius: 4px; padding: 0.08em 0.34em;
}}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 1.1rem; margin: 1.6rem 0; }}
.panel {{
  margin: 0; padding: 1rem 1rem 0.4rem; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 10px; overflow-x: auto;
}}
.panel figcaption p {{ margin: 0 0 0.5rem; font-size: 12.5px; color: var(--muted);
  font-family: system-ui, sans-serif; }}
.chart {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--rule); stroke-width: 1; }}
.tick {{ font-family: ui-monospace, Menlo, monospace; font-size: 10px; fill: var(--muted); }}
.tick-y {{ text-anchor: end; }}
.tick-x {{ text-anchor: middle; }}
.axis-label {{ font-family: ui-monospace, Menlo, monospace; font-size: 10px;
  fill: var(--muted); letter-spacing: 0.05em; }}
.legend {{ display: flex; gap: 0.9rem; margin: 0 0 0.3rem;
  font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: var(--muted); }}
.key {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
.key i {{ width: 14px; height: 3px; border-radius: 2px; display: inline-block; }}
.win {{ fill: none; stroke: var(--rule); stroke-width: 1.5; }}
.win-a {{ stroke: var(--greedy); stroke-dasharray: 4 3; }}
.win-b {{ stroke: var(--random); stroke-dasharray: 4 3; }}
.wlabel {{ font-family: ui-monospace, Menlo, monospace; font-size: 10.5px;
  letter-spacing: 0.06em; text-anchor: middle; }}
.node {{ fill: var(--surface); stroke: var(--ink); stroke-width: 1.6; }}
.priv-a {{ stroke: var(--greedy); }}
.priv-b {{ stroke: var(--random); }}
.shared {{ stroke: var(--ink); }}
.nlabel {{ font-family: system-ui, sans-serif; font-size: 13px; font-weight: 600;
  text-anchor: middle; fill: var(--ink); }}
.sub {{ font-size: 9px; baseline-shift: -3px; }}
.edge {{ fill: none; stroke: var(--muted); stroke-width: 1.6; }}
.confound {{ stroke: var(--accent); stroke-width: 2; }}
.clabel, .caption {{ font-family: system-ui, sans-serif; font-size: 10.5px;
  fill: var(--muted); }}
.caption {{ text-anchor: middle; }}
table {{ border-collapse: collapse; width: 100%; margin: 1.4rem 0; font-size: 14px;
  font-variant-numeric: tabular-nums;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
th, td {{ padding: 0.5rem 0.6rem; text-align: right; border-bottom: 1px solid var(--rule); }}
th:first-child, td:first-child {{ text-align: left; }}
thead th {{ font-family: ui-monospace, Menlo, monospace; font-size: 11px;
  letter-spacing: 0.04em; color: var(--muted); font-weight: 500; }}
.scroll {{ overflow-x: auto; }}
.callout {{
  background: var(--warn-bg); border-left: 3px solid var(--warn-line);
  border-radius: 0 8px 8px 0; padding: 0.9rem 1.15rem; margin: 1.8rem 0;
}}
.callout p:first-child {{ margin-top: 0; }} .callout p:last-child {{ margin-bottom: 0; }}
.callout .tag {{ font-family: ui-monospace, Menlo, monospace; font-size: 10.5px;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--warn-line);
  display: block; margin-bottom: 0.35rem; }}
footer {{ margin-top: 4rem; padding-top: 1.4rem; border-top: 1px solid var(--rule);
  font-size: 13px; color: var(--muted);
  font-family: system-ui, sans-serif; }}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">Measurement report &middot; 19 August 2026 &middot; 200 episodes per arm</p>
  <h1>The Budget Cliff</h1>
  <p class="standfirst">Every hyperparameter sweep this project has run measured the
  intervention budget as a null. It is not a null &mdash; it was being measured in the one
  region where nothing can matter.</p>
</header>

<h2>What was measured</h2>
<p>Two policies &mdash; a myopic information-gain oracle (<em>greedy</em>) and uniform random
choice &mdash; run over 200 episodes at every intervention budget from 1 to 20, in the
single-agent setting and the two-agent federated one.</p>

<p>Two changes from earlier budget measurements. First, the metric: the old
<code>episode_costs</code> charged unsolved episodes at the full budget, so a larger budget
multiplied the penalty for identical failures &mdash; the "budget effect" it reported was
mostly definitional. Here <strong>solve rate within budget</strong> and <strong>steps among
solved</strong> are reported separately and never combined. Second, the regime: earlier work
swept only at <code>n_obs = 20000</code>, where the median episode is finished in two moves.</p>

<p>One run per policy covers the entire budget curve exactly, because neither baseline reads
its remaining budget &mdash; a smaller budget is precisely a truncation of the same
trajectory. That shortcut would silently break for a learned policy, which does observe
budget, so it is confined to baselines.</p>

<h2>The single-agent picture</h2>
<div class="grid2">{sa_panels}</div>

<h3>The gap between them is the whole story</h3>
<p>Plotting greedy minus random makes the structure visible: a peak at budget 2&ndash;3,
then decay to nothing. Above budget 10 the two policies are indistinguishable, because
with enough interventions you can simply act on every variable and the quality of your
choices stops mattering.</p>
{gap_chart}

<div class="callout">
  <span class="tag">What this corrects</span>
  <p>The standing conclusion was <em>"budget is largely a metric artifact and must not be
  read as a lever"</em> &mdash; in the 13-lever sweep, moving the budget from 10 to 40 shifted
  the result by less than 0.2, against the largest observed effect of 0.288.</p>
  <p>That measurement was correct and its interpretation was wrong. Budget 10&ndash;40 is
  entirely inside the flat region on the chart above. The sweep was not discovering that
  budget does not matter; it was operating past the point where <em>anything</em> matters.
  The operating point has been moved to budget 5, with gates at 2&ndash;3.</p>
</div>

<h3>Greedy has an irreducible failure set</h3>
<p>At <code>d = 7</code> with scarce observational data, the two curves <strong>cross</strong>.
Greedy dominates early &mdash; 0.530 against 0.235 at budget 3 &mdash; then plateaus at 0.905
and never improves, while random climbs past it to 0.960.</p>
<p>So roughly <strong>9% of episodes are ones the myopic oracle never solves at any
budget</strong>, and random solves them by accident. Whatever those episodes are, one-step
information gain is systematically blind to them. This is the clearest evidence yet that
there is real headroom above greedy &mdash; and it is a failure of <em>myopia</em>, not of
sample size, because more budget does not fix it.</p>

<h2>The two-agent picture</h2>
{setup}
<div class="grid2">{ma_panels}</div>

<h3>The oracle never clamps</h3>
<p>Interventions come in two modes. <strong>Vary</strong> assigns a randomly redrawn value
and is far more informative about your own structure. <strong>Clamp</strong> holds a variable
at a constant, which is a much weaker experiment for you but is the only thing that removes
you as a hidden confounder for your partner.</p>
<p>The greedy oracle clamps <strong>0.000</strong> of the time. That is not a bug: clamping is
strictly worse for its own next-step information gain, and one-step gain is the entire
objective. It has no term for what its partner needs.</p>
<p>The consequence is stark. Greedy's solve rate is flat at <strong>0.190</strong> from budget
3 onward and <strong>exactly zero on confounded episodes at every budget tested</strong>.
Random, which clamps half the time by construction, reaches 0.755 overall and 0.444 on
confounded episodes. <em>Extra budget cannot rescue a policy that never takes the action
which unlocks the problem.</em></p>

<div class="callout">
  <span class="tag">A limit of this measurement</span>
  <p>This sweep was designed partly to answer whether a tight budget suppresses clamping the
  way an explicit clamp price did. <strong>It cannot answer that</strong>, and the reason is a
  flaw in the design rather than in the result.</p>
  <p>Neither baseline's clamp rate is behavioural. Random's is fixed near 0.50 by
  construction, since half of its actions are clamp-mode. Greedy's is fixed at 0.00 by its
  objective. Both are constant across every budget and both <code>n_obs</code> settings,
  exactly as they must be. Only a learned policy can respond to budget pressure, so the
  question stays open until Phase 5.</p>
</div>

<h2>What changes as a result</h2>
<table>
<thead><tr><th>Decision</th><th>Was</th><th>Now</th></tr></thead>
<tbody>
<tr><td>Default episode budget</td><td>10</td><td>5</td></tr>
<tr><td>Budget at which gates run</td><td>10&ndash;20</td><td>2&ndash;3</td></tr>
<tr><td>Two-agent per-agent budget</td><td>8</td><td>5</td></tr>
<tr><td>Reported metric</td><td><code>episode_costs</code></td><td>solve rate and steps, separately</td></tr>
</tbody>
</table>

<p>The gate change matters most. <strong>GATE 2 asks whether choices matter</strong>, and at
budget 10 it would have passed trivially while measuring nothing at all &mdash; both arms sit
at 0.99. It now runs where the policies are actually distinguishable.</p>

<p>One further consequence for the scaling ladder: <strong>dimension buys more headroom than
data scarcity does</strong>. At budget 8, moving from <code>d=5</code> to <code>d=7</code>
keeps a gap of 0.100 open, while dropping <code>n_obs</code> from 20000 to 100 at
<code>d=5</code> leaves only 0.035. If the task needs to stay hard, grow the graph rather
than starve it of data.</p>

<footer>
<p>200 episodes per arm, <code>n_int = 100</code>, identification threshold 0.7,
Erdős&ndash;Rényi prior at p = 0.5. Two-agent topology (1,1,3) under the
<code>joint_conf</code> belief rule; 9% of episodes were confounded. Source:
<code>scripts/budget_sweep.py</code>, raw data <code>results/budget/budget_sweep.json</code>.</p>
</footer>
</div>
"""


if __name__ == "__main__":
    main()
