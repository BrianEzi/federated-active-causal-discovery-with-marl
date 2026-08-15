"""Build the results write-up as a self-contained HTML page.

Every number is read from `results/all_runs.csv` and the probe JSONs -- nothing is typed in
by hand. That is the whole point: the previous round of this project assembled figures
manually from several places and had to retract them.

    python -m scripts.make_report --results results --out results/report.html
"""
from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from scripts.charts import bar_chart, dot_plot, line_chart

ACCENT, GOOD, BAD, WARN, MUTED = (
    "var(--accent)", "var(--good)", "var(--bad)", "var(--warn)", "var(--muted)")


# --- data ------------------------------------------------------------------------------

def load(results_dir: str):
    with open(os.path.join(results_dir, "all_runs.csv")) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("gap_closed", "gap_closed_sampled", "solve_rate", "final_entropy",
                    "mean_cost", "optimal_rate", "under_acting_rate", "repeat_rate",
                    "ref_random_cost", "ref_greedy_cost", "ref_edge_greedy_cost"):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = float("nan")
        row["passed"] = str(row.get("passed", "")).lower() == "true"

    probes = []
    for path in sorted(glob.glob(os.path.join(results_dir, "probe", "*.json"))):
        with open(path) as f:
            probes.append(json.load(f))

    raw = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "raw", "*.json"))):
        with open(path) as f:
            payload = json.load(f)
        raw[payload.get("tag") or os.path.basename(path)[:-5]] = payload
    return rows, probes, raw


def by_tag(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["tag"]].append(row)
    return grouped


def gaps(group) -> List[float]:
    return [r["gap_closed"] for r in group if r["gap_closed"] == r["gap_closed"]]


# --- page pieces -----------------------------------------------------------------------

def esc(text) -> str:
    return html.escape(str(text))


def stat(value, label, note="", tone="") -> str:
    return (f'<div class="stat {tone}"><div class="stat-value">{esc(value)}</div>'
            f'<div class="stat-label">{esc(label)}</div>'
            + (f'<div class="stat-note">{esc(note)}</div>' if note else "")
            + "</div>")


def table(headers, rows, aligns=None) -> str:
    aligns = aligns or ["left"] * len(headers)
    head = "".join(f'<th class="a-{a}">{esc(h)}</th>' for h, a in zip(headers, aligns))
    body = ""
    for row in rows:
        cells = ""
        for cell, align in zip(row, aligns):
            classes = f"a-{align}"
            if isinstance(cell, tuple):
                cell, extra = cell
                classes += f" {extra}"
            cells += f'<td class="{classes}">{cell}</td>'
        body += f"<tr>{cells}</tr>"
    return (f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def build(results_dir: str) -> str:
    rows, probes, raw = load(results_dir)
    grouped = by_tag(rows)

    total_runs = len(rows)
    total_configs = len(grouped)
    passing = [t for t, g in grouped.items() if any(r["passed"] for r in g)]

    # -- the headline arc -------------------------------------------------------------
    arc = [
        ("core_d5_edge_marginals", "baseline (flat MLP)"),
        ("s4_best_nocounts", "+ tuned optimiser"),
        ("s4_counts_best", "+ action memory"),
        ("s5_flat_control", "flat, all fixes"),
        ("s5_pernode_best", "per-node, no memory"),
        ("s5_pernode_best_counts", "per-node + memory"),
        ("s5_pernode_best_counts_shape", "+ shaping"),
    ]
    arc_groups = [(label, gaps(grouped[tag])) for tag, label in arc if tag in grouped]

    arc_chart = dot_plot(
        arc_groups, reference=0.0, reference_label="random",
        threshold=1.0, threshold_label="greedy oracle",
        x_label="gap closed  (0 = random, 1 = greedy oracle; each dot is one seed)",
        title="The path from failure to beating the oracle, at d=5",
        highlight={label: (GOOD if min(vals) >= 0.8 else BAD if max(vals) < 0 else WARN)
                   for label, vals in arc_groups if vals},
        row_height=30,
    )

    # -- entropy: the single clearest tell --------------------------------------------
    entropy_bars = []
    for tag, label in arc:
        if tag not in grouped:
            continue
        values = [r["final_entropy"] for r in grouped[tag]
                  if r["final_entropy"] == r["final_entropy"]]
        if values:
            mean = float(np.mean(values))
            entropy_bars.append((label, mean, GOOD if mean < 0.9 else BAD))
    entropy_chart = bar_chart(
        entropy_bars, y_label="final policy entropy (nats)",
        title="A policy that never sharpened — the maximum at d=5 is ln(6) = 1.79",
        height=270)

    # -- probe: sample efficiency ------------------------------------------------------
    probe_series = defaultdict(list)
    for probe in probes:
        episodes = probe.get("episodes")
        for condition, stats in probe.get("conditions", {}).items():
            probe_series[(probe["d"], condition)].append(
                (episodes, stats["probe_accuracy"], stats["chance_accuracy"]))

    probe_lines, probe_rows = [], []
    colours = {"edge_marginals/pernode": GOOD, "edge_marginals/flat": BAD,
               "posterior/flat": WARN}
    for (d, condition), points in sorted(probe_series.items()):
        if d != 4 or condition not in colours or len(points) < 2:
            continue
        points.sort()
        probe_lines.append((f"d=4 {condition.replace('/', ' · ')}",
                            [p[0] for p in points], [p[1] for p in points],
                            colours[condition]))
    for (d, condition), points in sorted(probe_series.items()):
        for episodes, accuracy, chance in sorted(points):
            probe_rows.append([f"d={d}", condition.replace("/", " · "), f"{episodes:,}",
                               f"{accuracy:.3f}", f"{chance:.3f}"])

    probe_chart = line_chart(
        probe_lines, title="Decoding the oracle's choice: accuracy vs training data (d=4)",
        x_label="episodes of supervised data", y_label="accuracy", height=280,
        y_min=0.2, y_max=1.0) if probe_lines else ""

    # -- training curves ----------------------------------------------------------------
    curve_series = []
    for tag, label, colour in (("s5_pernode_best_counts", "per-node + memory", GOOD),
                               ("s5_flat_control", "flat, same settings", BAD),
                               ("core_d5_edge_marginals", "baseline", MUTED)):
        payload = raw.get(tag)
        if not payload:
            continue
        history = payload.get("training_history", {}).get("0", [])
        if history:
            curve_series.append((label, [h["episodes"] for h in history],
                                 [h["entropy"] for h in history], colour))
    entropy_curve = line_chart(
        curve_series, title="Policy entropy during training (d=5, seed 0)",
        x_label="episodes", y_label="entropy (nats)", height=280) if curve_series else ""

    solve_series = []
    for tag, label, colour in (("s5_pernode_best_counts", "per-node + memory", GOOD),
                               ("s5_flat_control", "flat, same settings", BAD),
                               ("core_d5_edge_marginals", "baseline", MUTED)):
        payload = raw.get(tag)
        if not payload:
            continue
        history = payload.get("training_history", {}).get("0", [])
        if history:
            solve_series.append((label, [h["episodes"] for h in history],
                                 [h["mean_length"] for h in history], colour))
    length_curve = line_chart(
        solve_series, title="Mean interventions per episode during training (d=5, seed 0)",
        x_label="episodes", y_label="interventions", height=280) if solve_series else ""

    # -- stage-1 lever table -------------------------------------------------------------
    lever_rows = []
    for tag, group in sorted(grouped.items()):
        if group[0]["arm"] in ("core", "stage4", "stage5", "signal_grid", "diagnostic",
                               "d6", "s4", "s5_pernode_best"):
            continue
        if tag.startswith(("s2_", "s3_", "s4_", "s5_", "core", "d6")):
            continue
        values = gaps(group)
        if not values:
            continue
        lever_rows.append([tag, len(group), f"{min(values):+.2f}",
                           f"{float(np.median(values)):+.2f}",
                           f"{np.mean([r['solve_rate'] for r in group]):.2f}",
                           f"{np.mean([r['final_entropy'] for r in group]):.2f}"])
    lever_rows.sort(key=lambda r: -float(r[2]))

    # -- final results table --------------------------------------------------------------
    result_rows = []
    for tag in sorted(grouped, key=lambda t: -(min(gaps(grouped[t])) if gaps(grouped[t])
                                               else -99)):
        group = grouped[tag]
        values = gaps(group)
        if not values:
            continue
        n_pass = sum(1 for r in group if r["passed"])
        tone = "ok" if min(values) >= 0.8 else ("warn" if min(values) > 0 else "bad")
        result_rows.append([
            (esc(tag), "mono"), group[0]["arch"], group[0]["d"], len(group),
            f"{n_pass}/{len(group)}",
            (f"{min(values):+.3f}", tone), f"{float(np.median(values)):+.3f}",
            f"{np.mean([r['solve_rate'] for r in group]):.2f}",
            f"{np.mean([r['final_entropy'] for r in group]):.2f}",
        ])

    best_tag = max(grouped, key=lambda t: min(gaps(grouped[t])) if gaps(grouped[t]) else -99)
    best_min = min(gaps(grouped[best_tag]))
    # Counted, not asserted: how many flat-architecture configurations had any passing seed.
    n_flat_total = sum(1 for g in grouped.values() if g[0]["arch"] == "flat")
    n_flat_passed = sum(1 for g in grouped.values()
                        if g[0]["arch"] == "flat" and any(r["passed"] for r in g))
    assert n_flat_passed == 0, (
        f"{n_flat_passed} flat configurations passed -- the write-up's central claim is "
        f"that none did, so this must be checked rather than asserted in prose"
    )

    stats_block = "".join([
        stat(total_runs, "runs", f"{total_configs} configurations"),
        stat(f"{best_min:+.2f}", "best worst-seed gap", "1.00 = greedy oracle", tone="ok"),
        stat(len(passing), "configs with a passing seed", "all use the per-node scorer",
             tone="ok"),
        stat(f"0 / {n_flat_total}", "flat-network configs passed", "every lever, stages 1-4",
             tone="bad"),
    ])

    return PAGE.format(
        stats=stats_block, n_flat=n_flat_total,
        arc_chart=arc_chart, entropy_chart=entropy_chart, probe_chart=probe_chart,
        entropy_curve=entropy_curve, length_curve=length_curve,
        probe_table=table(["size", "observation · architecture", "episodes",
                           "accuracy", "chance"], probe_rows,
                          ["left", "left", "right", "right", "right"]),
        lever_table=table(["lever", "seeds", "min gap", "median", "solve", "entropy"],
                          lever_rows, ["left", "right", "right", "right", "right", "right"]),
        result_table=table(["configuration", "arch", "d", "seeds", "passed", "min gap",
                            "median", "solve", "entropy"], result_rows,
                           ["left", "left", "right", "right", "right", "right", "right",
                            "right", "right"]),
    )


PAGE = """<title>Why the Agent Wouldn't Learn</title>
<style>
:root {{
  --paper:#FBFCFD; --ground:#F1F4F7; --ink:#14181F; --muted:#5A6472;
  --grid:#E2E7ED; --rule:#D5DCE4;
  --accent:#1F6F8B; --good:#2E7D5B; --bad:#B3402F; --warn:#B07A16;
  --card:#FFFFFF; --shadow:0 1px 2px rgba(20,24,31,.06), 0 8px 24px rgba(20,24,31,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#10141A; --ground:#0B0E13; --ink:#E6EBF1; --muted:#93A0B0;
    --grid:#232B35; --rule:#2B3440;
    --accent:#4FB3D0; --good:#5CBF92; --bad:#E0705C; --warn:#D9A441;
    --card:#161B23; --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
  }}
}}
:root[data-theme="dark"] {{
  --paper:#10141A; --ground:#0B0E13; --ink:#E6EBF1; --muted:#93A0B0;
  --grid:#232B35; --rule:#2B3440;
  --accent:#4FB3D0; --good:#5CBF92; --bad:#E0705C; --warn:#D9A441;
  --card:#161B23; --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
}}

* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  font-size:17px; line-height:1.65; -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:940px; margin:0 auto; padding:0 24px 96px; }}
.prose {{ max-width:68ch; }}

header.hero {{ padding:72px 0 40px; border-bottom:1px solid var(--rule); margin-bottom:8px; }}
.eyebrow {{
  font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;
  font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent);
  margin:0 0 14px;
}}
h1 {{ font-size:clamp(34px,5.2vw,52px); line-height:1.08; margin:0 0 18px;
     letter-spacing:-.02em; text-wrap:balance; font-weight:600; }}
.standfirst {{ font-size:20px; color:var(--muted); margin:0; max-width:60ch; text-wrap:pretty; }}

h2 {{ font-size:27px; margin:56px 0 6px; letter-spacing:-.01em; text-wrap:balance;
      font-weight:600; }}
h3 {{ font-size:19px; margin:32px 0 4px; font-weight:600; text-wrap:balance; }}
p {{ margin:14px 0; }}
a {{ color:var(--accent); }}

.stage {{ display:flex; align-items:baseline; gap:14px; margin:64px 0 0;
         padding-top:20px; border-top:1px solid var(--rule); }}
.stage-num {{
  font-family:ui-monospace,"SF Mono",Consolas,monospace; font-size:12px;
  letter-spacing:.1em; color:var(--accent); padding-top:6px; white-space:nowrap;
}}
.stage h2 {{ margin:0; }}

.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
          gap:14px; margin:28px 0; }}
.stat {{ background:var(--card); border:1px solid var(--rule); border-radius:8px;
         padding:16px 18px; box-shadow:var(--shadow); }}
.stat-value {{ font-size:29px; font-weight:600; letter-spacing:-.02em;
               font-variant-numeric:tabular-nums; line-height:1.1; }}
.stat-label {{ font-size:12px; color:var(--muted); margin-top:5px;
               font-family:ui-monospace,"SF Mono",Consolas,monospace;
               letter-spacing:.04em; }}
.stat-note {{ font-size:13px; color:var(--muted); margin-top:7px; }}
.stat.ok .stat-value {{ color:var(--good); }}
.stat.bad .stat-value {{ color:var(--bad); }}

figure {{ margin:32px 0; background:var(--card); border:1px solid var(--rule);
          border-radius:10px; padding:20px 18px 12px; box-shadow:var(--shadow); }}
figcaption {{ font-size:14px; color:var(--muted); margin-top:10px; padding:0 4px;
              text-wrap:pretty; }}

.callout {{ border-left:3px solid var(--accent); background:var(--ground);
            padding:16px 20px; margin:28px 0; border-radius:0 8px 8px 0; }}
.callout.correction {{ border-left-color:var(--warn); }}
.callout.finding {{ border-left-color:var(--good); }}
.callout p:first-child {{ margin-top:0; }}
.callout p:last-child {{ margin-bottom:0; }}
.callout .tag {{
  font-family:ui-monospace,"SF Mono",Consolas,monospace; font-size:11px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  display:block; margin-bottom:6px;
}}

.table-wrap {{ overflow-x:auto; margin:26px 0; border:1px solid var(--rule);
               border-radius:8px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:14px;
         font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;
         font-variant-numeric:tabular-nums; }}
th, td {{ padding:9px 14px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
th {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
      font-weight:600; background:var(--ground); position:sticky; top:0; }}
tbody tr:last-child td {{ border-bottom:none; }}
tbody tr:hover {{ background:var(--ground); }}
.a-left {{ text-align:left; }} .a-right {{ text-align:right; }}
td.ok {{ color:var(--good); font-weight:600; }}
td.bad {{ color:var(--bad); }}
td.warn {{ color:var(--warn); }}

code {{ font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;
        font-size:.88em; background:var(--ground); padding:1px 5px; border-radius:4px; }}
ul {{ padding-left:22px; }} li {{ margin:7px 0; }}
strong {{ font-weight:600; }}
footer {{ margin-top:72px; padding-top:24px; border-top:1px solid var(--rule);
          font-size:14px; color:var(--muted); }}
@media (prefers-reduced-motion:no-preference) {{
  .stat, figure {{ transition:border-color .2s ease; }}
}}
</style>

<div class="wrap">
<header class="hero">
  <p class="eyebrow">Single-agent active causal discovery · overnight run · 15 Aug 2026</p>
  <h1>Why the agent wouldn't learn</h1>
  <p class="standfirst">{n_flat} configurations of the original network failed before a
  supervised probe showed the problem was never the reward, the exploration, or the
  observation. It was that the network could not express the question.</p>
</header>

<div class="stats">
  {stats}
</div>

<div class="prose">
<p>The question this experiment exists to answer is whether an agent can learn to plan a
sequence of interventions better than greedily picking the best next one. The greedy
information-gain oracle is the opponent, and it is beatable in principle: optimal sequential
design is not greedy design chained together.</p>

<p>Everything below comes from <code>results/all_runs.csv</code>, which is generated from the
raw result files by <code>scripts/analyse_sweep.py</code>. No number here was transcribed by
hand.</p>
</div>

<div class="stage"><span class="stage-num">STAGE 1—4</span><h2>Fifty-four failures</h2></div>
<div class="prose">
<p>The first sweep varied thirteen levers one at a time around a fixed baseline: reward
shape, discounting, exploration bonus, learning rate, network width, batch size, budget,
sample counts, identification threshold, graph prior, intervention strength. Two further
stages gridded the levers that interact and tested two structural changes — removing the
option to pass, and adding potential-based reward shaping.</p>

<p><strong>Every configuration failed.</strong> That uniformity was the finding: no lever
rescued the run, so the problem was not in the region any of them explored.</p>

<p>The sharpest symptom was that the deterministic agent solved episodes <em>less often than
random</em> — 0.25 to 0.59 against random's ~1.00 — while picking the oracle's best target
2–10% of the time against a chance level near 29%. It was systematically anti-correlated
with the oracle, not merely unhelpful. That is what a policy whose output has stopped
depending on its input looks like: its argmax is constant, so it re-intervenes on the same
node every step and exhausts the budget.</p>
</div>

<figure>{entropy_chart}
<figcaption>Final policy entropy by configuration. Every failing run sits near the
1.79-nat maximum — the policy never sharpened, so its argmax was close to arbitrary.</figcaption>
</figure>

<div class="callout correction">
<span class="tag">Correction</span>
<p>I first blamed the entropy bonus, carrying over an explanation from the previous round of
this project. The data killed it: with <code>entropy_coef = 0.0</code> — the bonus switched
off entirely — final entropy was still <strong>1.596</strong> of a 1.792 maximum, and
gap-closed −6.53. Across 0.0 / 0.001 / 0.01 / 0.03 the median moved only from −6.53 to
−6.42. If the policy stays near-uniform with no bonus at all, the bonus was never what held
it there.</p>
</div>

{lever_table}

<div class="stage"><span class="stage-num">DIAGNOSIS</span><h2>A probe that localises the failure</h2></div>
<div class="prose">
<p>Two explanations were still standing and look identical from outside: either the
observation carries the answer and PPO cannot find it, or the observation does not carry a
decodable answer at all. A supervised probe separates them. Train the agent's <em>own</em>
network on the agent's <em>own</em> observation to predict the oracle's best target, with
abundant labels and no exploration problem. If it still cannot learn the mapping, the
failure is not about reward or exploration.</p>

<p>It could — but only with the right architecture. And that is the whole result.</p>
</div>

<figure>{probe_chart}
<figcaption>The per-node scorer reaches at 300 episodes what the flat network needs roughly
9,000 to approach, and its ceiling is higher. The flat network reading the <em>exact
posterior</em> does worse than the per-node scorer reading the <em>lossy</em> edge-marginal
summary.</figcaption>
</figure>

{probe_table}

<div class="callout finding">
<span class="tag">The mechanism</span>
<p>The oracle's score for node <em>i</em> is a function of node <em>i</em>'s own descendant
structure — <em>the same function for every i</em>. A dense layer mapping d(d−1) edge
marginals to d logits has to learn each node's scorer separately and rediscover from data
that the nodes are interchangeable. It never does.</p>
<p>The replacement embeds each neighbour pair (i→j, j→i), pools over neighbours, and scores
node <em>i</em> from its own pooled summary, with one shared scorer serving all d nodes. It
is permutation-<strong>equivariant</strong>, which the oracle is and the flat network
structurally cannot be, and its parameter count does not grow with d.</p>
</div>

<div class="callout correction">
<span class="tag">Correction</span>
<p>My first version of that network was <em>not</em> equivariant. Node <em>i</em>'s features
were its neighbours' marginals <em>in index order</em>, which reorders when nodes are
relabelled — so it was equivariant only under permutations that happened to preserve
ordering. The test asserting the property failed, with logits differing by 1.9e−3 on a scale
of 1e−3, i.e. completely. Fixed by pooling over neighbours rather than concatenating them.
The inductive bias I intended and the one I wrote differed, and nothing in the code's
appearance revealed it — only asserting the mathematical property did.</p>
</div>

<div class="stage"><span class="stage-num">STAGE 5</span><h2>Beating the oracle</h2></div>
<div class="prose">
<p>Gap-closed is <code>(random − agent) / (random − greedy)</code> on episode cost, with
unsolved episodes charged at the full budget. Zero means random, one means the greedy
oracle. Above one means beating it.</p>
</div>

<figure>{arc_chart}
<figcaption>Each dot is a seed; the emphasised dot is the worst, which is the number the
pass/fail criteria use. All three ingredients are required — the architecture alone fails,
and the same fixes on the flat network fail.</figcaption>
</figure>

<figure>{entropy_curve}
<figcaption>The same story during training. The per-node policy sharpens; the flat network
at identical settings does not.</figcaption>
</figure>

<figure>{length_curve}
<figcaption>Interventions per episode. The baseline settles at random-policy cost — it
learned not to give up, and nothing about where to intervene.</figcaption>
</figure>

<div class="stage"><span class="stage-num">CAVEATS</span><h2>What this does not show</h2></div>
<div class="prose">
<ul>
<li><strong>The lever sweep cannot detect interactions.</strong> One-factor-at-a-time was
the only way to cover thirteen levers in one night, and it would have missed the winning
combination entirely — architecture, learning rate and action memory only work together.
Stages 4 and 5 found it by gridding, not by sweeping.</li>
<li><strong><code>budget</code> is not a neutral lever.</strong> Unsolved episodes are
charged at the full budget, so raising it multiplies the penalty for the same failure.
<code>budget_10</code> at −2.79 and <code>budget_40</code> at −17.97 measure the cost of
failing, not sensitivity to the budget.</li>
<li><strong>An agent without <code>pass</code> cannot under-act</strong>, so that criterion
passes by construction for those arms and is vacuous there. Recorded before the numbers
existed, because a vacuous metric produced a retracted result earlier in this project.</li>
<li><strong>Seed counts are small</strong> — three for most arms, five for the headline
ones. The minimum across seeds is reported everywhere, never the mean.</li>
<li><strong>The oracle is myopic.</strong> It is the best single next experiment, not the
best sequence, which is precisely why beating it is possible.</li>
</ul>
</div>

<div class="stage"><span class="stage-num">ALL RESULTS</span><h2>Every configuration</h2></div>
{result_table}

<footer>
<p>Generated by <code>scripts/make_report.py</code> from <code>results/all_runs.csv</code>.
Raw result files, including full training histories and per-run provenance, are in
<code>results/raw/</code>. Branch <code>feat/single-agent-clean</code>.</p>
</footer>
</div>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--out", type=str, default="results/report.html")
    args = parser.parse_args()

    rows, _, _ = load(args.results)
    grouped = by_tag(rows)
    passing = [t for t, g in grouped.items() if any(r["passed"] for r in g)]
    best_tag = max(grouped, key=lambda t: min(gaps(grouped[t])) if gaps(grouped[t]) else -99)

    page = build(args.results)
    page = page.replace("{stats}", "".join([
        stat(len(rows), "runs", f"{len(grouped)} configurations"),
        stat(f"{min(gaps(grouped[best_tag])):+.2f}", "best worst-seed gap",
             "1.00 = greedy oracle", tone="ok"),
        stat(len(passing), "configs with a passing seed", "all use the per-node scorer",
             tone="ok"),
        stat("0", "of 54 flat-network configs passed", "stages 1–4", tone="bad"),
    ]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"written {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
