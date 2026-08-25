"""Render episode traces as a step-through page.

Takes the JSON written by `scripts/trace_episode.py` and emits ONE self-contained HTML
file: every trace embedded, no network, no build step. The point is to make an episode
inspectable at the level the aggregate numbers hide -- which node each agent intervened on,
and what that did to every individual claim.

    python scripts/trace_view.py results/traces/*.json --out results/traces/viewer.html
"""
from __future__ import annotations

import argparse
import json
import pathlib

TEMPLATE = """<title>Episode Inspector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {
  --ground: #f6f7f9;
  --panel: #ffffff;
  --panel-2: #eef0f4;
  --line: #d7dbe3;
  --line-soft: #e6e9ef;
  --ink: #131820;
  --ink-2: #4a5462;
  --ink-3: #79828f;
  --accent: #2f6fd0;
  --accent-soft: #dfe9f9;
  --right: #0f7d64;
  --right-soft: #dbefe9;
  --wrong: #c0402c;
  --wrong-soft: #f8e2de;
  --unsure: #a67806;
  --unsure-soft: #f6ecd5;
  --shadow: 0 1px 2px rgba(19, 24, 32, .06), 0 6px 16px rgba(19, 24, 32, .05);
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --sans: "IBM Plex Sans", system-ui, -apple-system, Segoe UI, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0e1116;
    --panel: #161b22;
    --panel-2: #1c222b;
    --line: #2a323d;
    --line-soft: #222933;
    --ink: #e6eaf0;
    --ink-2: #a3adbb;
    --ink-3: #737f8d;
    --accent: #5b93e8;
    --accent-soft: #1b2c46;
    --right: #46b899;
    --right-soft: #14302a;
    --wrong: #e8705c;
    --wrong-soft: #38201c;
    --unsure: #d4a63c;
    --unsure-soft: #33281166;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 20px rgba(0, 0, 0, .3);
  }
}
:root[data-theme="dark"] {
  --ground: #0e1116;
  --panel: #161b22;
  --panel-2: #1c222b;
  --line: #2a323d;
  --line-soft: #222933;
  --ink: #e6eaf0;
  --ink-2: #a3adbb;
  --ink-3: #737f8d;
  --accent: #5b93e8;
  --accent-soft: #1b2c46;
  --right: #46b899;
  --right-soft: #14302a;
  --wrong: #e8705c;
  --wrong-soft: #38201c;
  --unsure: #d4a63c;
  --unsure-soft: #33281166;
  --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 20px rgba(0, 0, 0, .3);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: var(--sans); font-size: 14px; line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3 { margin: 0; text-wrap: balance; font-weight: 600; }
.label {
  font-family: var(--mono); font-size: 10px; letter-spacing: .09em;
  text-transform: uppercase; color: var(--ink-3); font-weight: 500;
}
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }

/* -- top bar ------------------------------------------------------------------ */
header {
  position: sticky; top: 0; z-index: 20; background: var(--panel);
  border-bottom: 1px solid var(--line); padding: 12px 20px;
  display: flex; flex-wrap: wrap; gap: 16px 24px; align-items: center;
}
header h1 { font-size: 15px; letter-spacing: -.01em; }
header h1 span { color: var(--ink-3); font-weight: 400; }
.controls { display: flex; gap: 8px; align-items: center; margin-left: auto; }
select, button {
  font-family: var(--sans); font-size: 13px; color: var(--ink);
  background: var(--panel-2); border: 1px solid var(--line);
  border-radius: 6px; padding: 6px 10px; cursor: pointer;
}
button:hover, select:hover { border-color: var(--accent); }
button:disabled { opacity: .4; cursor: default; border-color: var(--line); }
button:focus-visible, select:focus-visible, .step-row:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
.facts { display: flex; gap: 18px; flex-wrap: wrap; }
.fact { display: flex; flex-direction: column; gap: 1px; }
.fact b { font-family: var(--mono); font-size: 13px; font-weight: 500; }

/* -- shell -------------------------------------------------------------------- */
.shell { display: grid; grid-template-columns: 268px minmax(0, 1fr); gap: 0; }
@media (max-width: 900px) { .shell { grid-template-columns: 1fr; } }

aside {
  border-right: 1px solid var(--line); background: var(--panel);
  min-height: calc(100vh - 56px); padding: 14px 0 40px;
}
.rail-head { padding: 0 16px 10px; display: flex; justify-content: space-between; }
.step-row {
  display: grid; grid-template-columns: 30px minmax(0, 1fr) auto; gap: 8px;
  align-items: center; width: 100%; text-align: left; border: 0; border-radius: 0;
  background: transparent; padding: 7px 16px; cursor: pointer;
  border-left: 3px solid transparent;
}
.step-row:hover { background: var(--panel-2); }
.step-row[aria-current="true"] {
  background: var(--accent-soft); border-left-color: var(--accent);
}
.step-n { font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
.step-act { font-family: var(--mono); font-size: 12px; color: var(--ink); }
.step-act em { font-style: normal; color: var(--ink-3); }
.delta { font-family: var(--mono); font-size: 11px; }
.delta.up { color: var(--right); }
.delta.down { color: var(--wrong); }
.delta.flat { color: var(--ink-3); }

main { padding: 18px 20px 60px; min-width: 0; }
.verdict {
  display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap;
  margin-bottom: 14px;
}
.cards { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); }
@media (max-width: 560px) { .cards { grid-template-columns: 1fr; } }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  box-shadow: var(--shadow); overflow: hidden;
}
.card-head {
  display: flex; align-items: center; gap: 10px; padding: 10px 14px;
  border-bottom: 1px solid var(--line-soft); background: var(--panel-2);
}
.card-head h2 { font-size: 13px; }
.card-body { display: grid; grid-template-columns: 208px minmax(0, 1fr); }
@media (max-width: 700px) { .card-body { grid-template-columns: 1fr; } }
.graph { border-right: 1px solid var(--line-soft); padding: 6px; }
@media (max-width: 700px) { .graph { border-right: 0; border-bottom: 1px solid var(--line-soft); } }

/* -- chips -------------------------------------------------------------------- */
.chip {
  font-family: var(--mono); font-size: 10.5px; font-weight: 500; letter-spacing: .04em;
  padding: 2px 7px; border-radius: 20px; text-transform: uppercase; white-space: nowrap;
}
.chip.right { background: var(--right-soft); color: var(--right); }
.chip.wrong { background: var(--wrong-soft); color: var(--wrong); }
.chip.unsure { background: var(--unsure-soft); color: var(--unsure); }
.chip.yes { background: var(--right-soft); color: var(--right); }
.chip.no { background: var(--panel-2); color: var(--ink-3); }
.chip.acted { background: var(--accent-soft); color: var(--accent); }

/* -- ledger ------------------------------------------------------------------- */
table { border-collapse: collapse; width: 100%; }
th {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--ink-3); text-align: left; font-weight: 500; padding: 7px 8px 5px;
  border-bottom: 1px solid var(--line-soft);
}
td { padding: 5px 8px; border-bottom: 1px solid var(--line-soft); font-size: 12.5px; }
tr:last-child td { border-bottom: 0; }
tr.changed td { background: var(--accent-soft); }
tr.req td:first-child { border-left: 2px solid var(--ink-3); }
.pair { font-family: var(--mono); font-weight: 500; }
.truth { color: var(--ink-2); font-size: 12px; }
.freq { width: 108px; }
.bar { position: relative; height: 14px; background: var(--panel-2); border-radius: 3px; overflow: hidden; }
.bar i { position: absolute; inset: 0 auto 0 0; display: block; }
.bar i.c { background: var(--right); opacity: .55; }
.bar i.w { background: var(--wrong); opacity: .55; }
.bar u {
  position: absolute; top: -2px; bottom: -2px; width: 1px; background: var(--ink-2);
  opacity: .65;
}
.freq-n { font-family: var(--mono); font-size: 11px; color: var(--ink-2); }
.legend { display: flex; gap: 14px; flex-wrap: wrap; padding: 10px 20px 0; }
.legend span { display: flex; align-items: center; gap: 5px; }
.swatch { width: 16px; height: 3px; border-radius: 2px; display: inline-block; }
.empty { padding: 30px; color: var(--ink-3); text-align: center; }
@media (prefers-reduced-motion: no-preference) {
  .step-row, button, select { transition: background .12s ease, border-color .12s ease; }
}
</style>

<header>
  <h1>Episode Inspector <span id="subtitle"></span></h1>
  <div class="facts" id="facts"></div>
  <div class="controls">
    <select id="trace"></select>
    <select id="episode"></select>
    <button id="prev">&larr; Prev</button>
    <button id="next">Next &rarr;</button>
  </div>
</header>

<div class="legend">
  <span><i class="swatch" style="background:var(--right)"></i> settled right</span>
  <span><i class="swatch" style="background:var(--wrong)"></i> settled wrong</span>
  <span><i class="swatch" style="background:var(--unsure)"></i> unsure</span>
  <span><i class="swatch" style="background:var(--accent)"></i> intervened this step</span>
  <span><i class="swatch" style="background:var(--ink-3);height:8px;width:8px;border-radius:50%"></i> private node</span>
  <span class="label">arrow keys step &middot; left bar marks required claims</span>
</div>

<div class="shell">
  <aside>
    <div class="rail-head"><span class="label">Steps</span><span class="label" id="rail-note"></span></div>
    <div id="steps"></div>
  </aside>
  <main>
    <div class="verdict" id="verdict"></div>
    <div class="cards" id="cards"></div>
  </main>
</div>

<script>
const TRACES = __TRACE_DATA__;
const state = {t: 0, e: 0, s: 0};

const $ = id => document.getElementById(id);
const trace = () => TRACES[state.t];
const episode = () => trace().episodes[state.e];
const step = () => episode().steps[state.s];

/* Node ring layout: k nodes evenly on a circle, so every window reads the same way and
   the eye can compare two agents' cards without re-learning the picture. */
function layout(k, r, cx, cy) {
  return Array.from({length: k}, (_, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / k;
    return {x: cx + r * Math.cos(a), y: cy + r * Math.sin(a)};
  });
}

function claimFor(claims, u, v, kind) {
  return claims.find(c => c.kind === kind && c.u === u && c.v === v);
}

function graphSvg(win, actedPositions) {
  const k = win.nodes.length, size = 196, pts = layout(k, 62, size / 2, size / 2 + 2);
  const bar = TRACES[state.t].config.claim_bar;
  const parts = [];
  // Colours go in `style`, never in fill=/stroke= presentation attributes: those are not
  // parsed as CSS values, so var() in them resolves to nothing and the whole graph
  // silently renders black.
  parts.push(`<defs>
    <marker id="ah-r" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L8 4 L0 8 z" style="fill:var(--right)"/></marker>
    <marker id="ah-w" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L8 4 L0 8 z" style="fill:var(--wrong)"/></marker>
    <marker id="ah-u" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L8 4 L0 8 z" style="fill:var(--unsure)"/></marker>
  </defs>`);

  const colorOf = o => o === "right" ? "var(--right)" : o === "wrong" ? "var(--wrong)" : "var(--unsure)";
  const markerOf = o => o === "right" ? "ah-r" : o === "wrong" ? "ah-w" : "ah-u";

  // True edges, drawn with the outcome colour of the claim that scores them.
  for (const e of win.true_edges) {
    const [a, b] = [pts[e.u], pts[e.v]];
    const type = claimFor(win.claims, Math.min(e.u, e.v), Math.max(e.u, e.v), "type");
    const adj = claimFor(win.claims, Math.min(e.u, e.v), Math.max(e.u, e.v), "adjacency");
    const outcome = adj && adj.outcome !== "right" ? adj.outcome : (type ? type.outcome : "unsure");
    const col = colorOf(outcome);
    // Shorten so the arrowhead lands outside the node disc.
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1, t = 17 / L;
    const x1 = a.x + dx * t, y1 = a.y + dy * t, x2 = b.x - dx * t, y2 = b.y - dy * t;
    const w = 1 + 2.4 * (adj ? Math.max(adj.freq_correct, 0) : 1);
    const dash = e.kind === "bidirected" ? "stroke-dasharray:5 3;" : "";
    const startMarker = e.kind === "bidirected"
      ? `marker-start="url(#${markerOf(outcome)})"` : "";
    parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
      style="stroke:${col};stroke-width:${w};${dash}"
      marker-end="url(#${markerOf(outcome)})" ${startMarker}/>`);
  }
  // Believed-but-false edges: the failure the truth picture alone cannot show.
  for (const c of win.claims) {
    if (c.kind !== "adjacency" || c.truth !== "not adjacent") continue;
    if (c.freq_wrong < bar) continue;
    const [a, b] = [pts[c.u], pts[c.v]];
    parts.push(`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"
      style="stroke:var(--wrong);stroke-width:1.5;stroke-dasharray:2 4;opacity:.85"/>`);
  }
  // Nodes.
  win.nodes.forEach((nodeId, p) => {
    const acted = actedPositions.includes(p);
    const priv = win.private_positions.includes(p);
    parts.push(`<circle cx="${pts[p].x}" cy="${pts[p].y}" r="15" style="fill:${
      acted ? "var(--accent-soft)" : "var(--panel-2)"};stroke:${
      acted ? "var(--accent)" : "var(--line)"};stroke-width:${acted ? 2.5 : 1}"/>`);
    parts.push(`<text x="${pts[p].x}" y="${pts[p].y + 4}" text-anchor="middle"
      style="font-family:var(--mono);font-size:12px;font-weight:500;fill:${
        acted ? "var(--accent)" : "var(--ink)"}">${nodeId}</text>`);
    if (priv) {
      parts.push(`<circle cx="${pts[p].x}" cy="${pts[p].y - 21}" r="3" style="fill:var(--ink-3)"/>`);
    }
  });
  return `<svg viewBox="0 0 ${size} ${size}" width="100%" height="196" role="img"
    aria-label="window graph">${parts.join("")}</svg>`;
}

function ledger(win, prev) {
  const bar = TRACES[state.t].config.claim_bar;
  const rows = win.claims.map(c => {
    const was = prev ? claimFor(prev.claims, c.u, c.v, c.kind) : null;
    const changed = was && was.outcome !== c.outcome;
    const cw = Math.round(c.freq_correct * 100), ww = Math.round(c.freq_wrong * 100);
    const lead = c.freq_correct >= c.freq_wrong;
    return `<tr class="${changed ? "changed" : ""} ${c.required ? "req" : ""}">
      <td class="pair">${win.nodes[c.u]}&#8202;&ndash;&#8202;${win.nodes[c.v]}</td>
      <td class="truth">${c.kind === "type" ? c.truth : (c.truth === "adjacent" ? "edge" : "no edge")}</td>
      <td class="freq">
        <div class="bar">
          <i class="${lead ? "c" : "w"}" style="width:${lead ? cw : ww}%"></i>
          <u style="left:${bar * 100}%"></u>
        </div>
      </td>
      <td class="freq-n">${(lead ? c.freq_correct : c.freq_wrong).toFixed(2)}</td>
      <td><span class="chip ${c.outcome}">${c.outcome}</span></td>
    </tr>`;
  }).join("");
  return `<table><thead><tr>
    <th>Pair</th><th>Truth</th><th>Belief</th><th></th><th>Outcome</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function render() {
  const tr = trace(), ep = episode(), st = step();
  const prevStep = state.s > 0 ? ep.steps[state.s - 1] : null;

  $("subtitle").textContent = `${tr.topology.name} · ${tr.policy}`;
  $("facts").innerHTML = [
    ["Budget", `${tr.config.budget} (${tr.config.per_agent_budget}/agent)`],
    ["Bootstrap", `B=${tr.config.cb_n_boot}`],
    ["Claim bar", tr.config.claim_bar.toFixed(2)],
    ["Episode seed", ep.seed],
  ].map(([k, v]) => `<div class="fact"><span class="label">${k}</span><b>${v}</b></div>`).join("");

  // Steps rail. The delta is the count of required claims settled right, which is the
  // quantity an intervention is supposed to move -- flat means the step bought nothing.
  const required = s => Object.values(s.windows).reduce((n, w) => n + w.score.required_right, 0);
  $("steps").innerHTML = ep.steps.map((s, i) => {
    const acts = s.actions.filter(a => !a.passed);
    const desc = i === 0 ? "<em>observational start</em>"
      : acts.length ? acts.map(a => `A${a.agent}&rarr;<b>${a.node}</b>`).join(" ")
      : "<em>all passed</em>";
    const d = i === 0 ? 0 : required(s) - required(ep.steps[i - 1]);
    const cls = d > 0 ? "up" : d < 0 ? "down" : "flat";
    return `<button class="step-row" data-step="${i}" aria-current="${i === state.s}">
      <span class="step-n">${i}</span><span class="step-act">${desc}</span>
      <span class="delta ${cls}">${i === 0 ? "" : (d > 0 ? "+" : "") + d}</span>
    </button>`;
  }).join("");
  $("rail-note").textContent = `${ep.steps.length - 1} rounds`;

  const done = ep.steps[ep.steps.length - 1];
  const jointChip = ep.joint_identified ? `<span class="chip yes">joint identified</span>`
                                        : `<span class="chip no">not identified</span>`;
  $("verdict").innerHTML = `<span class="label">Step ${state.s} of ${ep.steps.length - 1}</span>
    ${jointChip}
    <span class="num" style="color:var(--ink-2)">reward ${st.reward >= 0 ? "+" : ""}${st.reward.toFixed(3)}</span>`;

  const acted = {};
  st.actions.forEach(a => { if (!a.passed) (acted[a.agent] = acted[a.agent] || []).push(a.node); });

  $("cards").innerHTML = Object.keys(ep.steps[0].windows).map(a => {
    const win = st.windows[a];
    const prevWin = prevStep ? prevStep.windows[a] : null;
    const actedPositions = (acted[a] || []).map(n => win.nodes.indexOf(n)).filter(p => p >= 0);
    const sc = win.score;
    return `<section class="card">
      <div class="card-head">
        <h2>Agent ${a}</h2>
        ${actedPositions.length ? `<span class="chip acted">acted</span>` : ""}
        <span style="margin-left:auto" class="chip ${sc.identified ? "yes" : "no"}">
          ${sc.identified ? "identified" : `${sc.required_right}/${sc.required_total} required`}</span>
        <span class="chip right">${sc.right}</span>
        <span class="chip wrong">${sc.wrong}</span>
        <span class="chip unsure">${sc.unsure}</span>
      </div>
      <div class="card-body">
        <div class="graph">${graphSvg(win, actedPositions)}</div>
        <div style="overflow-x:auto">${ledger(win, prevWin)}</div>
      </div>
    </section>`;
  }).join("");

  document.querySelectorAll(".step-row").forEach(b => {
    b.onclick = () => { state.s = +b.dataset.step; render(); };
  });
  $("prev").disabled = state.s === 0;
  $("next").disabled = state.s === ep.steps.length - 1;
}

function fillSelectors() {
  $("trace").innerHTML = TRACES.map((t, i) =>
    `<option value="${i}">${t.label}</option>`).join("");
  $("trace").value = state.t;
  $("episode").innerHTML = trace().episodes.map((e, i) =>
    `<option value="${i}">ep ${i} · seed ${e.seed} · ${e.joint_identified ? "solved" : "failed"}</option>`).join("");
  $("episode").value = state.e;
}

$("trace").onchange = e => { state.t = +e.target.value; state.e = 0; state.s = 0; fillSelectors(); render(); };
$("episode").onchange = e => { state.e = +e.target.value; state.s = 0; render(); };
$("prev").onclick = () => { if (state.s > 0) { state.s--; render(); } };
$("next").onclick = () => { if (state.s < episode().steps.length - 1) { state.s++; render(); } };
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { $("prev").click(); }
  if (e.key === "ArrowRight") { $("next").click(); }
});

fillSelectors();
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", help="trace JSON files from trace_episode.py")
    ap.add_argument("--out", default="results/traces/viewer.html")
    args = ap.parse_args()

    payload = []
    for name in args.traces:
        path = pathlib.Path(name)
        trace = json.loads(path.read_text())
        trace["label"] = path.stem
        payload.append(trace)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__TRACE_DATA__", json.dumps(payload)))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB, {len(payload)} traces)")


if __name__ == "__main__":
    main()
