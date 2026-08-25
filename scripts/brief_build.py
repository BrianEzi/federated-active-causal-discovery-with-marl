"""Build the morning brief: results, and episodes you can actually look at.

Reads whatever result JSONs exist and emits one self-contained page. Episodes are rendered
as FILMSTRIPS -- one row per agent, one column per round -- so the shape of a policy's
behaviour is visible at a glance: which node each agent took, and which claims that
resolved. The step-through inspector remains the tool for detail; this is the overview.

    python scripts/brief_build.py --out results/brief.html
"""
from __future__ import annotations

import argparse
import json
import pathlib

TEMPLATE = """<title>Overnight Brief</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {
  --ground: #f6f7f9; --panel: #fff; --panel-2: #eef0f4; --line: #d7dbe3;
  --line-soft: #e6e9ef; --ink: #131820; --ink-2: #4a5462; --ink-3: #79828f;
  --accent: #2f6fd0; --accent-soft: #dfe9f9;
  --right: #0f7d64; --right-soft: #dbefe9; --wrong: #c0402c; --wrong-soft: #f8e2de;
  --unsure: #a67806; --unsure-soft: #f6ecd5;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  --sans: "IBM Plex Sans", system-ui, -apple-system, sans-serif;
  --shadow: 0 1px 2px rgba(19,24,32,.06), 0 6px 16px rgba(19,24,32,.05);
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --ground: #0e1116; --panel: #161b22; --panel-2: #1c222b; --line: #2a323d;
  --line-soft: #222933; --ink: #e6eaf0; --ink-2: #a3adbb; --ink-3: #737f8d;
  --accent: #5b93e8; --accent-soft: #1b2c46;
  --right: #46b899; --right-soft: #14302a; --wrong: #e8705c; --wrong-soft: #38201c;
  --unsure: #d4a63c; --unsure-soft: #332811;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 20px rgba(0,0,0,.3);
} }
:root[data-theme="dark"] {
  --ground: #0e1116; --panel: #161b22; --panel-2: #1c222b; --line: #2a323d;
  --line-soft: #222933; --ink: #e6eaf0; --ink-2: #a3adbb; --ink-3: #737f8d;
  --accent: #5b93e8; --accent-soft: #1b2c46;
  --right: #46b899; --right-soft: #14302a; --wrong: #e8705c; --wrong-soft: #38201c;
  --unsure: #d4a63c; --unsure-soft: #332811;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 20px rgba(0,0,0,.3);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink); font-family: var(--sans);
       font-size: 15px; line-height: 1.55; -webkit-font-smoothing: antialiased; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 34px 22px 80px; }
h1 { font-size: 30px; letter-spacing: -.02em; margin: 0 0 6px; text-wrap: balance; }
h2 { font-size: 19px; margin: 40px 0 12px; letter-spacing: -.01em; }
h3 { font-size: 15px; margin: 22px 0 8px; }
p { margin: 0 0 12px; max-width: 74ch; }
.lede { color: var(--ink-2); font-size: 16px; max-width: 74ch; }
.label { font-family: var(--mono); font-size: 10px; letter-spacing: .09em;
         text-transform: uppercase; color: var(--ink-3); }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
code { font-family: var(--mono); font-size: 13px; background: var(--panel-2);
       padding: 1px 5px; border-radius: 4px; }
a { color: var(--accent); }

.cards { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(210px,1fr));
         margin: 20px 0 8px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
        padding: 14px 16px; box-shadow: var(--shadow); }
.card b { display: block; font-family: var(--mono); font-size: 25px; font-weight: 600;
          letter-spacing: -.02em; margin: 4px 0 2px; }
.card span { color: var(--ink-2); font-size: 13px; }
.card.good b { color: var(--right); } .card.bad b { color: var(--wrong); }

table { border-collapse: collapse; width: 100%; margin: 10px 0 6px; font-size: 14px; }
th { font-family: var(--mono); font-size: 10px; letter-spacing: .07em; text-transform: uppercase;
     color: var(--ink-3); text-align: left; font-weight: 500; padding: 8px 10px 6px;
     border-bottom: 1px solid var(--line); }
td { padding: 7px 10px; border-bottom: 1px solid var(--line-soft);
     font-variant-numeric: tabular-nums; }
td.n { font-family: var(--mono); }
tr.hi td { background: var(--accent-soft); }
.scroll { overflow-x: auto; }

.strip { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
         box-shadow: var(--shadow); padding: 12px 14px; margin: 14px 0; }
.strip h3 { margin: 0 0 2px; }
.strip .meta { color: var(--ink-3); font-size: 12.5px; font-family: var(--mono); margin-bottom: 8px; }
.grid { display: grid; gap: 6px; align-items: center; }
.cell { text-align: center; }
.cell .cap { font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); }
.rowlab { font-family: var(--mono); font-size: 11px; color: var(--ink-2); text-align: right;
          padding-right: 6px; white-space: nowrap; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0 2px; font-size: 12.5px;
          color: var(--ink-2); }
.legend i { display: inline-block; width: 15px; height: 3px; border-radius: 2px;
            vertical-align: middle; margin-right: 5px; }
.note { border-left: 3px solid var(--line); padding: 2px 0 2px 14px; color: var(--ink-2);
        margin: 14px 0; max-width: 74ch; }
.note.warn { border-left-color: var(--unsure); }
ul { max-width: 74ch; padding-left: 20px; } li { margin-bottom: 7px; }
</style>
<div class="wrap" id="app"></div>
<script>
const DATA = __BRIEF_DATA__;

/* ---- episode filmstrip -------------------------------------------------------------- */
function layout(k, r, cx, cy) {
  return Array.from({length: k}, (_, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / k;
    return {x: cx + r * Math.cos(a), y: cy + r * Math.sin(a)};
  });
}
function claimFor(claims, u, v, kind) {
  return claims.find(c => c.kind === kind && c.u === u && c.v === v);
}
function miniGraph(win, acted, size) {
  const k = win.nodes.length, pts = layout(k, size * 0.30, size / 2, size / 2);
  const col = o => o === "right" ? "var(--right)" : o === "wrong" ? "var(--wrong)" : "var(--unsure)";
  const parts = [];
  for (const e of win.true_edges) {
    const lo = Math.min(e.u, e.v), hi = Math.max(e.u, e.v);
    const type = claimFor(win.claims, lo, hi, "type");
    const adj = claimFor(win.claims, lo, hi, "adjacency");
    const outcome = adj && adj.outcome !== "right" ? adj.outcome : (type ? type.outcome : "unsure");
    const a = pts[e.u], b = pts[e.v];
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1, t = (size * 0.075) / L;
    parts.push(`<line x1="${a.x + dx * t}" y1="${a.y + dy * t}" x2="${b.x - dx * t}" y2="${b.y - dy * t}"
      style="stroke:${col(outcome)};stroke-width:${e.kind === "bidirected" ? 2.2 : 1.6};${
      e.kind === "bidirected" ? "stroke-dasharray:3 2;" : ""}"/>`);
  }
  win.nodes.forEach((id, p) => {
    const on = acted.includes(p);
    parts.push(`<circle cx="${pts[p].x}" cy="${pts[p].y}" r="${size * 0.085}" style="fill:${
      on ? "var(--accent-soft)" : "var(--panel-2)"};stroke:${on ? "var(--accent)" : "var(--line)"};
      stroke-width:${on ? 2 : 1}"/>`);
    parts.push(`<text x="${pts[p].x}" y="${pts[p].y + size * 0.032}" text-anchor="middle"
      style="font-family:var(--mono);font-size:${size * 0.09}px;fill:${
      on ? "var(--accent)" : "var(--ink-2)"}">${id}</text>`);
  });
  return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">${parts.join("")}</svg>`;
}
function filmstrip(trace, episodeIndex, title, blurb) {
  const ep = trace.episodes[episodeIndex];
  const agents = Object.keys(ep.steps[0].windows);
  const size = 84;
  const cols = ep.steps.length;
  let html = `<div class="strip"><h3>${title}</h3>
    <div class="meta">${trace.policy} &middot; seed ${ep.seed} &middot; ${
      ep.joint_identified ? "all windows identified" : "not all identified"}</div>
    <div class="scroll"><div class="grid" style="grid-template-columns:auto repeat(${cols}, ${size}px)">`;
  html += `<div></div>`;
  ep.steps.forEach((s, i) => {
    const acts = s.actions.filter(a => !a.passed);
    html += `<div class="cell"><span class="cap">${
      i === 0 ? "start" : acts.map(a => `A${a.agent}&rarr;${a.node}`).join(" ")}</span></div>`;
  });
  for (const a of agents) {
    html += `<div class="rowlab">agent ${a}</div>`;
    ep.steps.forEach(s => {
      const win = s.windows[a];
      const acted = s.actions.filter(x => !x.passed && String(x.agent) === a)
        .map(x => win.nodes.indexOf(x.node)).filter(p => p >= 0);
      const sc = win.score;
      html += `<div class="cell">${miniGraph(win, acted, size)}
        <span class="cap" style="color:${sc.identified ? "var(--right)" : "var(--ink-3)"}">${
          sc.required_right}/${sc.required_total}${sc.identified ? " done" : ""}</span></div>`;
    });
  }
  html += `</div></div>
    <div class="legend">
      <span><i style="background:var(--right)"></i>resolved correctly</span>
      <span><i style="background:var(--unsure)"></i>still open</span>
      <span><i style="background:var(--wrong)"></i>settled wrong</span>
      <span><i style="background:var(--accent)"></i>intervened this round</span>
      <span>dashed = confounded pair</span>
    </div>
    ${blurb ? `<p style="margin:8px 0 0;color:var(--ink-2);font-size:14px">${blurb}</p>` : ""}
    </div>`;
  return html;
}

/* ---- learning curve ------------------------------------------------------------------ */
function curve(series, height, ceiling, greedy) {
  const w = 660, h = height, pad = 30;
  const all = series.flatMap(s => s.points);
  const maxY = Math.max(ceiling || 0, ...all, 0.05) * 1.1;
  const maxX = Math.max(...series.map(s => s.points.length)) - 1 || 1;
  const X = i => pad + (i / maxX) * (w - pad - 8);
  const Y = v => h - pad - (v / maxY) * (h - pad - 10);
  let out = `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px">`;
  for (const gy of [0, maxY / 2, maxY]) {
    out += `<line x1="${pad}" y1="${Y(gy)}" x2="${w - 8}" y2="${Y(gy)}"
      style="stroke:var(--line-soft);stroke-width:1"/>
      <text x="4" y="${Y(gy) + 4}" style="font-family:var(--mono);font-size:10px;fill:var(--ink-3)">${gy.toFixed(2)}</text>`;
  }
  const ref = (v, color, label) => v == null ? "" :
    `<line x1="${pad}" y1="${Y(v)}" x2="${w - 8}" y2="${Y(v)}"
       style="stroke:${color};stroke-width:1.5;stroke-dasharray:5 4"/>
     <text x="${w - 10}" y="${Y(v) - 5}" text-anchor="end"
       style="font-family:var(--mono);font-size:10.5px;fill:${color}">${label}</text>`;
  out += ref(ceiling, "var(--right)", `ceiling ${ceiling != null ? ceiling.toFixed(3) : ""}`);
  out += ref(greedy, "var(--unsure)", `greedy ${greedy != null ? greedy.toFixed(3) : ""}`);
  series.forEach(s => {
    const d = s.points.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    out += `<path d="${d}" fill="none" style="stroke:${s.color};stroke-width:2"/>`;
  });
  out += `</svg><div class="legend">${series.map(s =>
    `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>`;
  return out;
}

/* ---- render -------------------------------------------------------------------------- */
const el = document.getElementById("app");
let html = `<span class="label">${DATA.date}</span><h1>${DATA.title}</h1>
  <p class="lede">${DATA.lede}</p>`;
html += `<div class="cards">${DATA.headline.map(c =>
  `<div class="card ${c.tone || ""}"><span>${c.label}</span><b>${c.value}</b><span>${c.note}</span></div>`
).join("")}</div>`;
for (const section of DATA.sections) {
  html += `<h2>${section.title}</h2>`;
  for (const block of section.blocks) {
    if (block.type === "text") html += `<p>${block.body}</p>`;
    if (block.type === "note") html += `<div class="note ${block.tone || ""}">${block.body}</div>`;
    if (block.type === "list") html += `<ul>${block.items.map(i => `<li>${i}</li>`).join("")}</ul>`;
    if (block.type === "table") {
      html += `<div class="scroll"><table><thead><tr>${
        block.columns.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${
        block.rows.map(r => `<tr class="${r.highlight ? "hi" : ""}">${
          r.cells.map((c, i) => `<td class="${i ? "n" : ""}">${c}</td>`).join("")}</tr>`).join("")
      }</tbody></table></div>`;
    }
    if (block.type === "curve") {
      html += curve(block.series, block.height || 230, block.ceiling, block.greedy);
    }
    if (block.type === "film") {
      const trace = DATA.traces[block.trace];
      if (trace) html += filmstrip(trace, block.episode || 0, block.title, block.blurb);
    }
  }
}
el.innerHTML = html;
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload", required=True, help="JSON describing the brief")
    ap.add_argument("--out", default="results/brief.html")
    args = ap.parse_args()

    payload = json.loads(pathlib.Path(args.payload).read_text())
    for key, path in list(payload.get("trace_files", {}).items()):
        payload.setdefault("traces", {})[key] = json.loads(pathlib.Path(path).read_text())
    payload.pop("trace_files", None)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__BRIEF_DATA__", json.dumps(payload)))
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
