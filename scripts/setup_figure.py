"""Draw the setup: one sampled system, the agent partitioning, and what each agent sees.

Self-contained SVG in one HTML file -- the repo has no matplotlib and no jupyter, and
`ma_graph_render.py` already establishes inline SVG as how this project draws graphs.

THREE PANELS, because the setup is three facts that are easy to state and hard to picture:

  1. THE SYSTEM      the true DAG over all d variables, with each agent's private block in
                     its own colour and the shared block in grey. The jointly-visible mask
                     means no edge ever runs between two different private blocks, and the
                     picture should make that obvious rather than merely true.
  2. WHAT AGENT i     the same graph with everything outside agent i's window drawn faint.
     ACTUALLY SEES    Faint rather than omitted, because "this agent is blind to that
                     variable" is the single most important fact about the setup.
  3. AGENT i's TRUE   the latent projection onto its window -- the MAG. A hidden common
     MAG              cause in someone else's private block becomes a BIDIRECTED edge here,
                     and that edge is the thing the whole thesis is about.

    .venv/bin/python scripts/setup_figure.py --seed 3 --out results/figures/setup.html
"""
from __future__ import annotations

import argparse
import html
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ma.env import MAConfig, TwoAgentEnv                             # noqa: E402
from ma.projection import BIDIRECTED, DIRECTED                       # noqa: E402
from ma.topology import federated_topology                           # noqa: E402

PALETTE = ["#2f6f9f", "#b5651d", "#4d8b5f", "#8b4a7a", "#a3903c"]
SHARED = "#6b6b76"
FAINT = "#d8d8dd"


def _layout(topology, width, height):
    """Shared block along the top, each agent's private block in a column beneath it.

    Deliberately not a force layout: the point of the picture is the PARTITION, and a
    spring embedding would place nodes by connectivity and hide exactly that.
    """
    pos = {}
    shared = list(topology.exposed)
    for i, node in enumerate(shared):
        x = width * (i + 1) / (len(shared) + 1)
        pos[node] = (x, height * 0.16)
    n = topology.n_agents
    for a, block in enumerate(topology.private):
        cx = width * (a + 0.5) / n
        for j, node in enumerate(block):
            spread = (j - (len(block) - 1) / 2) * min(46, width / (n * max(len(block), 1)))
            pos[node] = (cx + spread, height * (0.52 + 0.30 * (j % 2)))
    return pos


def _svg(nodes, pos, edges, owner, visible, width, height, title, bidirected=()):
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
           f'style="max-width:{width}px;height:auto">',
           '<defs>'
           '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
           'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>'
           '<marker id="af" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
           'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#d8d8dd"/></marker>'
           '<marker id="b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
           'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#c0392b"/></marker>'
           '</defs>',
           f'<text x="{width/2}" y="22" text-anchor="middle" font-size="15" '
           f'font-weight="600" fill="#222">{html.escape(title)}</text>']
    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        seen = visible is None or (u in visible and v in visible)
        x1, y1 = pos[u]; x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        # stop short of the node so the arrowhead is visible outside the circle
        x2 -= dx / length * 15; y2 -= dy / length * 15
        x1 += dx / length * 15; y1 += dy / length * 15
        colour = "#555" if seen else FAINT
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                   f'stroke="{colour}" stroke-width="{1.6 if seen else 1.0}" '
                   f'marker-end="url(#{"a" if seen else "af"})"/>')
    for u, v in bidirected:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 46
        out.append(f'<path d="M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {x2:.1f},{y2:.1f}" '
                   f'fill="none" stroke="#c0392b" stroke-width="2" stroke-dasharray="6 4" '
                   f'marker-end="url(#b)" marker-start="url(#b)"/>')
    for node in nodes:
        if node not in pos:
            continue
        x, y = pos[node]
        seen = visible is None or node in visible
        fill = (PALETTE[owner[node] % len(PALETTE)] if owner[node] is not None else SHARED)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" '
                   f'fill="{fill if seen else FAINT}" stroke="#fff" stroke-width="2"/>')
        out.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="11" '
                   f'font-weight="600" fill="{"#fff" if seen else "#aaa"}">{node}</text>')
    out.append('</svg>')
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--private", type=int, default=3)
    ap.add_argument("--shared", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results/figures/setup.html")
    args = ap.parse_args(argv)

    cfg = MAConfig(topology=federated_topology(args.agents, args.private, args.shared),
                   n_obs=200, n_int=40, budget=24, turn_order="round_robin",
                   belief_backend="factored", action_modes=("vary",), claim_bar=1.0,
                   reward_criterion="claims", policy_arch="gnn_portable", graph_model="sf",
                   sf_m=2, episode_mix="confounded", vs_evidence="oracle")
    env = TwoAgentEnv(cfg)
    env.reset(seed=args.seed)
    topology, adjacency = env.topology, env.true_adjacency
    d = topology.d
    owner = {node: None for node in range(d)}
    for a, block in enumerate(topology.private):
        for node in block:
            owner[node] = a
    edges = [(u, v) for u in range(d) for v in range(d) if adjacency[u, v]]
    W, H = 760, 380
    pos = _layout(topology, W, H)

    panels = [("<h2>1. The system &mdash; the true DAG over every variable</h2>"
               "<p>Each agent's private block has its own colour; the shared block is grey. "
               "<b>No edge ever runs between two different private blocks</b> &mdash; that is "
               "the jointly-visible mask, and at this topology it forbids about half of all "
               "ordered pairs.</p>"
               + _svg(range(d), pos, edges, owner, None, W, H, "true DAG"))]

    for a in topology.agents:
        window = set(topology.observed_by(a))
        mag = env._take_mag(a) if hasattr(env, "_take_mag") else env._true_mag(a)
        nodes = list(topology.observed_by(a))
        wpos = {n: pos[n] for n in nodes}
        directed, bidir = [], []
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i < j:
                    if mag[i, j] == BIDIRECTED and mag[j, i] == BIDIRECTED:
                        bidir.append((u, v))
                    elif mag[i, j] == DIRECTED:
                        directed.append((u, v))
                    elif mag[j, i] == DIRECTED:
                        directed.append((v, u))
        panels.append(
            f"<h2>2.{a+1} Agent {a} &mdash; what it can see, and what it must conclude</h2>"
            "<p>Left: the same system, with everything outside this agent's window faint. "
            "Right: the agent's true MAG. A hidden common cause sitting in someone else's "
            "private block projects to a <b style='color:#c0392b'>dashed red bidirected "
            "edge</b> &mdash; the agent sees the dependence but not the variable causing it. "
            "That edge is what the thesis is about.</p>"
            '<div class="row">'
            + _svg(range(d), pos, edges, owner, window, W, H, f"agent {a}: visible")
            + _svg(nodes, wpos, directed, owner, None, W, H,
                   f"agent {a}: its true MAG", bidirected=bidir)
            + "</div>")

    body = "\n".join(panels)
    page = f"""<!doctype html><meta charset="utf-8"><title>The setup</title>
<style>
 body{{font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
   max-width:1600px;margin:0 auto;padding:28px;color:#222;background:#fbfbfc}}
 h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:34px 0 6px}}
 p{{color:#555;max-width:820px;margin:0 0 12px}}
 .row{{display:flex;gap:20px;flex-wrap:wrap}}
 .meta{{color:#777;font-size:13px;margin-bottom:18px}}
 svg{{background:#fff;border:1px solid #e6e6ea;border-radius:8px}}
</style>
<h1>Federated active causal discovery &mdash; the setup</h1>
<div class="meta">{args.agents} agents &middot; {args.private} private variables each &middot;
{args.shared} shared &middot; d={d} &middot; window k={args.private+args.shared} &middot;
scale-free (m=2) &middot; seed {args.seed}</div>
{body}
"""
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    print(f"wrote {path}  ({d} variables, {len(edges)} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
