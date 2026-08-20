"""Draw the worked examples: truth, each agent's conclusion, and the union.

Every panel uses the SAME node coordinates, so the four graphs in a case can be compared by
eye without re-reading the labels. Nodes an agent cannot see are drawn faint rather than
omitted, because "this agent is blind to that variable" is the single most important fact
about the setup and hiding it would make the panels look like ordinary graph estimates.

Edges are coloured against the truth: correct, missed, or spurious. Confounding claims are
drawn as dashed double-headed arcs, which is the standard MAG notation for "these two share
an unobserved common cause" -- and deliberately NOT drawn as a directed edge, since the
whole retraction this month came from confusing the augmented graph with the causal one.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np

# Triangle of shared variables, privates on the flanks. Cross-private edges are structurally
# impossible under the topology mask, so nothing has to route around the middle.
POS = {0: (30, 100), 1: (270, 100), 2: (150, 30), 3: (96, 172), 4: (204, 172)}
LABEL = {0: "P<tspan baseline-shift='sub' font-size='8'>A</tspan>",
         1: "P<tspan baseline-shift='sub' font-size='8'>B</tspan>",
         2: "X1", 3: "X2", 4: "X3"}
R = 19.0
VB = (300, 205)


def _edge_geometry(u, v, shrink=R + 3.0, bow=0.0):
    (x1, y1), (x2, y2) = POS[u], POS[v]
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    sx, sy = x1 + ux * shrink, y1 + uy * shrink
    tx, ty = x2 - ux * shrink, y2 - uy * shrink
    if bow:
        mx, my = (sx + tx) / 2 - uy * bow, (sy + ty) / 2 + ux * bow
        return (sx, sy), (tx, ty), (mx, my), (ux, uy)
    return (sx, sy), (tx, ty), None, (ux, uy)


def _arrowhead(tip, direction, colour, size=7.0):
    ux, uy = direction
    px, py = -uy, ux
    x, y = tip
    pts = "%.1f,%.1f %.1f,%.1f %.1f,%.1f" % (
        x, y,
        x - ux * size + px * size * 0.45, y - uy * size + py * size * 0.45,
        x - ux * size - px * size * 0.45, y - uy * size - py * size * 0.45)
    return '<polygon points="%s" fill="%s"/>' % (pts, colour)


def directed_edge(u, v, colour, dashed=False, width=2.0):
    (sx, sy), (tx, ty), _, direction = _edge_geometry(u, v)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return ('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%.1f"'
            ' stroke-linecap="round"%s/>' % (sx, sy, tx, ty, colour, width, dash)
            + _arrowhead((tx, ty), direction, colour))


def bidirected_edge(u, v, colour):
    """MAG notation: a dashed arc with an arrowhead at BOTH ends, meaning a hidden common
    cause. Bowed away from the straight line so it never sits on top of a causal edge
    between the same pair -- which is exactly the case that matters, since a confounding
    claim and a real edge can coexist."""
    (sx, sy), (tx, ty), (mx, my), (ux, uy) = _edge_geometry(u, v, shrink=R + 4.0, bow=26.0)
    path = ('<path d="M %.1f %.1f Q %.1f %.1f %.1f %.1f" fill="none" stroke="%s" '
            'stroke-width="2" stroke-dasharray="4 3"/>' % (sx, sy, mx, my, tx, ty, colour))
    d1 = ((sx - mx), (sy - my))
    n1 = max((d1[0] ** 2 + d1[1] ** 2) ** 0.5, 1e-6)
    d2 = ((tx - mx), (ty - my))
    n2 = max((d2[0] ** 2 + d2[1] ** 2) ** 0.5, 1e-6)
    return (path
            + _arrowhead((sx, sy), (d1[0] / n1, d1[1] / n1), colour, 6.0)
            + _arrowhead((tx, ty), (d2[0] / n2, d2[1] / n2), colour, 6.0))


def node(i, visible=True, role="shared"):
    x, y = POS[i]
    if not visible:
        return ('<circle cx="%d" cy="%d" r="%.1f" fill="none" stroke="var(--rule)" '
                'stroke-width="1.5" stroke-dasharray="3 3"/>'
                '<text x="%d" y="%d" class="nlabel hidden">%s</text>'
                % (x, y, R, x, y + 4, LABEL[i]))
    fill = {"a": "var(--priv-a)", "b": "var(--priv-b)", "shared": "var(--shared)"}[role]
    return ('<circle cx="%d" cy="%d" r="%.1f" fill="%s" stroke="var(--ink)" '
            'stroke-width="1.2" stroke-opacity=".25"/>'
            '<text x="%d" y="%d" class="nlabel">%s</text>'
            % (x, y, R, fill, x, y + 4, LABEL[i]))


def role_of(i, topo):
    if i in topo["a_private"]:
        return "a"
    if i in topo["b_private"]:
        return "b"
    return "shared"


def panel(adjacency, topo, visible: Optional[List[int]] = None,
          truth: Optional[np.ndarray] = None,
          confounded: Optional[List[List[int]]] = None,
          true_confounded: Optional[List[List[int]]] = None) -> str:
    """One graph. If `truth` is given, edges are coloured by agreement with it."""
    adjacency = np.asarray(adjacency)
    d = adjacency.shape[0]
    visible = list(range(d)) if visible is None else visible
    parts = ['<svg viewBox="0 0 %d %d" class="graph" role="img">' % VB]

    truth_arr = None if truth is None else np.asarray(truth)
    for u in range(d):
        for v in range(d):
            if u == v:
                continue
            has = bool(adjacency[u, v])
            in_truth = bool(truth_arr[u, v]) if truth_arr is not None else has
            if not has and not in_truth:
                continue
            if truth_arr is None:
                parts.append(directed_edge(u, v, "var(--edge)"))
            elif has and in_truth:
                parts.append(directed_edge(u, v, "var(--ok)"))
            elif has and not in_truth:
                parts.append(directed_edge(u, v, "var(--wrong)", width=2.2))
            else:
                parts.append(directed_edge(u, v, "var(--missing)", dashed=True, width=1.6))

    tc = {tuple(sorted(p)) for p in (true_confounded or [])}
    for pair in (confounded or []):
        key = tuple(sorted(pair))
        colour = "var(--conf-ok)" if key in tc else "var(--wrong)"
        parts.append(bidirected_edge(pair[0], pair[1], colour))
    # A confounded pair the agent MISSED is as informative as one it invented.
    for pair in tc - {tuple(sorted(p)) for p in (confounded or [])}:
        parts.append(bidirected_edge(pair[0], pair[1], "var(--missing)"))

    for i in range(d):
        parts.append(node(i, visible=i in visible, role=role_of(i, topo)))
    parts.append("</svg>")
    return "".join(parts)


def moves_strip(moves, topo) -> str:
    """The move sequence, so clamping behaviour is visible rather than inferred."""
    names = {0: "P_A", 1: "P_B", 2: "X1", 3: "X2", 4: "X3"}
    rows = []
    for agent in ("A", "B"):
        cells = []
        for m in moves:
            entry = m[agent]
            if entry["node"] is None:
                cells.append('<span class="mv pass">&mdash;</span>')
            else:
                cls = "clamp" if entry["mode"] == "clamp" else "vary"
                cells.append('<span class="mv %s">%s<i>%s</i></span>'
                             % (cls, names[entry["node"]], entry["mode"][0]))
        rows.append('<div class="mrow"><b>%s</b>%s</div>' % (agent, "".join(cells)))
    return '<div class="moves">%s</div>' % "".join(rows)


def case_block(ex, i) -> str:
    topo = ex["topology"]
    truth = np.asarray(ex["true_adjacency"])
    A, B = ex["agents"]["A"], ex["agents"]["B"]

    def agent_panel(agent, other_private):
        # The agent's window truth and claim live in WINDOW indices; lift them back to
        # global indices so every panel shares one coordinate system.
        nodes = agent["nodes"]
        g_causal = np.zeros_like(truth)
        g_truth = np.zeros_like(truth)
        for a, u in enumerate(nodes):
            for b, v in enumerate(nodes):
                if a != b:
                    g_causal[u, v] = agent["map_causal"][a][b]
                    g_truth[u, v] = agent["truth"][a][b]
        claimed = [[nodes[p[0]], nodes[p[1]]] for p in agent["claimed_confounded"]]
        true_conf = [[nodes[p[0]], nodes[p[1]]] for p in agent["true_confounded_pairs"]]
        return panel(g_causal, topo, visible=nodes, truth=g_truth,
                     confounded=claimed, true_confounded=true_conf)

    verdict = "solved" if ex["success"] else "not solved"
    tags = []
    tags.append('<span class="tag %s">%s</span>'
                % ("conf" if ex["confounded"] else "unconf",
                   "confounded" if ex["confounded"] else "unconfounded"))
    tags.append('<span class="tag %s">%s</span>'
                % ("good" if ex["success"] else "bad", verdict))

    def stat(agent):
        return ('<div class="stat"><span>credit mass</span><b class="%s">%.3f</b>'
                '<span class="bar"><i style="width:%.0f%%"></i></span></div>'
                % ("good" if agent["credit_ok"] else "bad",
                   agent["credit_mass"], 100 * min(agent["credit_mass"], 1.0)))

    return """
<section class="case">
  <div class="chead"><h3>Episode %d</h3><div>%s</div></div>
  <div class="panels">
    <figure><figcaption>True graph</figcaption>%s
      <p class="cap">What actually generated the data. <b>P<sub>A</sub></b> is A's private
      variable, <b>P<sub>B</sub></b> is B's, <b>X1&ndash;X3</b> are shared.</p></figure>
    <figure><figcaption>Agent A concluded</figcaption>%s%s
      <p class="cap">Dotted circle = a variable A cannot see. %s</p></figure>
    <figure><figcaption>Agent B concluded</figcaption>%s%s
      <p class="cap">Dotted circle = a variable B cannot see. %s</p></figure>
  </div>
  %s
</section>""" % (
        ex["seed"] - 900_000, "".join(tags),
        panel(truth, topo),
        agent_panel(A, 1), stat(A),
        "Exactly right." if A["exact_match"] else
        ("Right up to Markov equivalence." if A["equivalent_to_truth"]
         else "Did not recover the structure."),
        agent_panel(B, 0), stat(B),
        "Exactly right." if B["exact_match"] else
        ("Right up to Markov equivalence." if B["equivalent_to_truth"]
         else "Did not recover the structure."),
        moves_strip(ex["moves"], topo))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", default="results/ma_examples/examples.json")
    ap.add_argument("--template", default="scripts/ma_graph_template.html")
    ap.add_argument("--out", default="results/graph_examples.html")
    args = ap.parse_args(argv)

    data = json.loads(pathlib.Path(args.examples).read_text(encoding="utf-8"))
    blocks = "".join(case_block(ex, i) for i, ex in enumerate(data["examples"]))
    s = data["summary"]
    summary = (
        "Across <b>%d</b> unseen episodes this policy pair solved <b>%.0f%%</b> "
        "&mdash; <b>%.0f%%</b> of the confounded ones and <b>%.0f%%</b> of the "
        "unconfounded ones. It spent <b>%.1f</b> moves per episode, and <b>%.0f%%</b> of "
        "those moves were clamps."
        % (s["n_episodes"], 100 * s["success_rate"], 100 * s["success_when_confounded"],
           100 * s["success_when_unconfounded"], s["mean_moves"],
           100 * s["clamp_fraction"]))

    html = pathlib.Path(args.template).read_text(encoding="utf-8")
    html = html.replace("{{SUMMARY}}", summary).replace("{{CASES}}", blocks)
    pathlib.Path(args.out).write_text(html, encoding="utf-8")
    print("wrote %s  (%d cases, %.1f KB)"
          % (args.out, len(data["examples"]), len(html) / 1024))


if __name__ == "__main__":
    main()
