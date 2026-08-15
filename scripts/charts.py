"""Inline SVG charts for the results write-up.

Hand-rolled rather than matplotlib, for one reason: the report is published as a page that
must render correctly in both light and dark themes, and a rasterised PNG carries its
background with it. These emit SVG that draws its colours from CSS custom properties, so
the charts follow the reader's theme instead of fighting it.

Deliberately plain: axes, gridlines, and the marks themselves. Every chart shows the
individual seeds rather than only their summary, because the seed spread IS the result at
this sample size -- a configuration is only as trustworthy as its worst run.
"""
from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence, Tuple

# Colours are CSS variables so the page's theme controls them. Any chart-specific hue
# still needs a light and dark value, which the page defines.
INK = "var(--ink)"
MUTED = "var(--muted)"
GRID = "var(--grid)"
ACCENT = "var(--accent)"
GOOD = "var(--good)"
BAD = "var(--bad)"
WARN = "var(--warn)"


def _esc(text) -> str:
    return html.escape(str(text))


def _text(x, y, content, size=11, fill=MUTED, anchor="middle", weight="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="ui-sans-serif, system-ui, sans-serif">{_esc(content)}</text>')


def _nice_ticks(low: float, high: float, count: int = 5) -> List[float]:
    """Round tick values spanning [low, high]."""
    if high <= low:
        high = low + 1.0
    raw = (high - low) / max(count, 1)
    magnitude = 10 ** int(f"{raw:e}".split("e")[1])
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw),
               default=magnitude)
    start = step * int(low / step) - (step if low < 0 else 0)
    ticks, value = [], start
    while value <= high + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks


def dot_plot(groups: Sequence[Tuple[str, Sequence[float]]],
             *, width: int = 720, row_height: int = 26,
             reference: Optional[float] = None, reference_label: str = "",
             threshold: Optional[float] = None, threshold_label: str = "",
             x_label: str = "", title: str = "",
             highlight: Optional[Dict[str, str]] = None) -> str:
    """Horizontal dot plot: one row per configuration, one dot per seed.

    The individual seeds are the point. A bar chart of means would be a smaller, prettier
    picture of a less honest quantity.
    """
    highlight = highlight or {}
    left, right, top, bottom = 190, 24, 40 if title else 18, 42
    plot_w = width - left - right
    height = top + row_height * len(groups) + bottom

    values = [v for _, vs in groups for v in vs if v == v]
    low = min(values + ([reference] if reference is not None else []) +
              ([threshold] if threshold is not None else []) + [0.0])
    high = max(values + ([reference] if reference is not None else []) +
               ([threshold] if threshold is not None else []) + [1.0])
    pad = (high - low) * 0.06 or 0.1
    low, high = low - pad, high + pad

    def sx(value: float) -> float:
        return left + (value - low) / (high - low) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="{_esc(title or x_label)}">']
    if title:
        parts.append(_text(left, 18, title, size=13, fill=INK, anchor="start", weight="600"))

    for tick in _nice_ticks(low, high):
        x = sx(tick)
        if not (left - 1 <= x <= left + plot_w + 1):
            continue
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                     f'y2="{top + row_height * len(groups)}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(_text(x, top + row_height * len(groups) + 16, f"{tick:g}"))

    for marker, colour, label in ((reference, ACCENT, reference_label),
                                  (threshold, GOOD, threshold_label)):
        if marker is None:
            continue
        x = sx(marker)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" '
                     f'y2="{top + row_height * len(groups)}" stroke="{colour}" '
                     f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        if label:
            parts.append(_text(x, top - 10, label, size=10, fill=colour))

    for i, (name, seed_values) in enumerate(groups):
        y = top + row_height * i + row_height / 2
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1" opacity="0.4"/>')
        parts.append(_text(left - 10, y + 4, name, size=11, fill=INK, anchor="end"))

        finite = [v for v in seed_values if v == v]
        colour = highlight.get(name, ACCENT)
        if finite:
            # The worst seed is drawn solid and larger: it is the number the pass/fail
            # criteria actually use.
            worst = min(finite)
            for v in finite:
                x = max(left, min(left + plot_w, sx(v)))
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}" '
                             f'opacity="0.45"/>')
            xw = max(left, min(left + plot_w, sx(worst)))
            parts.append(f'<circle cx="{xw:.1f}" cy="{y:.1f}" r="5.5" fill="{colour}" '
                         f'stroke="var(--bg)" stroke-width="1.5"/>')
        else:
            parts.append(_text(left + 6, y + 4, "no data", size=10, anchor="start"))

    if x_label:
        parts.append(_text(left + plot_w / 2, height - 6, x_label, size=11))
    parts.append("</svg>")
    return "".join(parts)


def line_chart(series: Sequence[Tuple[str, Sequence[float], Sequence[float], str]],
               *, width: int = 720, height: int = 260,
               x_label: str = "", y_label: str = "", title: str = "",
               y_min: Optional[float] = None, y_max: Optional[float] = None) -> str:
    """Multiple (label, xs, ys, colour) series on shared axes."""
    left, right, top, bottom = 58, 110, 34 if title else 14, 40
    plot_w, plot_h = width - left - right, height - top - bottom

    all_x = [x for _, xs, _, _ in series for x in xs]
    all_y = [y for _, _, ys, _ in series for y in ys if y == y]
    if not all_x or not all_y:
        return ""
    x_lo, x_hi = min(all_x), max(all_x)
    y_lo = min(all_y) if y_min is None else y_min
    y_hi = max(all_y) if y_max is None else y_max
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    def sx(v):
        return left + (v - x_lo) / max(x_hi - x_lo, 1e-9) * plot_w

    def sy(v):
        return top + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="{_esc(title or y_label)}">']
    if title:
        parts.append(_text(left, 16, title, size=13, fill=INK, anchor="start", weight="600"))

    for tick in _nice_ticks(y_lo, y_hi):
        if not (y_lo - 1e-9 <= tick <= y_hi + 1e-9):
            continue
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(_text(left - 8, y + 4, f"{tick:g}", size=10, anchor="end"))
    for tick in _nice_ticks(x_lo, x_hi, 6):
        if not (x_lo - 1e-9 <= tick <= x_hi + 1e-9):
            continue
        parts.append(_text(sx(tick), top + plot_h + 16, f"{tick:g}", size=10))

    for i, (label, xs, ys, colour) in enumerate(series):
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}"
                          for x, y in zip(xs, ys) if y == y)
        if not points:
            continue
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.8" stroke-linejoin="round"/>')
        ly = top + 12 + i * 16
        parts.append(f'<line x1="{left + plot_w + 10}" y1="{ly - 4}" '
                     f'x2="{left + plot_w + 26}" y2="{ly - 4}" stroke="{colour}" '
                     f'stroke-width="2.5"/>')
        parts.append(_text(left + plot_w + 30, ly, label, size=10, anchor="start", fill=INK))

    if y_label:
        parts.append(f'<text x="14" y="{top + plot_h / 2:.1f}" font-size="11" fill="{MUTED}" '
                     f'text-anchor="middle" font-family="ui-sans-serif, system-ui, sans-serif" '
                     f'transform="rotate(-90 14 {top + plot_h / 2:.1f})">{_esc(y_label)}</text>')
    if x_label:
        parts.append(_text(left + plot_w / 2, height - 6, x_label, size=11))
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(bars: Sequence[Tuple[str, float, str]], *, width: int = 720,
              height: int = 240, y_label: str = "", title: str = "",
              value_format: str = "{:.2f}") -> str:
    """Vertical bars: (label, value, colour)."""
    left, right, top, bottom = 54, 16, 34 if title else 14, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for _, v, _ in bars if v == v]
    if not values:
        return ""
    y_hi = max(values + [0.0]) * 1.12 or 1.0
    y_lo = min(values + [0.0]) * 1.12

    def sy(v):
        return top + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h

    slot = plot_w / max(len(bars), 1)
    bar_w = min(slot * 0.62, 54)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="{_esc(title or y_label)}">']
    if title:
        parts.append(_text(left, 16, title, size=13, fill=INK, anchor="start", weight="600"))
    for tick in _nice_ticks(y_lo, y_hi):
        if not (y_lo - 1e-9 <= tick <= y_hi + 1e-9):
            continue
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(_text(left - 8, y + 4, f"{tick:g}", size=10, anchor="end"))

    for i, (label, value, colour) in enumerate(bars):
        cx = left + slot * (i + 0.5)
        y0, y1 = sy(0), sy(value)
        parts.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{min(y0, y1):.1f}" '
                     f'width="{bar_w:.1f}" height="{abs(y1 - y0):.1f}" fill="{colour}" '
                     f'rx="2"/>')
        # Value label above a positive bar, below a negative one -- gap-closed goes well
        # below zero for a failing agent, and a label inside the bar is unreadable.
        label_y = min(y0, y1) - 5 if value >= 0 else max(y0, y1) + 12
        parts.append(_text(cx, label_y, value_format.format(value), size=10, fill=INK))
        parts.append(_text(cx, top + plot_h + 16, label, size=10))

    if y_label:
        parts.append(f'<text x="13" y="{top + plot_h / 2:.1f}" font-size="11" fill="{MUTED}" '
                     f'text-anchor="middle" font-family="ui-sans-serif, system-ui, sans-serif" '
                     f'transform="rotate(-90 13 {top + plot_h / 2:.1f})">{_esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)
