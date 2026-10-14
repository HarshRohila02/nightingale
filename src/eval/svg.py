"""Report figures written straight to SVG, with no plotting library (3e, EXP-023).

matplotlib is not installed, and the laptop takes no installs (D-4), so the few figures the report
needs are drawn here as plain SVG text: reliability diagrams, a confusion heatmap and a dot chart of
per-condition recall. They are static figures for a printed report, so each one carries everything a
reader needs inside it: the model label, the evidence level, the split, "DDXPlus is synthetic", and
how to read the marks. Every data mark also carries an SVG ``<title>`` with its exact values, which a
browser shows on hover and a screen reader reads.

Design, following the dataviz method (validated, not eyeballed):

* **The three evidence levels** take the first three slots of the reference categorical palette,
  blue, orange and aqua, which pass every colour-vision check on all pairs (`validate_palette.js`,
  light mode, on white: worst CVD ΔE 9.2, normal-vision 24.0). Aqua is below 3:1 contrast on white,
  so the levels are never told apart by colour alone: each also has its own marker shape (circle,
  square, triangle), and a legend or a text label names it.
* **Magnitude** (a confusion share) is one hue, blue, light to dark, in five classes with a key.
* **Text is ink**, never a series colour; grid and axes are solid hairlines; the background is
  white.

Nothing here reads data files or computes a metric: each function takes aggregates and returns the
SVG as a string.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

__all__ = [
    "LEVELS",
    "Scale",
    "confusion_figure",
    "recall_figure",
    "reliability_figure",
    "reliability_grid",
    "share_class",
]

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EMPTY_CELL = "#f5f4f1"
DIAGONAL_CELL = "#dcdbd5"

LEVELS: dict[str, dict[str, str]] = {
    "1.0": {"colour": "#2a78d6", "shape": "circle", "name": "100%"},
    "0.5": {"colour": "#eb6834", "shape": "square", "name": "50%"},
    "0.25": {"colour": "#1baf7a", "shape": "triangle", "name": "25%"},
}
"""Each evidence level's colour (categorical slots 1–3), marker shape and name."""

SEQUENTIAL = ("#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b")
"""Blue steps 150, 300, 450, 550, 700 of the reference sequential ramp, lightest first."""
SHARE_EDGES = (0.01, 0.03, 0.10, 0.30)
"""The confusion shares' class boundaries: under 1%, 1–3%, 3–10%, 10–30%, 30% and over."""
SHARE_NAMES = ("< 1%", "1–3%", "3–10%", "10–30%", "≥ 30%")
DARK_FROM = 2
"""Classes from this one on are dark enough to need white text."""
MIN_POINT_CASES = 20
"""A reliability point over fewer patients than this is drawn hollow: it says little."""


@dataclass(frozen=True)
class Scale:
    """A linear map from a data interval to a pixel interval (``r1 < r0`` flips it, as SVG's y
    axis needs)."""

    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, value: float) -> float:
        return self.r0 + (value - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def share_class(share: float) -> int:
    """The colour class of a confusion share (0–4); 0 for any share under 1%, including 0."""
    return sum(share >= edge for edge in SHARE_EDGES)


def _n(value: float) -> str:
    """A coordinate with one decimal, without a trailing ``.0``: it keeps the files small."""
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _e(text: str) -> str:
    return escape(str(text), {'"': "&quot;"})


def _width(text: str, size: float) -> float:
    """An estimate of a text's rendered width in the system sans: enough to lay out labels."""
    return len(text) * size * 0.56


def _wrap(text: str, width: float, size: float) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if line and _width(candidate, size) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def compact(count: int) -> str:
    """A bin count short enough to sit above a 20-pixel bar: 53, 2.1k, 33k."""
    if count < 1000:
        return str(count)
    if count < 10_000:
        return f"{count / 1000:.1f}k"
    return f"{round(count / 1000)}k"


class _Canvas:
    """The SVG under construction: a list of element strings and a fixed size."""

    def __init__(self, width: float, height: float, title: str, description: str) -> None:
        self.width = math.ceil(width)
        self.height = math.ceil(height)
        self.title = title
        self.description = description
        self.parts: list[str] = []

    @staticmethod
    def _close(tag: str, attributes: str, title: str | None) -> str:
        if title is None:
            return f"<{tag} {attributes}/>"
        return f"<{tag} {attributes}><title>{_e(title)}</title></{tag}>"

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        *,
        title: str | None = None,
        extra: str = "",
    ) -> None:
        attributes = f'x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" fill="{fill}"'
        self.parts.append(self._close("rect", f"{attributes}{extra}", title))

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        width: float = 1,
        *,
        extra: str = "",
    ) -> None:
        self.parts.append(
            f'<line x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" stroke="{stroke}" '
            f'stroke-width="{width:g}"{extra}/>'
        )

    def polyline(
        self, points: Sequence[tuple[float, float]], stroke: str, width: float = 2
    ) -> None:
        if len(points) < 2:
            return
        path = " ".join(f"{_n(x)},{_n(y)}" for x, y in points)
        self.parts.append(
            f'<polyline points="{path}" fill="none" stroke="{stroke}" stroke-width="{width:g}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = 11,
        fill: str = INK,
        anchor: str = "start",
        bold: bool = False,
        rotate: float | None = None,
        extra: str = "",
    ) -> None:
        attributes = f'x="{_n(x)}" y="{_n(y)}" font-size="{size:g}" fill="{fill}"'
        if anchor != "start":
            attributes += f' text-anchor="{anchor}"'
        if bold:
            attributes += ' font-weight="600"'
        if rotate is not None:
            attributes += f' transform="rotate({rotate:g} {_n(x)} {_n(y)})"'
        self.parts.append(f"<text {attributes}{extra}>{_e(text)}</text>")

    def marker(
        self,
        shape: str,
        x: float,
        y: float,
        colour: str,
        *,
        size: float = 8,
        hollow: bool = False,
        title: str | None = None,
        extra: str = "",
    ) -> None:
        """A data point: filled with a 1.5-pixel white ring that keeps overlapping points apart,
        or hollow (white inside, outlined in its colour)."""
        fill, stroke = (SURFACE, colour) if hollow else (colour, SURFACE)
        paint = f' fill="{fill}" stroke="{stroke}" stroke-width="1.5"{extra}'
        half = size / 2
        if shape == "circle":
            body = f'cx="{_n(x)}" cy="{_n(y)}" r="{_n(half)}"{paint}'
            self.parts.append(self._close("circle", body, title))
        elif shape == "square":
            s = size * 0.88
            body = (
                f'x="{_n(x - s / 2)}" y="{_n(y - s / 2)}" width="{_n(s)}" height="{_n(s)}"{paint}'
            )
            self.parts.append(self._close("rect", body, title))
        elif shape == "triangle":
            h = size * 1.05
            points = (
                f"{_n(x)},{_n(y - h * 0.6)} {_n(x + h * 0.58)},{_n(y + h * 0.4)} "
                f"{_n(x - h * 0.58)},{_n(y + h * 0.4)}"
            )
            self.parts.append(self._close("polygon", f'points="{points}"{paint}', title))
        else:
            raise ValueError(f"unknown marker shape {shape!r}")

    def paragraph(
        self, x: float, y: float, text: str, width: float, *, size: float = 10, fill: str = INK_2
    ) -> float:
        """Wrapped text from ``y`` (the first baseline) down; returns the next free baseline."""
        step = size * 1.35
        for line in _wrap(text, width, size):
            self.text(x, y, line, size=size, fill=fill)
            y += step
        return y

    def render(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="t d" '
            f'font-family="{FONT}">'
            f'<title id="t">{_e(self.title)}</title><desc id="d">{_e(self.description)}</desc>'
            f'<rect width="100%" height="100%" fill="{SURFACE}"/>'
        )
        return head + "".join(self.parts) + "</svg>\n"


def _header(
    canvas: _Canvas, title: str, subtitle: str, width: float, x: float = 16, y: float = 24
) -> float:
    """The figure's title and subtitle; returns the next free baseline."""
    canvas.text(x, y, title, size=14, bold=True)
    return canvas.paragraph(x, y + 19, subtitle, width, size=11, fill=INK_2) + 4


def _level_legend(canvas: _Canvas, x: float, y: float, levels: Sequence[str]) -> None:
    for level in levels:
        style = LEVELS[level]
        canvas.marker(style["shape"], x + 4, y - 4, style["colour"])
        label = f"{style['name']} evidence"
        canvas.text(x + 13, y, label, size=11, fill=INK)
        x += 13 + _width(label, 11) + 18


def _reliability_axes(
    canvas: _Canvas,
    x0: float,
    y0: float,
    size: float,
    ticks: Sequence[float],
    *,
    tick_size: float = 9.5,
) -> tuple[Scale, Scale]:
    """A square 0–1 by 0–1 plot with hairline grid, axes and the diagonal of perfect calibration."""
    sx = Scale(0, 1, x0, x0 + size)
    sy = Scale(0, 1, y0 + size, y0)
    for t in ticks:
        canvas.line(sx(t), y0, sx(t), y0 + size, GRID)
        canvas.line(x0, sy(t), x0 + size, sy(t), GRID)
        label = f"{t:g}"
        canvas.text(
            sx(t), y0 + size + tick_size + 3, label, size=tick_size, fill=MUTED, anchor="middle"
        )
        canvas.text(
            x0 - 4, sy(t) + tick_size * 0.35, label, size=tick_size, fill=MUTED, anchor="end"
        )
    canvas.line(x0, y0 + size, x0 + size, y0 + size, AXIS)
    canvas.line(x0, y0, x0, y0 + size, AXIS)
    canvas.line(sx(0), sy(0), sx(1), sy(1), MUTED, 1, extra=' class="diagonal"')
    return sx, sy


def _reliability_series(
    canvas: _Canvas,
    sx: Scale,
    sy: Scale,
    rows: Sequence[Mapping[str, float]],
    level: str,
    what: str,
    *,
    size: float = 8,
) -> None:
    style = LEVELS[level]
    points = [(sx(r["confidence"]), sy(r["accuracy"])) for r in rows]
    canvas.polyline(points, style["colour"], 2)
    for r, (x, y) in zip(rows, points, strict=True):
        title = (
            f"{what}, {style['name']} evidence, bin {r['bin_low']:.1f}–{r['bin_high']:.1f}: "
            f"{r['cases']:,} patients, mean probability {r['confidence']:.3f}, "
            f"observed {r['accuracy']:.3f}"
        )
        canvas.marker(
            style["shape"],
            x,
            y,
            style["colour"],
            size=size,
            hollow=r["cases"] < MIN_POINT_CASES,
            title=title,
            extra=' class="point"',
        )


def _ece_text(ece: float, ci: Sequence[float]) -> str:
    return f"ECE {ece:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def reliability_figure(
    panels: Sequence[Mapping[str, Any]],
    *,
    title: str,
    subtitle: str,
    notes: str,
    what: str = "Top-ranked condition",
) -> str:
    """Reliability diagrams side by side, one panel per evidence level, each with its bin counts.

    Args:
        panels: Each ``{"level": "1.0" | "0.5" | "0.25", "rows": reliability table,
            "ece": float, "ci": [low, high], "cases": int}``. A table's rows are
            :func:`~src.eval.metrics.reliability_rows`'s: ``bin_low``, ``bin_high``, ``cases``,
            ``confidence`` and ``accuracy``, empty bins left out.
        title: The figure's title; name the model.
        subtitle: What the scores are and on which patients.
        notes: How to read it, printed under the panels.
        what: The probability's name in each point's hover text.
    """
    size, gap, left, strip = 210.0, 56.0, 64.0, 34.0
    width = left + len(panels) * size + (len(panels) - 1) * gap + 28
    canvas = _Canvas(width, 10, title, f"{subtitle} {notes}")
    y = _header(canvas, title, subtitle, width - 32)
    y0 = y + 40
    biggest = max((r["cases"] for p in panels for r in p["rows"]), default=1)
    for i, panel in enumerate(panels):
        x0 = left + i * (size + gap)
        style = LEVELS[panel["level"]]
        canvas.marker(style["shape"], x0 + 4, y0 - 30, style["colour"])
        canvas.text(x0 + 13, y0 - 26, f"{style['name']} evidence", size=12, bold=True)
        canvas.text(x0, y0 - 9, _ece_text(panel["ece"], panel["ci"]), size=11, fill=INK_2)
        sx, sy = _reliability_axes(canvas, x0, y0, size, (0, 0.2, 0.4, 0.6, 0.8, 1))
        _reliability_series(canvas, sx, sy, panel["rows"], panel["level"], what)
        canvas.text(
            x0 + size / 2,
            y0 + size + 28,
            "mean probability given",
            size=10,
            fill=INK_2,
            anchor="middle",
        )
        if i == 0:
            canvas.text(
                x0 - 40,
                y0 + size / 2,
                "observed share correct",
                size=10,
                fill=INK_2,
                anchor="middle",
                rotate=-90,
            )
        # Patients per bin: bar height on a log scale, the count above it.
        base = y0 + size + 44 + strip
        slot = size / 10
        for r in panel["rows"]:
            b = min(int(r["bin_low"] * 10 + 1e-9), 9)
            h = max(1.5, strip * math.log10(r["cases"] + 1) / math.log10(biggest + 1))
            canvas.rect(
                x0 + b * slot + 1,
                base - h,
                slot - 2,
                h,
                style["colour"],
                title=f"{style['name']} evidence, bin {r['bin_low']:.1f}–{r['bin_high']:.1f}: "
                f"{r['cases']:,} patients",
                extra=' class="count"',
            )
            canvas.text(
                x0 + b * slot + slot / 2,
                base - h - 3,
                compact(int(r["cases"])),
                size=8.5,
                fill=INK_2,
                anchor="middle",
            )
        canvas.line(x0, base, x0 + size, base, AXIS)
        if i == 0:
            canvas.text(x0 - 6, base - 2, "patients", size=9, fill=MUTED, anchor="end")
            canvas.text(x0 - 6, base + 9, "per bin (log)", size=9, fill=MUTED, anchor="end")
    y = canvas.paragraph(16, y0 + size + 44 + strip + 26, notes, width - 32)
    canvas.height = math.ceil(y)
    return canvas.render()


def reliability_grid(
    panels: Sequence[Mapping[str, Any]],
    *,
    title: str,
    subtitle: str,
    notes: str,
    columns: int = 5,
) -> str:
    """Small multiples of one-vs-rest reliability, one panel per condition, the levels overlaid.

    Args:
        panels: Each ``{"label": condition name, "series": {level: {"rows", "ece", "ci"}}}``.
            Under each panel, every level's ECE is printed with its interval, which doubles as
            the legend.
    """
    size, cell_w, left = 132.0, 196.0, 52.0
    rows_n = math.ceil(len(panels) / columns)
    levels = [lv for lv in LEVELS if any(lv in p["series"] for p in panels)]
    cell_h = 20 + size + 18 + 13 * len(levels) + 22
    width = left + columns * cell_w
    canvas = _Canvas(width, 10, title, f"{subtitle} {notes}")
    y = _header(canvas, title, subtitle, width - 32)
    _level_legend(canvas, 16, y + 6, levels)
    top = y + 34
    for i, panel in enumerate(panels):
        x0 = left + (i % columns) * cell_w
        y0 = top + (i // columns) * cell_h + 20
        canvas.text(x0, y0 - 8, panel["label"], size=11, bold=True)
        sx, sy = _reliability_axes(canvas, x0, y0, size, (0, 0.5, 1), tick_size=9)
        for level in levels:
            if level in panel["series"]:
                _reliability_series(
                    canvas, sx, sy, panel["series"][level]["rows"], level, panel["label"], size=7
                )
        ty = y0 + size + 28
        for level in levels:
            if level not in panel["series"]:
                continue
            style, entry = LEVELS[level], panel["series"][level]
            canvas.marker(style["shape"], x0 + 3, ty - 3.5, style["colour"], size=6.5)
            canvas.text(x0 + 11, ty, f"{style['name']}", size=9.5, fill=INK_2)
            canvas.text(x0 + 38, ty, _ece_text(entry["ece"], entry["ci"]), size=9.5, fill=INK_2)
            ty += 13
    y = canvas.paragraph(16, top + rows_n * cell_h + 4, notes, width - 32)
    canvas.height = math.ceil(y)
    return canvas.render()


def confusion_figure(
    matrix: Sequence[Sequence[int]],
    *,
    rows: Sequence[str],
    columns: Sequence[str],
    row_labels: Mapping[str, str],
    column_codes: Mapping[str, str],
    title: str,
    subtitle: str,
    notes: str,
) -> str:
    """A confusion heatmap: true condition (row) against first-ranked condition (column).

    Off the diagonal, a cell's colour is the share of the row's patients ranked first as that
    column, in five classes of one blue (:data:`SHARE_EDGES`), and its text is the count. The
    diagonal is grey, with the row's top-1 recall. A row's total is printed at its right.
    """
    cw, ch, label_w = 34.0, 24.0, 172.0
    x0 = 16 + label_w
    width = x0 + len(columns) * cw + 64
    canvas = _Canvas(width, 10, title, f"{subtitle} {notes}")
    y = _header(canvas, title, subtitle, width - 32)
    canvas.text(
        x0 + len(columns) * cw / 2, y + 12, "ranked first", size=10, fill=INK_2, anchor="middle"
    )
    y0 = y + 34
    for j, column in enumerate(columns):
        canvas.text(
            x0 + j * cw + cw / 2, y0 - 6, column_codes[column], size=9.5, fill=INK, anchor="middle"
        )
    canvas.text(x0 + len(columns) * cw + 8, y0 - 6, "patients", size=9.5, fill=INK_2)
    canvas.text(16, y0 - 6, "true condition", size=10, fill=INK_2)
    for i, row in enumerate(rows):
        total = int(sum(matrix[i]))
        ry = y0 + i * ch
        canvas.text(16, ry + ch / 2 + 4, row_labels[row], size=10.5)
        canvas.text(x0 + len(columns) * cw + 8, ry + ch / 2 + 4, f"{total:,}", size=9.5, fill=INK_2)
        for j, column in enumerate(columns):
            count = int(matrix[i][j])
            share = count / total if total else 0.0
            cell = f' class="cell" data-r="{i}" data-c="{j}"'
            label = (
                f"{row_labels[row]} ranked first as {column_codes[column]}: {count:,} of {total:,}"
            )
            cx, cy = x0 + j * cw + 1, ry + 1
            if row == column:
                canvas.rect(
                    cx,
                    cy,
                    cw - 2,
                    ch - 2,
                    DIAGONAL_CELL,
                    title=f"{label} ({share:.1%}, top-1 recall)",
                    extra=cell,
                )
                canvas.text(
                    cx + (cw - 2) / 2, cy + ch / 2 + 2.5, f"{share:.1%}", size=8.5, anchor="middle"
                )
            elif count == 0:
                canvas.rect(cx, cy, cw - 2, ch - 2, EMPTY_CELL, extra=cell)
            else:
                k = share_class(share)
                canvas.rect(
                    cx,
                    cy,
                    cw - 2,
                    ch - 2,
                    SEQUENTIAL[k],
                    title=f"{label} ({share:.2%})",
                    extra=cell,
                )
                canvas.text(
                    cx + (cw - 2) / 2,
                    cy + ch / 2 + 2.5,
                    f"{count:,}",
                    size=8.5,
                    fill=SURFACE if k >= DARK_FROM else INK,
                    anchor="middle",
                )
    # The key: the five share classes and the diagonal.
    ky = y0 + len(rows) * ch + 22
    canvas.text(16, ky, "Off the diagonal, the share of the row's patients:", size=10, fill=INK_2)
    kx = 16 + _width("Off the diagonal, the share of the row's patients:", 10) + 10
    for k, name in enumerate(SHARE_NAMES):
        canvas.rect(kx, ky - 10, 18, 12, SEQUENTIAL[k])
        canvas.text(kx + 22, ky, name, size=10, fill=INK_2)
        kx += 22 + _width(name, 10) + 12
    canvas.rect(16, ky + 8, 18, 12, DIAGONAL_CELL)
    canvas.text(
        38, ky + 18, "diagonal: the true condition ranked first (top-1 recall)", size=10, fill=INK_2
    )
    y = canvas.paragraph(16, ky + 40, notes, width - 32)
    canvas.height = math.ceil(y)
    return canvas.render()


def _ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    step = 0.05 if span <= 0.25 else 0.1 if span <= 0.5 else 0.2
    first = math.ceil(lo / step - 1e-9) * step
    out, t = [], first
    while t <= hi + 1e-9:
        out.append(round(t, 4))
        t += step
    return out


def recall_figure(
    panels: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    labels: Mapping[str, str],
    title: str,
    subtitle: str,
    notes: str,
    lowest: float | None = None,
) -> str:
    """A dot chart of per-condition recall: a row per condition, a panel per system, a marker per
    evidence level with its 95% interval as a horizontal line.

    Args:
        panels: Each ``{"label": system name, "rows": {condition: {level: {"value", "ci"}}}}``.
        conditions: The rows, top to bottom.
        labels: Each condition's printed name.
        lowest: The left end of the shared x axis; by default the lowest interval end, rounded
            down to 0.05. Every panel shares the axis, so the systems can be compared.
    """
    if lowest is None:
        ends = [
            entry["ci"][0]
            for p in panels
            for by_level in p["rows"].values()
            for entry in by_level.values()
        ]
        lowest = max(0.0, math.floor(min(ends, default=0.0) / 0.05) * 0.05)
    lowest = min(lowest, 0.95)
    label_w, pw, gap, rh = 178.0, 196.0, 22.0, 27.0
    x_start = 16 + label_w
    width = x_start + len(panels) * pw + (len(panels) - 1) * gap + 20
    levels = [
        lv
        for lv in LEVELS
        if any(lv in by_level for p in panels for by_level in p["rows"].values())
    ]
    canvas = _Canvas(width, 10, title, f"{subtitle} {notes}")
    y = _header(canvas, title, subtitle, width - 32)
    _level_legend(canvas, 16, y + 6, levels)
    y0 = y + 46
    ticks = _ticks(lowest, 1.0)
    offsets = {lv: (k - (len(levels) - 1) / 2) * 7 for k, lv in enumerate(levels)}
    plot_h = len(conditions) * rh
    for i, condition in enumerate(conditions):
        canvas.text(16, y0 + i * rh + rh / 2 + 4, labels[condition], size=10.5)
    for p_i, panel in enumerate(panels):
        px = x_start + p_i * (pw + gap)
        sx = Scale(lowest, 1.0, px + 6, px + pw - 6)
        canvas.text(px + pw / 2, y0 - 10, panel["label"], size=11.5, bold=True, anchor="middle")
        for t in ticks:
            canvas.line(sx(t), y0, sx(t), y0 + plot_h, GRID)
            canvas.text(sx(t), y0 + plot_h + 13, f"{t:g}", size=9, fill=MUTED, anchor="middle")
        canvas.line(px, y0 + plot_h, px + pw, y0 + plot_h, AXIS)
        for i, condition in enumerate(conditions):
            cy = y0 + i * rh + rh / 2
            if i:
                canvas.line(px, y0 + i * rh, px + pw, y0 + i * rh, EMPTY_CELL)
            for level in levels:
                entry = panel["rows"].get(condition, {}).get(level)
                if entry is None:
                    continue
                style = LEVELS[level]
                yy = cy + offsets[level]
                lo, hi = (max(lowest, v) for v in entry["ci"])
                canvas.line(sx(lo), yy, sx(hi), yy, style["colour"], 1.5)
                canvas.marker(
                    style["shape"],
                    sx(max(lowest, entry["value"])),
                    yy,
                    style["colour"],
                    size=7,
                    title=f"{panel['label']}, {labels[condition]}, {style['name']} evidence: "
                    f"{entry['value']:.3f} [{entry['ci'][0]:.3f}, {entry['ci'][1]:.3f}]",
                    extra=f' class="dot" data-r="{i}"',
                )
    y = canvas.paragraph(16, y0 + plot_h + 34, notes, width - 32)
    canvas.height = math.ceil(y)
    return canvas.render()
