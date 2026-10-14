"""The SVG renderer (src/eval/svg.py): well-formed XML, and every value placed where it belongs.

The figures go into a printed report, so the tests check what a reader would: that a point sits
where its numbers say, that a cell has the colour of its class and shows its count, and that text is
escaped rather than breaking the file.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from src.eval import svg

NS = "{http://www.w3.org/2000/svg}"
MI = "COND:nstemi_stemi"
UA = "COND:unstable_angina"
SA = "COND:stable_angina"


def parse(text: str) -> ET.Element:
    root = ET.fromstring(text)
    assert root.tag == f"{NS}svg"
    width, height = float(root.get("width")), float(root.get("height"))
    assert root.get("viewBox") == f"0 0 {root.get('width')} {root.get('height')}"
    assert width > 0 and height > 0
    assert root.find(f"{NS}title") is not None and root.find(f"{NS}desc") is not None
    return root


def text_of(root: ET.Element) -> str:
    return " ".join(t.text or "" for t in root.iter(f"{NS}text"))


def row(low: float, cases: int, confidence: float, accuracy: float) -> dict:
    return {
        "bin_low": low,
        "bin_high": low + 0.1,
        "cases": cases,
        "confidence": confidence,
        "accuracy": accuracy,
    }


def center(element: ET.Element) -> tuple[float, float]:
    """A marker's centre, whatever its shape."""
    tag = element.tag.removeprefix(NS)
    if tag == "circle":
        return float(element.get("cx")), float(element.get("cy"))
    if tag == "rect":
        return (
            float(element.get("x")) + float(element.get("width")) / 2,
            float(element.get("y")) + float(element.get("height")) / 2,
        )
    points = [tuple(map(float, p.split(","))) for p in element.get("points").split()]
    xs, ys = zip(*points, strict=True)
    return sum(xs) / 3, (min(ys) * 0.4 + max(ys) * 0.6)  # the triangle's drawn centre


def test_scale_maps_linearly_and_can_flip():
    s = svg.Scale(0, 1, 100, 300)
    assert (s(0), s(0.5), s(1)) == (100, 200, 300)
    flipped = svg.Scale(0, 1, 300, 100)
    assert flipped(0.25) == 250


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        (0.0, 0),
        (0.0099, 0),
        (0.01, 1),
        (0.029, 1),
        (0.03, 2),
        (0.1, 3),
        (0.2999, 3),
        (0.3, 4),
        (1, 4),
    ],
)
def test_share_classes(share, expected):
    assert svg.share_class(share) == expected


def test_compact_counts():
    assert [svg.compact(n) for n in (7, 999, 1000, 2140, 9999, 33512)] == [
        "7",
        "999",
        "1.0k",
        "2.1k",
        "10.0k",
        "34k",
    ]


def reliability(rows: list[dict], level: str = "1.0") -> ET.Element:
    return parse(
        svg.reliability_figure(
            [{"level": level, "rows": rows, "ece": 0.0123, "ci": [0.01, 0.015]}],
            title="Model X: top-label reliability",
            subtitle="Raw scores.",
            notes="Validation; DDXPlus is synthetic.",
        )
    )


def test_reliability_points_sit_where_their_numbers_say():
    root = reliability([row(0.2, 50, 0.25, 0.75), row(0.4, 50, 0.5, 0.5), row(0.7, 50, 0.75, 0.25)])
    points = [e for e in root.iter() if e.get("class") == "point"]
    assert len(points) == 3
    (x1, y1), (x2, y2), (x3, y3) = (center(p) for p in points)
    # Equal steps in the data are equal steps on the page; y grows downwards in SVG.
    assert x2 - x1 == pytest.approx(x3 - x2, abs=0.25)
    assert y1 < y2 < y3
    assert (x3 - x1) == pytest.approx(y3 - y1, abs=0.25)  # a square plot
    # The middle point, (0.5, 0.5), lies on the diagonal of perfect calibration.
    (diagonal,) = [e for e in root.iter(f"{NS}line") if e.get("class") == "diagonal"]
    ax, ay = float(diagonal.get("x1")), float(diagonal.get("y1"))
    bx, by = float(diagonal.get("x2")), float(diagonal.get("y2"))
    assert x2 == pytest.approx((ax + bx) / 2, abs=0.25)
    assert y2 == pytest.approx((ay + by) / 2, abs=0.25)


def test_reliability_shows_the_ece_with_its_interval_counts_and_titles():
    root = reliability([row(0.9, 33512, 0.97, 0.96), row(0.5, 7, 0.55, 0.4)])
    text = text_of(root)
    assert "ECE 0.0123 [0.0100, 0.0150]" in text
    assert "34k" in text and "7" in text  # the bin counts above the bars
    assert "100% evidence" in text
    titles = [t.text for t in root.iter(f"{NS}title")]
    assert any("33,512 patients" in (t or "") for t in titles)
    points = [e for e in root.iter() if e.get("class") == "point"]
    # Fewer than 20 patients: hollow (white inside, outlined in the level's colour).
    small = [p for p in points if p.get("fill") == svg.SURFACE]
    assert len(small) == 1 and small[0].get("stroke") == svg.LEVELS["1.0"]["colour"]


def test_each_level_has_its_own_colour_and_shape():
    shapes = {}
    for level in ("1.0", "0.5", "0.25"):
        root = reliability([row(0.4, 50, 0.45, 0.4)], level)
        (point,) = [e for e in root.iter() if e.get("class") == "point"]
        assert point.get("fill") == svg.LEVELS[level]["colour"]
        shapes[level] = point.tag.removeprefix(NS)
    assert shapes == {"1.0": "circle", "0.5": "rect", "0.25": "polygon"}


def test_text_is_escaped():
    text = svg.reliability_figure(
        [{"level": "0.5", "rows": [row(0.1, 30, 0.15, 0.2)], "ece": 0.0, "ci": [0.0, 0.0]}],
        title='A <b> & "quoted" title',
        subtitle="x < y",
        notes="a & b",
    )
    root = parse(text)
    assert root.find(f"{NS}title").text == 'A <b> & "quoted" title'
    assert "x < y" in text_of(root)


def test_reliability_grid_draws_every_condition_and_level():
    series = {
        "1.0": {"rows": [row(0.9, 100, 0.95, 0.96)], "ece": 0.001, "ci": [0.0005, 0.002]},
        "0.25": {
            "rows": [row(0.0, 900, 0.02, 0.01), row(0.5, 12, 0.55, 0.5)],
            "ece": 0.01,
            "ci": [0.008, 0.012],
        },
    }
    panels = [{"label": name, "series": series} for name in ("MI *", "GERD", "PE *")]
    root = parse(svg.reliability_grid(panels, title="Grid", subtitle="s", notes="n", columns=2))
    points = [e for e in root.iter() if e.get("class") == "point"]
    assert len(points) == 3 * 3
    text = text_of(root)
    assert text.count("ECE 0.0010 [0.0005, 0.0020]") == 3
    assert text.count("ECE 0.0100 [0.0080, 0.0120]") == 3
    assert "MI *" in text and "PE *" in text


def confusion(matrix: list[list[int]]) -> ET.Element:
    return parse(
        svg.confusion_figure(
            matrix,
            rows=[MI, UA],
            columns=[MI, UA, SA],
            row_labels={MI: "MI *", UA: "Unstable angina *"},
            column_codes={MI: "MI", UA: "UA", SA: "SA"},
            title="Confusions",
            subtitle="s",
            notes="n",
        )
    )


def cells(root: ET.Element) -> dict[tuple[int, int], ET.Element]:
    return {
        (int(e.get("data-r")), int(e.get("data-c"))): e
        for e in root.iter(f"{NS}rect")
        if e.get("class") == "cell"
    }


def test_confusion_cells_take_their_class_colour_and_show_their_count():
    # MI row: 90 right, 2 to UA (2.0%: class 1), 8 to SA (8%: class 2). UA row: 50 to MI (50%).
    root = confusion([[90, 2, 8], [50, 50, 0]])
    grid = cells(root)
    assert len(grid) == 6
    assert grid[(0, 0)].get("fill") == svg.DIAGONAL_CELL
    assert grid[(0, 1)].get("fill") == svg.SEQUENTIAL[1]
    assert grid[(0, 2)].get("fill") == svg.SEQUENTIAL[2]
    assert grid[(1, 0)].get("fill") == svg.SEQUENTIAL[4]
    assert grid[(1, 2)].get("fill") == svg.EMPTY_CELL
    text = text_of(root)
    assert "90.0%" in text and "50.0%" in text  # the diagonal: top-1 recall
    assert re.search(r"\b8\b", text) and re.search(r"\b2\b", text)
    assert "100" in text  # each row's total
    # A dark cell's count is white, a light one's ink.
    counts = {t.text: t.get("fill") for t in root.iter(f"{NS}text")}
    assert counts["50"] == svg.SURFACE
    assert counts["2"] == svg.INK


def test_confusion_cells_are_laid_out_by_row_and_column():
    grid = cells(confusion([[1, 2, 3], [4, 5, 6]]))
    x = {c: float(grid[(0, c)].get("x")) for c in range(3)}
    y = {r: float(grid[(r, 0)].get("y")) for r in range(2)}
    assert x[0] < x[1] < x[2] and x[1] - x[0] == pytest.approx(x[2] - x[1])
    assert y[0] < y[1]


def test_recall_dots_are_placed_on_a_shared_axis():
    def rows(value: float) -> dict:
        return {
            MI: {
                "1.0": {"value": 1.0, "ci": [1.0, 1.0]},
                "0.25": {"value": value, "ci": [value - 0.02, value + 0.02]},
            },
            UA: {"1.0": {"value": 0.9, "ci": [0.88, 0.92]}},
        }

    root = parse(
        svg.recall_figure(
            [{"label": "A", "rows": rows(0.8)}, {"label": "B", "rows": rows(0.6)}],
            conditions=[MI, UA],
            labels={MI: "MI *", UA: "UA *"},
            title="Recall",
            subtitle="s",
            notes="n",
        )
    )
    dots = [e for e in root.iter() if e.get("class") == "dot"]
    assert len(dots) == 6
    titles = [d.find(f"{NS}title").text for d in dots]
    assert "A, MI *, 25% evidence: 0.800 [0.780, 0.820]" in titles
    by_title = {t: center(d) for t, d in zip(titles, dots, strict=True)}
    a_one = by_title["A, MI *, 100% evidence: 1.000 [1.000, 1.000]"]
    a_low = by_title["A, MI *, 25% evidence: 0.800 [0.780, 0.820]"]
    a_ua = by_title["A, UA *, 100% evidence: 0.900 [0.880, 0.920]"]
    assert a_low[0] < a_ua[0] < a_one[0]
    # Linear: 0.9 is halfway between 0.8 and 1.0.
    assert a_ua[0] - a_low[0] == pytest.approx(a_one[0] - a_ua[0], abs=0.2)
    # The axis starts at the lowest interval end rounded down to 0.05: 0.58 → 0.55.
    ticks = [
        float(t.text) for t in root.iter(f"{NS}text") if re.fullmatch(r"0\.\d+|1", t.text or "")
    ]
    assert min(ticks) >= 0.55 and max(ticks) == 1.0
    assert a_one[1] < a_ua[1]  # MI's row above UA's


def test_markers_refuse_an_unknown_shape():
    canvas = svg._Canvas(10, 10, "t", "d")
    with pytest.raises(ValueError, match="shape"):
        canvas.marker("star", 1, 1, "#000000")
