"""Draw the report's exploratory-data-analysis figures from the validation parquet.

Reads ``data/interim/ddxplus_chestpain_validate.parquet`` (the 13 in-scope conditions; validation only, never
the test split) and writes three PNGs and a values file into ``docs/report/bmu/``. Nothing is trained, fitted or
chosen: the figures show counts, quartiles and a histogram of the data as it stands.

    ./.venv/Scripts/python.exe scripts/make_report_eda_figures.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import BY_ID  # noqa: E402

FONT_FILES = ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf")
INK = (30, 30, 30)
GRID = (215, 215, 215)
MUST = (176, 42, 48)
OTHER = (70, 110, 160)
W, H = 1800, 1050


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf") if bold else FONT_FILES
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 36), title, fill=INK, font=font(40, True))
    d.text((60, 90), subtitle, fill=(90, 90, 90), font=font(26))
    return img, d


def text_w(d: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> int:
    box = d.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def class_balance(df: pd.DataFrame, out: Path) -> list[str]:
    counts = df["label_condition_id"].value_counts()
    rows = [(BY_ID[cid].label, int(n), BY_ID[cid].is_must_not_miss) for cid, n in counts.items()]
    img, d = canvas(
        "Patients per condition, validation split",
        f"{len(df):,} synthetic patients, 13 conditions. Red bars are must-not-miss conditions.",
    )
    left, right, top, bottom = 470, W - 150, 150, H - 110
    top_n = max(n for _, n, _ in rows)
    bar_h = (bottom - top) / len(rows)
    for tick in range(0, top_n + 1, 1000):
        x = left + (right - left) * tick / (top_n * 1.05)
        d.line([(x, top), (x, bottom)], fill=GRID, width=2)
        d.text((x - 20, bottom + 12), f"{tick:,}", fill=INK, font=font(22))
    d.text((left, bottom + 55), "Patients", fill=INK, font=font(26))
    for i, (label, n, mnm) in enumerate(rows):
        y = top + i * bar_h
        x1 = left + (right - left) * n / (top_n * 1.05)
        d.rectangle([left, y + 8, x1, y + bar_h - 8], fill=MUST if mnm else OTHER)
        d.text(
            (left - 14 - text_w(d, label, font(25)), y + bar_h / 2 - 15),
            label,
            fill=INK,
            font=font(25),
        )
        d.text((x1 + 10, y + bar_h / 2 - 15), f"{n:,}", fill=INK, font=font(24))
    img.save(out)
    return [
        f"- {label}: {n:,} patients" + (" (must-not-miss)" if m else "") for label, n, m in rows
    ]


def age_by_condition(df: pd.DataFrame, out: Path) -> list[str]:
    stats = []
    for cid, group in df.groupby("label_condition_id"):
        q1, med, q3 = group["age"].quantile([0.25, 0.5, 0.75]).tolist()
        stats.append((BY_ID[cid].label, q1, med, q3, BY_ID[cid].is_must_not_miss))
    stats.sort(key=lambda s: s[2])
    img, d = canvas(
        "Age by condition, validation split",
        "Median (dot) and interquartile range (bar), in years. Ages are generated, not epidemiological.",
    )
    left, right, top, bottom = 470, W - 120, 150, H - 110
    lo, hi = 0, 80
    bar_h = (bottom - top) / len(stats)
    for tick in range(lo, hi + 1, 10):
        x = left + (right - left) * (tick - lo) / (hi - lo)
        d.line([(x, top), (x, bottom)], fill=GRID, width=2)
        d.text((x - 12, bottom + 12), str(tick), fill=INK, font=font(22))
    d.text((left, bottom + 55), "Age (years)", fill=INK, font=font(26))
    lines = []
    for i, (label, q1, med, q3, mnm) in enumerate(stats):
        y = top + i * bar_h + bar_h / 2
        x1 = left + (right - left) * (q1 - lo) / (hi - lo)
        x3 = left + (right - left) * (q3 - lo) / (hi - lo)
        xm = left + (right - left) * (med - lo) / (hi - lo)
        d.rectangle([x1, y - 11, x3, y + 11], fill=MUST if mnm else OTHER)
        d.ellipse([xm - 9, y - 9, xm + 9, y + 9], fill="white", outline=INK, width=3)
        d.text((left - 14 - text_w(d, label, font(25)), y - 15), label, fill=INK, font=font(25))
        lines.append(f"- {label}: median {med:.0f}, interquartile range {q1:.0f} to {q3:.0f}")
    img.save(out)
    return lines


def findings_per_patient(df: pd.DataFrame, out: Path) -> list[str]:
    sizes = df["positive_codes"].map(len)
    hist = Counter(int(s) for s in sizes)
    top_k = max(hist)
    img, d = canvas(
        "Positive findings per patient, validation split",
        "Number of answers other than no or not applicable, per patient (full history).",
    )
    left, right, top, bottom = 150, W - 100, 160, H - 150
    peak = max(hist.values())
    span = top_k + 1
    bw = (right - left) / span
    for tick in range(0, peak + 1, 1000):
        y = bottom - (bottom - top) * tick / (peak * 1.05)
        d.line([(left, y), (right, y)], fill=GRID, width=2)
        d.text(
            (left - 18 - text_w(d, f"{tick:,}", font(22)), y - 12),
            f"{tick:,}",
            fill=INK,
            font=font(22),
        )
    for k in range(span):
        n = hist.get(k, 0)
        x0 = left + k * bw
        y1 = bottom - (bottom - top) * n / (peak * 1.05)
        d.rectangle([x0 + 2, y1, x0 + bw - 2, bottom], fill=OTHER)
        if k % 5 == 0:
            d.text((x0 + bw / 2 - 8, bottom + 12), str(k), fill=INK, font=font(22))
    d.text(
        (left, bottom + 60), "Positive findings in the patient's history", fill=INK, font=font(26)
    )
    d.text((24, top - 40), "Patients", fill=INK, font=font(26))
    img.save(out)
    return [
        f"- Positive findings per patient: mean {sizes.mean():.1f}, median {sizes.median():.0f}, "
        f"minimum {sizes.min()}, maximum {sizes.max()}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/interim/ddxplus_chestpain_validate.parquet")
    parser.add_argument("--out-dir", default="docs/report/bmu")
    args = parser.parse_args()
    df = pd.read_parquet(args.parquet)
    if set(df["split"].unique()) != {"validate"}:
        print("refusing: this script reads the validation split only", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    sections = {
        "Patients per condition (`eda_class_balance.png`)": class_balance(
            df, out / "img" / "eda_class_balance.png"
        ),
        "Age by condition (`eda_age_by_condition.png`)": age_by_condition(
            df, out / "img" / "eda_age_by_condition.png"
        ),
        "Findings per patient (`eda_findings_per_patient.png`)": findings_per_patient(
            df, out / "img" / "eda_findings_per_patient.png"
        ),
    }
    sex = df["sex"].value_counts()
    sections["Sex"] = [f"- {k}: {int(v):,} patients ({v / len(df):.1%})" for k, v in sex.items()]
    lines = [
        "# Values behind the exploratory figures",
        "",
        f"Computed from `{args.parquet}` ({len(df):,} validation patients) by `scripts/make_report_eda_figures.py`.",
        "These are descriptive counts of the generated data, not experimental results; no interval applies.",
        "",
    ]
    for title, rows in sections.items():
        lines += [f"## {title}", "", *rows, ""]
    (out / "eda_values.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out / "eda_values.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
